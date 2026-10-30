"""Commands and orchestration: run / stop / rm / list / help."""
import os
import queue
import signal
import sys
import threading
from pathlib import Path

import psutil

from . import applog, clients, hf, llmfit, models, monitor, picker, servers, startup, ui
from .applog import log
from .config import Config

USAGE = """\
local-llm — bring up a local model + an API for claude / pi / omp / opencode.

What it does:
  1. Shows local/recommended models (TUI: scroll, search, load-more from
     HuggingFace, sorting by hardware fit, g toggles MLX / GGUF), downloads
     the chosen one.
  2. Starts the model server (mlx_lm.server for MLX, llama-server for GGUF;
     OpenAI API), a tiny helper model (:8081, alias ollmlx/small — Claude
     Code's background calls like title generation, any client's quick
     asks) and a litellm proxy (OpenAI /v1/chat/completions + Anthropic
     /v1/messages for Claude Code) — as foreground children, no daemons:
     exiting the monitor or Ctrl-C stops everything at once.
  3. Registers the model (+ the fixed alias ollmlx/local) in
     ~/.pi/agent/models.json, ~/.omp/agent/models.json,
     ~/.config/opencode/opencode.json and generates
     ~/.ollmlx/claude-local.json.
  4. Shows an htop-style TUI monitor: per-core CPU + a history graph,
     RAM/swap, server status and the client launch commands.

Usage (symlink: ~/bin/local-llm):
  local-llm                # interactive TUI model picker + monitor
  local-llm <org/model>    # no questions asked (MLX or GGUF repo)
  local-llm <org/model> <quant>   # GGUF: quant file to run
                           # (Q4_K_M, iq4_xs, …; default: best for free RAM)
  local-llm list           # show downloaded models (with sizes)
  local-llm serve <org/model> [quant]  # headless: no TUI, block until SIGTERM
  local-llm rm [org/repo…] # delete downloaded models from disk (no args — menu)
  local-llm token [hf_…]   # HF token: show / save / --clear (or env HF_TOKEN)

Env: MLX_KV_BITS=8 (KV-cache quantization, 0=off), MLX_PROMPT_CACHE_BYTES (0=off),
     MLX_MAX_CTX=110000 (hard context ceiling for clients + server refusal),
     MLX_PORT, LITELLM_PORT, LOAD_TIMEOUT, OLLMLX_HOME, HF_HUB_CACHE,
     OLLMLX_SMALL_MODEL / MLX_SMALL_PORT (tiny helper model for fast calls),
     OLLMLX_LLAMA_SERVER (path to a llama-server binary; default: managed
     download into ~/.ollmlx/llama.cpp for GGUF models)"""


def main(argv: list[str] | None = None) -> int:
    if argv is None:   # the console script calls main() with no args
        argv = sys.argv[1:]
    applog.setup()   # first thing: a crash anywhere must already be logged
    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    interactive = sys.stdout.isatty() and sys.stdin.isatty()
    if interactive:
        # the tab title is owned by the whole app run: set it the moment we
        # start, hand it back exactly once at exit
        ui.push_title()
        ui.set_title("🟢 ollmlx")
    try:
        return _dispatch(argv, interactive)
    finally:
        if interactive:
            ui.pop_title()


def _dispatch(argv: list[str], interactive: bool) -> int:
    cfg = Config.from_env()
    if not argv:
        model = picker.pick_model(cfg)
        if model is None:
            ui.warn("selection cancelled — exiting")
            return 0
        if interactive:
            return run_stack_dialog(cfg, model)
        return run_stack(cfg, model)
    cmd, rest = argv[0], argv[1:]
    if cmd == "stop":
        servers.stop_all(cfg.mlx_port)
        return 0
    if cmd == "token":
        return cmd_token(cfg, rest)
    if cmd in ("rm", "remove", "del"):
        return cmd_rm(cfg, rest)
    if cmd in ("list", "ls"):
        return cmd_list(cfg)
    if cmd == "serve":
        if not rest or "/" not in rest[0]:
            raise SystemExit("   usage: local-llm serve <org/model> [quant]")
        return run_serve(cfg, rest[0], rest[1] if len(rest) > 1 else None)
    if "/" in cmd:
        # second word (when given) is the GGUF quant: file name, quant name
        # or any substring of it — run_stack sorts MLX vs GGUF out itself
        return run_stack(cfg, cmd, quant=rest[0] if rest else None)
    ui.warn(f"'{cmd}' doesn't look like org/model, opening the picker")
    model = picker.pick_model(cfg)
    if model is None:
        ui.warn("selection cancelled — exiting")
        return 0
    return run_stack(cfg, model)


def _pick_gguf_group(model: str, groups, quant: str | None):
    """(label, [shard names]) of the quant to serve: the user's substring
    (Q4_K_M, q4, full file name), else the largest that fits free RAM with
    headroom — clamping to the smallest beats refusing to start."""
    if quant:
        q = quant.lower()
        for label, _total, names in groups:
            if q in label.lower():
                return label, names
        known = ", ".join(g[0] for g in groups[:8])
        raise SystemExit(f"   no GGUF file matching '{quant}' in {model};"
                         f" available: {known}")
    # 15% weights headroom + 2G floor for KV/runtime, like the crash guard
    budget = psutil.virtual_memory().available - 2 * 1024 ** 3
    for label, total, names in groups:          # largest first
        if total * 1.15 <= budget:
            return label, names
    ui.warn("no GGUF quant fits in free RAM with headroom —"
            " taking the smallest")
    return groups[-1][0], groups[-1][2]


def _gguf_prepare(cfg: Config, model: str, quant: str | None = None):
    """Local path of the GGUF file to serve, downloading just the chosen
    quant when needed. None — not a GGUF model (the MLX path continues).

    Local snapshot first (works offline, and the run before this one
    already chose a quant); the repo listing only when the user asks for
    a quant we don't have yet, or nothing local says GGUF."""
    local = hf.local_gguf_files(cfg.hf_hub, model)
    sizes = {name: size for name, size in local}
    groups = hf.gguf_groups(local)
    need_remote = (not groups
                   or not any(hf.gguf_set_complete(g[2]) for g in groups)
                   or (quant and not any(quant.lower() in g[0].lower()
                                         for g in groups)))
    if need_remote:
        remote = hf.repo_gguf_files(model, cfg.hf_token)
        if remote is None:
            if quant and not groups:
                raise SystemExit(
                    f"   HF API unreachable — can't list {model}'s GGUF"
                    " files; retry, or pass a quant that's already"
                    " downloaded")
        else:
            sizes.update({n: s for n, s in remote if n not in sizes})
            groups = hf.gguf_groups(list(sizes.items()))
    if not any(hf.gguf_set_complete(g[2]) for g in groups):
        # shard2 without shard1 (fetch in flight or dead) must not reach
        # llama-server — it refuses anything but the first split
        raise SystemExit(f"   {model}: no complete GGUF shard set locally"
                         " and the HF API is unreachable — retry when"
                         " online; a running fetch keeps its progress")
    if not groups:
        if quant:
            ui.warn(f"'{quant}' ignored — {model} has no GGUF files"
                    " (MLX model)")
        return None
    label, names = _pick_gguf_group(model, groups, quant)
    have = {name for name, _size in local}   # local lists base names
    if {n.rsplit("/", 1)[-1] for n in names} <= have:
        ui.ok(f"GGUF {label} already on disk")
    else:
        ui.info(f"Downloading GGUF {label} (via DoH-pinned DNS)")
        hf.download(model, cfg.hf_token, cfg.hf_hub, allow_patterns=names)
    path = hf.gguf_local_path(cfg.hf_hub, model, label)
    if path is None:
        raise SystemExit(f"   {model}: {label} not in the snapshot after"
                         " download — see the log above")
    return path


def run_stack(cfg: Config, model: str, launch_monitor: bool = True,
              quant: str | None = None, meta: dict | None = None) -> int:
    # a port collision is a config error — fail before touching anything
    # (the second server would die on bind and _wait_ready would kill the run)
    if (cfg.mlx_port == cfg.mlx_small_port
            or cfg.litellm_port in (cfg.mlx_port, cfg.mlx_small_port)):
        raise SystemExit(
            f"   port collision: mlx={cfg.mlx_port},"
            f" small={cfg.mlx_small_port}, litellm={cfg.litellm_port}"
            " — all three must differ (MLX_PORT / MLX_SMALL_PORT / LITELLM_PORT)")
    # GGUF repos run llama-server in the mlx slot; everything above the
    # port (litellm, clients, monitor) doesn't care which one it is
    gguf_path = _gguf_prepare(cfg, model, quant)
    backend = "llama" if gguf_path else "mlx"
    if meta is not None:      # the startup dialog reads it after the boot
        meta["backend"] = backend
    servers.ensure_litellm()
    log.info("stack start: model=%s backend=%s mlx=:%d small=:%d litellm=:%d",
             model, backend, cfg.mlx_port, cfg.mlx_small_port,
             cfg.litellm_port)
    ui.info(f"Model: {model}"
            + (f" · {gguf_path.name} (GGUF)" if gguf_path else ""))
    if gguf_path:
        # the quant file is the whole download; _gguf_prepare fetched it
        weights = gguf_path.stat().st_size
    else:
        missing = hf.missing_files(cfg.hf_hub, model, cfg.hf_token)
        if missing is None:
            # HF API unreachable: can't verify completeness, trust the local
            # snapshot (all links resolve = best we can tell without network)
            if hf.is_cached(cfg.hf_hub, model):
                ui.ok("already downloaded (completeness not verified —"
                      " HF API unreachable)")
            else:
                ui.info("Downloading (via DoH-pinned DNS)")
                hf.download(model, cfg.hf_token, cfg.hf_hub)
                ui.ok("download complete")
        elif missing:
            # partial model: some shards done, hub reuses finished blobs and
            # fetches only the rest
            ui.warn(f"incomplete download: {len(missing)} file(s) missing"
                    f" (first: {missing[0]}) — fetching the rest")
            hf.download(model, cfg.hf_token, cfg.hf_hub)
            missing = hf.missing_files(cfg.hf_hub, model, cfg.hf_token)
            if missing:
                raise SystemExit(f"   still incomplete after download:"
                                 f" {len(missing)} file(s) missing,"
                                 f" e.g. {missing[0]}")
            ui.ok("download complete")
        else:
            ui.ok("already downloaded")
        weights = hf.model_disk_bytes(cfg.hf_hub, model)

    # tiny helper model for fast/background calls — if it can't be fetched
    # (HF unreachable, cache cold) we degrade instead of blocking the stack:
    # everything then runs on the big model like before
    small_ok = True
    if not hf.is_cached(cfg.hf_hub, cfg.small_model):
        ui.info(f"Downloading helper model {cfg.small_model} (~0.5GB)")
        try:
            hf.download(cfg.small_model, cfg.hf_token, cfg.hf_hub)
        except (Exception, SystemExit):  # noqa: BLE001 — degrade, not die
            pass
        small_ok = hf.is_cached(cfg.hf_hub, cfg.small_model)
        if not small_ok:
            ui.warn("helper model unavailable — continuing without it"
                    " (fast/background calls will use the main model)")

    # memory-pressure guard: weights + heavy swap is exactly how a session
    # dies mid-work (Metal OOM kills the generation thread -> 404s until
    # restart). Warn before loading, not after.
    avail = psutil.virtual_memory().available
    if weights and avail < weights + 4 * 1024**3:
        ui.warn(f"only {ui.human_bytes(avail)} RAM available, the model's"
                f" weights are {ui.human_bytes(weights)} — under this"
                " pressure a Metal OOM kills the server until restart")
        if launch_monitor and sys.stdin.isatty():
            try:
                answer = input("?# Start anyway? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if not answer.startswith("y"):
                ui.ok("cancelled — free memory first (`local-llm stop` kills"
                      " leftovers)")
                return 0
        else:
            ui.warn("non-interactive start — continuing")

    try:
        state = {"big": None, "small": None, "backend": backend,
                 "gguf_path": str(gguf_path) if gguf_path else None}
        max_input = hf.safe_context(cfg.hf_hub, model, cfg.prompt_cache_bytes,
                                    cfg.max_ctx)
        if max_input < hf.model_ctx(cfg.hf_hub, model):
            ui.info(f"model context capped at {max_input} tokens"
                    " (prompt-cache budget) — oversized requests are refused,"
                    " clients compact before it OOMs")
        state["big"] = (servers.start_llama(cfg, state["gguf_path"], model,
                                            max_input) if gguf_path
                        else servers.start_mlx(cfg, model, max_input))
        if small_ok:
            state["small"] = servers.start_mlx_small(cfg)
        servers.start_litellm(cfg, model, small_ok)
        clients.write_client_configs(cfg, model, small_ok, max_input)
        servers.warmup(cfg)
        log.info("stack ready: %s (max_input=%s, helper=%s)", model,
                 max_input, cfg.small_model if small_ok else "off")
        servers.supervise(cfg, model, state, max_input=max_input)
        print()
        ui.info(f"Ready. OpenAI and Anthropic APIs on"
                f" http://127.0.0.1:{cfg.litellm_port}")
        print(f"  pi:      pi --model ollmlx/local   ({model})")
        print(f"  omp:     omp --model ollmlx/local   ({model})")
        print(f"  opencode: opencode --model ollmlx/local   ({model})")
        print(f"  claude:  claude --settings {cfg.claude_cfg}")
        if small_ok:
            print(f"  fast/bg: ollmlx/small  ({cfg.small_model} on"
                  f" :{cfg.mlx_small_port} — titles, quick asks)")
        print()
        if not launch_monitor:
            # handoff: the caller keeps the stack running and opens the UI
            return cfg.mlx_small_port if small_ok else 0
        ui.info("System monitor (q or Ctrl-C stops everything)")
        home = str(Path.home())
        monitor.run(model, cfg.mlx_port, cfg.litellm_port,
                    str(cfg.claude_cfg).replace(home, "~"),
                    cfg.mlx_small_port if small_ok else 0, backend)
    finally:
        if launch_monitor:
            servers.stop_watchdog()
            servers.stop_children()
    print()
    log.info("stack stopped")
    ui.ok(f"stopped — {'llama-server' if backend == 'llama' else 'mlx_lm.server'}"
          " and litellm are down")
    return 0



def run_serve(cfg: Config, model: str, quant: str | None = None) -> int:
    """Headless stack for bots/cron: same boot as the interactive run
    (quant pick, servers, watchdog) but no TUI — blocks until SIGTERM or
    SIGINT, then stops the children cleanly. Port collisions and other
    startup failures exit non-zero without killing a foreign stack."""
    stop = threading.Event()

    def _signal(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT, _signal)
    try:
        run_stack(cfg, model, launch_monitor=False, quant=quant)
        log.info("serve: stack ready (%s) — waiting for SIGTERM/SIGINT", model)
        print(f"serve: ready — model {model}; OpenAI/Anthropic on"
              f" http://127.0.0.1:{cfg.litellm_port}; stop with SIGTERM",
              flush=True)
    except BaseException as exc:  # startup failed: nothing to supervise
        servers.stop_watchdog()
        servers.stop_children()
        if isinstance(exc, SystemExit) and isinstance(exc.code, int):
            return exc.code
        raise
    stop.wait()
    log.info("serve: signal received — stopping the stack")
    print("serve: stopping", flush=True)
    servers.stop_watchdog()
    servers.stop_children()
    return 0

def run_stack_dialog(cfg: Config, model: str) -> int:
    """Picker -> fullscreen startup dialog -> straight into the monitor.
    The stack boots in a worker thread with its output fed to the dialog;
    on success the dialog hands the terminal to the monitor, on failure it
    waits for a keypress over the error."""
    lines: "queue.Queue[str]" = queue.Queue()
    done = threading.Event()
    meta: dict = {}      # run_stack reports which backend it booted
    result: dict = {}    # exit code / error / handoff from the worker

    def work():
        old = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = startup.QueueWriter(lines)
        try:
            result["handoff"] = run_stack(cfg, model, launch_monitor=False,
                                          meta=meta)
            result["code"] = 0
        except SystemExit as exc:
            result["code"] = exc.code if isinstance(exc.code, int) else 1
            if exc.code and not isinstance(exc.code, int):
                result["error"] = str(exc.code)
            log.error("stack start failed: %s", exc.code)
        except KeyboardInterrupt:
            result["code"] = 130
            log.info("stack start interrupted")
        except BaseException as exc:  # noqa: BLE001 — surfaced in the dialog
            result["code"] = 1
            result["error"] = f"{type(exc).__name__}: {exc}"
            log.error("stack start crashed", exc_info=True)
        finally:
            sys.stdout, sys.stderr = old
            if result.get("code"):
                # failed — nothing to hand off, take the servers down
                servers.stop_watchdog()
                servers.stop_children()
            done.set()

    worker = threading.Thread(target=work, name="stack-start", daemon=True)
    worker.start()
    # the tab shows the boot right away; the monitor refines it per frame
    ui.set_title(f"🟡 {model.split('/')[-1]}")
    try:
        startup.dialog(model, lines, done, result)
    except KeyboardInterrupt:
        ui.warn("interrupted — stopping the stack")
        log.info("stack stopped (interrupted during startup)")
        servers.stop_watchdog()
        servers.stop_children()
        worker.join(timeout=15)
        return 130
    worker.join(timeout=15)
    code = result.get("code", 1)
    if code:
        if result.get("error"):
            ui.warn(result["error"])
        raise SystemExit(code)
    try:
        home = str(Path.home())
        monitor.run(model, cfg.mlx_port, cfg.litellm_port,
                    str(cfg.claude_cfg).replace(home, "~"),
                    result.get("handoff", 0), meta.get("backend", "mlx"))
    finally:
        servers.stop_watchdog()
        servers.stop_children()
    log.info("stack stopped")
    print()
    what = ("llama-server" if meta.get("backend") == "llama"
            else "mlx_lm.server")
    ui.ok(f"stopped — {what} and litellm are down")
    return 0


def cmd_token(cfg: Config, args: list[str]) -> int:
    """Show/save/clear the HF token. Precedence: env HF_TOKEN > file."""
    token_file = cfg.ollmlx_home / "hf-token"
    if args and args[0] == "--clear":
        if token_file.exists():
            token_file.unlink()
            ui.ok("token deleted")
        else:
            ui.ok("there was no saved token")
        return 0
    if args:
        cfg.ollmlx_home.mkdir(parents=True, exist_ok=True)
        token_file.write_text(args[0].strip() + "\n")
        token_file.chmod(0o600)
        ui.ok(f"token saved to {token_file} (chmod 600)")
        return 0
    token = cfg.hf_token
    if not token:
        ui.warn("no token set: local-llm token <hf_...> or env HF_TOKEN")
        return 0
    from_env = bool(os.environ.get("HF_TOKEN") or
                    os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    src = "env HF_TOKEN" if from_env else str(token_file)
    masked = f"{token[:5]}…{token[-4:]}" if len(token) > 12 else "…"
    ui.info(f"HF token: {masked} (source: {src})")
    return 0


def cmd_list(cfg: Config) -> int:
    installed = hf.scan_installed(cfg.hf_hub)
    if not installed:
        ui.warn(f"No downloaded LLM models ({cfg.hf_hub})")
        return 0
    lf = llmfit.load(cfg.ollmlx_home) or {}
    sys_ram = ui.total_ram_gb()
    ui.info("💿 Installed models:")
    for rid, size, dling in installed:
        mark = "⬇ " if dling else "  "
        e = lf.get(rid)
        ram = models.ram_estimate_gb(rid, e, size=size)
        moe = " · MoE" if models.parse_model(rid)[4] else ""
        ctx = llmfit.ctx_str(e) if e else None
        lfsc = llmfit.score(e, sys_ram) if e else None
        extra = (f" · ctx {ctx}" if ctx else "")
        extra += f" · ⚡{lfsc['score']:.0f}" if lfsc else ""
        extra += " · 🔒 helper" if rid == cfg.small_model else ""
        warn = "  ⚠won't fit" if ram and ram > sys_ram else ""
        ram_txt = f"~{ram}GB RAM" if ram else "RAM ?"
        print(f"  {mark}{rid:<62} {size:>6}  {ram_txt}{moe}{extra}{warn}")
    return 0


def cmd_rm(cfg: Config, args: list[str]) -> int:
    installed = hf.scan_installed(cfg.hf_hub)
    ids = [t[0] for t in installed]
    if not installed:
        ui.warn(f"No downloaded LLM models ({cfg.hf_hub})")
        return 0

    if args:
        targets = list(args)
    else:
        ui.info("💿 Installed models:")
        for n, (rid, size, dling) in enumerate(installed, 1):
            mark = "⬇ " if dling else "  "
            print(f"  {n:2d}) {mark}{rid:<60} {size}")
        try:
            raw = input("?# Delete (numbers or org/repo, space-separated,"
                        " empty — cancel): ").strip()
        except EOFError:
            raw = ""
        if not raw:
            ui.ok("cancelled")
            return 0
        targets = []
        for t in raw.split():
            if t.isdigit():
                idx = int(t) - 1
                if 0 <= idx < len(ids):
                    targets.append(ids[idx])
                else:
                    ui.warn(f"no item #{t}")
            elif "/" in t:
                targets.append(t)
            else:
                ui.warn(f"can't parse: {t} (need a number or org/repo)")

    if not targets:
        ui.ok("nothing selected")
        return 0

    for d in targets:
        if "/" not in d:
            ui.warn(f"{d} — expected org/repo")
            continue
        if d == cfg.small_model:
            ui.warn(f"{d} — helper model for fast/background calls;"
                    " point OLLMLX_SMALL_MODEL elsewhere first")
            continue
        if not hf.model_dir(cfg.hf_hub, d).is_dir():
            ui.warn(f"{d} — not found in cache")
            continue
        size = next((s for i, s, _ in installed if i == d), "?")
        try:
            answer = input(f"?# Delete {d} ({size})? [y/N] ").strip()
        except EOFError:
            ui.ok("cancelled")
            return 0
        if answer.lower().startswith("y"):
            try:
                hf.delete_model(cfg.hf_hub, d)
                ui.ok(f"deleted: {d} (freed {size})")
            except OSError as exc:
                ui.warn(f"could not delete: {exc}")
        else:
            ui.ok(f"skipped: {d}")
    return 0


if __name__ == "__main__":  # python -m local_llm.cli
    sys.exit(main(sys.argv[1:]))
