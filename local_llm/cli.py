"""Commands and orchestration: run / stop / rm / list / help."""
import os
import sys
from pathlib import Path

import psutil

from . import clients, hf, models, monitor, picker, servers, ui
from .config import Config

USAGE = """\
local-llm — bring up a local MLX model + an API for claude / pi / omp.

What it does:
  1. Shows local/recommended models (TUI: scroll, search, load-more from
     HuggingFace, sorting by hardware fit), downloads the chosen one.
  2. Starts mlx_lm.server (OpenAI API), a tiny helper model (:8081, alias
     ollmlx/small — Claude Code's background calls like title generation,
     any client's quick asks) and a litellm proxy (OpenAI
     /v1/chat/completions + Anthropic /v1/messages for Claude Code) — as
     foreground children, no daemons: exiting the monitor or Ctrl-C stops
     everything at once.
  3. Registers the model (+ the fixed alias ollmlx/local) in
     ~/.pi/agent/models.json, ~/.omp/agent/models.json and generates
     ~/.ollmlx/claude-local.json.
  4. Shows an htop-style TUI monitor: per-core CPU + a history graph,
     RAM/swap, server status and the pi/omp/claude launch commands.

Usage (symlink: ~/bin/local-llm):
  local-llm                # interactive TUI model picker + monitor
  local-llm <org/model>    # no questions asked
  local-llm list           # show downloaded models (with sizes)
  local-llm stop           # kill processes leaked from a previous run
  local-llm rm [org/repo…] # delete downloaded models from disk (no args — menu)
  local-llm token [hf_…]   # HF token: show / save / --clear (or env HF_TOKEN)

Env: MLX_KV_BITS=8 (KV-cache quantization, 0=off), MLX_PROMPT_CACHE_BYTES (0=off),
     MLX_PORT, LITELLM_PORT, LOAD_TIMEOUT, OLLMLX_HOME, HF_HUB_CACHE,
     OLLMLX_SMALL_MODEL / MLX_SMALL_PORT (tiny helper model for fast calls)"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cfg = Config.from_env()
    if not argv:
        model = picker.pick_model(cfg)
        if model is None:
            ui.warn("selection cancelled — exiting")
            return 0
        return run_stack(cfg, model)
    cmd, rest = argv[0], argv[1:]
    if cmd == "stop":
        servers.stop_all()
        return 0
    if cmd == "token":
        return cmd_token(cfg, rest)
    if cmd in ("rm", "remove", "del"):
        return cmd_rm(cfg, rest)
    if cmd in ("list", "ls"):
        return cmd_list(cfg)
    if "/" in cmd:
        return run_stack(cfg, cmd)
    ui.warn(f"'{cmd}' doesn't look like org/model, opening the picker")
    model = picker.pick_model(cfg)
    if model is None:
        ui.warn("selection cancelled — exiting")
        return 0
    return run_stack(cfg, model)


def run_stack(cfg: Config, model: str) -> int:
    # a port collision is a config error — fail before touching anything
    # (the second server would die on bind and _wait_ready would kill the run)
    if (cfg.mlx_port == cfg.mlx_small_port
            or cfg.litellm_port in (cfg.mlx_port, cfg.mlx_small_port)):
        raise SystemExit(
            f"   port collision: mlx={cfg.mlx_port},"
            f" small={cfg.mlx_small_port}, litellm={cfg.litellm_port}"
            " — all three must differ (MLX_PORT / MLX_SMALL_PORT / LITELLM_PORT)")
    servers.ensure_litellm()
    ui.info(f"Model: {model}")
    missing = hf.missing_files(cfg.hf_hub, model, cfg.hf_token)
    if missing is None:
        # HF API unreachable: can't verify completeness, trust the local
        # snapshot (all links resolve = best we can tell without the network)
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
                             f" {len(missing)} file(s) missing, e.g. {missing[0]}")
        ui.ok("download complete")
    else:
        ui.ok("already downloaded")

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
    weights = hf.model_disk_bytes(cfg.hf_hub, model)
    avail = psutil.virtual_memory().available
    if weights and avail < weights + 4 * 1024**3:
        ui.warn(f"only {ui.human_bytes(avail)} RAM available, the model's"
                f" weights are {ui.human_bytes(weights)} — under this"
                " pressure a Metal OOM kills the server until restart")
        if sys.stdin.isatty():
            try:
                answer = input("?# Start anyway? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if not answer.startswith("y"):
                ui.ok("cancelled — free memory first (`local-llm stop` kills"
                      " leftovers)")
                return 0
        else:
            ui.warn("non-interactive — starting anyway")

    try:
        state = {"big": None, "small": None}
        state["big"] = servers.start_mlx(cfg, model)
        if small_ok:
            state["small"] = servers.start_mlx_small(cfg)
        servers.start_litellm(cfg, model, small_ok)
        clients.write_client_configs(cfg, model, small_ok)
        servers.warmup(cfg)
        servers.supervise(cfg, model, state)
        print()
        ui.info(f"Ready. OpenAI and Anthropic APIs on"
                f" http://127.0.0.1:{cfg.litellm_port}")
        print(f"  pi:      pi --model ollmlx/local   ({model})")
        print(f"  omp:     omp --model ollmlx/local   ({model})")
        print(f"  claude:  claude --settings {cfg.claude_cfg}")
        if small_ok:
            print(f"  fast/bg: ollmlx/small  ({cfg.small_model} on"
                  f" :{cfg.mlx_small_port} — titles, quick asks)")
        print()
        ui.info("System monitor (q or Ctrl-C stops everything)")
        home = str(Path.home())
        monitor.run(model, cfg.mlx_port, cfg.litellm_port,
                    str(cfg.claude_cfg).replace(home, "~"),
                    cfg.mlx_small_port if small_ok else 0)
    finally:
        servers.stop_watchdog()
        servers.stop_children()
    print()
    ui.ok("stopped — mlx_lm.server and litellm are down")
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
    ui.info("💿 Installed models:")
    for rid, size, dling in installed:
        mark = "⬇ " if dling else "  "
        ram = models.ram_estimate_gb(rid, size=size)
        ram_txt = f"~{ram}GB RAM" if ram else "RAM ?"
        print(f"  {mark}{rid:<62} {size}  {ram_txt}")
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
