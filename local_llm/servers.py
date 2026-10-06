"""Lifecycle of mlx_lm.server (:8080, OpenAI) and litellm (:4000, OpenAI +
Anthropic bridge): litellm config, start, readiness wait, warmup, stop,
plus a watchdog that restarts mlx after a crash or the post-OOM zombie.

The servers run as foreground children of our process (no nohup/daemons):
exiting the monitor or Ctrl-C kills everything at once (see cli.run_stack ->
finally)."""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import hf, ui

MLX_LOG = "/tmp/mlx-server.log"
MLX_SMALL_LOG = "/tmp/mlx-small.log"
LITELLM_LOG = "/tmp/litellm.log"
WATCHDOG_LOG = "/tmp/mlx-watchdog.log"

# Popen handles of live children — killed in stop_children()
children: list[subprocess.Popen] = []


# our wrapper around mlx_lm.server (tok/s logging) or a raw server from an
# older local-llm — `stop` must kill either
_MLX_PATTERN = r"local_llm\.mlxwrap|mlx_lm\.server"


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def tail(path: str, n: int = 30):
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except OSError:
        return []


def _pkill(pattern: str) -> bool:
    return subprocess.run(["pkill", "-f", pattern],
                          capture_output=True).returncode == 0


def _open_log(path: str):
    """Keep one generation of history: the previous run's log moves to
    path.1 before the fresh one truncates it — otherwise every restart
    destroys the evidence of whatever killed the run before it."""
    try:
        os.replace(path, path + ".1")
    except OSError:
        pass  # no previous log (or unwritable dir) — not fatal
    return open(path, "wb")


def stop_all() -> None:
    """For `local-llm stop`: kill processes leaked from a previous run."""
    ui.info("Stopping servers")
    ui.ok("mlx_lm.server stopped" if _pkill(_MLX_PATTERN)
          else "mlx_lm.server not running")
    ui.ok("litellm stopped" if _pkill(f"litellm --config")
          else "litellm not running")


def stop_children() -> None:
    for p in children:
        if p.poll() is None:
            p.terminate()
    for p in children:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


# --- watchdog --------------------------------------------------------------------
#
# mlx dies in two ways: the process exits, or a Metal OOM kills the
# generation thread while the HTTP server lives on — the post-OOM zombie
# (seen 2026-10-05 23:58 with a 186k-token session): /v1/models keeps
# answering 200 and every completion 404s until restarted. The watchdog
# catches both and brings the model back without user intervention.

_watchdog_stop = threading.Event()


def _note(msg: str) -> None:
    """An event line in the watchdog's own log — the monitor tails it.
    NOT the mlx log: the server child holds that file at its own non-append
    offset and overwrites whatever anyone else appends after it (the
    13:19:35 restart note was lost exactly this way)."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(WATCHDOG_LOG, "a") as f:
            f.write(f"{stamp} - WATCHDOG - {msg}\n")
    except OSError:
        pass


def _probe_generation(port: int, model: str, timeout: float = 30.0) -> str:
    """'ok' | 'bad' | 'busy' — health is a real 1-token completion, not
    /v1/models (that one stays 200 in the zombie state). 'busy' = no
    answer in time: mid-prefill of a huge prompt, not proof of anything."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": model, "max_tokens": 1, "stream": False,
                         "messages": [{"role": "user", "content": "ping"}]}
                        ).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return "ok"
    except urllib.error.HTTPError:
        return "bad"     # 404/500 — the zombie signature
    except (OSError, urllib.error.URLError):
        return "busy"


def supervise(cfg, model: str, state: dict, interval: float = 45.0) -> None:
    """Start the watchdog thread. `state` = {"big": Popen, "small": Popen|None};
    on a dead process or 2 consecutive bad probes it restarts both mlx
    servers in place — litellm and the client configs keep pointing at the
    same ports, so clients just see a pause."""
    _watchdog_stop.clear()

    def restart(reason: str) -> None:
        _pkill(_MLX_PATTERN)   # takes the helper down too; it comes right back
        time.sleep(1)
        try:
            state["big"] = start_mlx(cfg, model)
            if state["small"] is not None:
                state["small"] = start_mlx_small(cfg)
            _note(f"mlx restarted ({reason}) — stack healthy again")
        except SystemExit as exc:
            _note(f"restart failed ({reason}): {exc}")

    def run():
        bad = 0
        while not _watchdog_stop.wait(interval):
            big = state.get("big")
            if big is None or big.poll() is not None:
                code = big.returncode if big is not None else "?"
                restart(f"process exited (code {code})")
                bad = 0
                continue
            if _probe_generation(cfg.mlx_port, model) == "bad":
                bad += 1
                if bad >= 2:
                    restart("generation thread dead — completions 404"
                            " (Metal OOM zombie)")
                    bad = 0
            else:
                bad = 0

    threading.Thread(target=run, daemon=True, name="mlx-watchdog").start()


def stop_watchdog() -> None:
    """Signal the watchdog to stand down before stop_children() — no
    restarts during shutdown."""
    _watchdog_stop.set()


def ensure_litellm() -> None:
    if shutil.which("litellm"):
        return
    ui.info("Installing litellm (uv tool)")
    subprocess.run(["uv", "tool", "install", "litellm"], check=True)


def _wait_ready(url: str, proc, timeout: int, log: str,
                what: str, step: int = 3) -> int:
    waited = 0
    while not http_ok(url):
        if proc.poll() is not None:
            ui.warn(f"{what} died:")
            print("\n".join(tail(log, 30)))
            raise SystemExit(1)
        if waited >= timeout:
            ui.warn(f"{what} did not become ready in {timeout}s")
            print("\n".join(tail(log, 30)))
            raise SystemExit(1)
        print(f"\r   loading… {waited}s", end="", flush=True)
        time.sleep(step)
        waited += step
    print(f"\r   ready in {waited}s        ")
    return waited


def _server_flags(python: str) -> set[str]:
    """Flags the installed mlx_lm.server actually accepts. The qwen4_exp PR
    fork dropped --kv-bits/--kv-group-size (its linear-attention cache can't
    be KV-quantized); passing them makes the server die on argparse."""
    try:
        out = subprocess.run([python, "-m", "mlx_lm.server", "--help"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return set()  # can't ask -> pass nothing optional
    # the usage block is ANSI-colored (flag names wrapped in escapes)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    return {w for line in plain.splitlines() for w in line.split()
            if w.startswith("--")}


def start_mlx(cfg, model: str) -> None:
    ui.info(f"Starting mlx_lm.server on :{cfg.mlx_port}")
    _pkill(_MLX_PATTERN)
    time.sleep(1)
    # by default mlx lives in ~/.ollmlx/venv (the launcher); under
    # pipx/uv-tool installs, run from the environment the package is in
    python = (str(cfg.venv_python) if cfg.venv_python.exists()
              else sys.executable)
    ctx = hf.model_ctx(cfg.hf_hub, model)
    supported = _server_flags(python)
    flags = []
    if cfg.kv_bits > 0:
        if {"--kv-bits", "--kv-group-size"} <= supported:
            flags += ["--kv-bits", str(cfg.kv_bits),
                      "--kv-group-size", str(cfg.kv_group_size)]
        else:
            ui.warn("KV-cache quantization not supported by this "
                    "mlx_lm.server — running without it")
    if cfg.prompt_cache_bytes > 0 and "--prompt-cache-bytes" in supported:
        flags += ["--prompt-cache-bytes", str(cfg.prompt_cache_bytes)]
    log = _open_log(MLX_LOG)
    env = dict(os.environ)
    # By the time we start mlx, the cache is complete (cli verified it against
    # the repo file list). Force offline: otherwise a partial cache would make
    # mlx start its own background download — plain DNS (no DoH pin), xet, no
    # token — which stalls for hours while /v1/models already answers 200.
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    if cfg.hf_token:
        env["HF_TOKEN"] = cfg.hf_token
    try:
        proc = subprocess.Popen(
            [python, "-m", "local_llm.mlxwrap",   # mlx_lm.server + tok/s log
             "--model", model, "--host", "127.0.0.1",
             "--port", str(cfg.mlx_port),
             "--max-tokens", str(min(ctx, 32768)), *flags],
            stdout=log, stderr=subprocess.STDOUT, env=env)
    finally:
        log.close()
    children.append(proc)
    _wait_ready(f"http://127.0.0.1:{cfg.mlx_port}/v1/models", proc,
                cfg.load_timeout, MLX_LOG, "mlx_lm.server")
    return proc


def start_mlx_small(cfg) -> None:
    """The tiny helper model on its own port: cheap classification/title
    calls (Claude Code's background requests have a short fixed timeout and
    would queue for minutes behind the big model's generation) plus a quick
    `ollmlx/small` for any client. Same flags as the big one are pointless
    here: no KV quant (nothing to save on ~0.4G weights), no prompt-cache
    cap (its caches are a few MB)."""
    ui.info(f"Starting helper model {cfg.small_model} on :{cfg.mlx_small_port}")
    # only stale instances of *this* model — start_mlx already swept the broad
    # pattern, and the main server must survive this pkill
    _pkill(f"local_llm\\.mlxwrap.*{re.escape(cfg.small_model)}")
    time.sleep(1)
    python = (str(cfg.venv_python) if cfg.venv_python.exists()
              else sys.executable)
    # thinking off by default: hybrid-Qwen3 would burn ~200 hidden tokens
    # pondering a one-word answer (0.9s); a request can still re-enable it
    # via chat_template_kwargs — per-request args override the CLI ones
    flags = (["--chat-template-args", '{"enable_thinking":false}']
             if "--chat-template-args" in _server_flags(python) else [])
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    log = _open_log(MLX_SMALL_LOG)
    try:
        proc = subprocess.Popen(
            [python, "-m", "local_llm.mlxwrap",
             "--model", cfg.small_model, "--host", "127.0.0.1",
             "--port", str(cfg.mlx_small_port),
             "--max-tokens", "4096", *flags],
            stdout=log, stderr=subprocess.STDOUT, env=env)
    finally:
        log.close()
    children.append(proc)
    _wait_ready(f"http://127.0.0.1:{cfg.mlx_small_port}/v1/models", proc,
                120, MLX_SMALL_LOG, "helper model")
    return proc


def _write_litellm_config(cfg, model: str, small: bool = True) -> None:
    small_block = (f"""\
  - model_name: "small"          # tiny helper: background calls / titles
    litellm_params:
      model: "openai/{cfg.small_model}"
      api_base: "http://127.0.0.1:{cfg.mlx_small_port}/v1"
      api_key: "none"
      request_timeout: 120
""" if small else "")
    cfg.litellm_cfg.write_text(f"""\
model_list:
  - model_name: "{model}"
    litellm_params:
      model: "openai/{model}"
      api_base: "http://127.0.0.1:{cfg.mlx_port}/v1"
      api_key: "none"
      request_timeout: 600
  - model_name: "local"          # slash-free alias: pi/omp send the bare catalog id to the API
    litellm_params:
      model: "openai/{model}"
      api_base: "http://127.0.0.1:{cfg.mlx_port}/v1"
      api_key: "none"
      request_timeout: 600
{small_block}
litellm_settings:
  drop_params: true

general:
  master_key: "{cfg.api_key}"
""")


def start_litellm(cfg, model: str, small: bool = True) -> None:
    health = f"http://127.0.0.1:{cfg.litellm_port}/health/liveliness"
    if http_ok(health):
        ui.ok(f"litellm already running on :{cfg.litellm_port}"
              " (restarting for new config)")
        _pkill(f"litellm --config")
        time.sleep(2)
    ui.info(f"Starting litellm on :{cfg.litellm_port}")
    _write_litellm_config(cfg, model, small)
    env = dict(os.environ)
    # Anthropic bridge /v1/messages -> chat/completions (mlx has no /v1/responses)
    env["LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES"] = "1"
    log = _open_log(LITELLM_LOG)
    try:
        proc = subprocess.Popen(
            ["litellm", "--config", str(cfg.litellm_cfg),
             "--host", "127.0.0.1", "--port", str(cfg.litellm_port)],
            stdout=log, stderr=subprocess.STDOUT, env=env)
    finally:
        log.close()
    children.append(proc)
    waited = _wait_ready(health, proc, 60, LITELLM_LOG, "litellm", step=1)
    ui.ok(f"litellm ready ({waited}s)")


def warmup(cfg) -> None:
    """End-to-end warmup: litellm (via the `local` alias, like the clients) -> mlx.

    Fatal on failure: /v1/models answers 200 before the weights are loaded,
    so this is the only real readiness gate — a stack that fails here would
    only serve errors (cli's finally still stops the servers)."""
    ui.info("Warmup request")
    req = urllib.request.Request(
        f"http://127.0.0.1:{cfg.litellm_port}/v1/chat/completions",
        data=json.dumps({"model": "local", "max_tokens": 8,
                         "messages": [{"role": "user", "content": "Say OK"}]}
                        ).encode(),
        headers={"Authorization": f"Bearer {cfg.api_key}",
                 "content-type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(300, cfg.load_timeout)) as r:
            r.read()
        ui.ok("e2e OK")
    except (OSError, urllib.error.URLError) as exc:
        ui.warn(f"warmup failed ({exc})")
        ui.warn("if it repeats: memory pressure can kill the generation "
                "thread (Metal OOM -> 404s until restart) — free RAM/swap, "
                "`local-llm stop`, retry; the previous run's log is "
                f"{MLX_LOG}.1")
        for path in (MLX_LOG, LITELLM_LOG):
            lines = tail(path, 15)
            if lines:
                print(f"--- last lines of {path} ---")
                print("\n".join(lines))
        raise SystemExit(1) from None
