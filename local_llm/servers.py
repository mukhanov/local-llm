"""Lifecycle of mlx_lm.server (:8080, OpenAI) and litellm (:4000, OpenAI +
Anthropic bridge): litellm config, start, readiness wait, warmup, stop.

The servers run as foreground children of our process (no nohup/daemons):
exiting the monitor or Ctrl-C kills everything at once (see cli.run_stack ->
finally)."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import hf, ui

MLX_LOG = "/tmp/mlx-server.log"
LITELLM_LOG = "/tmp/litellm.log"

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
    log = open(MLX_LOG, "wb")
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


def _write_litellm_config(cfg, model: str) -> None:
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

litellm_settings:
  drop_params: true

general:
  master_key: "{cfg.api_key}"
""")


def start_litellm(cfg, model: str) -> None:
    health = f"http://127.0.0.1:{cfg.litellm_port}/health/liveliness"
    if http_ok(health):
        ui.ok(f"litellm already running on :{cfg.litellm_port}"
              " (restarting for new config)")
        _pkill(f"litellm --config")
        time.sleep(2)
    ui.info(f"Starting litellm on :{cfg.litellm_port}")
    _write_litellm_config(cfg, model)
    env = dict(os.environ)
    # Anthropic bridge /v1/messages -> chat/completions (mlx has no /v1/responses)
    env["LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES"] = "1"
    log = open(LITELLM_LOG, "wb")
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
        for path in (MLX_LOG, LITELLM_LOG):
            lines = tail(path, 15)
            if lines:
                print(f"--- last lines of {path} ---")
                print("\n".join(lines))
        raise SystemExit(1) from None
