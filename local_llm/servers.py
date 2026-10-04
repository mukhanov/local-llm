"""Жизненный цикл mlx_lm.server (:8080, OpenAI) и litellm (:4000, OpenAI +
Anthropic-мост): конфиг litellm, старт, ожидание готовности, прогрев, стоп.

Серверы — foreground-дети нашего процесса (без nohup/демонов): выход из
монитора или Ctrl-C убивает всё разом (см. cli.run_stack -> finally).
"""
import json
import os
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

# Popen'ы живых детей — убиваются в stop_children()
children: list[subprocess.Popen] = []

_MLX_PATTERN = r"mlx_lm\.server"


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
    """Для `local-llm stop`: прибить утёкшие с прошлого запуска процессы."""
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


def _wait_ready(url: str, proc: subprocess.Popen, timeout: int, log: str,
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


def start_mlx(cfg, model: str) -> None:
    ui.info(f"Starting mlx_lm.server on :{cfg.mlx_port}")
    _pkill(_MLX_PATTERN)
    time.sleep(1)
    # из коробки mlx живёт в ~/.ollmlx/venv (лаунчер); при установке через
    # pipx/uv-tool запускаемся из того окружения, где стоит пакет
    python = (str(cfg.venv_python) if cfg.venv_python.exists()
              else sys.executable)
    ctx = hf.model_ctx(cfg.hf_hub, model)
    flags = []
    if cfg.kv_bits > 0:
        flags += ["--kv-bits", str(cfg.kv_bits),
                  "--kv-group-size", str(cfg.kv_group_size)]
    if cfg.prompt_cache_bytes > 0:
        flags += ["--prompt-cache-bytes", str(cfg.prompt_cache_bytes)]
    log = open(MLX_LOG, "wb")
    try:
        proc = subprocess.Popen(
            [python, "-m", "mlx_lm.server",
             "--model", model, "--host", "127.0.0.1",
             "--port", str(cfg.mlx_port),
             "--max-tokens", str(min(ctx, 32768)), *flags],
            stdout=log, stderr=subprocess.STDOUT)
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
  - model_name: "local"          # алиас без слэша: pi/omp шлют в API голый id из каталога
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
    # Anthropic-мост /v1/messages -> chat/completions (mlx не умеет /v1/responses)
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
    """Сквозной прогрев: litellm (через алиас local, как клиенты) -> mlx."""
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
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        ui.ok("e2e OK")
    except (OSError, urllib.error.URLError) as exc:
        ui.warn(f"warmup failed ({exc}) — см. {LITELLM_LOG} и {MLX_LOG}")
