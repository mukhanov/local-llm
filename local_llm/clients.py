"""Конфиги клиентов: pi/omp (models.json, провайдер ollmlx) и claude
(claude-local.json с env на litellm). Клиенты всегда запускаются одинаково."""
import json
from pathlib import Path

from . import hf, ui


def _entry(mid: str, name: str, ctx: int, reasoning: bool) -> dict:
    return {
        "id": mid,
        "name": name,
        "reasoning": reasoning,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": ctx,
        "maxTokens": min(ctx, 32768),
    }


def write_client_configs(cfg, model: str) -> None:
    ctx = hf.model_ctx(cfg.hf_hub, model)
    reasoning = "Qwen3" in model
    ui.info("Writing client configs")

    for f in (Path("~/.pi/agent/models.json").expanduser(),
              Path("~/.omp/agent/models.json").expanduser()):
        f.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if f.exists():
            try:
                data = json.loads(f.read_text())
            except Exception:
                data = {}
        providers = data.setdefault("providers", {})
        providers["ollmlx"] = {
            "baseUrl": f"http://127.0.0.1:{cfg.litellm_port}/v1",
            "api": "openai-completions",
            "apiKey": cfg.api_key,
            "models": [
                _entry(model, f"{model.split('/')[-1]} (local MLX)", ctx, reasoning),
                _entry("local", "Local MLX (whatever is running)", ctx, reasoning),
            ],
        }
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        ui.ok(str(f))

    # omp кэширует каталог в models.yml после первого чтения models.json — сбрасываем
    omp_cache = Path("~/.omp/agent/models.yml").expanduser()
    if omp_cache.exists():
        omp_cache.unlink()
        ui.ok(f"{omp_cache} (cached)")

    home = str(Path.home())
    cfg.claude_cfg.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{cfg.litellm_port}",
            "ANTHROPIC_AUTH_TOKEN": cfg.api_key,
            "ANTHROPIC_MODEL": "local",
            "ANTHROPIC_SMALL_FAST_MODEL": "local",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "local",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "local",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "local",
            "API_TIMEOUT_MS": "600000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }, indent=2) + "\n")
    ui.ok(str(cfg.claude_cfg).replace(home, "~"))
