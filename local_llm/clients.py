"""Client configs: pi/omp (models.json, provider "ollmlx") and claude
(claude-local.json with env pointing at litellm). Clients are always
launched the same way."""
import json
from pathlib import Path

from . import hf, ui


def _write_omp_smol_role(small: bool) -> None:
    """omp routes its background calls — commit messages, memory notes,
    prewalk/plan-yolo execution — through the `smol` model role; pin it to
    our helper. Without a configured role omp guesses from the whole model
    catalog, and the guess may be a cloud model. pi has no role harness
    (v1.0.3), so for pi the catalog entry is all there is."""
    f = Path("~/.omp/agent/config.yml").expanduser()
    want = "ollmlx/small" if small else None
    try:
        import yaml
    except ImportError:
        ui.warn("pyyaml unavailable — skip ~/.omp/agent/config.yml (smol role)")
        return
    data = {}
    if f.exists():
        try:
            loaded = yaml.safe_load(f.read_text())
            data = loaded if isinstance(loaded, dict) else {}
        except Exception:
            data = {}
    roles = data.get("modelRoles")
    roles = dict(roles) if isinstance(roles, dict) else {}
    if want:
        roles["smol"] = want
    elif roles.get("smol") == "ollmlx/small":  # ours and the helper is gone
        roles.pop("smol")
    else:
        return
    if roles:
        data["modelRoles"] = roles
    else:
        data.pop("modelRoles", None)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(yaml.safe_dump(data, sort_keys=False))
    ui.ok(f"{f} (smol role)")


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


def write_client_configs(cfg, model: str, small: bool = True,
                         max_input: int | None = None) -> None:
    # the advertised context is the KV-safe budget, not the config.json
    # maximum: past the prompt-cache ceiling the model dies on the next
    # cache extension (Metal OOM), so clients must compact earlier
    if max_input is None:
        max_input = hf.safe_context(cfg.hf_hub, model, cfg.prompt_cache_bytes)
    ctx = max_input
    reasoning = "Qwen3" in model
    ui.info("Writing client configs")

    # pi/omp have no small/fast-model hook — the tiny helper is exposed as a
    # plain catalog entry (`--model ollmlx/small`, subagents, omp modes)
    models = [
        _entry(model, f"{model.split('/')[-1]} (local MLX)", ctx, reasoning),
        _entry("local", "Local MLX (whatever is running)", ctx, reasoning),
    ]
    if small:
        # no thinking mode: these calls want latency, not reasoning depth
        models.append(_entry("small", "Local MLX small — fast helper"
                             " (background calls, titles)",
                             hf.model_ctx(cfg.hf_hub, cfg.small_model), False))

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
            "models": models,
        }
        f.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        ui.ok(str(f))

    # omp caches the catalog in models.yml after first reading models.json — reset it
    omp_cache = Path("~/.omp/agent/models.yml").expanduser()
    if omp_cache.exists():
        omp_cache.unlink()
        ui.ok(f"{omp_cache} (cached)")
    _write_omp_smol_role(small)

    home = str(Path.home())
    # small-fast + haiku go to the tiny helper — those background calls
    # (session titles and the like) have a short fixed timeout and used
    # to time out queued behind the big model; opus/sonnet (the actual
    # coding traffic) stay on it. The auto-mode Bash safety classifier
    # ignores these slots: on non-Anthropic providers it runs on the
    # session model by design and pays the prefill delta itself.
    cfg.claude_cfg.write_text(json.dumps({
        "env": {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{cfg.litellm_port}",
            "ANTHROPIC_AUTH_TOKEN": cfg.api_key,
            "ANTHROPIC_MODEL": "local",
            "ANTHROPIC_SMALL_FAST_MODEL": "small" if small else "local",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "local",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "local",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "small" if small else "local",
            "API_TIMEOUT_MS": "600000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    }, indent=2) + "\n")
    ui.ok(str(cfg.claude_cfg).replace(home, "~"))
