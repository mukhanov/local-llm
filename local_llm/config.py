"""Paths, ports and settings. Everything is configured via env (see --help)."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ollmlx_home: Path
    hf_hub: Path
    mlx_port: int
    litellm_port: int
    api_key: str
    # Second, tiny model for cheap fast calls (Claude Code's background
    # title generator and other haiku-slot calls, which use a short fixed
    # timeout and would otherwise queue behind the big model's generation).
    small_model: str
    mlx_small_port: int
    # KV-cache quantization: halves KV memory but, on mlx_lm 0.32, forces the
    # sequential serving path where the prompt cache cannot match long
    # conversations (QuantizedKVCache isn't trimmable) — every request then
    # re-prefills the whole 100k+ prompt, minutes per call. Enable only for
    # models whose weights (~100G of 128G RAM) plus fp16 KV would OOM Metal
    # (the generation thread dies and the server 404s until restarted).
    kv_bits: int
    kv_group_size: int
    # Prompt cache cap (bytes), 0 = no cap. Must exceed the KV of the
    # biggest session you keep alive, or every call re-prefills it from
    # scratch (~96KB/token on the 122B: a 123k-token context is ~12GB —
    # the old 8GB cap never held it, costing a 5-minute prefill per call).
    # 16GB = ~136k tokens of KV; with 65GB weights that's the budget for
    # one mega-session on a 128GB Mac (two concurrent ones risk Metal OOM,
    # which kills the generation thread until restart).
    prompt_cache_bytes: int
    load_timeout: int  # seconds to wait for weights to load
    # hard context ceiling (tokens) advertised to clients and enforced by
    # the server; the KV-budget clamp still applies if stricter. 0 = off.
    # 110k: the KV ceiling of the 122B-class on a 16GB cache is ~174k, but
    # sessions that actually live near it OOM the machine under pressure.
    max_ctx: int

    @property
    def venv_python(self) -> Path:
        return self.ollmlx_home / "venv" / "bin" / "python"

    @property
    def litellm_cfg(self) -> Path:
        return self.ollmlx_home / "litellm-config.yaml"

    @property
    def claude_cfg(self) -> Path:
        return self.ollmlx_home / "claude-local.json"

    # HF token for gated models and higher API limits: env takes precedence,
    # fallback is the private file ~/.ollmlx/hf-token (set via `local-llm token`).
    @property
    def hf_token(self) -> str | None:
        env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if env and env.strip():
            return env.strip()
        try:
            t = (self.ollmlx_home / "hf-token").read_text().strip()
            return t or None
        except OSError:
            return None

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            ollmlx_home=Path(os.environ.get("OLLMLX_HOME", "~/.ollmlx")).expanduser(),
            hf_hub=Path(
                os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")
            ).expanduser(),
            mlx_port=int(os.environ.get("MLX_PORT", "8080")),
            litellm_port=int(os.environ.get("LITELLM_PORT", "4000")),
            api_key="sk-local-llm",
            small_model=os.environ.get(
                "OLLMLX_SMALL_MODEL", "mlx-community/Qwen3-0.6B-4bit"),
            mlx_small_port=int(os.environ.get("MLX_SMALL_PORT", "8081")),
            kv_bits=int(os.environ.get("MLX_KV_BITS", "0")),
            kv_group_size=int(os.environ.get("MLX_KV_GROUP_SIZE", "64")),
            prompt_cache_bytes=int(
                os.environ.get("MLX_PROMPT_CACHE_BYTES", str(16 * 1024**3))
            ),
            load_timeout=int(os.environ.get("LOAD_TIMEOUT", "900")),
            max_ctx=int(os.environ.get("MLX_MAX_CTX", "110000")),
        )
