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
    # KV-cache quantization: long prompts (claude sends ~30k tokens) on large
    # models (~101G of 128G RAM) eat the remainder -> Metal OOM -> the
    # generation thread dies and the server answers 404 to everything until
    # restarted. 8 bits = half the memory. 0 = off.
    kv_bits: int
    kv_group_size: int
    # Prompt cache cap (bytes), 0 = no cap.
    prompt_cache_bytes: int
    load_timeout: int  # seconds to wait for weights to load

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
            kv_bits=int(os.environ.get("MLX_KV_BITS", "8")),
            kv_group_size=int(os.environ.get("MLX_KV_GROUP_SIZE", "64")),
            prompt_cache_bytes=int(
                os.environ.get("MLX_PROMPT_CACHE_BYTES", str(8 * 1024**3))
            ),
            load_timeout=int(os.environ.get("LOAD_TIMEOUT", "900")),
        )
