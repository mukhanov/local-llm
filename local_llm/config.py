"""Пути, порты и настройки. Всё настраивается через env (см. --help)."""
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
    # Квант KV-кэша: длинные промпты (claude шлёт ~30k токенов) на больших
    # моделях (~101G из 128G RAM) выедают остаток -> Metal OOM -> generation
    # thread умирает и сервер до рестарта отвечает 404 на всё. 8 бит = вдвое
    # меньше памяти. 0 = off.
    kv_bits: int
    kv_group_size: int
    # Cap prompt cache (байт), 0 = без потолка.
    prompt_cache_bytes: int
    load_timeout: int  # сек на загрузку весов в память

    @property
    def venv_python(self) -> Path:
        return self.ollmlx_home / "venv" / "bin" / "python"

    @property
    def litellm_cfg(self) -> Path:
        return self.ollmlx_home / "litellm-config.yaml"

    @property
    def claude_cfg(self) -> Path:
        return self.ollmlx_home / "claude-local.json"

    # HF-токен для gated-моделей и бóльших лимитов API: env приоритетнее,
    # фолбек — приватный файл ~/.ollmlx/hf-token (ставится `local-llm token`).
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
