"""Per-model stats from the llmfit catalog (github.com/AlexsJones/llmfit).

llmfit ships a HuggingFace catalog (llmfit-core/data/hf_models.json, ~16k
entries; quality/speed scores are computed client-side and not included, but
context length, RAM guidance and capabilities are). We keep the MLX slice —
mlx-community plus format=mlx repos from other authors (personal re-quants) —
in a local gzip cache, refresh it weekly and surface context length +
tool-use capability in the model picker."""
import gzip
import json
import time
import urllib.request
from pathlib import Path

from . import ui

CATALOG_URL = ("https://raw.githubusercontent.com/AlexsJones/llmfit/main/"
               "llmfit-core/data/hf_models.json")
CACHE_NAME = "llmfit-mlx.json.gz"
TTL_DAYS = 7
PREFIX = "mlx-community/"
# bump when the cached subset changes — a cache written by an older build
# (without the marker) is treated as stale and refetched once
SLICE = 2


def _is_mlx(m: dict) -> bool:
    return (str(m.get("format", "")).lower() == "mlx"
            or str(m.get("name", "")).startswith(PREFIX))


def load(ollmlx_home: Path) -> dict | None:
    """{repo_id: catalog entry} for MLX-format models, or None when nothing
    is available (no cache and no network). A failed refresh falls back to
    the stale cache."""
    cache = ollmlx_home / CACHE_NAME
    models: dict | None = None
    fetched_at = 0.0
    try:
        with gzip.open(cache, "rt") as f:
            blob = json.load(f)
        if blob.get("slice") == SLICE:
            fetched_at, models = blob["fetched_at"], blob["models"]
    except Exception:
        pass
    if models is not None and time.time() - fetched_at < TTL_DAYS * 86400:
        return models
    ui.info("Fetching the llmfit model catalog (MLX slice)")
    try:
        with urllib.request.urlopen(
                urllib.request.Request(CATALOG_URL), timeout=120) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        if models is not None:
            ui.warn(f"llmfit catalog refresh failed ({exc}) — using the cache")
            return models
        return None
    models = {m["name"]: m for m in data
              if isinstance(m, dict) and m.get("name") and _is_mlx(m)}
    try:
        ollmlx_home.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache, "wt") as f:
            json.dump({"slice": SLICE, "fetched_at": int(time.time()),
                       "models": models}, f)
    except OSError:
        pass
    return models


def ctx_str(entry: dict) -> str | None:
    """Context length as a short label: 262144 -> '262K', 1048576 -> '1M'."""
    n = entry.get("context_length")
    if not isinstance(n, int) or n <= 0:
        return None
    if n >= 1_000_000:
        return f"{round(n / 1e6)}M"
    if n >= 1000:
        return f"{round(n / 1000)}K"
    return str(n)


def has_tools(entry: dict) -> bool:
    return "tool_use" in (entry.get("capabilities") or [])
