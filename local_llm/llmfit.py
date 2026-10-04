"""Per-model stats from the llmfit catalog (github.com/AlexsJones/llmfit).

llmfit ships a HuggingFace catalog (llmfit-core/data/hf_models.json, ~16k
entries; quality/speed scores are computed client-side and not included, but
context length, RAM guidance and capabilities are). We keep the MLX slice —
mlx-community plus format=mlx repos from other authors (personal re-quants) —
in a local gzip cache, refresh it weekly and surface context length +
tool-use capability in the model picker.

Also a faithful Python port of llmfit's composite fit score (llmfit-core
src/fit.rs + src/models.rs, Apache-2.0), scoped to our platform: Apple
Silicon, unified memory, MLX runtime. The picker sorts by it — same ranking
the llmfit CLI shows for this machine."""
import datetime
import gzip
import json
import math
import re
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


# ── llmfit composite score port (fit.rs / models.rs) ─────────────────────────
#
# Scoped to Apple Silicon (unified memory, Metal, MLX): RunMode is always Gpu
# with the whole RAM as the pool, and speed comes from the bandwidth roofline
# of the chip. Everything else — quality tiers, bonuses, quant tables, MoE
# calibrations, weights — mirrors the Rust source.

DEFAULT_ESTIMATION_CTX = 8192   # fit.rs: runtimes use far less than the max window
EFFICIENCY = 0.55               # CalcConfig default: kernel/KV/controller overhead
MOE_FIXED_EFFECTIVE_BPP = 3.2   # models.rs: fixed-component bandwidth equivalent
_PRESSURE_THRESHOLD = 0.60
_PRESSURE_FLOOR = 0.30
_PRESSURE_DEFAULT_RATIO = 0.50

# hardware.rs gpu_memory_bandwidth_gbps, Apple Silicon slice (GB/s), longest
# match first within a generation. Unknown chip → the Metal/MLX fallback
# constant (k=250 ≈ 227 GB/s implied bandwidth).
_CHIP_BW: tuple[tuple[str, float], ...] = (
    ("m5 max", 614.0), ("m5 pro", 307.0), ("m5", 153.6),
    ("m4 ultra", 819.0), ("m4 max", 546.0), ("m4 pro", 273.0), ("m4", 120.0),
    ("m3 ultra", 800.0), ("m3 max", 400.0), ("m3 pro", 150.0), ("m3", 100.0),
    ("m2 ultra", 800.0), ("m2 max", 400.0), ("m2 pro", 200.0), ("m2", 100.0),
    ("m1 ultra", 800.0), ("m1 max", 400.0), ("m1 pro", 200.0), ("m1", 68.0),
)
_DEFAULT_BW = 227.0

# models.rs quant tables: quality penalty / bits-per-param / bytes-per-param.
_UD_TIERS = {"Q2_K": (-12.0, 0.37, 0.25), "Q3_K": (-8.0, 0.48, 0.375),
             "Q4_K": (-5.0, 0.58, 0.5), "Q5_K": (-2.0, 0.68, 0.625),
             "Q6_K": (-1.0, 0.80, 0.75), "Q8_K": (0.0, 1.05, 1.0)}
_Q_TABLE = {
    "F32": (4.0, 4.0), "F16": (0.0, 2.0, 2.0), "BF16": (0.0, 2.0, 2.0),
    "Q8_0": (0.0, 1.05, 1.0), "Q6_K": (-1.0, 0.80, 0.75), "Q5_K_M": (-2.0, 0.68, 0.625),
    "Q4_K_M": (-5.0, 0.58, 0.5), "Q4_0": (-5.0, 0.58, 0.5), "Q3_K_M": (-8.0, 0.48, 0.375),
    "Q2_K": (-12.0, 0.37, 0.25), "I2_S": (-6.0, 0.42, 0.40), "TQ2_0": (-6.0, 0.42, 0.40),
    "TQ1_0": (-6.0, 0.42, 0.40), "MXFP4": (0.0, 0.55, 0.53),
    "mlx-4bit": (-4.0, 0.55, 0.5), "mlx-8bit": (0.0, 1.0, 1.0),
    "AWQ-4bit": (-3.0, 0.5, 0.5), "AWQ-8bit": (0.0, 1.0, 1.0),
    "GPTQ-Int4": (-3.0, 0.5, 0.5), "GPTQ-Int8": (0.0, 1.0, 1.0),
    "AutoRound-4bit": (-3.0, 0.5, 0.5), "AutoRound-8bit": (0.0, 1.0, 1.0),
}
for _k, (_p, _b, _by) in _UD_TIERS.items():
    for _s in ("XL", "L", "M", "S"):
        _Q_TABLE[f"UD-{_k}_{_s}"] = (_p, _b, _by)
_Q_DEFAULT = (-5.0, 0.58, 0.5)   # unrecognized label: Q4-class

# use_case -> (weights (quality, speed, fit, context), speed target tok/s, ctx target)
_USE_CASES = {
    "General":    ((0.45, 0.30, 0.15, 0.10), 40.0, 4096),
    "Coding":     ((0.50, 0.20, 0.15, 0.15), 40.0, 8192),
    "Reasoning":  ((0.55, 0.15, 0.15, 0.15), 25.0, 8192),
    "Chat":       ((0.40, 0.35, 0.15, 0.10), 40.0, 4096),
    "Multimodal": ((0.50, 0.20, 0.15, 0.15), 40.0, 4096),
    "Embedding":  ((0.30, 0.40, 0.20, 0.10), 200.0, 512),
}

_chip_bw: float | None = None


def chip_bandwidth() -> float:
    """This machine's unified-memory bandwidth (GB/s); resolved once."""
    global _chip_bw
    if _chip_bw is None:
        name = ui.gpu_name().lower()
        _chip_bw = next((bw for pat, bw in _CHIP_BW if pat in name), _DEFAULT_BW)
    return _chip_bw


def _use_case(e: dict) -> str:
    """models.rs UseCase::from_model."""
    name = str(e.get("name", "")).lower()
    uc = str(e.get("use_case", "")).lower()
    if "embedding" in uc or "embed" in name or "bge" in name:
        return "Embedding"
    if "code" in name or "code" in uc:
        return "Coding"
    if "vision" in uc or "multimodal" in uc:
        return "Multimodal"
    if "reason" in uc or "chain-of-thought" in uc or "deepseek-r1" in name:
        return "Reasoning"
    if "chat" in uc or "instruction" in uc:
        return "Chat"
    return "General"


def _params_b(e: dict) -> float:
    raw = e.get("parameters_raw")
    if isinstance(raw, (int, float)) and raw > 0:
        return raw / 1e9
    m = re.fullmatch(r"([\d.]+)\s*[Bb]", str(e.get("parameter_count") or "").strip())
    return float(m.group(1)) if m else 0.0


def _months_since(d) -> int | None:
    if not isinstance(d, str) or len(d) < 7:
        return None
    try:
        y, m = int(d[:4]), int(d[5:7])
    except ValueError:
        return None
    t = datetime.date.today()
    return max(0, (t.year - y) * 12 + t.month - m)


def _qwen_minor(name_lower: str) -> float | None:
    for pat, gen in (("qwen3.8", 3.8), ("qwen3_8", 3.8), ("qwen3.6", 3.6),
                     ("qwen3_6", 3.6), ("qwen3.5", 3.5), ("qwen3_5", 3.5),
                     ("qwen2.5", 2.5), ("qwen2_5", 2.5)):
        if pat in name_lower:
            return gen
    return None


def _generation(e: dict) -> float | None:
    """models.rs parse_generation: architecture first, then the repo name."""
    a = str(e.get("architecture") or "").lower()
    n = str(e.get("name") or "").lower()
    if a:
        if a.startswith("deepseek"):
            return 4.0 if "v4" in a else 3.0 if "v3" in a else 2.0 if "v2" in a else 1.0
        if a.startswith("qwen"):
            if (g := _qwen_minor(n)) is not None:
                return g
            s = a[4:]
            return (3.5 if s.startswith(("3_5", "3.5")) else
                    3.8 if s.startswith(("3_next", "3next")) else
                    3.0 if s.startswith("3") else
                    2.0 if s.startswith("2") else 1.0)
        if a.startswith("llama4"):
            return 4.0
        if a.startswith("gemma"):
            s = a[5:]
            return 4.0 if s.startswith("4") else 3.0 if s.startswith("3") \
                else 2.0 if s.startswith("2") else 1.0
        if a.startswith("phi"):
            s = a[3:]
            return 4.0 if s.startswith("4") else 3.0 if s.startswith(("3", "moe")) \
                else 2.0 if s.startswith("2") else 1.0
        if a.startswith(("mistral", "mixtral")):
            return 1.0
        if a.startswith("cohere"):
            return 2.0 if a[6:].startswith("2") else 1.0
        if a.startswith("falcon"):
            return 3.0 if a[6:].startswith("3") else 1.0
        if a.startswith("granite"):
            return 4.0 if a[7:].startswith("4") else 1.0
    if (g := _qwen_minor(n)) is not None:
        return g
    if "qwen3" in n:
        return 3.0
    if "qwen2" in n:
        return 2.0
    for pat, gen in (("llama-4", 4.0), ("llama4", 4.0), ("llama-3.3", 3.3),
                     ("llama3.3", 3.3), ("llama-3.2", 3.2), ("llama3.2", 3.2),
                     ("llama-3.1", 3.1), ("llama3.1", 3.1), ("llama-3", 3.0),
                     ("llama3", 3.0), ("llama-2", 2.0), ("llama2", 2.0),
                     ("gemma-4", 4.0), ("gemma4", 4.0), ("gemma-3", 3.0),
                     ("gemma3", 3.0), ("gemma-2", 2.0), ("gemma2", 2.0),
                     ("deepseek-v4", 4.0), ("deepseekv4", 4.0),
                     ("deepseek-v3", 3.0), ("deepseekv3", 3.0),
                     ("deepseek-v2", 2.0), ("deepseekv2", 2.0),
                     ("phi-4", 4.0), ("phi4", 4.0), ("phi-3", 3.0), ("phi3", 3.0)):
        if pat in n:
            return gen
    return None


def _quality(e: dict, uc: str) -> float:
    """fit.rs quality_score: base tier on *active* params (MoE), + family,
    generation, recency, quant penalty, task alignment."""
    params = _params_b(e)
    active = e.get("active_parameters")
    qp = active / 1e9 if isinstance(active, (int, float)) and active else params
    base = (30.0 if qp < 1 else 45.0 if qp < 3 else 60.0 if qp < 7 else
            75.0 if qp < 10 else 82.0 if qp < 20 else 89.0 if qp < 40 else 95.0)
    n = str(e.get("name") or "").lower()
    family = (2.0 if "qwen" in n else 3.0 if "deepseek" in n else 2.0 if "llama" in n
              else 1.0 if "mistral" in n or "mixtral" in n else 1.0 if "gemma" in n
              else 1.0 if "starcoder" in n else 0.0)
    gen = _generation(e)
    gen_bonus = min(9.0, max(0.0, (gen - 1.0) * 3.0)) if gen is not None else 0.0
    months = _months_since(e.get("release_date"))
    recency = (3.0 if months < 3 else 1.5 if months < 9 else 0.0) if months is not None else 0.0
    penalty = _Q_TABLE.get(e.get("quantization"), _Q_DEFAULT)[0]
    # task alignment: name heuristics (llmfit's curated bench table isn't in
    # the catalog, so Coding/Reasoning/Multimodal get the heuristic bump only)
    if uc == "Coding":
        task = 6.0 if any(k in n for k in ("code", "starcoder", "wizard")) else 0.0
    elif uc == "Reasoning":
        task = 5.0 if params >= 13.0 else 0.0
    elif uc == "Multimodal" and ("vision" in n
                                 or "vision" in str(e.get("use_case", "")).lower()):
        task = 6.0
    else:
        task = 0.0
    return max(0.0, min(100.0, base + family + gen_bonus + recency + penalty + task))


def _moe_tier1_bytes(e: dict, bpp: float) -> float | None:
    """models.rs moe_bandwidth_decomposition → per-token GB, or None when
    the architecture metadata is missing (most catalog entries)."""
    def num(k):
        v = e.get(k)
        return v if isinstance(v, (int, float)) and v else None
    hidden, layers = num("hidden_size"), num("num_hidden_layers")
    a_exp, inter, vocab = num("active_experts"), num("moe_intermediate_size"), num("vocab_size")
    if None in (hidden, layers, a_exp, inter, vocab):
        return None
    heads = num("num_attention_heads") or 1
    kv = num("num_key_value_heads") or heads
    hd = num("head_dim") or hidden / heads
    n_exp = num("num_experts") or 8
    active_ffn = layers * a_exp * 3 * hidden * inter
    attn = layers * (2 * heads * hd * hidden + 2 * kv * hd * hidden)
    shared = layers * 3 * hidden * (num("shared_expert_intermediate_size") or 0)
    router = layers * n_exp * hidden
    fixed = attn + shared + router + 2 * vocab * hidden
    arch = str(e.get("architecture") or "").lower()
    fixed_bpp = 1.43 if arch.startswith(("gpt_oss", "gptoss")) else MOE_FIXED_EFFECTIVE_BPP
    return active_ffn * bpp / 1e9 + fixed * fixed_bpp / 1e9


def _estimate_tps(e: dict, bw: float, ram_gb: float) -> float:
    """fit.rs estimate_tps, GPU mode on unified memory (pool = total RAM)."""
    params = _params_b(e)
    if params <= 0:
        return 0.1
    quant = e.get("quantization")
    _, bpp, bypp = _Q_TABLE.get(quant, _Q_DEFAULT)
    active = e.get("active_parameters")
    a_b = active / 1e9 if isinstance(active, (int, float)) and active and e.get("is_moe") else params
    if not e.get("is_moe"):
        return max(0.1, bw / (params * bypp) * EFFICIENCY)
    # cache-pressure penalty from inactive experts polluting the pool
    util = params * bpp / ram_gb if ram_gb > 0 else 0.0
    if util <= _PRESSURE_THRESHOLD or util > 1.0:
        pressure = 1.0
    else:
        n_exp = e.get("num_experts")
        ratio = (1.0 - (e.get("active_experts") or 1) / n_exp if isinstance(n_exp, (int, float))
                 else _PRESSURE_DEFAULT_RATIO)
        pressure = max(_PRESSURE_FLOOR, 1.0 - (util - _PRESSURE_THRESHOLD) * ratio)
    if (per_token := _moe_tier1_bytes(e, bpp)) is not None:
        return max(0.1, bw / per_token * pressure)
    arch = str(e.get("architecture") or "").lower()
    if arch.startswith(("gpt_oss", "gptoss")):
        eff, ovh = 0.72, 0.80
    elif arch.startswith(("deepseek_v3", "deepseek_v4")):
        eff, ovh = 0.62, 0.70
    else:
        eff = EFFICIENCY
        n_exp = e.get("num_experts")
        ovh = (0.90 if n_exp is not None and n_exp <= 8 else
               0.85 if n_exp is not None and n_exp <= 16 else
               0.80 if n_exp is not None and n_exp <= 32 else
               0.70 if n_exp is not None and n_exp <= 64 else
               0.40 if n_exp is not None else 0.60)
    return max(0.1, bw / (a_b * bpp) * eff * ovh * pressure)


def _kv_gb(e: dict, ctx: int) -> float:
    layers, hd = e.get("num_hidden_layers"), e.get("head_dim")
    if isinstance(layers, (int, float)) and layers and isinstance(hd, (int, float)) and hd:
        kvh = e.get("num_key_value_heads") or e.get("num_attention_heads") or 8
        return 2 * kvh * hd * ctx * 2 / 1024 ** 3   # K+V, fp16
    return 0.000008 * _params_b(e) * ctx           # coarse fallback


def _mem_gb(e: dict) -> float:
    """models.rs estimate_memory_gb at the estimation context: weights + KV + 0.5."""
    ctx = min(DEFAULT_ESTIMATION_CTX, e.get("context_length") or DEFAULT_ESTIMATION_CTX)
    return _params_b(e) * _Q_TABLE.get(e.get("quantization"), _Q_DEFAULT)[1] \
        + _kv_gb(e, ctx) + 0.5


def score(entry: dict, ram_gb: float) -> dict | None:
    """llmfit's composite score of a catalog entry on THIS machine (Apple
    Silicon, unified memory, MLX). None when the entry lacks the data for it.
    Components mirror fit.rs: quality, speed (bandwidth roofline), fit
    (Gaussian falloff past 70% of the pool), context — weighted per use case."""
    if _params_b(entry) <= 0:
        return None
    uc = _use_case(entry)
    weights, spd_target, ctx_target = _USE_CASES[uc]
    q = _quality(entry, uc)
    tps = _estimate_tps(entry, chip_bandwidth(), ram_gb)
    s = max(0.0, min(100.0, tps / spd_target * 100.0))
    mem = _mem_gb(entry)
    f = 0.0 if mem > ram_gb else max(
        0.0, min(100.0, 100.0 * math.exp(-0.5 * max(0.0, (mem / ram_gb - 0.70) / 0.20) ** 2)))
    cl = entry.get("context_length") or 0
    c = 100.0 if cl >= ctx_target else 70.0 if cl >= ctx_target / 2 else 30.0
    total = round((q * weights[0] + s * weights[1] + f * weights[2] + c * weights[3]) * 10) / 10
    return {"score": total, "quality": round(q, 1), "speed": round(s, 1),
            "fit": round(f, 1), "context": round(c, 1), "tps": round(tps, 1),
            "mem": round(mem, 1), "params": round(_params_b(entry), 1), "use_case": uc}
