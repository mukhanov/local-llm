"""Model knowledge: repo-name parsing, hardware-fit scoring, RAM
recommendations and human-readable descriptions."""
import math
import re

# Top pick per RAM class (GB -> model)
TOP_MODELS = {
    4: "mlx-community/Qwen3-8B-4bit",
    8: "mlx-community/Qwen3-8B-4bit",
    16: "mlx-community/Qwen3-32B-4bit",
    24: "mlx-community/Qwen3-30B-A3B-4bit",
    32: "mlx-community/Qwen3-30B-A3B-4bit",
    64: "mlx-community/Qwen3-235B-A22B-3bit",
    128: "mlx-community/Qwen3-235B-A22B-3bit",
}

# Fallback when the HF API is unreachable: org/repo | downloads | likes
FALLBACK_TRENDING = [
    ("mlx-community/Qwen3-8B-4bit", 150000, 450),
    ("mlx-community/Qwen3-32B-4bit", 85000, 320),
    ("mlx-community/Qwen3-30B-A3B-4bit", 45000, 180),
    ("mlx-community/Qwen3-235B-A22B-3bit", 25000, 95),
    ("mlx-community/Qwen3-1.7B-4bit", 200000, 520),
    ("mlx-community/Qwen3-0.6B-4bit", 180000, 480),
]

# family -> bonus for quality / MLX support
FAMILIES = (
    ("qwen3", 8), ("qwen2.5", 4), ("qwen2", 2), ("llama-3.3", 6),
    ("llama-3.2", 5), ("llama-3.1", 5), ("llama-3", 3), ("gpt-oss", 7),
    ("gemma-3", 6), ("gemma-2", 4), ("gemma", 3), ("kimi", 5),
    ("mixtral", 4), ("mistral", 3), ("deepseek", 4), ("glm", 3), ("phi", 3),
)

# quantization -> bytes per parameter in the weights
QUANTS = (
    ("4bit", 0.58), ("8bit", 1.10), ("6bit", 0.82), ("7bit", 0.95),
    ("5bit", 0.70), ("3bit", 0.45), ("mxfp4", 0.60),
    ("bf16", 2.05), ("fp16", 2.05), ("f16", 2.05),
)

_DESCRIPTIONS = (
    ("Qwen3-0.6B", "Ultra-light, ~0.6B params, instant load"),
    ("Qwen3-1.7B", "Light, ~1.7B params, fast generation"),
    ("Qwen3-4B", "Compact, ~4B params, good balance"),
    ("Qwen3-8B", "Optimal balance, ~8B params, high quality"),
    ("Qwen3-32B", "High quality, ~32B params, needs 20GB+ RAM"),
    ("Qwen3-30B-A3B", "MoE, ~30B total with 3B active, efficient"),
    ("Qwen3-235B", "Huge model, ~235B params, best quality"),
    ("Llama-3.1-8B", "Llama 3.1 8B, popular Meta model"),
    ("Llama-3.2-1B", "Llama 3.2 1B, ultra-light version"),
    ("gemma-2-9b", "Gemma 2 9B by Google, solid open model"),
    ("Kimi-K2.5", "Kimi K2.5, strong long-context model"),
    ("gpt-oss-20b", "Open-source GPT ~20B, MXFP4 quantized"),
    ("Qwen2.5", "Qwen 2.5 series, predecessor of Qwen3"),
    ("Qwen2", "Qwen 2 series, the classic version"),
    ("Llama-3", "Llama 3 series by Meta"),
)

# rough RAM table (GB) for names without a recognized size
_RAM_TABLE = (
    ("0.5B", 2), ("0.6B", 2), ("1B", 4), ("1.7B", 4), ("3B", 6), ("4B", 6),
    ("7B", 8), ("8B", 8), ("9B", 8), ("14B", 12), ("20B", 16), ("27B", 16),
    ("30B", 20), ("32B", 20), ("30B-A3B", 24), ("70B", 40), ("235B", 80),
    ("gpt-oss-20b", 16),
)


def recommend_for_ram(ram: int) -> str:
    if ram >= 80:
        return TOP_MODELS[64]
    if ram >= 32:
        return TOP_MODELS[24]
    if ram >= 24:
        return TOP_MODELS[24]
    if ram >= 16:
        return TOP_MODELS[16]
    if ram >= 8:
        return TOP_MODELS[8]
    return TOP_MODELS[4]


def describe(model: str) -> str:
    for pat, text in _DESCRIPTIONS:
        if pat in model:
            return text
    return "MLX model for Apple Silicon"


def parse_model(mid: str):
    """(total_B, active_B, bpp, family_bonus, moe, instruct) from repo name."""
    s = mid.lower()
    total = active = None
    m = re.search(r"(\d+(?:\.\d+)?)b[-_ ]?a(\d+(?:\.\d+)?)b", s)   # 235B-A22B
    if m:
        total, active = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"[-_](\d+(?:\.\d+)?)b(?:[-_]|$)", s)        # -8B- / -0.6B
        if m:
            total = float(m.group(1))
    bpp = 0.58  # most mlx-community repos are 4bit
    for q, b in QUANTS:
        if q in s:
            bpp = b
            break
    fam = 0
    for name, bonus in FAMILIES:
        if name in s:
            fam = bonus
            break
    moe = active is not None or "moe" in s or "mixtral" in s
    instruct = any(t in s for t in ("instruct", "-it", "it-", "chat"))
    return total, active, bpp, fam, moe, instruct


def ram_need(total_b, bpp: float):
    """Weights + headroom: without it Metal easily OOMs on the KV cache."""
    return None if total_b is None else total_b * bpp + 1.5


def ram_need_gb(mid: str) -> int:
    """RAM estimate in GB: from the name (params × quant), table as fallback."""
    total, _active, bpp, *_ = parse_model(mid)
    rn = ram_need(total, bpp)
    if rn is not None:
        return max(1, round(rn))
    for pat, gb in _RAM_TABLE:
        if pat in mid:
            return gb
    return 8


def score(sys_ram: int, mid: str, downloads: int):
    """Hardware-fit score for this machine: higher is better.

    RAM fit (weights = params × bytes/param of the quant), quality ~ √params
    (while it fits), MoE bonus (few active params — fast on Apple Silicon),
    family, instruct variant; downloads are only a soft tiebreak."""
    total, active, bpp, fam, moe, instruct = parse_model(mid)
    rn = ram_need(total, bpp)
    sc = 0.0
    if rn is not None:
        if rn <= sys_ram * 0.85:
            sc += 40          # fits with headroom
        elif rn <= sys_ram:
            sc += 25          # fits tightly
        elif rn <= sys_ram * 1.2:
            sc += 5           # swap only — will be slow
        else:
            sc -= 35          # won't fit
    if total:
        sc += min(22.0, 6.0 * math.sqrt(total))
    if moe:
        sc += 10
    sc += fam
    if instruct:
        sc += 4
    sc += 2.0 * math.log10(downloads + 1)
    return sc, rn, total, active
