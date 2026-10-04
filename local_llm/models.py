"""Знания о моделях: парсинг имени репо, скоринг совместимости с железом,
рекомендации по RAM и человекочитаемые описания."""
import math
import re

# Топ под конфигурации по RAM (GB -> модель)
TOP_MODELS = {
    4: "mlx-community/Qwen3-8B-4bit",
    8: "mlx-community/Qwen3-8B-4bit",
    16: "mlx-community/Qwen3-32B-4bit",
    24: "mlx-community/Qwen3-30B-A3B-4bit",
    32: "mlx-community/Qwen3-30B-A3B-4bit",
    64: "mlx-community/Qwen3-235B-A22B-3bit",
    128: "mlx-community/Qwen3-235B-A22B-3bit",
}

# Фолбек, когда HF API недоступен: org/repo | downloads | likes
FALLBACK_TRENDING = [
    ("mlx-community/Qwen3-8B-4bit", 150000, 450),
    ("mlx-community/Qwen3-32B-4bit", 85000, 320),
    ("mlx-community/Qwen3-30B-A3B-4bit", 45000, 180),
    ("mlx-community/Qwen3-235B-A22B-3bit", 25000, 95),
    ("mlx-community/Qwen3-1.7B-4bit", 200000, 520),
    ("mlx-community/Qwen3-0.6B-4bit", 180000, 480),
]

# семейство -> бонус за качество/поддержку в MLX
FAMILIES = (
    ("qwen3", 8), ("qwen2.5", 4), ("qwen2", 2), ("llama-3.3", 6),
    ("llama-3.2", 5), ("llama-3.1", 5), ("llama-3", 3), ("gpt-oss", 7),
    ("gemma-3", 6), ("gemma-2", 4), ("gemma", 3), ("kimi", 5),
    ("mixtral", 4), ("mistral", 3), ("deepseek", 4), ("glm", 3), ("phi", 3),
)

# квант -> байт на параметр в весах
QUANTS = (
    ("4bit", 0.58), ("8bit", 1.10), ("6bit", 0.82), ("7bit", 0.95),
    ("5bit", 0.70), ("3bit", 0.45), ("mxfp4", 0.60),
    ("bf16", 2.05), ("fp16", 2.05), ("f16", 2.05),
)

_DESCRIPTIONS = (
    ("Qwen3-0.6B", "Супер-лёгкая, ~0.6B параметров, мгновенная загрузка"),
    ("Qwen3-1.7B", "Лёгкая, ~1.7B параметров, быстрая генерация"),
    ("Qwen3-4B", "Компактная, ~4B параметров, хороший баланс"),
    ("Qwen3-8B", "Оптимальный баланс, ~8B параметров, высокое качество"),
    ("Qwen3-32B", "Высокое качество, ~32B параметров, требует 20GB+ RAM"),
    ("Qwen3-30B-A3B", "MoE архитектура, ~30B активных из 235B, эффективная"),
    ("Qwen3-235B", "Гигантская модель, ~235B параметров, лучшее качество"),
    ("Llama-3.1-8B", "Llama 3.1 8B, популярная модель от Meta"),
    ("Llama-3.2-1B", "Llama 3.1 1B, супер-лёгкая версия"),
    ("gemma-2-9b", "Gemma 2 9B от Google, качественная открытая модель"),
    ("Kimi-K2.5", "Kimi K2.5, мощная модель с длинным контекстом"),
    ("gpt-oss-20b", "Open-source GPT ~20B, MFP4 квантование"),
    ("Qwen2.5", "Qwen 2.5 серия, предшественник Qwen3"),
    ("Qwen2", "Qwen 2 серия, классическая версия"),
    ("Llama-3", "Llama 3 серия от Meta"),
)

# грубая таблица RAM (GB) для имён без распознанного размера
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
    return "MLX-модель для Apple Silicon"


def parse_model(mid: str):
    """(total_B, active_B, bpp, family_bonus, moe, instruct) из имени репо."""
    s = mid.lower()
    total = active = None
    m = re.search(r"(\d+(?:\.\d+)?)b[-_ ]?a(\d+(?:\.\d+)?)b", s)   # 235B-A22B
    if m:
        total, active = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"[-_](\d+(?:\.\d+)?)b(?:[-_]|$)", s)        # -8B- / -0.6B
        if m:
            total = float(m.group(1))
    bpp = 0.58  # большинство mlx-community репо — 4bit
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
    """Веса + запас: без него Metal легко ловит OOM на KV-кэше."""
    return None if total_b is None else total_b * bpp + 1.5


def ram_need_gb(mid: str) -> int:
    """Оценка RAM в GB: из имени (параметры × квант), фолбек — таблица."""
    total, _active, bpp, *_ = parse_model(mid)
    rn = ram_need(total, bpp)
    if rn is not None:
        return max(1, round(rn))
    for pat, gb in _RAM_TABLE:
        if pat in mid:
            return gb
    return 8


def score(sys_ram: int, mid: str, downloads: int):
    """Скор совместимости с данным железом: выше — лучше.

    fit по RAM (вес = параметры × байт/параметр кванта), качество ~ √параметров
    (пока влезает), MoE-бонус (мало активных параметров — быстро на Apple
    Silicon), семейство, instruct-вариант; загрузки — только мягкий тайбрейк."""
    total, active, bpp, fam, moe, instruct = parse_model(mid)
    rn = ram_need(total, bpp)
    sc = 0.0
    if rn is not None:
        if rn <= sys_ram * 0.85:
            sc += 40          # влезает с запасом
        elif rn <= sys_ram:
            sc += 25          # влезает впритык
        elif rn <= sys_ram * 1.2:
            sc += 5           # только со swap — тормоза
        else:
            sc -= 35          # не влезет
    if total:
        sc += min(22.0, 6.0 * math.sqrt(total))
    if moe:
        sc += 10
    sc += fam
    if instruct:
        sc += 4
    sc += 2.0 * math.log10(downloads + 1)
    return sc, rn, total, active
