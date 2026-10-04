"""Мелкие helpers: цветной вывод, терминал/локаль, железо, форматирование."""
import locale
import os
import platform
import subprocess


def info(msg: str) -> None:
    print(f"\033[1;34m== {msg}\033[0m")


def ok(msg: str) -> None:
    print(f"\033[1;32m   {msg}\033[0m")


def warn(msg: str) -> None:
    print(f"\033[1;33m   {msg}\033[0m")


def force_utf8_locale() -> None:
    """Без UTF-8-локали ncurses молча рисует '?' вместо █/●/—. Берём локаль
    из окружения, а если она не UTF-8 — форсим en_US.UTF-8. Звать до initscr."""
    cands = [os.environ.get(k) for k in ("LC_ALL", "LC_CTYPE", "LANG")]
    cands += ["en_US.UTF-8", "C.UTF-8"]
    for cand in filter(None, cands):
        try:
            locale.setlocale(locale.LC_ALL, cand)
        except locale.Error:
            continue
        if "utf" in locale.nl_langinfo(locale.CODESET).lower():
            return


def force_compatible_term() -> None:
    """TERM=xterm-ghostty включает terminfo-опцию rep (CSI Ps b «повтор глифа»),
    с которой ncurses портит multibyte-символы: шлёт один байт из трёх `█` ->
    agterm рендерит это как �. У xterm-256color rep нет — вывод чистый
    (сверено побайтовым захватом вывода монитора). Звать до initscr."""
    os.environ["TERM"] = "xterm-256color"


def total_ram_gb() -> int:
    """Полная RAM машины в GB (macOS: hw.memsize — байты)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return int(out.stdout.strip()) // 1024**3
    except Exception:
        return 8


def gpu_name() -> str:
    return "Apple Silicon" if platform.machine() == "arm64" else "Intel/Other"


def human_bytes(n: float) -> str:
    for u in "BKMG":
        if n < 1024 or u == "G":
            return f"{n:.1f}{u}" if u == "G" else f"{n:.0f}{u}"
        n /= 1024


def human_downloads(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)
