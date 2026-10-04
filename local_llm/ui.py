"""Small helpers: colored output, terminal/locale handling, hardware, formatting."""
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
    """Without a UTF-8 locale ncurses silently draws '?' instead of █/●/—.
    Try locales from the environment; if none is UTF-8, force en_US.UTF-8.
    Call before initscr."""
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
    """TERM=xterm-ghostty enables the terminfo 'rep' capability (CSI Ps b
    "repeat glyph"), with which ncurses corrupts multibyte characters: it
    sends one byte out of a three-byte `█`, and agterm renders it as �.
    xterm-256color has no 'rep' — output is clean (verified by capturing
    the monitor's output byte-by-byte). Call before initscr."""
    os.environ["TERM"] = "xterm-256color"


def total_ram_gb() -> int:
    """Total machine RAM in GB (macOS: hw.memsize is in bytes)."""
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
