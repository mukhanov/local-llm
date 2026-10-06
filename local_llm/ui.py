"""Small helpers: colored output, terminal/locale handling, hardware, formatting."""
import locale
import os
import platform
import subprocess
import sys


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


def push_title() -> None:
    """Save the terminal tab title (xterm stack) — pair with pop_title() at
    exit so the user's own title comes back. Writes to the original stdout
    so a redirected sys.stdout can't swallow it."""
    try:
        out = sys.__stdout__ or sys.stdout
        out.write("\x1b[22t")
        out.flush()
    except (OSError, ValueError):
        pass


def set_title(text: str) -> None:
    """Name the terminal tab right away (OSC 0)."""
    try:
        out = sys.__stdout__ or sys.stdout
        out.write(f"\x1b]0;{text}\x07")
        out.flush()
    except (OSError, ValueError):
        pass


def pop_title() -> None:
    """Restore the title saved by push_title()."""
    try:
        out = sys.__stdout__ or sys.stdout
        out.write("\x1b[23t")
        out.flush()
    except (OSError, ValueError):
        pass


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
    """Chip name for the header — the real one on Apple Silicon
    (machdep.cpu.brand_string, e.g. "Apple M5 Max": llmfit scoring keys its
    memory-bandwidth table off this string)."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        if out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
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
