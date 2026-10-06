"""Fullscreen startup dialog: while the stack boots, a centered panel with
a spinner and the live boot log replaces scrolling terminal text; when the
boot finishes, the caller takes the terminal straight to the monitor.

The boot runs with stdout/stderr redirected into a QueueWriter, so every
existing print (download progress, server readiness, warmup) shows up in
the panel without touching the curses screen."""
import curses
import io
import queue
import re
import time
from collections import deque

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class QueueWriter(io.TextIOBase):
    """stdout/stderr sink for the boot: text is stripped of ANSI codes and
    fed to the dialog line by line (\\r progress ticks replace the line in
    place); nothing reaches the terminal while the panel is up."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self._partial = ""

    def writable(self):
        return True

    def isatty(self):
        # hf.download and ui print live progress only on a tty
        return True

    def write(self, s):
        for ch in _ANSI.sub("", s):
            if ch in "\r\n":
                if self._partial.strip():
                    self.q.put(self._partial.strip())
                self._partial = ""
            else:
                self._partial += ch
        return len(s)

    def flush(self):
        pass


def dialog(model: str, q: queue.Queue, done, result: dict) -> None:
    """Draw the panel until `done` is set. Success shows the final state for
    a moment and returns (the caller then opens the monitor); failure waits
    for a keypress so the error stays readable."""

    def _render(scr):
        curses.curs_set(0)
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = 0
        curses.init_pair(1, curses.COLOR_CYAN, bg)
        scr.timeout(80)
        frame = 0
        keep = deque(maxlen=400)
        while True:
            try:
                while True:
                    keep.append(q.get_nowait())
            except queue.Empty:
                pass

            h, w = scr.getmaxyx()
            short = model.split("/")[-1]
            short = short[:25] + "…" if len(short) > 26 else short
            code = result.get("code") if done.is_set() else None
            ok = code == 0
            failed = done.is_set() and not ok
            bw = min(max(56, len(short) + 22), max(24, w - 2))
            rows = max(5, min(h - 4, 12))
            y0 = max(0, (h - rows - 2) // 2)
            x0 = max(0, (w - bw) // 2)

            def add(yy, xx, text, attr=0):
                try:
                    scr.addstr(yy, xx, text[: max(0, w - 1 - xx)], attr)
                except curses.error:
                    pass

            def clip(text, width):
                return text if len(text) <= width else text[: max(0, width - 1)] + "…"

            scr.erase()
            mark = ("✓" if ok else "✗" if failed
                    else SPINNER[frame % len(SPINNER)])
            state = ("ready" if ok else "failed" if failed else "booting")
            head = f" {mark}  {state}: {short} "
            add(y0, x0, "┌" + head + "─" * max(0, bw - 2 - len(head)) + "┐",
                curses.color_pair(1) | curses.A_BOLD)
            body = list(keep)[-(rows - 2):]
            blank = rows - 2 - len(body)
            for i in range(rows - 2):
                yy = y0 + 1 + i
                add(yy, x0, "│", curses.A_DIM)
                if i >= blank:
                    add(yy, x0 + 2, clip(body[i - blank], bw - 4), curses.A_DIM)
                add(yy, x0 + bw - 1, "│", curses.A_DIM)
            add(y0 + rows - 1, x0, "└" + "─" * (bw - 2) + "┘", curses.A_DIM)
            scr.refresh()

            if done.is_set():
                if failed:
                    scr.timeout(-1)
                    scr.getch()
                else:
                    time.sleep(0.9)
                return
            frame += 1
            scr.getch()   # frame pacing (80ms timeout); stray keys ignored
            time.sleep(0.02)

    curses.wrapper(_render)
