"""htop-style TUI monitor for the stack: tokens-per-second graphs for the
big model and the helper (current rate + history), per-core CPU + a
history graph, RAM/swap, port status, mlx/litellm processes with RSS,
the pi/omp/claude launch commands and a tail of the logs with errors.
Blocks until q/Ctrl-C (cli stops the servers).

The tok/s numbers come from TOKPS lines that our mlx_lm.server wrapper
(local_llm.mlxwrap) prints into /tmp/mlx-server.log and
/tmp/mlx-small.log ~once a second."""
import collections
import curses
import os
import re
import socket
import sys
import time

import psutil

from . import ui
from .servers import MLX_SMALL_LOG, WATCHDOG_LOG

LOGS = (("mlx", "/tmp/mlx-server.log"), ("litellm", "/tmp/litellm.log"))


class _UiSink:
    """Captures stray prints while the curses UI owns the terminal (watchdog
    restarts, readiness ticks) and appends them to the watchdog log — the
    logs box tails it, so the events stay visible instead of scribbling
    over the screen."""

    def __init__(self, path: str):
        self.path = path

    def write(self, s):
        lines = [l.strip() for l in re.sub(
            r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)",
            "", s).splitlines()]
        stamp = time.strftime("%H:%M:%S")
        for line in lines:
            if not line:
                continue
            try:
                with open(self.path, "a") as f:
                    f.write(f"{stamp} [ui] {line}\n")
            except OSError:
                pass
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def run(model: str, mlx_port: int, lite_port: int, claude_cfg: str,
        small_port: int = 0) -> None:
    """Blocks until exit (q / Ctrl-C). small_port=0 — no helper model."""
    global LOGS
    if small_port:
        LOGS = LOGS + (("mlx-small", MLX_SMALL_LOG),)
    LOGS = LOGS + (("watchdog", WATCHDOG_LOG),)
    ui.force_utf8_locale()
    ui.force_compatible_term()
    # while the curses UI owns the screen, stray prints must not reach the
    # terminal (a watchdog restart printing "ready in 3s" over the grid
    # shredded the display) — route them into the watchdog log instead
    sink = _UiSink(WATCHDOG_LOG)
    old = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = sink
    try:
        curses.wrapper(lambda scr: _main(
            scr, model, mlx_port, lite_port, claude_cfg, small_port))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout, sys.stderr = old
    # the tab title's save/restore belongs to the app level (cli.main
    # pushed at startup and pops at exit), not to the monitor


LOW_AVAILABLE_GB = 10   # MEM bar pulses red with a warning below this


def _low_mem(vm) -> bool:
    """True when free memory is low enough that a Metal eval can OOM the
    model — the condition behind every generation-thread death so far."""
    return vm.available < LOW_AVAILABLE_GB * 1024 ** 3


def top_eaters(k: int = 3, exclude_pids=()) -> list:
    """Top-k third-party processes by RSS as (rss, pid, name) — the kill
    candidates under memory pressure. Our own mlx/litellm processes are
    excluded: they are listed separately with their own RSS."""
    me = os.getpid()
    best = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = p.info
            if info["pid"] in (0, me) or info["pid"] in exclude_pids:
                continue
            rss = info["memory_info"].rss if info["memory_info"] else 0
            if rss < 300 * 1024 * 1024:   # sub-300MB is not a lever
                continue
            best.append((rss, info["pid"], os.path.basename(info["name"] or "?")))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    best.sort(reverse=True)
    return best[:k]


LOW_AVAILABLE_GB = 10   # MEM bar pulses red with a warning below this
_LOW_PUSH_COOLDOWN = 600  # seconds between repeat low-memory pushes

_low_latched = False    # a push is already out for the current low phase
_last_push = 0.0


def _push(title: str, message: str) -> None:
    """macOS notification via the built-in osascript (no dependencies).
    First delivery may need Script Editor allowed in System Settings →
    Notifications; failures are silent — the on-screen warning remains."""
    message = message.replace('"', "'").replace("\\", "")
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{message}" with title "{title}"'],
                       capture_output=True, timeout=10)
    except Exception:
        pass


def _maybe_notify_low(low: bool, eaters, vm, now: float) -> None:
    """Push once when free memory crosses the danger line, then at most
    every _LOW_PUSH_COOLDOWN while it stays there; reset on recovery."""
    global _low_latched, _last_push
    if not low:
        _low_latched = False
        return
    if _low_latched and now - _last_push < _LOW_PUSH_COOLDOWN:
        return
    _low_latched, _last_push = True, now
    avail = ui.human_bytes(vm.available)
    who = f" Kill: {eaters[0][2]} ({ui.human_bytes(eaters[0][0])})" if eaters else ""
    _push("ollmlx ⚠ OOM risk",
          f"Only {avail} free — the model can OOM.{who}")


def port_ok(port: int) -> bool:
    try:
        with socket.socket() as s:
            s.settimeout(0.4)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def watch_procs(small_port: int = 0):
    """[(label, Process)] for mlx/litellm; process_iter caches instances,
    so cpu_percent() between frames yields meaningful deltas."""
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or ())
        except Exception:
            continue
        if "mlx_lm.server" in cmd or "local_llm.mlxwrap" in cmd:
            # the helper is told apart by its --port (the big model's id
            # never contains it)
            label = ("mlx-small" if small_port and f"--port {small_port}" in cmd
                     else "mlx_lm.server")
            out.append((label, p))
        elif "litellm" in cmd and "--config" in cmd:
            out.append(("litellm", p))
    return out


def heat(pct: float) -> int:  # green -> yellow -> red
    return 1 if pct < 60 else 2 if pct < 85 else 3


def _set_title(text: str) -> None:
    """Name the terminal tab (OSC 0, like Claude Code names its session).
    Written straight to the tty — curses manages only the screen grid, so
    the sequence passes through untouched. Uses the original stdout: while
    the monitor runs, sys.stdout is redirected to the watchdog log."""
    try:
        out = sys.__stdout__ or sys.stdout
        out.write(f"\x1b]0;{text}\x07")
        out.flush()
    except (OSError, ValueError):
        pass


def load_emoji(load_pct: float) -> str:
    """Colored circle for the tab title: green/yellow/red by load —
    emoji glyphs keep their color where OSC color codes don't survive."""
    return "🟢" if load_pct < 60 else "🟡" if load_pct < 85 else "🔴"


def _tab_title(model: str, procs, load_pct: float = 0.0) -> str:
    """Tab text: load circle + short model name + the RAM of the model
    processes (mlx and the helper; litellm is a proxy, not a model — not
    counted). Nothing else."""
    ram = 0
    for label, p in procs:
        if label == "litellm":
            continue
        try:
            ram += p.memory_info().rss
        except Exception:
            pass
    short = model.split("/")[-1]
    short = short[:25] + "…" if len(short) > 26 else short
    tail = f" · {ui.human_bytes(ram)}" if ram else ""
    return f"{load_emoji(load_pct)} {short}{tail}"


def tail_errors(path: str, k: int = 2):
    """Last k error lines of a log, skipping benign disconnect noise
    (a client dropping mid-response logs a scary-but-harmless
    BrokenPipeError traceback in the servers). Only the tail of the file
    is read — cheap on a live log, no multi-megabyte scans."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    tokens = ("ERROR", "CRITICAL", "Traceback", "Exception")
    benign = ("BrokenPipeError", "ConnectionResetError", "ConnectionAbortedError")
    hits = []
    # logging writes each record between dash separators — and the final
    # exception line of a traceback can land in the block after one, so
    # check the block and its successor before calling it a real error
    blocks = data.split("-" * 40)
    for i, block in enumerate(blocks):
        if not any(t in block for t in tokens):
            continue
        window = block + (blocks[i + 1] if i + 1 < len(blocks) else "")
        if any(b in l for l in window.splitlines() for b in benign):
            continue  # client hung up mid-response — the request was fine
        hits.extend(l for l in block.splitlines()
                    if any(t in l for t in tokens))
    return hits[-k:]


def tail_lines(path: str, k: int = 2, window: int = 262144):
    """Last k log lines (not only errors): generation progress, requests.
    Reads up to `window` bytes of tail — enough history for scrolling."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - window))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [l for l in data.splitlines() if l.strip()]
    return lines[-k:]


_TOKPS_LIVE = re.compile(r"^TOKPS live #(\d+) ([\d.]+)")
_TOKPS_PRE = re.compile(r"^TOKPS prefill #(\d+) (\d+)")
_TOKPS_DONE = re.compile(r"^TOKPS done (.+)")


def tail_tokps(path: str):
    """(live_seq, live_rate, pre_seq, pre_rate, summary) from the last TOKPS
    lines of the log tail. The seq numbers advance ~1/s while the model
    generates, so an unchanged seq means "not generating right now" — that's
    how the monitor tells a live rate from a stale line."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 16384))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return -1, 0.0, -1, 0.0, ""
    live, pre = (-1, 0.0), (-1, 0.0)
    summary = ""
    for l in data.splitlines():
        if m := _TOKPS_LIVE.match(l):
            live = (int(m[1]), float(m[2]))
        elif m := _TOKPS_PRE.match(l):
            pre = (int(m[1]), int(m[2]))
        elif m := _TOKPS_DONE.match(l):
            summary = m[1]
    return live[0], live[1], pre[0], pre[1], summary


def _main(stdscr, model, mlx_port, lite_port, claude_cfg, small_port) -> None:
    client_cmds = (
        ("pi", "pi --model ollmlx/local"),
        ("omp", "omp --model ollmlx/local"),
        ("claude", f"claude --settings {claude_cfg}"),
    )
    start = time.time()
    cpu_hist = collections.deque(maxlen=240)
    tok_hist = collections.deque(maxlen=240)      # the big model
    tok_hist_s = collections.deque(maxlen=240)    # the helper
    tok = {"live_seq": -1, "pre_seq": -1, "idle": 1 << 30,
           "cur": 0.0, "txt": "idle", "summary": "", "peak": 0.0}
    tok_s = dict(tok)
    prev_title = ""
    log_off = 0   # lines from the live tail; 0 = following

    curses.curs_set(0)
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = 0
    for i, fg in ((1, curses.COLOR_GREEN), (2, curses.COLOR_YELLOW),
                  (3, curses.COLOR_RED), (4, curses.COLOR_CYAN)):
        curses.init_pair(i, fg, bg)
    B = curses.A_BOLD
    psutil.cpu_percent(percpu=True)  # priming: the first call is always 0
    watch_procs(small_port)
    stdscr.timeout(1000)
    curses.flushinp()  # drop terminal replies to curses init queries (ESC[...])

    def uptime() -> str:
        s = int(time.time() - start)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()
        y = [0]

        def add(yy, xx, text, attr=0):
            try:
                stdscr.addstr(yy, xx, text[: max(0, w - 1 - xx)], attr)
            except curses.error:
                pass

        def put(text="", attr=0):
            add(y[0], 0, text, attr)
            y[0] += 1

        def put_bar(yy, xx, label, bw, frac, suffix="", attr=0, fill=None):
            """label[███ filled with color, dim background] suffix — only
            glyphs present in any terminal font (no ░/▁▂▃)."""
            f = int(bw * max(0.0, min(1.0, frac)) + 0.5)
            add(yy, xx, label)
            add(yy, xx + len(label), "█" * f,
                fill if fill is not None
                else curses.color_pair(heat(frac * 100)))
            add(yy, xx + len(label) + f, "█" * (bw - f), curses.A_DIM)
            if suffix:
                add(yy, xx + len(label) + bw, suffix, attr)

        percpu = psutil.cpu_percent(percpu=True)
        total = sum(percpu) / len(percpu) if percpu else 0.0
        cpu_hist.append(total)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        procs = watch_procs(small_port)

        # --- header ---
        put(f" ollmlx — {model}", B | curses.color_pair(4))
        mlx = f"mlx :{mlx_port} ●up" if port_ok(mlx_port) else f"mlx :{mlx_port} ○down"
        small = (f"small :{small_port} ●up" if port_ok(small_port)
                 else f"small :{small_port} ○down") if small_port else ""
        lit = (f"litellm :{lite_port} ●up" if port_ok(lite_port)
               else f"litellm :{lite_port} ○down")
        chips = "   ".join(c for c in (mlx, small, lit) if c)
        put(f" uptime {uptime()}   {chips}   cores {len(percpu)}")
        put()

        # two-column grid the whole screen follows: left boxes span
        # x=0 .. rx-1, right ones start at rx and stop at the screen edge
        rx = w // 2 + 1
        lw = rx - 1
        rw = w - rx - 1

        # --- tok/s graphs: the big model and the helper side by side (a
        # single full-width graph when no helper runs). Each keeps its own
        # state, history and scale — 30 tok/s and 300 tok/s would flatten
        # each other on a shared axis. ---
        graphs = [(tok, tok_hist, "TOK/S main", LOGS[0][1])]
        if small_port:
            graphs.append((tok_s, tok_hist_s, "TOK/S small", MLX_SMALL_LOG))
        gh = min(4, max(0, (h - y[0]) - 28))

        def draw_toks(x0, width, st, hist, label, path):
            ls, lr, ps, pr, summary = tail_tokps(path)
            live_new, pre_new = ls != st["live_seq"], ps != st["pre_seq"]
            if live_new:
                st.update(idle=0, cur=lr, txt=f"{lr:5.1f} tok/s")
                st["peak"] = max(st["peak"], lr)   # run-wide, never resets
            elif pre_new:
                st.update(idle=0, cur=pr, txt=f"{pr:4.0f} tok/s prefill")
            else:
                st["idle"] += 1
                if st["idle"] > 3:   # a couple of grace ticks smooths skips
                    st.update(cur=0.0, txt="idle")
            st["live_seq"], st["pre_seq"], st["summary"] = ls, ps, summary
            # the graph is decode-only: prefill's 1000s of tok/s would blow
            # the scale; the zeros mark idle/prefill periods
            hist.append(lr if live_new else 0.0)
            scale = 25.0
            if tmax := max(hist):
                scale = max(25.0, -(-int(tmax) // 25) * 25)   # ceil to a 25 step
            # the bar line and the frame both fill the graph's column
            # exactly, so the two graphs line up with the boxes below
            pre = f" {label} ["
            xb = x0 + len(pre)
            suffix = f"] {st['txt']}"
            bw = max(4, width - len(pre) - len(suffix))
            f = int(bw * max(0.0, min(1.0, st["cur"] / scale)) + 0.5)
            add(y[0], x0, pre, B)
            add(y[0], xb, "█" * f, curses.color_pair(1))
            add(y[0], xb + f, "█" * (bw - f), curses.A_DIM)
            add(y[0], xb + bw, suffix, B)
            y[0] += 1
            if gh:
                # avg over active samples only — the zeros in the hist mark
                # idle/prefill, not real speed; max is the run-wide peak
                # (survives the rolling window; the scale stays window-based)
                active = [v for v in hist if v > 0]
                avg = f"{sum(active) / len(active):.0f}" if active else "–"
                stats = (f"· avg {avg} · max {st['peak']:.0f} "
                         if st["peak"] > 0 else "")
                base = f" tok/s 0–{scale:.0f} "
                t = (base + stats if len(base) + len(stats) <= width - 2
                     else base if len(base) <= width - 2 else "")
                add(y[0], x0,
                    "┌" + t + "─" * max(0, width - 2 - len(t)) + "┐",
                    curses.A_DIM)
                y[0] += 1
                data = list(hist)[-(width - 4):]
                off = (width - 4) - len(data)
                for i, v in enumerate(data):
                    ch = int(gh * v / scale + 0.5)
                    for r in range(gh - ch, gh):
                        add(y[0] + r, x0 + 2 + off + i, "█",
                            curses.color_pair(1))
                for r in range(gh):
                    add(y[0] + r, x0, "│", curses.A_DIM)
                    add(y[0] + r, x0 + width - 1, "│", curses.A_DIM)
                y[0] += gh
                add(y[0], x0, "└" + "─" * (width - 2) + "┘", curses.A_DIM)
                y[0] += 1
            txt = (f"last: {st['summary']}" if st["summary"]
                   else "no completions yet — tok/s appears during generation")
            add(y[0], x0 + 2, txt[: max(1, width - 4)], curses.A_DIM)
            y[0] += 1

        # both graphs share the same rows and line up with the column
        # grid below (left half / right half): draw each from the same
        # top line, then land below — they're always equal-height
        gcols = [(0, lw), (rx, rw)] if small_port else [(0, w - 1)]
        y_top = y[0]
        for (st, hist, label, path), (gx0, gwd) in zip(graphs, gcols):
            y[0] = y_top
            draw_toks(gx0, gwd, st, hist, label, path)
        y[0] = y_top + (gh + 4 if gh else 2)
        put()

        # --- tab title: load circle, model, its process RAM — nothing
        # else. The circle colors by load: max of CPU and RAM pressure,
        # the thing that actually kills the model is memory. ---
        load = max(total, vm.percent)
        title = _tab_title(model, procs, load)
        if title != prev_title:
            prev_title = title
            _set_title(title)

        # --- lower half, one grid: CPU left / cores right, then clients
        # left / processes right (boxed), then the RAM/swap bars and the
        # logs box at the very bottom; the q hint owns the last row ---
        top = y[0]
        usable = h - 1            # the last row is the q hint

        def clip(text, width):
            return text if len(text) <= width else text[:max(0, width - 1)] + "…"

        def box_top(yy, x0, wt, title=""):
            t = f" {title} " if title else ""
            add(yy, x0, "┌" + t + "─" * max(0, wt - 2 - len(t)) + "┐",
                curses.A_DIM)

        def box_bottom(yy, x0, wt):
            add(yy, x0, "└" + "─" * max(0, wt - 2) + "┘", curses.A_DIM)

        def box_sides(yy, x0, wt):
            add(yy, x0, "│", curses.A_DIM)
            add(yy, x0 + wt - 1, "│", curses.A_DIM)

        # --- CPU (left) / cores (right) ---
        ly = top
        bw_l = max(8, lw - 16)   # room for the CPU bar's "] 100.0%" suffix
        put_bar(ly, 0, " CPU [", bw_l, total / 100, f"] {total:5.1f}%", B)
        ly += 1
        cw = 6                   # "C00[██████]  30%" cell, 16 chars, pitch 18
        rcols = max(1, (rw - 4) // (cw + 12))
        crows = (len(percpu) + rcols - 1) // rcols
        band = usable - ly
        show_cores = band >= crows + 2
        cpu_g = 4 if band >= max(6, crows + 2) else (
            2 if band >= max(4, crows + 2) else 0)
        if cpu_g:
            box_top(ly, 0, lw, " CPU history ")
            gw_l = max(4, lw - 4)                  # data columns inside
            data = list(cpu_hist)[-gw_l:]
            off = gw_l - len(data)
            for i, v in enumerate(data):
                ch = int(cpu_g * v / 100 + 0.5)
                col = curses.color_pair(heat(v))
                for r in range(cpu_g - ch, cpu_g):
                    add(ly + 1 + r, 2 + off + i, "█", col)
            for r in range(cpu_g):
                box_sides(ly + 1 + r, 0, lw)
            ly += 1 + cpu_g
            box_bottom(ly, 0, lw)
            ly += 1
        ry = top
        if show_cores:
            box_top(ry, rx, rw, " cores ")
            for i in range(0, len(percpu), rcols):
                rr = i // rcols
                for side, v in enumerate(percpu[i:i + rcols]):
                    f = int(cw * v / 100 + 0.5)
                    add(ry + 1 + rr, rx + 2 + side * (cw + 12),
                        f"C{i + side:02d}[{'█' * f}{' ' * (cw - f)}]{v:4.0f}%",
                        curses.color_pair(heat(v)))
                box_sides(ry + 1 + rr, rx, rw)
            ry += 1 + crows
            box_bottom(ry, rx, rw)
            ry += 1

        # --- client launch commands (left) / processes (right) ---
        ours = 0
        proc_rows = []
        for name, p in procs:
            try:
                cpu, rss = p.cpu_percent(), p.memory_info().rss
                ours += rss
                proc_rows.append(
                    (f"{name:<14} pid {p.pid:<7} {cpu:5.1f}% cpu"
                     f"  {ui.human_bytes(rss):>7} rss"
                     f" {rss / vm.total * 100:4.1f}% ram"
                     f"  {p.num_threads()} thr",
                     curses.color_pair(heat(min(100.0, cpu)))))
            except Exception:
                pass
        if proc_rows:
            # RAM ledger: our procs' share vs the rest of the machine;
            # "available" is what macOS can still hand out before swapping.
            others = max(0, vm.used - ours)
            proc_rows.append(
                (f"llm {ui.human_bytes(ours)} · {ours / vm.total:.0%} of RAM"
                 f" · others {ui.human_bytes(others)}"
                 f" · available {ui.human_bytes(vm.available)}",
                 curses.color_pair(heat(vm.percent))))
        else:
            proc_rows.append(("no mlx/litellm processes found",
                              curses.color_pair(3)))
        gy = max(ly, ry) + 1
        band2 = usable - gy
        n_cli = min(len(client_cmds), band2 - 2) if band2 >= 3 else 0
        n_proc = min(len(proc_rows), band2 - 2) if band2 >= 3 else 0
        if n_cli:
            box_top(gy, 0, lw, " clients ")
            for i, (name, cmd) in enumerate(client_cmds[:n_cli]):
                add(gy + 1 + i, 2, clip(f"{name:<6}  {cmd}", lw - 4), B)
                box_sides(gy + 1 + i, 0, lw)
            box_bottom(gy + 1 + n_cli, 0, lw)
        if n_proc:
            box_top(gy, rx, rw, " processes ")
            for i, (text, attr) in enumerate(proc_rows[:n_proc]):
                add(gy + 1 + i, rx + 2, clip(text, rw - 4), attr)
                box_sides(gy + 1 + i, rx, rw)
            box_bottom(gy + 1 + n_proc, rx, rw)

        # --- RAM/swap bars right below the boxes; clamped so they never
        # run onto the q hint row on terminals too short for everything.
        # Available memory under LOW_AVAILABLE_GB pulses the MEM bar red —
        # that pressure is what OOMs the model (seen 2026-10-05/06). ---
        ended = max(n_cli, n_proc)
        by = gy + ended + 2 if ended else gy
        bars = 2 if sw.total else 1
        by = min(by, usable - bars - 1)   # + the top-mem line below the bars
        bw_b = max(8, w - 30)    # room for "]  77%  95.1G/128.0G"
        low_mem = _low_mem(vm)
        pulse = curses.A_BOLD if int(time.time()) % 2 else curses.A_REVERSE
        mem_suffix = (f"] {vm.percent:3.0f}%  ⚠ OOM risk: "
                      f"{ui.human_bytes(vm.available)} free" if low_mem else
                      f"] {vm.percent:3.0f}%"
                      f"  {ui.human_bytes(vm.used)}/{ui.human_bytes(vm.total)}")
        put_bar(by, 0, " MEM [", bw_b - (18 if low_mem else 0),
                vm.percent / 100, mem_suffix,
                curses.color_pair(3) if low_mem else B,
                curses.color_pair(3) | pulse if low_mem else None)
        if sw.total:
            put_bar(by + 1, 0, " SWP [", bw_b, sw.percent / 100,
                    f"] {sw.percent:3.0f}%"
                    f"  {ui.human_bytes(sw.used)}/{ui.human_bytes(sw.total)}")

        # heaviest third-party apps — the kill candidates under pressure
        eaters = top_eaters(3, {p.pid for _, p in procs})
        if eaters:
            txt = " · ".join(f"{name} {ui.human_bytes(rss)}({pid})"
                             for rss, pid, name in eaters)
            add(by + bars, 0,
                clip(f" top mem (kill <pid>): {txt}", w - 1),
                curses.color_pair(3) if low_mem else curses.A_DIM)
        _maybe_notify_low(low_mem, eaters, vm, time.monotonic())

        # --- logs: the very bottom box, its tail sitting just above the
        # bottom border; grows/shrinks with the terminal. ↑/↓ PgUp/PgDn
        # scroll back into history (the frame title shows the distance
        # from live), End/G returns to the live tail. ---
        lt = by + bars + (2 if eaters else 1)
        height = usable - lt                 # rows incl. top/bottom borders
        errs = [(tag, l) for tag, path in LOGS for l in tail_errors(path)][-3:]
        log_rows = [(f"[{tag}] {l}", curses.color_pair(3)) for tag, l in errs]
        if height >= 3:
            content = height - 2
            room = max(0, content - len(log_rows))
            merged = [(tag, l) for tag, path in LOGS
                      for l in tail_lines(path, room + log_off)] if room else []
            merged = merged[-(room + log_off):]
            log_off = min(log_off, max(0, len(merged) - room))
            if log_off:
                merged = merged[:len(merged) - log_off]
            view = merged[-room:]
            log_rows += [(f"[{tag}] {l}", curses.A_DIM) for tag, l in view]
            if not log_rows:
                log_rows = [("no errors · logs empty", curses.color_pair(1))]
            n_log = min(len(log_rows), content)
            box_top(lt, 0, w - 1,
                    f" logs  ·  {log_off} from live " if log_off else " logs ")
            first = lt + 1 + (content - n_log)   # tail hugs the bottom border
            for i, (text, attr) in enumerate(log_rows[-n_log:]):
                add(first + i, 2, clip(text, w - 5), attr)
                box_sides(first + i, 0, w - 1)
            box_bottom(lt + height - 1, 0, w - 1)

        logs = ("/tmp/mlx-server.log, /tmp/mlx-small.log, /tmp/litellm.log"
                ", /tmp/mlx-watchdog.log"
                if small_port else
                "/tmp/mlx-server.log, /tmp/litellm.log, /tmp/mlx-watchdog.log")
        add(h - 1, 0,
            f" q — quit · ↑↓ PgUp/PgDn scroll logs | logs: {logs}",
            curses.A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return
        page = max(1, (h - 6) // 2)
        if ch in (curses.KEY_UP, ord("k")):
            log_off += 1
        elif ch in (curses.KEY_DOWN, ord("j")):
            log_off = max(0, log_off - 1)
        elif ch == curses.KEY_PPAGE:
            log_off += page
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            log_off = max(0, log_off - page)
        elif ch in (curses.KEY_END, ord("G")):
            log_off = 0
