"""htop-style TUI monitor for the stack: tokens-per-second the model is
generating right now (current rate + history graph), per-core CPU + a
history graph, RAM/swap, port status, mlx/litellm processes with RSS, the
pi/omp/claude launch commands and a tail of the logs with errors. Blocks
until q/Ctrl-C (cli stops the servers).

The tok/s numbers come from TOKPS lines that our mlx_lm.server wrapper
(local_llm.mlxwrap) prints into /tmp/mlx-server.log ~once a second."""
import collections
import curses
import os
import re
import socket
import time

import psutil

from . import ui

LOGS = (("mlx", "/tmp/mlx-server.log"), ("litellm", "/tmp/litellm.log"))


def run(model: str, mlx_port: int, lite_port: int, claude_cfg: str) -> None:
    """Blocks until exit (q / Ctrl-C)."""
    ui.force_utf8_locale()
    ui.force_compatible_term()
    try:
        curses.wrapper(lambda scr: _main(scr, model, mlx_port, lite_port, claude_cfg))
    except KeyboardInterrupt:
        pass


def port_ok(port: int) -> bool:
    try:
        with socket.socket() as s:
            s.settimeout(0.4)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def watch_procs():
    """[(label, Process)] for mlx/litellm; process_iter caches instances,
    so cpu_percent() between frames yields meaningful deltas."""
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or ())
        except Exception:
            continue
        if "mlx_lm.server" in cmd or "local_llm.mlxwrap" in cmd:
            out.append(("mlx_lm.server", p))
        elif "litellm" in cmd and "--config" in cmd:
            out.append(("litellm", p))
    return out


def heat(pct: float) -> int:  # green -> yellow -> red
    return 1 if pct < 60 else 2 if pct < 85 else 3


def tail_errors(path: str, k: int = 2):
    """Last k error lines of a log. Only the tail of the file is read — cheap
    on a live log, no multi-megabyte scans."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    hits = [l for l in data.splitlines()
            if any(t in l for t in ("ERROR", "CRITICAL", "Traceback", "Exception"))]
    return hits[-k:]


def tail_lines(path: str, k: int = 2):
    """Last k log lines (not only errors): generation progress, requests."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - 16384))
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
            pre = (int(m[1]), float(m[2]))
        elif m := _TOKPS_DONE.match(l):
            summary = m[1]
    return live[0], live[1], pre[0], pre[1], summary


def _main(stdscr, model, mlx_port, lite_port, claude_cfg) -> None:
    client_cmds = (
        ("pi", "pi --model ollmlx/local"),
        ("omp", "omp --model ollmlx/local"),
        ("claude", f"claude --settings {claude_cfg}"),
    )
    start = time.time()
    cpu_hist = collections.deque(maxlen=240)
    tok_hist = collections.deque(maxlen=240)
    tok = {"live_seq": -1, "pre_seq": -1, "idle": 1 << 30,
           "cur": 0.0, "txt": "idle", "summary": ""}

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
    watch_procs()
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

        def put_bar(label, bw, frac, suffix="", attr=0):
            """label[███ filled with color, dim background] suffix — only
            glyphs present in any terminal font (no ░/▁▂▃)."""
            f = int(bw * max(0.0, min(1.0, frac)) + 0.5)
            add(y[0], 0, label)
            add(y[0], len(label), "█" * f, curses.color_pair(heat(frac * 100)))
            add(y[0], len(label) + f, "█" * (bw - f), curses.A_DIM)
            if suffix:
                add(y[0], len(label) + bw, suffix, attr)
            y[0] += 1

        percpu = psutil.cpu_percent(percpu=True)
        total = sum(percpu) / len(percpu) if percpu else 0.0
        cpu_hist.append(total)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        procs = watch_procs()

        # --- header ---
        put(f" ollmlx — {model}", B | curses.color_pair(4))
        mlx = f"mlx :{mlx_port} ●up" if port_ok(mlx_port) else f"mlx :{mlx_port} ○down"
        lit = (f"litellm :{lite_port} ●up" if port_ok(lite_port)
               else f"litellm :{lite_port} ○down")
        put(f" uptime {uptime()}   {mlx}   {lit}   cores {len(percpu)}")

        # height budget: the bottom of the window (mem/clients/procs/errors/
        # logs) is always visible; the tok/s + CPU graphs and per-core rows
        # share what's left. Fixed chrome: 2 status-bar rows with blanks (4),
        # 2 graph frames (4), the tok/s summary line (1), and cores' blank +
        # frame (3, counted with cores_h below).
        tail = 2 + (1 if sw.total else 0) + 5 + 1 + max(1, len(procs)) + 6 + 6 + 2
        mid = h - y[0] - tail
        bw = max(10, w - 27)
        half = (len(percpu) + 1) // 2
        cores_h = half if mid >= half + 4 else 0
        avail = max(0, mid - cores_h - 12)
        tokps_h = min(3, avail) if avail >= 5 else 0
        cpu_g = min(8, avail - tokps_h)
        if cpu_g < 2:
            cpu_g = 0

        # --- tok/s: what the model is producing right now ---
        ls, lr, ps, pr, summary = tail_tokps(LOGS[0][1])
        live_new, pre_new = ls != tok["live_seq"], ps != tok["pre_seq"]
        if live_new:
            tok.update(idle=0, cur=lr, txt=f"{lr:5.1f} tok/s")
        elif pre_new:
            tok.update(idle=0, cur=pr, txt=f"{pr:4.0f} tok/s prefill")
        else:
            tok["idle"] += 1
            if tok["idle"] > 3:   # a couple of grace ticks smooths cadence skips
                tok.update(cur=0.0, txt="idle")
        tok["live_seq"], tok["pre_seq"], tok["summary"] = ls, ps, summary
        # the graph is decode-only: prefill's 1000s of tok/s would blow the
        # scale; the zeros mark idle/prefill periods
        tok_hist.append(lr if live_new else 0.0)
        scale = 25.0
        if tmax := max(tok_hist):
            scale = max(25.0, -(-int(tmax) // 25) * 25)   # ceil to a 25 step

        put()
        f = int((bw - 2) * max(0.0, min(1.0, tok["cur"] / scale)) + 0.5)
        add(y[0], 0, " TOK/S [")   # 8 wide: the bar ends where the CPU bar's does
        add(y[0], 8, "█" * f, curses.color_pair(1))
        add(y[0], 8 + f, "█" * (bw - 2 - f), curses.A_DIM)
        add(y[0], 6 + bw, f"] {tok['txt']}", B)
        y[0] += 1
        if tokps_h:
            gl, gr = 5, 6 + bw
            title = f" tok/s 0–{scale:.0f} " if bw >= 20 else ""
            add(y[0], gl,
                "┌" + title + "─" * max(0, gr - gl + 1 - 2 - len(title)),
                curses.A_DIM)
            y[0] += 1
            data = list(tok_hist)[-bw:]
            off = bw - len(data)
            for i, v in enumerate(data):
                ch = int(tokps_h * v / scale + 0.5)
                for r in range(tokps_h - ch, tokps_h):
                    add(y[0] + r, 6 + off + i, "█", curses.color_pair(1))
            for r in range(tokps_h):
                add(y[0] + r, gl, "│", curses.A_DIM)
                add(y[0] + r, gr, "│", curses.A_DIM)
            y[0] += tokps_h
            add(y[0], gl, "└" + "─" * (gr - gl - 1) + "┘", curses.A_DIM)
            y[0] += 1
        add(y[0], 0, f" last: {tok['summary']}" if tok["summary"]
            else " no completions yet — tok/s appears during generation",
            curses.A_DIM)
        y[0] += 1

        # --- cpu: total + framed history graph (bars, like htop) ---
        put()
        put_bar(" CPU [", bw, total / 100, f"] {total:5.1f}%", B)
        if cpu_g:
            # frame: │ at col 5 (under '[' of the total bar), data 6..6+bw-1,
            # │ at 6+bw (under ']'). Without it the graph edges are unreadable.
            gl, gr = 5, 6 + bw
            title = " CPU history " if bw >= 15 else ""
            add(y[0], gl,
                "┌" + title + "─" * max(0, gr - gl + 1 - 2 - len(title)) + "┐",
                curses.A_DIM)
            y[0] += 1
            data = list(cpu_hist)[-bw:]
            off = bw - len(data)
            for i, v in enumerate(data):
                ch = int(cpu_g * v / 100 + 0.5)
                col = curses.color_pair(heat(v))
                for r in range(cpu_g - ch, cpu_g):
                    add(y[0] + r, 6 + off + i, "█", col)
            for r in range(cpu_g):
                add(y[0] + r, gl, "│", curses.A_DIM)
                add(y[0] + r, gr, "│", curses.A_DIM)
            y[0] += cpu_g
            add(y[0], gl, "└" + "─" * (gr - gl - 1) + "┘", curses.A_DIM)
            y[0] += 1

        # --- cpu: per core, two columns in a shared frame ---
        if cores_h:
            put()
            cw = max(6, (w - 26) // 2)
            cl, cr = 0, 2 * cw + 23   # │ at 0, cores from 1, │ after the second column
            title = " cores " if cr >= 12 else ""
            add(y[0], cl,
                "┌" + title + "─" * max(0, cr - cl + 1 - 2 - len(title)) + "┐",
                curses.A_DIM)
            y[0] += 1
            for i in range(half):
                for side, idx in enumerate((i, i + half)):
                    if idx >= len(percpu):
                        continue
                    v = percpu[idx]
                    f = int(cw * v / 100 + 0.5)
                    add(y[0], 1 + side * (cw + 12),
                        f"C{idx:02d}[{'█' * f}{' ' * (cw - f)}]{v:4.0f}%",
                        curses.color_pair(heat(v)))
                add(y[0], cl, "│", curses.A_DIM)
                add(y[0], cr, "│", curses.A_DIM)
                y[0] += 1
            add(y[0], cl, "└" + "─" * (cr - cl - 1) + "┘", curses.A_DIM)
            y[0] += 1

        # --- memory ---
        put()
        put_bar(" MEM [", bw, vm.percent / 100,
                f"] {vm.percent:3.0f}%  {ui.human_bytes(vm.used)}/{ui.human_bytes(vm.total)}")
        if sw.total:
            put_bar(" SWP [", bw, sw.percent / 100,
                    f"] {sw.percent:3.0f}%  {ui.human_bytes(sw.used)}/{ui.human_bytes(sw.total)}")

        # --- client launch commands: always in view ---
        put()
        put(" client launch commands:", curses.A_DIM)
        for name, cmd in client_cmds:
            put(f"   {name:<6}  {cmd}", B)

        # --- our processes ---
        put()
        for name, p in procs:
            try:
                cpu, rss = p.cpu_percent(), p.memory_info().rss
                put(f" {name:<14} pid {p.pid:<7} {cpu:5.1f}% cpu"
                    f"  {ui.human_bytes(rss):>7} rss  {p.num_threads()} thr",
                    curses.color_pair(heat(min(100.0, cpu))))
            except Exception:
                pass
        if not procs:
            put(" no mlx/litellm processes found", curses.color_pair(3))

        # --- recent errors from the stack logs ---
        put()
        put(" recent errors:", curses.A_DIM)
        errs = [(tag, l) for tag, path in LOGS for l in tail_errors(path)]
        if errs:
            for tag, l in errs[-4:]:
                put(f" [{tag}] {l}", curses.color_pair(3))
        else:
            put(" no errors", curses.color_pair(1))

        # --- log tail: what the servers are doing right now ---
        put()
        put(" latest logs:", curses.A_DIM)
        shown = [(tag, l) for tag, path in LOGS for l in tail_lines(path)]
        if shown:
            for tag, l in shown[-4:]:
                put(f" [{tag}] {l}", curses.A_DIM)
        else:
            put(" logs empty", curses.A_DIM)

        # --- footer ---
        put()
        put(" q — quit | logs: /tmp/mlx-server.log, /tmp/litellm.log", curses.A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return
