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
from .servers import MLX_SMALL_LOG

LOGS = (("mlx", "/tmp/mlx-server.log"), ("litellm", "/tmp/litellm.log"))


def run(model: str, mlx_port: int, lite_port: int, claude_cfg: str,
        small_port: int = 0) -> None:
    """Blocks until exit (q / Ctrl-C). small_port=0 — no helper model."""
    global LOGS
    if small_port:
        LOGS = LOGS + (("mlx-small", MLX_SMALL_LOG),)
    ui.force_utf8_locale()
    ui.force_compatible_term()
    try:
        curses.wrapper(lambda scr: _main(
            scr, model, mlx_port, lite_port, claude_cfg, small_port))
    except KeyboardInterrupt:
        pass
    finally:
        # hand the tab back its previous title (xterm save/restore stack;
        # emulators without it just keep our last text)
        try:
            sys.stdout.write("\x1b[23t")
            sys.stdout.flush()
        except OSError:
            pass


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
    the sequence passes through untouched."""
    try:
        sys.stdout.write(f"\x1b]0;{text}\x07")
        sys.stdout.flush()
    except OSError:
        pass


def _tab_title(model: str, procs) -> str:
    """Tab text: short model name + RAM of the model processes (mlx and
    the helper; litellm is a proxy, not a model — not counted)."""
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
    return f"ollmlx · {short} · {ui.human_bytes(ram)}" if ram else f"ollmlx · {short}"


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
    try:
        sys.stdout.write("\x1b[22t")  # push the tab's title; run() pops it
        sys.stdout.flush()
    except OSError:
        pass
    prev_title = ""

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

        def put_bar(yy, xx, label, bw, frac, suffix="", attr=0):
            """label[███ filled with color, dim background] suffix — only
            glyphs present in any terminal font (no ░/▁▂▃)."""
            f = int(bw * max(0.0, min(1.0, frac)) + 0.5)
            add(yy, xx, label)
            add(yy, xx + len(label), "█" * f, curses.color_pair(heat(frac * 100)))
            add(yy, xx + len(label) + f, "█" * (bw - f), curses.A_DIM)
            if suffix:
                add(yy, xx + len(label) + bw, suffix, attr)

        percpu = psutil.cpu_percent(percpu=True)
        total = sum(percpu) / len(percpu) if percpu else 0.0
        cpu_hist.append(total)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        procs = watch_procs(small_port)

        # --- tab title: what this tab runs, with live RAM of the models ---
        title = _tab_title(model, procs)
        if title != prev_title:
            prev_title = title
            _set_title(title)

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

        # --- tok/s graphs: the big model and the helper side by side (a
        # single full-width graph when no helper runs). Each keeps its own
        # state, history and scale — 30 tok/s and 300 tok/s would flatten
        # each other on a shared axis. ---
        graphs = [(tok, tok_hist, "TOK/S main", LOGS[0][1])]
        if small_port:
            graphs.append((tok_s, tok_hist_s, "TOK/S small", MLX_SMALL_LOG))
        gw = (w - 2) // 2 if len(graphs) > 1 else w
        # the graphs take what's left after the fixed lower half (the
        # right column is ~18 rows; the left one shrinks itself separately)
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
            # leave room for the "] 1234 tok/s prefill" suffix so a
            # side-by-side neighbour never gets written over
            bw = max(12, width - 26)
            gl, gr = x0 + 5, x0 + 6 + bw     # frame edges, data between
            xb = x0 + len(f" {label} [")     # bar data starts after '['
            bar_w = max(2, gr - 1 - xb)
            f = int(bar_w * max(0.0, min(1.0, st["cur"] / scale)) + 0.5)
            add(y[0], x0, f" {label} [")
            add(y[0], xb, "█" * f, curses.color_pair(1))
            add(y[0], xb + f, "█" * (bar_w - f), curses.A_DIM)
            add(y[0], gr, f"] {st['txt']}", B)
            y[0] += 1
            if gh:
                # avg over active samples only — the zeros in the hist mark
                # idle/prefill, not real speed; max is the run-wide peak
                # (survives the rolling window; the scale stays window-based)
                active = [v for v in hist if v > 0]
                avg = f"{sum(active) / len(active):.0f}" if active else "–"
                stats = (f" · avg {avg} · max {st['peak']:.0f} "
                         if st["peak"] > 0 else "")
                base = f" tok/s 0–{scale:.0f} "
                t = (base[:-1] + stats if len(base) + len(stats) <= bw
                     else base if len(base) <= bw else "")
                add(y[0], gl,
                    "┌" + t + "─" * max(0, gr - gl + 1 - 2 - len(t)) + "┐",
                    curses.A_DIM)
                y[0] += 1
                data = list(hist)[-bw:]
                off = bw - len(data)
                for i, v in enumerate(data):
                    ch = int(gh * v / scale + 0.5)
                    for r in range(gh - ch, gh):
                        add(y[0] + r, gl + 1 + off + i, "█",
                            curses.color_pair(1))
                for r in range(gh):
                    add(y[0] + r, gl, "│", curses.A_DIM)
                    add(y[0] + r, gr, "│", curses.A_DIM)
                y[0] += gh
                add(y[0], gl, "└" + "─" * (gr - gl - 1) + "┘", curses.A_DIM)
                y[0] += 1
            txt = (f"last: {st['summary']}" if st["summary"]
                   else "no completions yet — tok/s appears during generation")
            add(y[0], gl + 1, txt[: max(1, gr - gl - 2)], curses.A_DIM)
            y[0] += 1

        # both graphs share the same rows: draw each from the same top
        # line (draw_toks advances y, so reset it per graph), then land
        # below whichever is taller — they're always equal-height
        y_top = y[0]
        for i, (st, hist, label, path) in enumerate(graphs):
            y[0] = y_top
            draw_toks(i * (gw + 2), min(gw, w - i * (gw + 2)),
                      st, hist, label, path)
        y[0] = y_top + (gh + 4 if gh else 2)
        put()

        # --- lower half: two columns instead of one long stack — the
        # system (CPU / cores / RAM) on the left, the stack itself
        # (clients, processes, errors, logs) on the right ---
        top = y[0]
        rx = w // 2 + 1

        # right column: clients, our processes, errors, logs
        ry = top
        add(ry, rx, " client launch commands:", curses.A_DIM)
        ry += 1
        for name, cmd in client_cmds:
            add(ry, rx, f"   {name:<6}  {cmd}", B)
            ry += 1
        ry += 1
        ours = 0
        for name, p in procs:
            try:
                cpu, rss = p.cpu_percent(), p.memory_info().rss
                ours += rss
                add(ry, rx, f" {name:<14} pid {p.pid:<7} {cpu:5.1f}% cpu"
                    f"  {ui.human_bytes(rss):>7} rss"
                    f" {rss / vm.total * 100:4.1f}% ram"
                    f"  {p.num_threads()} thr",
                    curses.color_pair(heat(min(100.0, cpu))))
                ry += 1
            except Exception:
                pass
        if not procs:
            add(ry, rx, " no mlx/litellm processes found", curses.color_pair(3))
            ry += 1
        else:
            # RAM ledger: our procs' share vs the rest of the machine.
            # "others" = the kernel's used minus us (system, apps, the
            # compressor — everything that keeps running when the stack is
            # down); "available" is what macOS can still hand out before
            # swapping starts.
            others = max(0, vm.used - ours)
            add(ry, rx,
                f" llm {ui.human_bytes(ours)} · {ours / vm.total:.0%} of RAM"
                f" · others {ui.human_bytes(others)}"
                f" · available {ui.human_bytes(vm.available)}",
                curses.color_pair(heat(vm.percent)))
            ry += 1
        ry += 1
        add(ry, rx, " recent errors:", curses.A_DIM)
        ry += 1
        errs = [(tag, l) for tag, path in LOGS for l in tail_errors(path)]
        if errs:
            for tag, l in errs[-4:]:
                add(ry, rx, f" [{tag}] {l}", curses.color_pair(3))
                ry += 1
        else:
            add(ry, rx, " no errors", curses.color_pair(1))
            ry += 1
        ry += 1
        add(ry, rx, " latest logs:", curses.A_DIM)
        ry += 1
        shown = [(tag, l) for tag, path in LOGS for l in tail_lines(path)]
        if shown:
            for tag, l in shown[-4:]:
                add(ry, rx, f" [{tag}] {l}", curses.A_DIM)
                ry += 1
        else:
            add(ry, rx, " logs empty", curses.A_DIM)
            ry += 1

        # left column: CPU bar + history graph + per-core grid + RAM bars;
        # shrinks itself to the height that remains (graph height first,
        # then the cores grid, then the graph) instead of pushing the
        # footer off-screen
        ly = top
        bw_l = max(8, rx - 28)   # room for the "]  86%  106.9G/128.0G" suffix
        put_bar(ly, 0, " CPU [", bw_l, total / 100, f"] {total:5.1f}%", B)
        ly += 1
        mem_rows = 2 if sw.total else 1
        avail_l = (h - 3) - ly
        # ~15-char "C00[██]100%" cells; a single cell per row is pointless
        cols_n = max(2, (bw_l - 6) // 15)
        cores_rows = (len(percpu) + cols_n - 1) // cols_n + 2

        def left_h(g, cores):
            return ((g + 2 if g else 0) + 1
                    + (cores_rows + 1 if cores else 0) + mem_rows)

        cpu_g = 4 if avail_l >= left_h(4, True) else 2
        cores = True
        if avail_l < left_h(cpu_g, cores):
            cpu_g = 2 if avail_l >= left_h(2, True) else 0
        if avail_l < left_h(cpu_g, cores):
            cores = False
        if cpu_g:
            gl, gr = 5, 6 + bw_l
            title = " CPU history " if bw_l >= 15 else ""
            add(ly, gl,
                "┌" + title + "─" * max(0, gr - gl + 1 - 2 - len(title)) + "┐",
                curses.A_DIM)
            ly += 1
            data = list(cpu_hist)[-bw_l:]
            off = bw_l - len(data)
            for i, v in enumerate(data):
                ch = int(cpu_g * v / 100 + 0.5)
                col = curses.color_pair(heat(v))
                for r in range(cpu_g - ch, cpu_g):
                    add(ly + r, 6 + off + i, "█", col)
            for r in range(cpu_g):
                add(ly + r, gl, "│", curses.A_DIM)
                add(ly + r, gr, "│", curses.A_DIM)
            ly += cpu_g
            add(ly, gl, "└" + "─" * (gr - gl - 1) + "┘", curses.A_DIM)
            ly += 1
        ly += 1
        if cores:
            cl, cr = 5, 6 + bw_l
            cw = max(2, (bw_l - 6) // cols_n - 12)
            title = " cores " if cr - cl >= 12 else ""
            add(ly, cl,
                "┌" + title + "─" * max(0, cr - cl + 1 - 2 - len(title)) + "┐",
                curses.A_DIM)
            ly += 1
            for i in range(0, len(percpu), cols_n):
                for side, v in enumerate(percpu[i:i + cols_n]):
                    f = int(cw * v / 100 + 0.5)
                    add(ly, 6 + side * (cw + 12),
                        f"C{i + side:02d}[{'█' * f}{' ' * (cw - f)}]{v:4.0f}%",
                        curses.color_pair(heat(v)))
                add(ly, cl, "│", curses.A_DIM)
                add(ly, cr, "│", curses.A_DIM)
                ly += 1
            add(ly, cl, "└" + "─" * (cr - cl - 1) + "┘", curses.A_DIM)
            ly += 1
            ly += 1
        put_bar(ly, 0, " MEM [", bw_l, vm.percent / 100,
                f"] {vm.percent:3.0f}%"
                f"  {ui.human_bytes(vm.used)}/{ui.human_bytes(vm.total)}")
        ly += 1
        if sw.total:
            put_bar(ly, 0, " SWP [", bw_l, sw.percent / 100,
                    f"] {sw.percent:3.0f}%"
                    f"  {ui.human_bytes(sw.used)}/{ui.human_bytes(sw.total)}")
            ly += 1

        # --- footer, under whichever column ended lower ---
        logs = ("/tmp/mlx-server.log, /tmp/mlx-small.log, /tmp/litellm.log"
                if small_port else
                "/tmp/mlx-server.log, /tmp/litellm.log")
        add(max(ly, ry) + 1, 0, f" q — quit | logs: {logs}", curses.A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            return
