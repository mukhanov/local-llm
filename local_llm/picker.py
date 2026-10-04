"""Model picker: a curses TUI (scroll/search/load-more/delete) and a plain
text fallback for terminals that can't run curses (not a tty, etc.).

Entry points:
  pick_model(cfg) -> str | None   — TUI, falls back to the text list on env errors
  pick_plain(cfg, ...) -> str | None
"""
import curses
import re
import sys

from . import hf, llmfit, models, ui


class EnvError(Exception):
    """Terminal/environment is not suitable for curses — use the text fallback."""


def pick_model(cfg) -> str | None:
    """TUI picker; None means the user cancelled. EnvError never escapes:
    it's caught here and switches to the plain-text fallback."""
    sys_ram = ui.total_ram_gb()
    recommended = models.recommend_for_ram(sys_ram)
    lf = llmfit.load(cfg.ollmlx_home)  # {repo: entry} | None
    try:
        return pick_tui(cfg, sys_ram, recommended, lf)
    except EnvError:
        ui.warn("TUI unavailable in this terminal — showing a plain list")
        return pick_plain(cfg, sys_ram, recommended, lf)


# --- curses TUI ----------------------------------------------------------------

LIMIT = 50  # HF API page size for load-more


def pick_tui(cfg, sys_ram: int, recommended: str, lf: dict | None) -> str | None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise EnvError("not a tty")
    ui.force_utf8_locale()
    ui.force_compatible_term()
    result = {}

    try:
        curses.wrapper(
            lambda scr: _tui(scr, cfg, sys_ram, recommended, lf, result))
    except curses.error as exc:
        raise EnvError(str(exc)) from exc

    return result.get("choice")


def _tui(stdscr, cfg, sys_ram: int, recommended: str, lf: dict | None,
         result: dict) -> None:
    installed = hf.scan_installed(cfg.hf_hub)   # (rid, size, downloading)
    inst_ids = {t[0] for t in installed}
    remote, skip, has_more = [], 0, True
    ranked: list[tuple[float, str, int, int]] = []   # (score, repo, dl, likes)
    sort_mode, query, searching = "fit", "", False
    scope = "community"   # "community": mlx-community org | "all": any author
    cursor = scroll_off = 0
    status, confirm = "", None

    curses.curs_set(0)
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = 0
    for i, fg in ((1, curses.COLOR_GREEN), (2, curses.COLOR_YELLOW),
                  (3, curses.COLOR_RED), (4, curses.COLOR_CYAN)):
        curses.init_pair(i, fg, bg)
    B, R, DIM = curses.A_BOLD, curses.A_REVERSE, curses.A_DIM
    stdscr.keypad(True)

    def col(n):
        return curses.color_pair(n)

    def entries():
        """ONE list ranked by score — downloaded and not-yet-downloaded
        together, so the sort is honest (the real #1 sits on the first
        row). Rows carry inst=True (✓, size on disk) for what's local."""

        def srt_key(e):
            # llmfit score when the catalog knows the model, the local
            # heuristic otherwise — both are 0–100, so mixing is fine
            return (-(e["lfsc"]["score"] if e["lfsc"] else e["sc"]), -e["dl"])

        ents = []
        q = query.lower()
        for rid, sz, dling in installed:
            if q and q not in rid.lower():
                continue
            e = (lf or {}).get(rid)
            sc, rn, total, active = models.score(sys_ram, rid, 0)
            rn = rn if rn is not None else (  # exotic quant names don't parse
                e.get("recommended_ram_gb") if e else None)
            ents.append(dict(id=rid, inst=True, size=sz, dling=dling,
                             dl=(e.get("hf_downloads") or 0) if e else 0,
                             likes=(e.get("hf_likes") or 0) if e else 0,
                             sc=sc, rn=rn, total=total, active=active,
                             ctx=llmfit.ctx_str(e) if e else None,
                             tools=bool(e and llmfit.has_tools(e)),
                             lfsc=llmfit.score(e, sys_ram) if e else None))
        for m in remote:
            if m["id"] in inst_ids or (q and q not in m["id"].lower()):
                continue
            e = (lf or {}).get(m["id"])
            sc, rn, total, active = models.score(sys_ram, m["id"], m["downloads"])
            rn = rn if rn is not None else (
                e.get("recommended_ram_gb") if e else None)
            ents.append(dict(id=m["id"], inst=False, size="", dling=False,
                             dl=m["downloads"],
                             likes=m["likes"], sc=sc, rn=rn, total=total,
                             active=active, ctx=llmfit.ctx_str(e) if e else None,
                             tools=bool(e and llmfit.has_tools(e)),
                             lfsc=llmfit.score(e, sys_ram) if e else None))
        ents.sort(key=srt_key if sort_mode == "fit"
                  else lambda e: (-e["dl"], -(e["lfsc"]["score"] if e["lfsc"] else e["sc"])))
        return ents

    def build_ranked():
        """The scope's catalog models, best llmfit score for THIS machine
        first — the same ordering the llmfit CLI prints, not just whatever
        HF's top-downloads page happens to contain. Empty when the catalog
        is unavailable (offline without a cache)."""
        nonlocal ranked
        ranked = []
        if not lf:
            return
        for rid, e in lf.items():
            if scope == "community" and not rid.startswith("mlx-community/"):
                continue
            s = llmfit.score(e, sys_ram)
            ranked.append((s["score"] if s else 0.0, rid,
                           e.get("hf_downloads") or 0, e.get("hf_likes") or 0))
        ranked.sort(key=lambda t: -t[0])

    def load_more():
        nonlocal skip, remote, has_more, status
        if not has_more:
            status = " all pages loaded"
            return
        label = "all MLX" if scope == "all" else "mlx-community"
        if ranked:
            # local ranking of the whole catalog slice — no network needed
            page = ranked[skip:skip + LIMIT]
            skip += LIMIT
            known = {m["id"] for m in remote}
            remote += [{"id": rid, "downloads": dl, "likes": likes}
                       for _sc, rid, dl, likes in page if rid not in known]
            has_more = skip < len(ranked)
            status = (f" llmfit-ranked ({label}): {len(remote)}/{len(ranked)}"
                      f"{'' if has_more else ' — all'} (l — more)")
            return
        status = f" loading models {skip + 1}–{skip + LIMIT} from HF ({label})…"
        draw()
        stdscr.refresh()
        try:
            page = hf.fetch_page(skip, LIMIT, cfg.hf_token, scope)
        except Exception as exc:
            status = f" HF API unavailable: {exc}"
            return
        skip += LIMIT
        known = {m["id"] for m in remote}
        remote += [m for m in page if m["id"] not in known]
        has_more = bool(page)
        status = (f" loaded from HF ({label}): {len(remote)}"
                  f"{'' if has_more else ' — all pages'} (l — more)")

    def run_search():
        """Fetch the search term server-side: local / filters only what's
        already loaded, but a fresh re-quant may not be on any page yet."""
        nonlocal remote, status
        status = f" searching HF for “{query}”…"
        draw()
        stdscr.refresh()
        try:
            found = hf.fetch_page(0, LIMIT, cfg.hf_token, scope, query)
        except Exception as exc:
            status = f" HF search failed: {exc}"
            return
        known = {m["id"] for m in remote}
        fresh = [m for m in found if m["id"] not in known]
        remote += fresh
        status = f" HF search: +{len(fresh)} model(s)"

    def draw():
        nonlocal cursor, scroll_off
        ents = entries()
        cursor = max(0, min(cursor, len(ents) - 1)) if ents else 0
        rows, cur_row = [], 0
        if ents:
            src = "all MLX" if scope == "all" else "mlx-community"
            how = "by score" if sort_mode == "fit" else "by downloads"
            rows.append(("hdr", f"✓ on disk · ⬇ downloading · {src} · {how}"))
        else:
            rows = [("hdr", "nothing found" if query else "list is empty")]
        for ei, e in enumerate(ents):
            rows.append(("ent", e))
            if ei == cursor:
                cur_row = len(rows) - 1

        h, w = stdscr.getmaxyx()
        stdscr.erase()

        def add(yy, xx, text, attr=0):
            try:
                stdscr.addstr(yy, xx, text[: max(0, w - 1 - xx)], attr)
            except curses.error:
                pass

        add(0, 0, f" model picker — RAM {sys_ram}GB · {ui.gpu_name()} · sort: "
            f"{'by score' if sort_mode == 'fit' else 'by downloads'} (s — toggle)",
            B | col(4))

        list_h = max(1, h - 5)
        if cur_row < scroll_off:
            scroll_off = cur_row
        if cur_row >= scroll_off + list_h:
            scroll_off = cur_row - list_h + 1
        scroll_off = max(0, min(scroll_off, max(0, len(rows) - list_h)))

        for i in range(scroll_off, min(len(rows), scroll_off + list_h)):
            kind, payload = rows[i]
            yy = 2 + (i - scroll_off)
            if kind == "hdr":
                add(yy, 0, f"── {payload} " + "─" * max(0, w - len(payload) - 6),
                    DIM | col(4))
                continue
            e = payload
            if e["inst"]:
                right = (f"{e['size']} so far" if e["dling"]
                         else f"{e['size']} on disk")
                if e["ctx"]:
                    right += f" · ctx {e['ctx']}"
            else:
                ram = f"~{e['rn']:.0f}GB" if e["rn"] is not None else "?GB"
                moe = " · MoE" if e["active"] else ""
                right = f"{ram} RAM{moe} · ↓{ui.human_downloads(e['dl'])} · ⭐{e['likes']}"
                if e["ctx"]:
                    right += f" · ctx {e['ctx']}"
                if e["tools"]:
                    right += " · tools"
                if e["rn"] is not None and e["rn"] > sys_ram:
                    right += " ⚠won't fit"
            if e["lfsc"]:
                right = f"⚡{e['lfsc']['score']:.0f} · " + right
            mark = ("▸ " if i == cur_row else "  ") + \
                ("⬇ " if e["inst"] and e["dling"]
                 else "✓ " if e["inst"] else "  ")
            rec = " ★" if e["id"] == recommended else ""
            name_w = max(10, w - len(mark) - len(right) - 3)
            nm = e["id"] if len(e["id"]) <= name_w else e["id"][:name_w - 1] + "…"
            pad = max(1, w - 1 - len(mark) - len(nm) - len(rec) - len(right))
            if i == cur_row:
                attr = R | B
            elif e["inst"] and e["dling"]:
                attr = col(2)   # in flight: yellow, not green
            elif e["inst"]:
                attr = col(1)
            elif e["rn"] is not None and e["rn"] > sys_ram:
                attr = col(3)
            else:
                attr = 0
            add(yy, 0, f"{mark}{nm}{rec}{' ' * pad}{right}", attr)

        # llmfit breakdown for the row under the cursor
        cur_e = ents[cursor] if ents else None
        if cur_e and cur_e["lfsc"]:
            d = cur_e["lfsc"]
            extra = ""
            if cur_e["dling"]:   # rough progress: bytes so far vs. weight estimate
                m = re.match(r"([\d.]+)([GM])", cur_e["size"])
                if m:
                    gb = float(m.group(1)) / (1 if m.group(2) == "G" else 1024)
                    if d["mem"] > 0:
                        extra = f" · ⬇ ~{min(99, gb / d['mem'] * 100):.0f}%"
            add(h - 3, 0, f" ⚡{d['score']:.1f} = quality {d['quality']:.0f} ·"
                f" speed {d['speed']:.0f} ({d['tps']:.0f} tok/s) ·"
                f" fit {d['fit']:.0f} · ctx {d['context']:.0f} ·"
                f" ~{d['mem']:.0f}GB of {sys_ram}GB · {d['use_case']}{extra}", DIM)

        if searching:
            hint = f" search: {query}█   Enter — apply (+HF) · Esc — clear"
        else:
            hint = (" j/k↑↓ PgUp/PgDn g/G · / search · l more · s sort ·"
                    " a all⇄community · d delete · Enter select · q cancel")
        add(h - 2, 0, hint, DIM)
        if confirm is not None:
            add(h - 1, 0, f" delete {confirm['id']} ({confirm['size']}) from disk? y/n",
                B | col(3))
        elif status:
            add(h - 1, 0, status, DIM)
        stdscr.refresh()

    status = " ranking MLX models for this machine…"
    draw()
    stdscr.refresh()
    build_ranked()
    load_more()   # the ranked catalog when available, HF pages otherwise

    while True:
        draw()
        ch = stdscr.getch()
        ents = entries()

        if confirm is not None:
            if ch in (ord("y"), ord("Y")):
                try:
                    hf.delete_model(cfg.hf_hub, confirm["id"])
                    status = f" deleted: {confirm['id']} (freed {confirm['size']})"
                except OSError as exc:
                    status = f" could not delete: {exc}"
                installed = hf.scan_installed(cfg.hf_hub)
                inst_ids = {t[0] for t in installed}
            else:
                status = ""
            confirm = None
        elif searching:
            if ch in (10, 13, curses.KEY_ENTER):
                searching = False
                cursor = scroll_off = 0
                if query:
                    run_search()   # local filter + HF query in one Enter
            elif ch == 27:
                searching, query = False, ""   # Esc clears the filter too
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                query = query[:-1]
            elif 32 <= ch < 127:
                query += chr(ch)
            cursor = scroll_off = 0
        elif ch in (curses.KEY_ENTER, 10, 13):
            if ents:
                result["choice"] = ents[cursor]["id"]
                return
        elif ch in (ord("q"), ord("Q"), 27, -1):
            return
        elif ch in (curses.KEY_UP, ord("k")):
            cursor = max(0, cursor - 1)
        elif ch in (curses.KEY_DOWN, ord("j")):
            if cursor < len(ents) - 1:
                cursor += 1
            elif has_more:
                load_more()   # past the end — fetch the next page
        elif ch == curses.KEY_PPAGE:
            cursor = max(0, cursor - max(1, len(ents) // 2))
        elif ch in (curses.KEY_NPAGE, ord(" ")):
            cursor = min(len(ents) - 1, cursor + max(1, len(ents) // 2)) if ents else 0
        elif ch in (curses.KEY_HOME, ord("g")):
            cursor = scroll_off = 0
        elif ch in (curses.KEY_END, ord("G")):
            cursor = max(0, len(ents) - 1)
        elif ch == ord("l"):
            load_more()
        elif ch == ord("s"):
            sort_mode = "dl" if sort_mode == "fit" else "fit"
            cursor = scroll_off = 0
        elif ch in (ord("a"), ord("A")):
            scope = "all" if scope == "community" else "community"
            remote, skip, has_more = [], 0, True
            cursor = scroll_off = 0
            build_ranked()
            load_more()
        elif ch == ord("/"):
            searching = True
        elif ch == ord("d"):
            if ents and ents[cursor]["inst"]:
                confirm = ents[cursor]


# --- plain-text fallback ---------------------------------------------------------

def pick_plain(cfg, sys_ram: int, recommended: str, lf: dict | None = None) -> str | None:
    """Static list: installed first, then the HF top-10. Reads stdin."""
    def lf_bits(repo):
        """llmfit extras: score with its highlights, context, tool use."""
        e = (lf or {}).get(repo)
        if not e:
            return ""
        bits = []
        if s := llmfit.score(e, sys_ram):
            bits.append(f"⚡{s['score']:.1f} llmfit ({s['use_case'].lower()},"
                        f" ~{s['tps']:.0f} tok/s, ~{s['mem']:.0f}GB)")
        if c := llmfit.ctx_str(e):
            bits.append(f"ctx {c}")
        if llmfit.has_tools(e):
            bits.append("tools")
        return f" | {' · '.join(bits)}" if bits else ""

    installed = hf.scan_installed(cfg.hf_hub)
    print()
    ui.info("🖥️  System info")
    print(f"   RAM: {sys_ram}GB | GPU: {ui.gpu_name()}")
    print(f"   💡 Recommended: {recommended} (~{models.ram_need_gb(recommended)}GB RAM)")
    print()
    ui.info("📊 Loading top MLX models from HuggingFace...")
    print()

    try:
        page = hf.fetch_page(0, 30, cfg.hf_token)
        entries = [(m["id"], m["downloads"], m["likes"]) for m in page]
    except Exception:
        ui.warn("HF API unavailable, using the fallback list")
        entries = models.FALLBACK_TRENDING

    shown: list[str] = []
    i = 1
    if installed:
        print("   💿 Installed models (start without downloading):\n")
        for rid, size, dling in installed:
            mark = " ★ " if rid == recommended else "   "
            print(f"{mark}{i:2d}) {rid}")
            print(f"      {models.describe(rid)}")
            state = f"⬇ {size} so far (still downloading)" if dling \
                else f"💾 {size} on disk"
            print(f"      {state} | ~{models.ram_need_gb(rid)}GB RAM"
                  f"{lf_bits(rid)}")
            shown.append(rid)
            i += 1
            print()
        print("   📥 Top MLX models from HuggingFace:\n")
    else:
        print("   📥 Top 10 MLX models by downloads (none installed yet):\n")

    shown_remote = 0
    for model, downloads, likes in entries:
        if shown_remote >= 10:
            break
        if model in shown:      # installed are already listed above — no duplicates
            continue
        rec = "  ★ " if model == recommended else "    "
        dl = ui.human_downloads(downloads)
        print(f"{rec}{i:2d}) {model}")
        print(f"      {models.describe(model)}")
        print(f"      📥 {dl} downloads | ⭐ {likes} likes |"
              f" ~{models.ram_need_gb(model)}GB RAM{lf_bits(model)}")
        shown.append(model)
        i += 1
        shown_remote += 1
        print()

    print("💡 Tip: pick a number or an org/repo from HuggingFace |"
          " delete an installed model: local-llm rm")
    try:
        choice = input("?# Pick a model: ").strip()
    except EOFError:
        return None
    if not choice:
        return None
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(shown):
            return shown[idx - 1]
    if "/" in choice:
        return choice
    ui.warn(f"Invalid choice, using the recommended one: {recommended}")
    return recommended
