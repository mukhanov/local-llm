"""Выбор модели: curses-TUI (скролл/поиск/догрузка/удаление) и текстовый
фолбек, когда терминал не тянет curses (не tty и т.п.).

Точки входа:
  pick_model(cfg) -> str | None   — TUI, при ошибке среды — текстовый список
  pick_plain(cfg, ...) -> str | None
"""
import curses
import sys

from . import hf, models, ui


class EnvError(Exception):
    """Терминал/среда не подходят для curses — зовите текстовый фолбек."""


def pick_model(cfg) -> str | None:
    """TUI-выбор; None — пользователь отменил. EnvError наружу не идёт:
    ловится здесь и включается текстовый фолбек."""
    sys_ram = ui.total_ram_gb()
    recommended = models.recommend_for_ram(sys_ram)
    try:
        return pick_tui(cfg, sys_ram, recommended)
    except EnvError:
        ui.warn("TUI-выбор недоступен в этом терминале — показываю простой список")
        return pick_plain(cfg, sys_ram, recommended)


# --- curses TUI ----------------------------------------------------------------

LIMIT = 50  # размер страницы HF API для догрузки


def pick_tui(cfg, sys_ram: int, recommended: str) -> str | None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise EnvError("не tty")
    ui.force_utf8_locale()
    ui.force_compatible_term()
    result = {}

    try:
        curses.wrapper(lambda scr: _tui(scr, cfg, sys_ram, recommended, result))
    except curses.error as exc:
        raise EnvError(str(exc)) from exc

    return result.get("choice")


def _tui(stdscr, cfg, sys_ram: int, recommended: str, result: dict) -> None:
    installed = hf.scan_installed(cfg.hf_hub)
    inst_ids = {rid for rid, _ in installed}
    remote, skip, has_more = [], 0, True
    sort_mode, query, searching = "fit", "", False
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
        """Скачанные сверху, ниже — удалённые в порядке текущей сортировки."""
        ents = []
        q = query.lower()
        for rid, sz in installed:
            if q and q not in rid.lower():
                continue
            sc, rn, total, active = models.score(sys_ram, rid, 0)
            ents.append(dict(id=rid, inst=True, size=sz, dl=0, likes=0, sc=sc,
                             rn=rn, total=total, active=active))
        rem = []
        for m in remote:
            if m["id"] in inst_ids or (q and q not in m["id"].lower()):
                continue
            dl = m["downloads"]
            sc, rn, total, active = models.score(sys_ram, m["id"], dl)
            rem.append(dict(id=m["id"], inst=False, size="", dl=dl,
                            likes=m["likes"], sc=sc, rn=rn, total=total,
                            active=active))
        rem.sort(key=lambda e: (-e["sc"], -e["dl"]) if sort_mode == "fit"
                 else (-e["dl"], -e["sc"]))
        return ents + rem

    def load_more():
        nonlocal skip, remote, has_more, status
        if not has_more:
            status = " HF: все доступные страницы загружены"
            return
        status = f" загрузка моделей {skip + 1}–{skip + LIMIT} с HF…"
        draw()
        stdscr.refresh()
        try:
            page = hf.fetch_page(skip, LIMIT, cfg.hf_token)
        except Exception as exc:
            status = f" HF API недоступен: {exc}"
            return
        skip += len(page)
        known = {m["id"] for m in remote}
        remote += [m for m in page if m["id"] not in known]
        has_more = len(page) == LIMIT
        status = f" с HF загружено: {len(remote)} (l — ещё)"

    def draw():
        nonlocal cursor, scroll_off
        ents = entries()
        cursor = max(0, min(cursor, len(ents) - 1)) if ents else 0
        rows, cur_row, prev, ei = [], 0, None, 0
        for e in ents:
            kind = "inst" if e["inst"] else "remote"
            if kind != prev:
                title = ("💿 Скачанные" if kind == "inst" else
                         ("☁ HuggingFace · по совместимости" if sort_mode == "fit"
                          else "☁ HuggingFace · по загрузкам"))
                rows.append(("hdr", title))
                prev = kind
            rows.append(("ent", e))
            if ei == cursor:
                cur_row = len(rows) - 1
            ei += 1
        if not rows:
            rows = [("hdr", "ничего не найдено" if query else "список пуст")]
            cur_row = 0

        h, w = stdscr.getmaxyx()
        stdscr.erase()

        def add(yy, xx, text, attr=0):
            try:
                stdscr.addstr(yy, xx, text[: max(0, w - 1 - xx)], attr)
            except curses.error:
                pass

        add(0, 0, f" выбор модели — RAM {sys_ram}GB · {ui.gpu_name()} · сортировка: "
            f"{'по совместимости' if sort_mode == 'fit' else 'по загрузкам'} (s — сменить)",
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
                right = f"{e['size']} на диске"
            else:
                ram = f"~{e['rn']:.0f}GB" if e["rn"] is not None else "?GB"
                moe = " · MoE" if e["active"] else ""
                right = f"{ram} RAM{moe} · ↓{ui.human_downloads(e['dl'])} · ⭐{e['likes']}"
                if e["rn"] is not None and e["rn"] > sys_ram:
                    right += " ⚠не влезет"
            mark = "▸ " if i == cur_row else "  "
            rec = " ★" if e["id"] == recommended else ""
            name_w = max(10, w - len(mark) - len(right) - 3)
            nm = e["id"] if len(e["id"]) <= name_w else e["id"][:name_w - 1] + "…"
            pad = max(1, w - 1 - len(mark) - len(nm) - len(rec) - len(right))
            if i == cur_row:
                attr = R | B
            elif e["inst"]:
                attr = col(1)
            elif e["rn"] is not None and e["rn"] > sys_ram:
                attr = col(3)
            else:
                attr = 0
            add(yy, 0, f"{mark}{nm}{rec}{' ' * pad}{right}", attr)

        if searching:
            hint = f" поиск: {query}█   Enter — применить · Esc — сброс"
        else:
            hint = (" j/k↑↓ PgUp/PgDn g/G · / поиск · l ещё с HF · s сортировка ·"
                    " d удалить · Enter выбрать · q отмена")
        add(h - 2, 0, hint, DIM)
        if confirm is not None:
            add(h - 1, 0, f" удалить {confirm['id']} ({confirm['size']}) с диска? y/n",
                B | col(3))
        elif status:
            add(h - 1, 0, status, DIM)
        stdscr.refresh()

    status = " загрузка топа MLX-моделей с HuggingFace…"
    draw()
    try:
        page = hf.fetch_page(0, LIMIT, cfg.hf_token)
        skip = len(page)
        remote = list(page)
        has_more = len(page) == LIMIT
        status = f" с HF загружено: {len(remote)} (l — ещё)"
    except Exception as exc:
        status = f" HF API недоступен: {exc} — показаны только скачанные"

    while True:
        draw()
        ch = stdscr.getch()
        ents = entries()

        if confirm is not None:
            if ch in (ord("y"), ord("Y")):
                try:
                    hf.delete_model(cfg.hf_hub, confirm["id"])
                    status = f" удалено: {confirm['id']} (освобождено {confirm['size']})"
                except OSError as exc:
                    status = f" не удалось удалить: {exc}"
                installed = hf.scan_installed(cfg.hf_hub)
                inst_ids = {rid for rid, _ in installed}
            else:
                status = ""
            confirm = None
        elif searching:
            if ch in (27, 10, 13, curses.KEY_ENTER):
                searching = False
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
                load_more()   # вниз за край — подгружаем следующую страницу
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
        elif ch == ord("/"):
            searching = True
        elif ch == ord("d"):
            if ents and ents[cursor]["inst"]:
                confirm = ents[cursor]


# --- текстовый фолбек ----------------------------------------------------------

def pick_plain(cfg, sys_ram: int, recommended: str) -> str | None:
    """Статический список: скачанные сверху, ниже топ-10 с HF. Читает stdin."""
    installed = hf.scan_installed(cfg.hf_hub)
    print()
    ui.info("🖥️  Системная информация")
    print(f"   RAM: {sys_ram}GB | GPU: {ui.gpu_name()}")
    print(f"   💡 Рекомендация: {recommended} (~{models.ram_need_gb(recommended)}GB RAM)")
    print()
    ui.info("📊 Загрузка топ MLX-моделей с HuggingFace...")
    print()

    try:
        page = hf.fetch_page(0, 30, cfg.hf_token)
        entries = [(m["id"], m["downloads"], m["likes"]) for m in page]
    except Exception:
        ui.warn("Не удалось получить данные из HF API, используем кэш")
        entries = models.FALLBACK_TRENDING

    shown: list[str] = []
    i = 1
    if installed:
        print("   💿 Скачанные модели (стартуют без загрузки):\n")
        for rid, size in installed:
            mark = " ★ " if rid == recommended else "   "
            print(f"{mark}{i:2d}) {rid}")
            print(f"      {models.describe(rid)}")
            print(f"      💾 {size} на диске | ~{models.ram_need_gb(rid)}GB RAM")
            shown.append(rid)
            i += 1
            print()
        print("   📥 Топ MLX-моделей с HuggingFace:\n")
    else:
        print("   📥 Топ 10 MLX-моделей по загрузкам (скачанных нет):\n")

    shown_remote = 0
    for model, downloads, likes in entries:
        if shown_remote >= 10:
            break
        if model in shown:      # скачанные уже показаны выше — не дублируем
            continue
        rec = "  ★ " if model == recommended else "    "
        dl = ui.human_downloads(downloads)
        print(f"{rec}{i:2d}) {model}")
        print(f"      {models.describe(model)}")
        print(f"      📥 {dl} downloads | ⭐ {likes} likes |"
              f" ~{models.ram_need_gb(model)}GB RAM")
        shown.append(model)
        i += 1
        shown_remote += 1
        print()

    print("💡 Подсказка: номер модели или org/repo из HuggingFace |"
          " удалить скачанную: local-llm rm")
    try:
        choice = input("?# Выберите модель: ").strip()
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
    ui.warn(f"Неверный выбор, используем рекомендованную: {recommended}")
    return recommended
