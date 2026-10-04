"""HuggingFace: DoH-обход сломанного DNS, топ моделей через API,
локальный кэш (~/.cache/huggingface/hub), загрузка и удаление."""
import json
import os
import shutil
import socket
import stat
import urllib.request
from pathlib import Path

from . import ui

# Локальный TUN-прокси выдаёт fake-ip для *.huggingface.co / *.hf.co, на которых
# умирает TLS. Резолвим их через DoH (1.1.1.1) и пинним результат.
DOH = "https://1.1.1.1/dns-query"
_pinned: dict = {}
_orig_gai = socket.getaddrinfo
_installed = False


def _doh(host: str):
    req = urllib.request.Request(
        f"{DOH}?name={host}&type=A", headers={"accept": "application/dns-json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        ans = [a["data"] for a in json.load(r).get("Answer", []) if a.get("type") == 1]
    return ans[0] if ans else None


def _gai(host, *a, **kw):
    if host and (host == "huggingface.co" or host.endswith((".hf.co", ".huggingface.co"))):
        if host not in _pinned:
            try:
                _pinned[host] = _doh(host)
            except Exception:
                _pinned[host] = None
        if _pinned[host]:
            return _orig_gai(_pinned[host], *a, **kw)
    return _orig_gai(host, *a, **kw)


def install_doh() -> None:
    """Патчит socket.getaddrinfo (идемпотентно). Звать до сетевых запросов."""
    global _installed
    if not _installed:
        socket.getaddrinfo = _gai
        _installed = True


def fetch_page(skip: int = 0, limit: int = 50, token: str | None = None):
    """Топ mlx-community с HF, только text-generation (LLM)."""
    install_doh()
    url = (
        f"https://huggingface.co/api/models?author=mlx-community"
        f"&sort=downloads&direction=-1&limit={limit}&skip={skip}"
    )
    req = urllib.request.Request(url, headers={"accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=20) as r:
        models = json.loads(r.read().decode())
    return [
        {"id": m.get("id", ""), "downloads": m.get("downloads", 0),
         "likes": m.get("likes", 0)}
        for m in models
        if m.get("pipeline_tag") == "text-generation" and m.get("id")
    ]


def _repo_file_meta(repo: str, token: str | None):
    """(суммарный размер репо в байтах, {sha256-blob: имя файла}, число файлов)
    из HF API. При ошибке — (0, {}, 0): прогресс покажется без процентов."""
    install_doh()
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}?blobs=true",
        headers={"accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return 0, {}, 0
    total = 0
    by_blob = {}
    siblings = data.get("siblings", [])
    for s in siblings:
        if s.get("size"):
            total += s["size"]
        # blob в кэше зовётся по lfs.sha256 (для LFS) либо по blobId (git-файлы)
        oid = (s.get("lfs") or {}).get("sha256") or s.get("blobId")
        if oid:
            by_blob[oid] = s.get("rfilename", "?")
    return total, by_blob, len(siblings)


def _scan_download(model_dir: Path):
    """(байт скачано, blob текущего файла, готовых файлов) по каталогу blobs:
    готовые файлы лежат в blobs/<sha>, качающиеся — blobs/<sha>.incomplete."""
    done = cur_size = files_done = 0
    cur_blob = None
    blobs = model_dir / "blobs"
    if not blobs.is_dir():
        return done, cur_blob, files_done
    for f in blobs.iterdir():
        try:
            size = f.stat().st_size
        except OSError:
            continue
        done += size
        if f.name.endswith(".incomplete"):
            if size > cur_size:
                # имя вида <sha256>.<случайный инфикс>.incomplete — для
                # сверки с lfs oid из API нужен только sha до первой точки
                cur_size = size
                cur_blob = f.name[:-len(".incomplete")].split(".")[0]
        else:
            files_done += 1
    return done, cur_blob, files_done


def _fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч{m:02d}м"
    return f"{m}м{s:02d}с" if m else f"{s}с"


def _prune_partials(model_dir: Path) -> int:
    """Убирает огрызки прошлых попыток (*.incomplete) и возвращает их размер.

    hub 1.x качает во временный файл с уникальным infix и не переиспользует
    его между запусками — такие огрызки бесполезны (могут остаться только
    после жёсткого убийства процесса)."""
    freed = 0
    blobs = model_dir / "blobs"
    if not blobs.is_dir():
        return 0
    for f in blobs.glob("*.incomplete"):
        try:
            freed += f.stat().st_size
            f.unlink()
        except OSError:
            pass
    return freed


def download(repo: str, token: str | None = None, hf_hub: Path | None = None,
             poll: float = 1.0) -> None:
    """snapshot_download с DoH-пином, без xet/hf_transfer и со своим прогрессом.

    Родные tqdm-бары huggingface_hub («Downloading bytes», «Reconstructing»,
    «Fetching N files») выключены: вместо них одна строка с баром, скоростью,
    ETA и текущим файлом (байты считаем по каталогу blobs, ожидаемый размер —
    из HF API). Токен уходит в env HF_TOKEN — huggingface_hub берёт его сам."""
    install_doh()
    if token:
        os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    import shutil
    import sys
    import threading
    import time

    print(f"==> fetching {repo}", flush=True)
    total, name_by_blob, total_files = _repo_file_meta(repo, token)
    if hf_hub is None:  # тот же резолв, что у snapshot_download
        hf_hub = Path(os.environ.get(
            "HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    model_dir = hf_hub / ("models--" + repo.replace("/", "--"))
    freed = _prune_partials(model_dir)
    if freed > 1024 * 1024:
        ui.info(f"очищено огрызков прошлых попыток: {ui.human_bytes(freed)}")
    is_tty = sys.stdout.isatty()
    term_w = shutil.get_terminal_size((100, 20)).columns

    state = {"error": None}

    def worker():
        try:
            snapshot_download(repo_id=repo)
        except BaseException as exc:  # noqa: BLE001 — прокидываем в главный поток
            state["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_done, last_t, rate, ticks = 0, time.monotonic(), 0.0, 0
    try:
        while thread.is_alive():
            time.sleep(poll)
            done, cur_blob, files_done = _scan_download(model_dir)
            now = time.monotonic()
            inst = (done - last_done) / max(1e-9, now - last_t)
            rate = inst if rate == 0 else 0.3 * inst + 0.7 * rate
            last_done, last_t = done, now

            parts = []
            if total:
                fill = int(20 * done / total + 0.5)
                parts += [f"{'█' * fill}{' ' * (20 - fill)} {done * 100 / total:5.1f}%",
                          f"{ui.human_bytes(done)}/{ui.human_bytes(total)}"]
            else:
                parts.append(ui.human_bytes(done))
            if rate > 1024:
                parts.append(f"{rate / 1024 / 1024:5.1f}MB/s")
                # при околонулевой скорости ETA бессмыслен и растёт до часов
                if total and rate > 50 * 1024:
                    parts.append(f"ETA {_fmt_eta((total - done) / rate)}")
            if cur_blob:
                name = name_by_blob.get(cur_blob, cur_blob[:12])
                if len(name) > 44:
                    name = name[:41] + "…"
                count = (f"{min(files_done + 1, total_files)}/{total_files} · "
                         if total_files else "")
                parts.append(f"файл {count}{name}")
            line = ("   ⬇ " + " │ ".join(parts))[: term_w - 1]
            if is_tty:
                print("\r" + line.ljust(term_w - 1), end="", flush=True)
            elif ticks % 15 == 0:  # в pipe — разовые строки раз в ~15с
                print(line, flush=True)
            ticks += 1
        thread.join()
    except KeyboardInterrupt:
        print("\n   прервано — докачки нет, при повторе файлы начнутся заново")
        raise

    if state["error"] is not None:
        raise SystemExit(f"   download failed: {state['error']}")
    if is_tty:
        print("\r" + " " * (term_w - 1) + "\r", end="", flush=True)


# --- локальный кэш -----------------------------------------------------------


def model_dir(hf_hub: Path, repo: str) -> Path:
    return hf_hub / ("models--" + repo.replace("/", "--"))


def is_cached(hf_hub: Path, repo: str) -> bool:
    """Полностью ли модель на диске: safetensors есть и все ссылки живы
    (у недокачанной части шардов цель отсутствует — надо качать заново)."""
    sts = list(model_dir(hf_hub, repo).glob("snapshots/*/*.safetensors"))
    return bool(sts) and all(p.exists() for p in sts)


def _dir_size(root: Path, hub_root: Path | None = None) -> int:
    """Реальный вес модели на диске.

    Симлинки считаем один раз по резолвнутой цели, а не по ссылке (снапшоты
    ссылаются в blobs). Без hub_root цели не резолвим: старый режим, когда все
    файлы лежали внутри каталога модели. С hub_root учитываем и общее
    хранилище hub/blobs/<xx>/<sha> (shared-CAS раскладка hub 1.x: у модели
    только симлинки, настоящие веса — в CAS)."""
    seen: set[str] = set()
    size = 0
    root_s = os.path.realpath(root)  # пути сравниваем после резолва всех
    hub_s = os.path.realpath(hub_root) if hub_root else None  # симлинков (/var -> /private/var)
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            if not stat.S_ISLNK(st.st_mode):
                size += st.st_size
                continue
            # симлинк: цель внутри каталога посчитается сама при обходе
            # (снапшот -> blobs), резолвим только цели вне него (CAS hub)
            try:
                tgt = os.path.realpath(fp)
                tst = os.stat(tgt)  # битая ссылка -> OSError
            except OSError:
                continue
            if (hub_s is None or tgt.startswith(root_s + os.sep)
                    or not tgt.startswith(hub_s + os.sep)):
                continue
            if tgt not in seen:
                seen.add(tgt)
                size += tst.st_size
    return size


def scan_installed(hf_hub: Path):
    """[(org/repo, human_size)] — только LLM: safetensors + токенизатор.

    safetensors-ссылки проверяются на резолв: у недокачанных моделей часть
    шардов битая (цели нет) — размер честный, с пометкой ⚠."""
    out = []
    if not hf_hub.is_dir():
        return out
    for d in sorted(hf_hub.iterdir()):
        if not d.name.startswith("models--"):
            continue
        parts = d.name[len("models--"):].split("--", 1)
        if len(parts) != 2:
            continue
        n_st = n_st_ok = 0
        has_tok = False
        for snap in (d / "snapshots").glob("*"):
            if not snap.is_dir():
                continue
            for f in snap.iterdir():
                if f.name.endswith(".safetensors"):
                    n_st += 1
                    if f.exists():
                        n_st_ok += 1
                elif (f.name in ("tokenizer.json", "tokenizer_config.json")
                      or f.name.startswith("chat_template")):
                    has_tok = has_tok or f.exists()
        if not (n_st_ok and has_tok):
            continue
        size = _dir_size(d, hf_hub)
        sz = ui.human_bytes(size) if size else "?"
        if n_st > n_st_ok:
            sz += f" ⚠{n_st - n_st_ok} файл(ов) нет"
        out.append(("/".join(parts), sz))
    return out


def delete_model(hf_hub: Path, repo: str) -> None:
    """Удаляет модель: каталог models--org--repo + её файлы в общем CAS
    hub/blobs/<xx>/<sha>, если на них не ссылается другая модель."""
    d = model_dir(hf_hub, repo)
    cas = os.path.realpath(hf_hub / "blobs")  # /var -> /private/var на macOS
    targets = set()
    for dirpath, _dirs, files in os.walk(d):
        for f in files:
            fp = os.path.join(dirpath, f)
            if os.path.islink(fp):
                try:
                    tgt = os.path.realpath(fp)
                    os.stat(tgt)
                except OSError:
                    continue
                if tgt.startswith(cas + os.sep):
                    targets.add(tgt)
    shutil.rmtree(d)
    if not targets:
        return
    still_used = set()
    for other in hf_hub.glob("models--*"):
        if other == d:
            continue
        for dirpath, _dirs, files in os.walk(other):
            for f in files:
                fp = os.path.join(dirpath, f)
                if os.path.islink(fp):
                    try:
                        tgt = os.path.realpath(fp)
                    except OSError:
                        continue
                    if tgt in targets:
                        still_used.add(tgt)
    for tgt in targets - still_used:
        try:
            os.unlink(tgt)
            try:
                os.rmdir(os.path.dirname(tgt))  # пустой <xx>/ приберём
            except OSError:
                pass
        except OSError:
            pass


def model_ctx(hf_hub: Path, repo: str) -> int:
    """Context window из config.json модели (включая text_config)."""
    for snap in (model_dir(hf_hub, repo) / "snapshots").glob("*"):
        cfg = snap / "config.json"
        if not cfg.is_file():
            continue
        try:
            c = json.loads(cfg.read_text())
        except Exception:
            continue

        def find(d):
            for k in ("max_position_embeddings", "seq_length"):
                if isinstance(d.get(k), int):
                    return d[k]
            for v in d.values():
                if isinstance(v, dict):
                    r = find(v)
                    if r:
                        return r
            return None

        r = find(c)
        if r:
            return r
    return 32768
