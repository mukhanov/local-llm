"""HuggingFace: DoH workaround for broken DNS, top models via the API,
local cache (~/.cache/huggingface/hub), download and deletion."""
import json
import os
import shutil
import socket
import stat
import urllib.parse
import urllib.request
from pathlib import Path

from . import ui

# A local TUN proxy hands out fake IPs for *.huggingface.co / *.hf.co, on
# which TLS dies. Resolve them via DoH (1.1.1.1) and pin the result.
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
    """Monkey-patch socket.getaddrinfo (idempotent). Call before network requests."""
    global _installed
    if not _installed:
        socket.getaddrinfo = _gai
        _installed = True


def fetch_page(skip: int = 0, limit: int = 50, token: str | None = None,
               scope: str = "community", search: str | None = None):
    """A page of MLX LLMs from HF sorted by downloads (up to 2*limit models:
    text-generation and image-text-to-text are disjoint tags, fetched
    separately and merged; an empty page means exhausted).

    scope "community" — the mlx-community org (canonical conversions);
    scope "all" — every author via filter=mlx, includes fresh personal
    re-quants. `search` runs the term server-side (matches repo names)."""
    install_doh()
    out = []
    for tag in ("text-generation", "image-text-to-text"):
        if scope == "community":
            url = (f"https://huggingface.co/api/models?author=mlx-community"
                   f"&pipeline_tag={tag}")
        else:
            url = f"https://huggingface.co/api/models?filter=mlx,{tag}"
        url += f"&sort=downloads&direction=-1&limit={limit}&skip={skip}"
        if search:
            url += f"&search={urllib.parse.quote(search)}"
        req = urllib.request.Request(url, headers={"accept": "application/json"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=20) as r:
            models = json.loads(r.read().decode())
        out += [{"id": m.get("id", ""), "downloads": m.get("downloads", 0),
                 "likes": m.get("likes", 0)} for m in models if m.get("id")]
    out.sort(key=lambda m: -m["downloads"])
    return out


def _repo_siblings(repo: str, token: str | None):
    """Repo file list from the HF API (?blobs=true), or None if unreachable."""
    install_doh()
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}?blobs=true",
        headers={"accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode()).get("siblings", [])
    except Exception:
        return None


def _repo_file_meta(repo: str, token: str | None):
    """(total repo size in bytes, {sha256-blob: file name}, file count) from
    the HF API. On failure — (0, {}, 0): progress shows without percentages."""
    siblings = _repo_siblings(repo, token) or []
    total = 0
    by_blob = {}
    for s in siblings:
        if s.get("size"):
            total += s["size"]
        # cache blobs are named by lfs.sha256 (for LFS) or blobId (git files)
        oid = (s.get("lfs") or {}).get("sha256") or s.get("blobId")
        if oid:
            by_blob[oid] = s.get("rfilename", "?")
    return total, by_blob, len(siblings)


def missing_files(hf_hub: Path, repo: str, token: str | None):
    """Repo files the local snapshot lacks, or None if the HF API is
    unreachable (completeness can't be verified).

    A partial download passes is_cached(): hub creates snapshot links only
    for finished files, so absent shards are invisible without the repo's
    file list. Broken links count as missing (is_file() doesn't follow them)."""
    siblings = _repo_siblings(repo, token)
    if siblings is None:
        return None
    have = set()
    for snap in (model_dir(hf_hub, repo) / "snapshots").glob("*"):
        if not snap.is_dir():
            continue
        for f in snap.rglob("*"):
            if f.is_file():
                have.add(str(f.relative_to(snap)))
    return [s.get("rfilename") for s in siblings
            if s.get("rfilename") and s["rfilename"] not in have]


def _scan_download(model_dir: Path):
    """(bytes downloaded, current file's blob, files done) from the blobs dir:
    finished files live in blobs/<sha>, in-flight ones in blobs/<sha>.incomplete."""
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
                # name looks like <sha256>.<random infix>.incomplete — matching
                # against the lfs sha256 from the API needs just the part
                # before the first dot
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
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _prune_partials(model_dir: Path) -> int:
    """Remove leftovers from previous attempts (*.incomplete), return their size.

    hub 1.x downloads into a temp file with a unique infix and never reuses it
    across runs — such leftovers are useless (they can only survive a hard
    kill of the process)."""
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
             poll: float = 1.0, attempts: int = 5) -> None:
    """snapshot_download with DoH pinning, without xet/hf_transfer, with our
    own progress display and flaky-network retries.

    huggingface_hub's native tqdm bars ("Downloading bytes", "Reconstructing",
    "Fetching N files") are disabled: instead there's a single line with a
    bar, speed, ETA and the current file (bytes are measured in the blobs
    dir, expected size comes from the HF API). The token goes into the
    HF_TOKEN env — huggingface_hub picks it up itself.

    On failure the whole snapshot_download is retried up to `attempts` times:
    hub instantly re-verifies finished blobs and re-fetches only the missing
    files, so a proxy that dies every few GB still converges."""
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
    if hf_hub is None:  # same resolution as snapshot_download
        hf_hub = Path(os.environ.get(
            "HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    model_dir = hf_hub / ("models--" + repo.replace("/", "--"))
    freed = _prune_partials(model_dir)
    if freed > 1024 * 1024:
        ui.info(f"pruned stale partials from previous attempts: {ui.human_bytes(freed)}")
    is_tty = sys.stdout.isatty()
    term_w = shutil.get_terminal_size((100, 20)).columns

    state = {"error": None, "attempt": 0}

    def worker():
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                # re-resolve: the pinned edge itself may be what died —
                # CloudFront rotates answer order, so a fresh DoH query
                # lands on a different IP
                _pinned.clear()
            try:
                # max_workers=2 (hub default is 8): TUN proxies choke on many
                # parallel streams — the initial burst saturates, then every
                # connection times out
                snapshot_download(repo_id=repo, max_workers=2)
                state["error"] = None
                return
            except BaseException as exc:  # noqa: BLE001 — re-raised in the main thread
                state["error"], state["attempt"] = exc, attempt
                if attempt < attempts:
                    time.sleep(min(60, 10 * attempt))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_done, last_t, rate, ticks, last_attempt = 0, time.monotonic(), 0.0, 0, 0
    try:
        while thread.is_alive():
            time.sleep(poll)
            if state["attempt"] > last_attempt:
                last_attempt = state["attempt"]
                print(f"\n   attempt {state['attempt']}/{attempts}"
                      f" after: {state['error']}", flush=True)
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
                # at near-zero speed an ETA is meaningless and grows to hours
                if total and rate > 50 * 1024:
                    parts.append(f"ETA {_fmt_eta((total - done) / rate)}")
            if cur_blob:
                name = name_by_blob.get(cur_blob, cur_blob[:12])
                if len(name) > 44:
                    name = name[:41] + "…"
                count = (f"{min(files_done + 1, total_files)}/{total_files} · "
                         if total_files else "")
                parts.append(f"file {count}{name}")
            line = ("   ⬇ " + " │ ".join(parts))[: term_w - 1]
            if is_tty:
                print("\r" + line.ljust(term_w - 1), end="", flush=True)
            elif ticks % 15 == 0:  # in a pipe — one line every ~15s
                print(line, flush=True)
            ticks += 1
        thread.join()
    except KeyboardInterrupt:
        print("\n   interrupted — there is no resume, files restart on the next run")
        raise

    if state["error"] is not None:
        raise SystemExit(f"   download failed after"
                         f" {max(state['attempt'], 1)} attempt(s): {state['error']}")
    if is_tty:
        print("\r" + " " * (term_w - 1) + "\r", end="", flush=True)


# --- local cache ----------------------------------------------------------------


def model_dir(hf_hub: Path, repo: str) -> Path:
    return hf_hub / ("models--" + repo.replace("/", "--"))


def is_cached(hf_hub: Path, repo: str) -> bool:
    """Is the model fully on disk: safetensors present and every link alive
    (in a partial download some shard targets are missing — needs a re-download)."""
    sts = list(model_dir(hf_hub, repo).glob("snapshots/*/*.safetensors"))
    return bool(sts) and all(p.exists() for p in sts)


def _dir_size(root: Path, hub_root: Path | None = None) -> int:
    """Real disk usage of a model.

    Symlinks are counted once by their resolved target, not by the link
    (snapshots point into blobs). Without hub_root targets are not resolved:
    the old mode where all files lived inside the model's own dir. With
    hub_root we also account for the shared store hub/blobs/<xx>/<sha>
    (the shared-CAS layout of hub 1.x: the model dir holds only symlinks,
    the actual weights live in the CAS)."""
    seen: set[str] = set()
    size = 0
    root_s = os.path.realpath(root)  # compare paths after resolving all
    hub_s = os.path.realpath(hub_root) if hub_root else None  # symlinks (/var -> /private/var)
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
            # symlink: a target inside the dir is counted by the walk itself
            # (snapshot -> blobs), resolve only targets outside it (hub CAS)
            try:
                tgt = os.path.realpath(fp)
                tst = os.stat(tgt)  # broken link -> OSError
            except OSError:
                continue
            if (hub_s is None or tgt.startswith(root_s + os.sep)
                    or not tgt.startswith(hub_s + os.sep)):
                continue
            if tgt not in seen:
                seen.add(tgt)
                size += tst.st_size
    return size


def model_disk_bytes(hf_hub: Path, repo: str) -> int:
    """Bytes a downloaded model occupies (shared CAS blobs counted once),
    0 when not on disk — the crash-guard's 'weights are this big' number."""
    d = model_dir(hf_hub, repo)
    return _dir_size(d, hf_hub) if d.is_dir() else 0


def scan_installed(hf_hub: Path):
    """[(org/repo, human_size, downloading)] — LLMs only: safetensors +
    tokenizer.

    A model counts as downloading when shards have unresolved links
    (interrupted) or blobs/ still holds *.incomplete files (an active
    download in another process writes those). The size stays honest —
    bytes on disk so far, with a ⚠ note for missing shards."""
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
            sz += f" ⚠{n_st - n_st_ok} missing"
        dling = n_st > n_st_ok or bool(list((d / "blobs").glob("*.incomplete")))
        out.append(("/".join(parts), sz, dling))
    return out


def delete_model(hf_hub: Path, repo: str) -> None:
    """Delete a model: the models--org--repo dir + its files in the shared
    CAS hub/blobs/<xx>/<sha>, unless another model still references them."""
    d = model_dir(hf_hub, repo)
    cas = os.path.realpath(hf_hub / "blobs")  # /var -> /private/var on macOS
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
                os.rmdir(os.path.dirname(tgt))  # clean up the empty <xx>/ shard dir
            except OSError:
                pass
        except OSError:
            pass


def model_ctx(hf_hub: Path, repo: str) -> int:
    """Context window from the model's config.json (including text_config)."""
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
