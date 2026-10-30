"""HuggingFace: DoH workaround for broken DNS, top models via the API,
local cache (~/.cache/huggingface/hub), download and deletion."""
import fnmatch
import hashlib
import json
import os
import re
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
               scope: str = "community", search: str | None = None,
               fmt: str = "mlx"):
    """A page of MLX LLMs from HF sorted by downloads (up to 2*limit models:
    text-generation and image-text-to-text are disjoint tags, fetched
    separately and merged; an empty page means exhausted).

    scope "community" — the mlx-community org (canonical conversions);
    scope "all" — every author via filter=mlx, includes fresh personal
    re-quants. fmt "gguf" — llama.cpp repos via filter=gguf (scope is
    MLX-only: GGUF conversions have no single canonical org). `search`
    runs the term server-side (matches repo names)."""
    install_doh()
    out = []
    for tag in ("text-generation", "image-text-to-text"):
        if fmt == "gguf":
            url = f"https://huggingface.co/api/models?filter=gguf,{tag}"
        elif scope == "community":
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


def repo_gguf_files(repo: str, token: str | None):
    """[(file name, bytes)] of quantized .gguf files in the repo, quants in
    subdirs included (SC117-style layouts); imatrix/mmproj/mtp draft files
    excluded (mtp can be a prefix too: `mtp-Qwen3….gguf` — draft heads for
    --model-draft, never the model). [] when the repo has none, None when
    the HF API is unreachable."""
    siblings = _repo_siblings(repo, token)
    if siblings is None:
        return None
    out = []
    for s in siblings:
        name = s.get("rfilename", "")
        if not name.endswith(".gguf"):
            continue
        base = name.rsplit("/", 1)[-1].lower()
        if ("imatrix" in base or "mmproj" in base or "-mtp-" in base
                or base.startswith("mtp-")):
            continue
        out.append((name, s.get("size") or 0))
    return out


def gguf_groups(files):
    """[(label, total_bytes, [shard names])] with split GGUFs
    (*-00001-of-00003.gguf) grouped into one entry, largest first. Labels
    are file base names (quant dirs like `IQ3_S/` dropped) — local and
    remote listings then produce identical labels; shard names keep their
    full repo paths for allow_patterns."""
    groups: dict = {}
    for name, size in files:
        base = name.rsplit("/", 1)[-1]
        base = re.sub(r"-\d{5}-of-\d{5}\.gguf$", ".gguf", base)
        g = groups.setdefault(base, [0, []])
        g[0] += size
        g[1].append(name)
    out = [(base, total, sorted(names))
           for base, (total, names) in groups.items()]
    out.sort(key=lambda t: -t[1])
    return out

def gguf_set_complete(names) -> bool:
    """Does the shard set carry its first split? Grouping strips the
    -NNNNN-of- suffixes, so a set that lost its 00001 file still looks
    whole from a partial local listing — and llama.cpp loads only from
    the first split (seen 2026-10-30: 'illegal split file idx: 1')."""
    return any("-00001-of-" in n for n in names) or not any(
        re.search(r"-\d{5}-of-\d{5}", n) for n in names)


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


def _repo_file_meta(repo: str, token: str | None, allow_patterns=None):
    """(total repo size in bytes, {sha256-blob: file name}, file count) from
    the HF API, restricted to allow_patterns when given (a GGUF fetch is
    one quant file, not the whole 100GB repo). On failure — (0, {}, 0):
    progress shows without percentages."""
    siblings = _repo_siblings(repo, token) or []
    if allow_patterns:
        siblings = [s for s in siblings
                    if any(fnmatch.fnmatch(s.get("rfilename", ""), p)
                           for p in allow_patterns)]
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


def _repo_sha(repo: str, token: str | None) -> str:
    """HEAD commit of main, to link a ranged-fetched blob into
    snapshots/<commit>/. Empty string when the API is unreachable."""
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}/revision/main",
        headers={"accept": "application/json"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode()).get("sha", "")
    except Exception:
        return ""


def _ranged_files(repo: str, token: str | None, allow_patterns):
    """[(name, size, lfs sha256)] of weight files to fetch via
    _fetch_ranged: every LFS file of at least 1GiB — safetensors and GGUF
    shards; tokenizer/config files stay on the hub path.

    hub 1.x cannot resume a big file across runs: its temp name carries a
    fresh uuid per call and is unlinked on failure, so a dropped connection
    meant the whole shard from zero (26GB lost to one DNS outage, seen
    2026-10-30). Files over the hub client's plain-HTTP cap it refuses to
    start at all."""
    out = []
    for s in _repo_siblings(repo, token) or []:
        name = s.get("rfilename", "")
        if allow_patterns is not None and name not in allow_patterns:
            continue
        sha = (s.get("lfs") or {}).get("sha256")
        if sha and (s.get("size") or 0) >= 1 << 30:
            out.append((name, s["size"], sha))
    return out

def _tail_matches(url: str, token: str | None, path: Path) -> bool:
    """Is `path` a true prefix of the file at `url`? Compare its last
    64KiB against the same byte range on the hub — hub's temp names carry
    hashes we can't always predict (xet-backed files use an internal one),
    but bytes don't lie. A wrong-quant leftover of the same repo fails
    here, a torn tail too."""
    n = path.stat().st_size
    if n == 0:
        return False
    lo = max(0, n - 65536)
    headers = {"Range": f"bytes={lo}-{n - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                return False
            remote = r.read()
    except Exception:
        return False
    with open(path, "rb") as f:
        f.seek(lo)
        return f.read(n - lo) == remote


def _fetch_ranged(repo: str, name: str, size: int, sha: str,
                  token: str | None, hf_hub: Path) -> None:
    """Fetch one big LFS file with plain Range requests straight into the
    hub cache layout (blobs/<sha> + snapshot symlink), resumable across
    runs and connection drops.

    The hub's own downloader is a dead end for this: over its plain-HTTP
    cap it refuses fresh starts, hf_xet's CAS chunks starve behind TUN
    proxies (31MB in 10 minutes, seen 2026-10-28), and its uuid temp files
    are never resumed in a later run. Range requests are the same wire hub
    uses internally — chunk it ourselves: every stall re-requests from the
    byte count on disk, a killed hub download's prefix is adopted, and the
    sha256 is verified before the blob is linked into the snapshot."""
    import time
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    model_dir = hf_hub / ("models--" + repo.replace("/", "--"))
    blobs = model_dir / "blobs"
    blob, part = blobs / sha, blobs / (sha + ".ranged")
    if blob.is_file():
        if blob.stat().st_size == size:
            part.unlink(missing_ok=True)  # a finished blob needs no part
            return
        # wrong size = a race victim: never leave it servable, the magic
        # check in local_gguf_files would bounce it anyway
        ui.warn(f"{blob.name[:12]}…: corrupt blob"
                f" ({ui.human_bytes(blob.stat().st_size)}, expected"
                f" {ui.human_bytes(size)}) — refetching")
        blob.unlink()
    blobs.mkdir(parents=True, exist_ok=True)
    have_now = part.stat().st_size if part.exists() else 0
    # adopt a killed hub download: its temps <oid>.<uuid>.incomplete hold a
    # clean prefix of the file but are never reused by hub itself. The oid
    # in the name is not always the lfs sha256 (xet-backed files hash
    # differently), so candidates are proven by content (_tail_matches)
    # and only a bigger-than-ours prefix is worth taking
    for left in sorted(blobs.glob("*.incomplete"),
                       key=lambda p: p.stat().st_size, reverse=True):
        if left.stat().st_size <= have_now:
            break   # sorted desc — nothing bigger remains
        if _tail_matches(url, token, left):
            os.replace(left, part)
            break
    refetched = False
    while True:
        have = part.stat().st_size if part.exists() else 0
        digest = hashlib.sha256()
        if have:                                # seed the running hash
            with open(part, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    digest.update(b)
            print(f"   ⬇ ranged {name.rsplit('/', 1)[-1]}: resuming"
                  f" from {have / 2**30:.1f}GiB", flush=True)
        failures, last_note = 0, 0.0
        while have < size:
            headers = {"Range": f"bytes={have}-"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as r, \
                        open(part, "ab") as f:
                    if r.status != 206:
                        raise OSError(f"HTTP {r.status} for a Range request")
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        digest.update(chunk)
                        failures = 0
                        now = time.monotonic()
                        if now - last_note > 30:
                            last_note = now
                            print(f"   ⬇ ranged {name.rsplit('/', 1)[-1]}:"
                                  f" {have * 100 / size:5.1f}%"
                                  f" ({have / 2**30:.0f}GiB)", flush=True)
            except KeyboardInterrupt:
                raise       # the .ranged part survives pruning and resumes
            except Exception:                   # noqa: BLE001 — flaky wire
                failures += 1
                if failures > 200:
                    raise SystemExit(f"   {name}: ranged fetch stalled at"
                                     f" {have} of {size} bytes — retry"
                                     " the run; finished bytes are kept")
                _pinned.clear()  # fresh CDN edge, like the retry worker
                time.sleep(min(60, 5 * failures))
        if digest.hexdigest() == sha:
            break
        if refetched:  # two bad hashes in a row = the source, not the wire
            part.unlink(missing_ok=True)
            raise SystemExit(f"   {name}: sha256 mismatch after the ranged"
                             " fetch — the hub file does not match its"
                             " advertised hash")
        refetched = True
        part.unlink(missing_ok=True)
        print(f"   {name.rsplit('/', 1)[-1]}: sha256 mismatch on the"
              " resumed prefix — refetching from scratch", flush=True)
    os.replace(part, blob)
    ref = model_dir / "refs" / "main"
    commit = (ref.read_text().strip() if ref.is_file()
              else _repo_sha(repo, token))
    if not commit:
        blob.unlink(missing_ok=True)
        raise SystemExit(f"   {name}: fetched, but no snapshot commit"
                         " known to link it into — retry the run")
    link = model_dir / "snapshots" / commit / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        os.symlink(os.path.relpath(blob, link.parent), link)


def _fetch_lock(md: Path):
    """Exclusive cross-process lock for one repo's download. Two
    concurrent fetchers interleave appends into the same temp files and
    both 'finish' with garbage — seen 2026-10-30: a background fetch
    raced the user's own local-llm and blobs came out wrong (one gained
    a second writer's bytes, another lost its header). The fd releases
    when download()'s frame drops on any exit path."""
    import fcntl
    md.mkdir(parents=True, exist_ok=True)
    f = open(md / "fetch.lock", "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit(f"   another download of this model is already"
                         " running — let it finish (its progress is kept)"
                         " and retry")
    return f


def download(repo: str, token: str | None = None, hf_hub: Path | None = None,
             poll: float = 1.0, attempts: int = 5,
             allow_patterns: list[str] | None = None) -> None:
    """snapshot_download with DoH pinning, without xet/hf_transfer, with our
    own progress display and flaky-network retries. allow_patterns limits
    the fetch to those files (GGUF: only the chosen quant); LFS files of
    1GiB and up go through _fetch_ranged, resumable across runs.


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
    total, name_by_blob, total_files = _repo_file_meta(repo, token,
                                                       allow_patterns)
    if hf_hub is None:  # same resolution as snapshot_download
        hf_hub = Path(os.environ.get(
            "HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
    model_dir = hf_hub / ("models--" + repo.replace("/", "--"))
    lock = _fetch_lock(model_dir)   # one fetcher per repo, any exit releases
    # resumable shards first — they adopt hub's *.incomplete leftovers as
    # resume prefixes, so this must run before _prune_partials sweeps them
    for name, size, sha in _ranged_files(repo, token, allow_patterns):
        ui.info(f"{name.rsplit('/', 1)[-1]}: resumable fetch — connection"
                " drops continue from the bytes already on disk")
        _fetch_ranged(repo, name, size, sha, token, hf_hub)
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
                snapshot_download(repo_id=repo, max_workers=2,
                                  allow_patterns=allow_patterns)
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
    (in a partial download some shard targets are missing — needs a
    re-download). GGUF repos: every .gguf link alive (they ship no
    tokenizer files — it's embedded in the weights; quants may sit in
    subdirs — rglob, not the top level)."""
    d = model_dir(hf_hub, repo)
    sts = list(d.glob("snapshots/*/*.safetensors"))
    if sts:
        return all(p.exists() for p in sts)
    gg = [f for snap in (d / "snapshots").glob("*") if snap.is_dir()
          for f in snap.rglob("*.gguf")]
    return bool(gg) and all(p.exists() for p in gg)


def local_gguf_files(hf_hub: Path, repo: str):
    """[(file name, bytes)] of servable .gguf files in the local snapshot
    with live links — subdirs included, imatrix/mmproj/mtp extras excluded
    (mmproj files are .gguf too, but vision projectors, not models)."""
    out = []
    for snap in (model_dir(hf_hub, repo) / "snapshots").glob("*"):
        if not snap.is_dir():
            continue
        for f in snap.rglob("*.gguf"):
            base = f.name.lower()
            if ("imatrix" in base or "mmproj" in base or "-mtp-" in base
                    or base.startswith("mtp-")):
                continue
            try:
                if f.exists():   # broken symlink -> skip, stat follows it
                    with open(f, "rb") as fh:
                        # every real .gguf part starts with the container
                        # magic; a raced/corrupt blob must not be served
                        if fh.read(4) != b"GGUF":
                            continue
                    out.append((f.name, f.stat().st_size))
            except OSError:
                continue
    return out


def gguf_local_path(hf_hub: Path, repo: str, prefer: str | None = None) -> Path | None:
    """Path of the .gguf to serve: the `prefer` group when given (label
    from gguf_groups), else the largest group; first shard of a split set."""
    files = local_gguf_files(hf_hub, repo)
    if not files:
        return None
    groups = gguf_groups(files)          # largest total first
    names = None
    if prefer:
        stem = prefer[:-len(".gguf")] if prefer.endswith(".gguf") else prefer
        names = next((g[2] for g in groups
                      if g[0] == prefer or g[0].startswith(stem)), None)
    if names is None:
        names = groups[0][2]
    want = {n.rsplit("/", 1)[-1] for n in names}   # match on base names
    for snap in (model_dir(hf_hub, repo) / "snapshots").glob("*"):
        if not snap.is_dir():
            continue
        for f in sorted(snap.rglob("*.gguf")):
            if f.name in want:           # sorted: shard -00001 first
                return f
    return None


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
    tokenizer, or .gguf files (the tokenizer is embedded in those).

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
        n_st = n_st_ok = n_gg = n_gg_ok = 0
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
                elif f.name.endswith(".gguf"):   # top-level GGUF repos
                    n_gg += 1
                    if f.exists():
                        n_gg_ok += 1
            # SC117-style layouts put quants in subdirs — count those too
            for f in snap.rglob("*/*.gguf"):
                n_gg += 1
                if f.exists():
                    n_gg_ok += 1
        if not (n_st_ok and has_tok) and not n_gg_ok:
            continue
        size = _dir_size(d, hf_hub)
        sz = ui.human_bytes(size) if size else "?"
        missing = (n_st - n_st_ok) + (n_gg - n_gg_ok)
        if missing:
            sz += f" ⚠{missing} missing"
        dling = missing > 0 or bool(list((d / "blobs").glob("*.incomplete")))
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
    """Context window from the model's config.json (including text_config);
    GGUF repos ship none — the .gguf header is parsed instead."""
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
    p = gguf_local_path(hf_hub, repo)
    if p:
        from .llamacpp import gguf_ctx_kv   # local import: keeps llamacpp
        return gguf_ctx_kv(p)[0]            # out of non-GGUF startup paths
    return 32768


def _config_of(hf_hub: Path, repo: str):
    """The model's parsed config.json, or None."""
    for snap in (model_dir(hf_hub, repo) / "snapshots").glob("*"):
        cfg = snap / "config.json"
        if cfg.is_file():
            try:
                return json.loads(cfg.read_text())
            except Exception:
                return None
    return None


def _dig(d, *keys):
    """First numeric value of `keys` anywhere in the config tree (top level,
    then text_config / nested blocks — MoE vision models nest the LM)."""
    for k in keys:
        if isinstance(d.get(k), (int, float)) and d.get(k):
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            r = _dig(v, *keys)
            if r:
                return r
    return None


def model_kv_per_token(hf_hub: Path, repo: str) -> float | None:
    """KV-cache bytes per token: 2 (K and V) x num_key_value_heads x
    head_dim x num_hidden_layers x 2 bytes (fp16). Verified against the
    field: a 122B at this formula's 96KB/token hit its 16GB cache ceiling
    at 183,993 tokens (91.2KB/token effective). GGUF repos: the same
    geometry from the .gguf header. None when nothing says."""
    c = _config_of(hf_hub, repo)
    if not c:
        p = gguf_local_path(hf_hub, repo)
        if p:
            from .llamacpp import gguf_ctx_kv
            return gguf_ctx_kv(p)[1]
        return None
    layers = _dig(c, "num_hidden_layers")
    heads = _dig(c, "num_attention_heads")
    kvh = _dig(c, "num_key_value_heads") or heads
    hd = _dig(c, "head_dim")
    if hd is None and heads and _dig(c, "hidden_size"):
        hd = _dig(c, "hidden_size") / heads
    if not all(isinstance(x, (int, float)) and x for x in (layers, kvh, hd)):
        return None
    return 2 * kvh * hd * layers * 2


def safe_context(hf_hub: Path, repo: str, prompt_cache_bytes: int,
                 cap: int = 110_000) -> int:
    """The context this model can actually serve on this machine: the
    config's window clamped to the hard cap and to what fits the
    prompt-cache budget with 10% headroom (a session past the ceiling dies
    on the next cache extension — Metal command-buffer OOM, the generation
    thread is gone). Advertised to clients so they compact BEFORE the model
    dies. cap=0 — KV budget only."""
    ctx = model_ctx(hf_hub, repo)
    lims = [ctx]
    if cap:
        lims.append(cap)
    if prompt_cache_bytes:
        kv = model_kv_per_token(hf_hub, repo)
        if kv:
            lims.append(int(prompt_cache_bytes / kv * 0.9))
    return max(4096, min(lims))
