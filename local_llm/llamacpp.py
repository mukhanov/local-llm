"""llama.cpp backend: a managed llama-server binary (GGUF models) and a
minimal GGUF header parser.

llama.cpp ships no pip package and its `/releases/latest` tag carries no
binaries — real builds are per-commit prereleases tagged b<N> with assets
like `llama-b<N>-bin-macos-arm64.tar.gz` (researched 2026-10-26). We
download one known-good build into ~/.ollmlx/llama.cpp/llama-b<N>/ and pin
it there: no silent upgrades, re-download only when the pin is missing.

The binary needs its dylibs next to it (rpath-relative), so the whole
extracted directory is kept, versioned by build id."""
import json
import os
import shutil
import struct
import tarfile
import urllib.request
import time
from pathlib import Path

from . import ui

_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
_ASSET = "llama-{tag}-bin-macos-arm64.tar.gz"

def _fetch(url, timeout: int, tries: int = 4):
    """urlopen with flaky-network retries: api.github.com and the release
    CDN die mid-TLS now and then on the same proxies hf.download retries
    around — a fresh attempt is usually enough."""
    last: Exception | None = None
    for n in range(1, tries + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — re-raised on the last try
            last = exc
            if n < tries:
                time.sleep(5 * n)
    raise last


def _llama_root() -> Path:
    return Path(os.environ.get("OLLMLX_HOME", "~/.ollmlx")).expanduser() / "llama.cpp"


def _latest_asset() -> tuple[str, str]:
    """(tag, download URL) of the newest prerelease with a macOS arm64
    asset — releases iterate newest-first, several per day."""
    req = urllib.request.Request(_RELEASES, headers={"accept": "application/json"})
    with _fetch(req, timeout=30) as r:
        releases = json.loads(r.read().decode())
    for rel in releases:
        tag = rel.get("tag_name", "")
        for a in rel.get("assets", []):
            if a.get("name") == _ASSET.format(tag=tag):
                return tag, a["browser_download_url"]
    raise RuntimeError("no macos-arm64 asset in the 10 newest llama.cpp releases")


def _download_build(root: Path) -> str:
    tag, url = _latest_asset()
    ui.info(f"Downloading llama.cpp {tag} (~16MB) from GitHub releases")
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / f"llama-{tag}.tar.gz"
    with _fetch(url, timeout=300) as r, open(tar_path, "wb") as f:
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r   {done * 100 / total:5.1f}%"
                      f" ({done // 1024 // 1024}MB)", end="", flush=True)
    print()
    try:
        # filter="data" (py3.12+): refuses path traversal / absolute members
        with tarfile.open(tar_path) as tf:
            tf.extractall(root, filter="data")
    finally:
        tar_path.unlink(missing_ok=True)
    server = root / f"llama-{tag}" / "llama-server"
    if not server.is_file():
        raise SystemExit(f"   llama-server missing in the extracted {tag} tarball")
    server.chmod(0o755)
    (root / "build.txt").write_text(tag + "\n")
    ui.ok(f"llama-server {tag} installed at {server}")
    return str(server)


def ensure_llama_server(override: str = "") -> str:
    """Absolute path to a llama-server binary: the explicit override >
    OLLMLX_LLAMA_SERVER > PATH > the managed download under
    ~/.ollmlx/llama.cpp (pinned to the build recorded in build.txt)."""
    override = override or os.environ.get("OLLMLX_LLAMA_SERVER", "")
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return str(p)
        raise SystemExit(f"   OLLMLX_LLAMA_SERVER={override} — not a file")
    found = shutil.which("llama-server")
    if found:
        return found
    root = _llama_root()
    pin = root / "build.txt"
    if pin.is_file():
        cand = root / f"llama-{pin.read_text().strip()}" / "llama-server"
        if cand.is_file():
            return str(cand)
        # the pinned dir was deleted — fall through and fetch fresh
    return _download_build(root)


# --- GGUF header -----------------------------------------------------------------
#
# GGUF repos ship no config.json; the context window and KV geometry live
# in the header of the .gguf itself. We parse just the metadata key/values
# (tensor tables are skipped) — everything the context math needs.

_MAGIC = b"GGUF"
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
           10: "<Q", 11: "<q"}   # u8..i64 by spec type id
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_str(f) -> str:
    n = _u64(f)
    return f.read(n).decode("utf-8", "replace")


def _skip_array(f, elem_type: int, count: int) -> None:
    if elem_type == 8:
        for _ in range(count):
            f.seek(_u64(f), 1)
    elif elem_type == 9:
        raise ValueError("nested arrays are not in the spec — bail out")
    else:
        f.seek(_SIZES[elem_type] * count, 1)


def gguf_meta(path) -> dict:
    """Scalar metadata from a GGUF v2/v3 header (arrays skipped);
    {} when the file is unreadable or truncated."""
    try:
        with open(path, "rb") as f:
            if f.read(4) != _MAGIC:
                return {}
            version = _u32(f)
            if version < 2:
                return {}
            f.seek(8, 1)          # tensor count — never read here
            out = {}
            for _ in range(_u64(f)):
                key = _read_str(f)
                t = _u32(f)
                if t == 8:
                    out[key] = _read_str(f)
                elif t == 9:
                    _skip_array(f, _u32(f), _u64(f))
                elif fmt := _SCALAR.get(t):
                    out[key] = struct.unpack(fmt, f.read(_SIZES[t]))[0]
                elif t == 6:
                    out[key] = struct.unpack("<f", f.read(4))[0]
                elif t in (7, 12):
                    out[key] = struct.unpack("<?" if t == 7 else "<d",
                                             f.read(1 if t == 7 else 8))[0]
                else:
                    break         # unknown type — the rest can't be skipped
            return out
    except (OSError, struct.error, ValueError):
        return {}


def gguf_ctx_kv(path) -> tuple[int, float | None]:
    """(context_length, kv_bytes_per_token) from a GGUF header, the same
    fp16-KV formula as the MLX path (2 x kv_heads x head_dim x layers x 2).
    (32768, None) when the header doesn't say — llama-server fills in the
    real defaults."""
    m = gguf_meta(path)
    arch = m.get("general.architecture")
    if not isinstance(arch, str):
        return 32768, None
    ctx = m.get(f"{arch}.context_length") or 32768
    layers = m.get(f"{arch}.block_count")
    kvh = m.get(f"{arch}.attention.head_count_kv") or m.get(f"{arch}.attention.head_count")
    head_dim = m.get(f"{arch}.attention.key_length")
    if not all(isinstance(x, (int, float)) and x for x in (layers, kvh, head_dim)):
        return int(ctx), None
    return int(ctx), 2 * kvh * head_dim * layers * 2
