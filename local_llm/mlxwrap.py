"""mlx_lm.server with live tokens-per-second logging for the monitor.

Launched instead of `python -m mlx_lm.server` (same CLI, same behavior —
see servers.start_mlx). One hook: ResponseGenerator.generate is wrapped so
that every generated token bumps an aggregate counter, and once a second a
"TOKPS ..." line goes to stdout — servers.py redirects it to
/tmp/mlx-server.log, where monitor.py picks it up for its tok/s graph.

Line formats (kept short and stable — the monitor parses them):
  TOKPS live #12 48.2          decode, ~1/s while tokens flow (aggregate
                               across concurrent requests; #seq lets the
                               monitor tell "still generating" from "stale")
  TOKPS prefill #3 4127        prompt-processing rate, uncached tokens
  TOKPS done 512 tok · decode 48.9 tok/s (10.5s) · prompt 8231 tok, 5179 cached
                               per-request summary when the stream ends
"""
import importlib
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import mlx_lm.server as mlxs

# Model types that shipped ahead of mlx_lm releases (e.g. qwen4_exp from the
# mlx-lm PR #1788 port, plus our loader fixes for the oMLX oQ layout). The
# file name is the model_type; registration only kicks in when the installed
# mlx_lm doesn't have the module itself — a later release wins automatically.
_VENDOR_DIR = Path(__file__).parent / "vendor"


def _vendor_models() -> None:
    for path in sorted(_VENDOR_DIR.glob("*.py")):
        name = f"mlx_lm.models.{path.stem}"
        try:
            importlib.import_module(name)
            continue  # the installed mlx_lm has it
        except ImportError:
            pass
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # importlib checks sys.modules first
        spec.loader.exec_module(mod)

EMIT_EVERY = 1.0   # seconds between live lines
IDLE_RESET = 5.0   # a longer gap since the last token starts a fresh window

_lock = threading.Lock()
# aggregate decode state: tokens since the last emitted line and when the
# window opened (shared by all in-flight requests — the generation thread
# interleaves them, so a per-request rate would under-report)
_live = {"tokens": 0, "t0": None, "seq": 0}


def _emit(text: str) -> None:
    print(f"TOKPS {text}", flush=True)   # stdout -> /tmp/mlx-server.log


def _tok_bump() -> None:
    """One more token: open the live window, or emit + reset when it's full."""
    now = time.monotonic()
    with _lock:
        st = _live
        if st["t0"] is None:
            st["t0"] = now
        st["tokens"] += 1
        dt = now - st["t0"]
        if dt >= IDLE_RESET:
            # long pause between requests: reporting the accumulated 2 tokens
            # over 30 s would read as ~0 — start over instead
            st["tokens"], st["t0"] = 0, now
        elif dt >= EMIT_EVERY:
            st["seq"] += 1
            _emit(f"live #{st['seq']} {st['tokens'] / dt:.1f}")
            st["tokens"], st["t0"] = 0, now


def _prefill_cb(orig_cb):
    """Wrap the handler's progress callback: emit prefill rate lines ~1/s."""
    state = {"last_p": 0, "t": time.monotonic(), "seq": 0}

    def cb(processed, total):
        now = time.monotonic()
        dt = now - state["t"]
        if dt >= EMIT_EVERY and processed > state["last_p"]:
            state["seq"] += 1
            _emit(f"prefill #{state['seq']} {(processed - state['last_p']) / dt:.0f}")
            state["last_p"], state["t"] = processed, now
        if orig_cb is not None:
            orig_cb(processed, total)

    return cb


def _counting(ctx, inner):
    """Yield what `inner` yields, counting tokens and timing the phases."""
    n = 0
    t0 = time.monotonic()
    t_first = None
    try:
        for resp in inner:
            if t_first is None:
                t_first = time.monotonic()
            n += 1
            _tok_bump()
            yield resp
    finally:
        t_end = time.monotonic()
        prompt = getattr(ctx, "prompt", ()) or ()
        cached = max(0, getattr(ctx, "prompt_cache_count", 0) or 0)
        parts = [f"done {n} tok"]
        if n and t_first:
            decode_t = max(1e-9, t_end - t_first)
            parts.append(f"decode {n / decode_t:.1f} tok/s ({decode_t:.1f}s)")
        if prompt:
            ttft = (t_first or t_end) - t0
            parts.append(f"prompt {len(prompt)} tok, {cached} cached"
                         f" ({ttft:.1f}s to first)")
        _emit(" · ".join(parts))


def _msg_text(content) -> str:
    """Message content as text: a string, or OpenAI-style parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(str(p.get("text") or p.get("content") or ""))
            else:
                out.append(str(p))
        return "".join(out)
    return str(content)


def _leading_system(messages):
    sys = []
    for m in messages:
        if m.get("role") == "system":
            sys.append(m)
        else:
            break
    return sys


def _trim_to_fit(messages, target_chars):
    """The newest messages whose text fits target_chars, always keeping the
    leading system messages and cutting only on a user boundary (never
    splitting a tool_call/tool_result chain). None when nothing can go."""
    total = sum(len(_msg_text(m.get("content"))) + 8 for m in messages)
    acc = 0
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        acc += len(_msg_text(messages[i].get("content"))) + 8
        if acc > target_chars:
            cut = i + 1
            break
    while cut < len(messages) and messages[cut].get("role") != "user":
        cut += 1
    if cut >= len(messages):
        return None
    trimmed = _leading_system(messages) + messages[cut:]
    return trimmed if len(trimmed) < len(messages) else None


def _install_hook() -> None:
    orig_generate = mlxs.ResponseGenerator.generate
    # KV budget for this machine (tokens); 0 = unlimited. Set by servers.
    # start_mlx from safe_context(). Past it the prompt eval OOMs the Metal
    # command buffer and kills the generation thread — the whole server
    # 404s until restarted. Refusing the request keeps the server alive.
    max_input = int(os.environ.get("OLLMLX_MAX_INPUT_TOKENS", "0") or 0)

    def generate(self, request, generation_args, progress_callback=None):
        ctx, inner = orig_generate(
            self, request, generation_args, _prefill_cb(progress_callback))
        n = len(getattr(ctx, "prompt", ()) or ())
        if max_input and n > max_input:
            # the session outgrew the machine's KV budget: shrink it here —
            # clients like GUI apps can't /compact, and a 404 leaves them
            # stuck. Drop the oldest messages, keep the system prompt and a
            # fresh tail; the tail starts on a user message so tool chains
            # never split.
            messages = getattr(request, "messages", None)
            orig_len = len(messages) if isinstance(messages, list) else 0
            if isinstance(messages, list) and orig_len > 2:
                total_chars = sum(len(_msg_text(m.get("content"))) + 8
                                  for m in messages)
                target_chars = int(total_chars * max_input * 0.8 / n)
                for _ in range(3):
                    trimmed = _trim_to_fit(messages, target_chars)
                    if trimmed is None:
                        break
                    dropped = len(messages) - len(trimmed)
                    # honest to the model: without this note it would
                    # confidently answer about context it no longer has
                    sys_n = len(_leading_system(trimmed))
                    first = trimmed[sys_n]
                    content = first.get("content")
                    if isinstance(content, str):
                        first["content"] = (
                            f"[{dropped} oldest messages of this conversation"
                            " were dropped to fit the context budget; earlier"
                            " details are unavailable.]\n" + content)
                    request.messages = trimmed
                    _emit(f"trim attempt: sending {len(trimmed)} of "
                          f"{orig_len} messages (dropped {dropped})")
                    ctx, inner = orig_generate(
                        self, request, generation_args,
                        _prefill_cb(progress_callback))
                    n = len(getattr(ctx, "prompt", ()) or ())
                    if n <= max_input:
                        _emit(f"context trimmed: dropped {dropped} oldest "
                              f"messages, {n} tokens fit the budget — "
                              "compact the client session")
                        return ctx, _counting(ctx, inner)
                    target_chars = int(target_chars * 0.7)
                    _emit(f"context trim pass: still {n} tokens over "
                          f"{max_input} — shrinking further")
            raise ValueError(
                f"context too long: {n} tokens > {max_input} (the KV budget"
                " of this machine) — compact the session and retry")
        return ctx, _counting(ctx, inner)

    mlxs.ResponseGenerator.generate = generate


def main() -> None:
    try:
        _vendor_models()
    except Exception as exc:  # vendoring must not kill the server either
        print(f"model vendoring failed: {exc}", file=sys.stderr, flush=True)
    try:
        _install_hook()
    except Exception as exc:  # never let stats kill the server
        print(f"TOKPS hook not installed: {exc}", file=sys.stderr, flush=True)
    mlxs.main()


if __name__ == "__main__":
    main()
