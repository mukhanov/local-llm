# AGENTS.md — development notes for coding agents

One-command local LLM stack for Apple-Silicon Macs: model picker (curses),
MLX server + litellm proxy, OpenAI/Anthropic APIs on localhost, htop-style
monitor. Python 3.12+, macOS-only (MLX), ~4k lines, no test suite — changes
are proven by running the real thing (see Verification).

User-facing docs: [README.md](README.md). This file is the repo map for
agents (and humans) changing the code.

## Layout — what lives where

```
local-llm            sh launcher: creates ~/.ollmlx/venv via uv, editable install
pyproject.toml       the single source of dependencies (mlx-lm, huggingface-hub, psutil)
local_llm/
  cli.py             entry point (local-llm = local_llm.cli:main) and orchestration
  config.py          Config.from_env() — all settings are env vars, no config files
  applog.py          ~/.ollmlx/logs/app.log + crash/thread/signal hooks
  ui.py              colored print helpers (ui.info/ok/warn), locale/term fixes
  picker.py          curses model picker (plain-text fallback when not a tty)
  startup.py         fullscreen boot dialog; QueueWriter feeds it worker output
  monitor.py         curses monitor; owns the terminal until q
  servers.py         spawn/readiness/warmup/stop of children + watchdog thread
  clients.py         writes ~/.pi / ~/.omp / opencode / claude client configs
  mlxwrap.py         `python -m local_llm.mlxwrap` = mlx_lm.server + TOKPS hook
  hf.py              HuggingFace API + cache (DoH-pinned DNS), downloads
  models.py          model-name parsing, RAM estimates, fit scoring
  llmfit.py          llmfit catalog cache + composite-score port (Apache-2.0)
  vendor/            model impls ahead of mlx_lm releases, registered only
                     when the installed mlx_lm lacks the module
```

## Invariants — breaking these breaks running setups

1. **Module names are process contracts.** Children are spawned as
   `python -m local_llm.mlxwrap` and swept with `pkill -f` on the same
   string; `monitor.watch_procs` matches cmdline by module path. Renaming
   or moving a module means updating `servers.py` (`_MLX_PATTERN`, spawn
   `cmd` lists) and `monitor.py` together — prefer not renaming.
2. **Servers are foreground children, never daemons.** The monitor's exit
   (`q`/Ctrl-C) and every `finally` in `cli.run_stack` must stop them.
   No `nohup`, no backgrounding, no pidfiles.
3. **Logs live in `~/.ollmlx/logs/`** (via `applog.logs_dir()` /
   `servers.MLX_LOG` …). Never `/tmp` — macOS wipes it on reboot, which
   once erased every trace of a crash. Server logs rotate by generation
   (`.log.1` = previous run, `_open_log`); `app.log` rotates by size.
4. **Terminal output goes through `ui.*`; events go through
   `applog.log`.** While curses owns the screen, `sys.stdout/stderr` are
   redirected to `_UiSink` — a stray `print` would shred the display.
   New diagnostics: `log.info/warning/error` (persists) + `ui.warn`
   (user sees it now).
5. **Config is env-only** (`Config.from_env`, defaults in `config.py`).
   Don't add config files; new knobs = new env var + README table row +
   `--help` text in `cli.USAGE`.
6. **vendor/ is a shim, not a fork**: a vendored model module is used only
   when `import mlx_lm.models.<name>` fails — a later mlx_lm release wins
   automatically. Don't edit vendored files to diverge from upstream.
7. **The watchdog never gives up.** Crash-loop → slower probing + a note
   to compact the session, but restarts continue (`servers.supervise`).
8. **Ports/paths must never collide silently** — the port-collision check
   at the top of `cli.run_stack` runs before anything is started.

## Conventions

- Comments explain **why**, with the incident when there was one
  (`# seen 2026-10-05: post-OOM zombie answers /v1/models 200 …`).
  The repo's memory lives in these comments — keep the style.
- Docstrings: one line of what, then the non-obvious constraints.
- Dependencies: only via `pyproject.toml` (the launcher reinstalls on its
  hash change). No version pins in code, no hidden imports of optional
  deps outside `try: import` (see `clients._write_omp_smol_role`).
- Errors the user can act on: `ui.warn` with the fix in the message;
  fatal paths: `raise SystemExit("   message")` (indented like siblings).
- English everywhere in code/comments/commit messages; commit style:
  one-line imperative, no prefixes (see `git log`).

## Verification — no test suite, prove it by running

Fast checks (seconds, no big model):

```console
$ .venv/bin/python -m py_compile local_llm/*.py     # syntax gate
$ .venv/bin/python -m local_llm list                # reads the real HF cache
$ OLLMLX_HOME=/tmp/smoke MLX_SMALL_PORT=18081 HF_HUB_OFFLINE=1 \
  .venv/bin/python -c "                              # real server, ~1GB model
import sys; sys.path.insert(0, '.')
from local_llm import applog; applog.setup()
from local_llm.config import Config
from local_llm import servers
servers._note('smoke')                               # → app.log + watchdog log
p = servers.start_mlx_small(Config.from_env())       # spawn/ready lines in app.log
servers.stop_children()"
$ tail /tmp/smoke/logs/app.log
```

- `OLLMLX_HOME=/tmp/...` isolates venv-path/config writes; the helper
  model (`Qwen3-0.6B`, ~1GB) loads in ~3s — use it instead of the big
  model for any server-path change.
- Crash-path changes: `applog.setup()` then raise (main thread and inside
  a `threading.Thread`) — the traceback must appear in `app.log`.
- Anything touching the picker/monitor needs a real terminal; run
  `./local-llm` in one and check the screen, not just the exit code.
- Full-stack smoke (big model, minutes + tens of GB RAM):
  `./local-llm mlx-community/Qwen3-0.6B-4bit` — swap in a small model id
  when the weights don't need to be the real ones.
- Leftover processes after a failed run: `local-llm stop`.

## Regenerating client-facing bits

- `llmfit.py`'s catalog cache is fetched weekly at runtime — nothing to
  regenerate in-repo.
- `README.md` section anchors are linked from `cli.USAGE` help text and
  this file — keep headings stable when renaming.
