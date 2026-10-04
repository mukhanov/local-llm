# local-llm

One command to run a local LLM on your Mac: pick a model that fits your RAM,
download it, serve it via OpenAI-compatible and Anthropic-compatible APIs, and
point your AI coding tools (`pi`, `omp`, Claude Code) at it.

```console
$ local-llm
```

```
 model picker — RAM 128GB · Apple M5 Max · sort: by score (s — toggle)
── 💿 Installed ────────────────────────────────────────────────────────────────
▸  mlx-community/Qwen3.5-122B-A10B-4bit                    ⚡91 · 64.8G on disk · ctx 262K
   mlx-community/GLM-4.5-Air-4bit                          ⚡79 · 49.3G on disk · ctx 131K
── ☁ HuggingFace · mlx-community · by score ───────────────────────────────────
   mlx-community/gemma-4-12B-it-qat-4bit             ⚡97 · ~8GB RAM · ↓24K · ⭐29 · ctx 262K
   mlx-community/gemma-4-12b-coder-fable5-composer2.5-8bit  ⚡97 · ~15GB RAM · ↓2K · ⭐37 · ctx 262K
   mlx-community/Llama-3.2-11B-Vision-Instruct-4bit  ⚡96 · ~8GB RAM · ↓1K · ⭐8 · ctx 131K · tools
   ...
 ⚡91.1 = quality 86 · speed 92 (37 tok/s) · fit 100 · ctx 100 · ~70GB of 128GB · Multimodal
 j/k↑↓ PgUp/PgDn g/G · / search · l more · s sort · a all⇄community · d delete · Enter select · q cancel
```

Behind the scenes it starts **mlx_lm.server** (OpenAI API on `:8080`) and a
**litellm** proxy (`:4000`) that adds an Anthropic `/v1/messages` endpoint, so
Claude Code can talk to the same local model. Both run as foreground children:
quitting the monitor (or Ctrl-C) stops everything — no daemons left behind.

## Features

- **Model picker (TUI)** — installed models first, then the HuggingFace top
  with paging, search, and score sorting (below). Models that won't fit your
  RAM are marked `⚠won't fit`. Press `a` to switch the catalog between the
  curated `mlx-community` org and **all MLX-format models** (any author —
  includes fresh personal re-quants); `/`-search queries HuggingFace
  server-side, so models beyond the loaded pages are findable too.
- **Readable download progress** — a single status line with a progress bar,
  percent, speed, ETA and the current file (native huggingface_hub's three
  interleaved tqdm bars are disabled).
- **API bridge** — OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`
  on one port, model exposed under a stable alias `ollmlx/local`.
- **Client config writer** — registers the model in `pi` and `omp` model
  catalogs and generates a Claude Code settings file.
- **System monitor** — htop-style curses UI: per-core CPU + history graph,
  RAM/swap, server status, process RSS, recent errors from the logs.
- **llmfit stats & score** — context length, tool-use capability and a RAM
  estimate for exotic quantization names next to each model in the picker,
  plus the **llmfit composite score** (a Python port of llmfit's fit
  calculator, Apache-2.0): quality × speed × memory-fit × context, weighted
  per use case and keyed to *your* chip's memory bandwidth. The list is the
  whole cached catalog ranked by that score, so the top of the list is
  llmfit's #1 for your machine (`a` widens it from `mlx-community` to all
  MLX authors) — and it renders instantly, offline: the weekly cache is
  enough for browsing, only `/`-search asks HuggingFace about models newer
  than the catalog. The row under the cursor gets a one-line breakdown
  (`⚡98.2 = quality 96 · speed 100 (53 tok/s) · …`), installed models
  included. Sourced from the [llmfit](https://github.com/AlexsJones/llmfit)
  model catalog (MLX-format slice, cached for a week in `~/.ollmlx`).
- **HF token support** — for gated models and API limits; stored outside the
  repo in `~/.ollmlx/hf-token` (chmod 600).
- **DoH-pinned DNS** — resolves `*.hf.co` via 1.1.1.1 when your TUN proxy
  hands out fake IPs and downloads die on TLS (see [Troubleshooting](#troubleshooting))).

## Requirements

- macOS on Apple Silicon (MLX runs only on M-series chips)
- [`uv`](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

That's the only manual prerequisite — Python itself and all dependencies are
installed automatically (see below). Run it and pick a model — no
configuration decisions needed.

## Install

From a git checkout — the launcher bootstraps everything on first run:

```console
$ git clone https://github.com/<you>/local-llm.git
$ cd local-llm
$ ./local-llm                 # creates the venv, installs deps, opens the picker
```

To have `local-llm` on your PATH, symlink the launcher:

```console
$ mkdir -p ~/bin && ln -s "$PWD/local-llm" ~/bin/local-llm   # if ~/bin is in PATH
```

Prefer standard tooling? The package is installable too:

```console
$ uv tool install .           # or: pipx install .
$ local-llm
# or run straight from the checkout without installing:
$ uv run local-llm
```

### Dependencies

Dependencies are declared once, in [`pyproject.toml`](pyproject.toml):
`mlx-lm` (pulls in MLX for your chip), `huggingface-hub`, `psutil`.
Nothing is installed globally:

| What                | Where                          | Installed when              |
|---------------------|--------------------------------|-----------------------------|
| Python 3.12         | managed by uv, if none found   | first run (`uv venv`)       |
| local-llm + deps    | `~/.ollmlx/venv` (editable)    | first run and whenever `pyproject.toml` changes |
| `litellm`           | uv tool (`~/.local/...`)       | first stack start, if missing |

The launcher reinstalls dependencies only when `pyproject.toml` changes, so
regular launches don't touch the network. `git pull` picks up code changes
immediately thanks to the editable install.

## Usage

```console
$ local-llm                     # interactive picker + monitor
$ local-llm mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
$ local-llm list                # downloaded models with real disk usage
$ local-llm rm                  # delete downloaded models (menu)
$ local-llm stop                # kill servers left over from a crashed run
$ local-llm token hf_xxx        # save HF token (or export HF_TOKEN)
$ local-llm token --clear
```

While the stack is up:

| Client     | Command                                       |
|------------|-----------------------------------------------|
| pi         | `pi --model ollmlx/local`                     |
| omp        | `omp --model ollmlx/local`                    |
| Claude Code| `claude --settings ~/.ollmlx/claude-local.json` |

Or talk to the API directly (both endpoints on the litellm port `:4000`):

```console
$ curl http://127.0.0.1:4000/v1/chat/completions \
    -H "Authorization: Bearer sk-local-llm" -H "Content-Type: application/json" \
    -d '{"model": "local", "messages": [{"role": "user", "content": "hi"}]}'

$ curl http://127.0.0.1:4000/v1/messages \
    -H "x-api-key: sk-local-llm" -H "Content-Type: application/json" \
    -d '{"model": "local", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}'
```

### Picker keybindings

| Key            | Action                                  |
|----------------|-----------------------------------------|
| `j/k` `↑/↓`    | move (past the end — load more from HF) |
| `PgUp/PgDn` `space` | page                                |
| `g` / `G`      | top / bottom                            |
| `/`            | search (Enter also queries HF server-side) |
| `l`            | load next page from HuggingFace         |
| `s`            | sort: hardware fit ⇄ downloads          |
| `a`            | catalog: `mlx-community` ⇄ all MLX models |
| `d`            | delete selected installed model         |
| `Enter`        | select and run                          |
| `q` `Esc`      | cancel                                  |

### Monitor

Shows CPU per core and a history graph, RAM/swap, port status, mlx/litellm
process stats, the client launch commands and the latest errors from
`/tmp/mlx-server.log` and `/tmp/litellm.log`. Press `q` (or Ctrl-C) to stop
the whole stack.

## Configuration

All via environment variables:

| Variable                  | Default      | Meaning                                    |
|---------------------------|--------------|--------------------------------------------|
| `MLX_PORT`                | `8080`       | mlx_lm.server port (OpenAI API)            |
| `LITELLM_PORT`            | `4000`       | litellm proxy port (OpenAI + Anthropic)    |
| `MLX_KV_BITS`             | `8`          | KV-cache quantization, `0` = off           |
| `MLX_KV_GROUP_SIZE`       | `64`         | KV-cache group size                        |
| `MLX_PROMPT_CACHE_BYTES`  | `8589934592` | prompt-cache cap, `0` = unlimited          |
| `LOAD_TIMEOUT`            | `900`        | seconds to wait for weights to load        |
| `OLLMLX_HOME`             | `~/.ollmlx`  | venv, litellm config, token, claude config |
| `HF_HUB_CACHE`            | `~/.cache/huggingface/hub` | model cache path           |
| `HF_TOKEN`                | —            | HuggingFace token (or `local-llm token`)   |

KV quantization matters on big models: long prompts (~30k tokens) with a full
16-bit cache can exhaust RAM on a model that "fits", which kills the Metal
generation thread — 8-bit keeps a 100GB-class model stable in 128GB RAM.

## Files

| Path                                  | Purpose                          |
|---------------------------------------|----------------------------------|
| `~/.ollmlx/venv`                      | isolated Python environment      |
| `~/.ollmlx/hf-token`                  | HF token, chmod 600              |
| `~/.ollmlx/llmfit-mlx.json.gz`        | llmfit catalog cache (weekly)    |
| `~/.ollmlx/litellm-config.yaml`       | generated proxy config           |
| `~/.ollmlx/claude-local.json`         | Claude Code settings             |
| `~/.pi/agent/models.json`             | `pi` model catalog (merged)      |
| `~/.omp/agent/models.json`            | `omp` model catalog (merged)     |
| `~/.cache/huggingface/hub`            | downloaded models                |
| `/tmp/mlx-server.log`, `/tmp/litellm.log` | server logs                 |

Deleting models (`local-llm rm`) is aware of the hub 1.x shared blob store
(`hub/blobs/<xx>/<sha>`): it frees the real weights and never touches blobs
shared with another downloaded model.

## Troubleshooting

- **Downloads hang / TLS errors on `*.hf.co`** — typically a TUN-mode proxy
  (fake-ip DNS) breaking name resolution. local-llm resolves HuggingFace hosts
  via DoH (1.1.1.1) and pins the result for the session. Plain `curl` from the
  same machine may still fail — that's expected.
- **Server didn't start** — check `/tmp/mlx-server.log` and `/tmp/litellm.log`;
  `local-llm stop` cleans up leftover processes.
- **Interrupted download** — huggingface_hub 1.x does not resume partial files
  across runs. local-llm detects an incomplete model before starting the stack
  (finished blobs are reused, stale `.incomplete` leftovers pruned, the rest
  re-downloaded), so a half-downloaded model never silently "serves".
- **Wrong model picked by the fit score** — press `s` to sort by downloads, or
  `/` to search, or just pass the full `org/repo` on the command line.

## Project layout

```
local-llm          # sh launcher: venv bootstrap from pyproject + exec
pyproject.toml     # dependencies, entry point, metadata
local_llm/
  cli.py           # commands and stack orchestration
  picker.py        # curses model picker (TUI + plain fallback)
  models.py        # model-string parsing, RAM estimates, fit scoring
  hf.py            # HuggingFace API, downloads, cache management, DoH
  llmfit.py        # llmfit catalog stats + composite score port for the picker
  servers.py       # mlx_lm.server + litellm lifecycle
  clients.py       # pi / omp / Claude Code config writer
  monitor.py       # curses system monitor
  config.py        # env-driven configuration
```

## License

[MIT](LICENSE)
