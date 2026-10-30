"""local-llm — local MLX stack: a model + OpenAI/Anthropic APIs + a monitor.

Modules:
  config   — paths, ports, env-driven settings
  applog   — persistent app log (~/.ollmlx/logs/app.log) + crash hooks
  ui       — colored output, terminal/locale handling, hardware info
  cli      — commands and orchestration: pick → download → start → monitor
  picker   — interactive model picker (curses TUI + plain-text fallback)
  startup  — fullscreen boot dialog between the picker and the monitor
  monitor  — htop-style stack monitor (owns the terminal until q)
  servers  — mlx_lm.server and litellm lifecycle + the restart watchdog
  mlxwrap  — mlx_lm.server entry wrapper: tok/s lines + context guard
             (spawned as its own process: `python -m local_llm.mlxwrap`)
  clients  — pi / omp / Claude Code config writer
  hf       — HuggingFace: DoH DNS workaround, catalog API, cache, download
  models   — model knowledge: name parsing, RAM estimates, hardware fit
  llmfit   — llmfit catalog cache + composite score port for the picker
  vendor/  — model impls ahead of mlx_lm releases (used only if missing)

Module names are load-bearing: servers are spawned and pkill-matched by
module path (see AGENTS.md) — don't rename them casually.
"""

__version__ = "2.0.0"
