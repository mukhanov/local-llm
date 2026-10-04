"""local-llm — local MLX stack: a model + OpenAI/Anthropic APIs + a monitor.

Modules:
  config   — paths, ports, env-driven settings
  ui       — colored output, terminal/locale handling, hardware info
  models   — model knowledge: name parsing, hardware-fit scoring, recommendations
  hf       — HuggingFace: DoH DNS workaround, top-list API, cache, download/delete
  picker   — interactive model picker (curses TUI + plain-text fallback)
  servers  — mlx_lm.server and litellm lifecycle
  clients  — pi / omp / claude config writer
  monitor  — htop-style stack monitor
  cli      — commands and orchestration
"""

__version__ = "2.0.0"
