"""local-llm — локальный MLX-стек: модель + OpenAI/Anthropic API + монитор.

Модули:
  config   — пути, порты, env-настройки
  ui       — цветной вывод, терминал/локаль, железо
  models   — знания о моделях: парсинг имени, скоринг совместимости, рекомендации
  hf       — HuggingFace: DoH-обход DNS, API топа, кэш, загрузка/удаление
  picker   — интерактивный выбор модели (curses TUI + текстовый фолбек)
  servers  — жизненный цикл mlx_lm.server и litellm
  clients  — конфиги pi / omp / claude
  monitor  — htop-подобный TUI-монитор стека
  cli      — команды и оркестрация
"""

__version__ = "2.0.0"
