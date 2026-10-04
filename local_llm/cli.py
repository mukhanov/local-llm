"""Команды и оркестрация: run / stop / rm / list / help."""
import os
import sys
from pathlib import Path

from . import clients, hf, models, monitor, picker, servers, ui
from .config import Config

USAGE = """\
local-llm — поднять локальную MLX-модель + API для claude / pi / omp.

Что делает:
  1. Показывает локальные/рекомендованные модели (TUI: скролл, поиск, догрузка
     с HuggingFace, сортировка по совместимости с железом), качает выбранную.
  2. Запускает mlx_lm.server (OpenAI API) и litellm-прокси (OpenAI
     /v1/chat/completions + Anthropic /v1/messages для Claude Code) — как
     foreground-дети, без демонов: выход из монитора или Ctrl-C останавливает
     всё разом.
  3. Прописывает модель (+ фиксированный алиас ollmlx/local) в
     ~/.pi/agent/models.json, ~/.omp/agent/models.json и генерирует
     ~/.ollmlx/claude-local.json.
  4. Показывает htop-подобный TUI-монитор: CPU по ядрам + график истории,
     RAM/swap, статус серверов и команды запуска pi/omp/claude.

Использование (symlink: ~/bin/local-llm):
  local-llm                # интерактивный TUI-выбор модели + монитор
  local-llm <org/model>    # без вопросов
  local-llm list           # показать скачанные модели (с размерами)
  local-llm stop           # прибить утёкшие с прошлого запуска процессы
  local-llm rm [org/repo…] # удалить скачанные модели с диска (без аргументов — меню)
  local-llm token [hf_…]   # HF-токен: показать / сохранить / --clear (или env HF_TOKEN)

Env: MLX_KV_BITS=8 (квант KV-кэша, 0=off), MLX_PROMPT_CACHE_BYTES (0=off),
     MLX_PORT, LITELLM_PORT, LOAD_TIMEOUT, OLLMLX_HOME, HF_HUB_CACHE"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cfg = Config.from_env()
    if not argv:
        model = picker.pick_model(cfg)
        if model is None:
            ui.warn("выбор отменён — выход")
            return 0
        return run_stack(cfg, model)
    cmd, rest = argv[0], argv[1:]
    if cmd == "stop":
        servers.stop_all()
        return 0
    if cmd == "token":
        return cmd_token(cfg, rest)
    if cmd in ("rm", "remove", "del"):
        return cmd_rm(cfg, rest)
    if cmd in ("list", "ls"):
        return cmd_list(cfg)
    if "/" in cmd:
        return run_stack(cfg, cmd)
    ui.warn(f"'{cmd}' — не похоже на org/model, открываю выбор")
    model = picker.pick_model(cfg)
    if model is None:
        ui.warn("выбор отменён — выход")
        return 0
    return run_stack(cfg, model)


def run_stack(cfg: Config, model: str) -> int:
    servers.ensure_litellm()
    ui.info(f"Model: {model}")
    if hf.is_cached(cfg.hf_hub, model):
        ui.ok("already downloaded")
    else:
        ui.info("Downloading (via DoH-pinned DNS)")
        hf.download(model, cfg.hf_token, cfg.hf_hub)
        ui.ok("download complete")
    try:
        servers.start_mlx(cfg, model)
        servers.start_litellm(cfg, model)
        clients.write_client_configs(cfg, model)
        servers.warmup(cfg)
        print()
        ui.info(f"Готово. API: OpenAI и Anthropic на"
                f" http://127.0.0.1:{cfg.litellm_port}")
        print(f"  pi:      pi --model ollmlx/local   ({model})")
        print(f"  omp:     omp --model ollmlx/local   ({model})")
        print(f"  claude:  claude --settings {cfg.claude_cfg}")
        print()
        ui.info("Монитор системы (q или Ctrl-C — остановить всё)")
        home = str(Path.home())
        monitor.run(model, cfg.mlx_port, cfg.litellm_port,
                    str(cfg.claude_cfg).replace(home, "~"))
    finally:
        servers.stop_children()
    print()
    ui.ok("остановлено — mlx_lm.server и litellm остановлены")
    return 0


def cmd_token(cfg: Config, args: list[str]) -> int:
    """Показать/сохранить/стереть HF-токен. Приоритет: env HF_TOKEN > файл."""
    token_file = cfg.ollmlx_home / "hf-token"
    if args and args[0] == "--clear":
        if token_file.exists():
            token_file.unlink()
            ui.ok("токен удалён")
        else:
            ui.ok("сохранённого токена не было")
        return 0
    if args:
        cfg.ollmlx_home.mkdir(parents=True, exist_ok=True)
        token_file.write_text(args[0].strip() + "\n")
        token_file.chmod(0o600)
        ui.ok(f"токен сохранён в {token_file} (chmod 600)")
        return 0
    token = cfg.hf_token
    if not token:
        ui.warn("токен не задан: local-llm token <hf_...> или env HF_TOKEN")
        return 0
    from_env = bool(os.environ.get("HF_TOKEN") or
                    os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    src = "env HF_TOKEN" if from_env else str(token_file)
    masked = f"{token[:5]}…{token[-4:]}" if len(token) > 12 else "…"
    ui.info(f"HF-токен: {masked} (источник: {src})")
    return 0


def cmd_list(cfg: Config) -> int:
    installed = hf.scan_installed(cfg.hf_hub)
    if not installed:
        ui.warn(f"Скачанных LLM-моделей нет ({cfg.hf_hub})")
        return 0
    ui.info("💿 Скачанные модели:")
    for rid, size in installed:
        print(f"  {rid:<62} {size}  ~{models.ram_need_gb(rid)}GB RAM")
    return 0


def cmd_rm(cfg: Config, args: list[str]) -> int:
    installed = hf.scan_installed(cfg.hf_hub)
    ids = [rid for rid, _ in installed]
    if not installed:
        ui.warn(f"Скачанных LLM-моделей нет ({cfg.hf_hub})")
        return 0

    if args:
        targets = list(args)
    else:
        ui.info("💿 Скачанные модели:")
        for n, (rid, size) in enumerate(installed, 1):
            print(f"  {n:2d}) {rid:<62} {size}")
        try:
            raw = input("?# Удалить (номера или org/repo через пробел,"
                        " пусто — отмена): ").strip()
        except EOFError:
            raw = ""
        if not raw:
            ui.ok("отмена")
            return 0
        targets = []
        for t in raw.split():
            if t.isdigit():
                idx = int(t) - 1
                if 0 <= idx < len(ids):
                    targets.append(ids[idx])
                else:
                    ui.warn(f"нет пункта №{t}")
            elif "/" in t:
                targets.append(t)
            else:
                ui.warn(f"непонятно: {t} (нужен номер или org/repo)")

    if not targets:
        ui.ok("ничего не выбрано")
        return 0

    for d in targets:
        if "/" not in d:
            ui.warn(f"{d} — ожидается org/repo")
            continue
        if not hf.model_dir(cfg.hf_hub, d).is_dir():
            ui.warn(f"{d} — не найден в кэше")
            continue
        size = next((s for i, s in installed if i == d), "?")
        try:
            answer = input(f"?# Удалить {d} ({size})? [y/N] ").strip()
        except EOFError:
            ui.ok("отмена")
            return 0
        if answer.lower().startswith("y"):
            try:
                hf.delete_model(cfg.hf_hub, d)
                ui.ok(f"удалено: {d} (освобождено {size})")
            except OSError as exc:
                ui.warn(f"не удалось удалить: {exc}")
        else:
            ui.ok(f"пропущено: {d}")
    return 0


if __name__ == "__main__":  # python -m local_llm.cli
    sys.exit(main(sys.argv[1:]))
