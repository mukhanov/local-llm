"""Native desktop monitor: local snapshot API and SwiftUI app launcher.

The widget is optional and reads the same logs/processes as monitor.py. Its
HTTP server lives only while the desktop window is open and binds localhost.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

from . import applog, hf, monitor, servers, ui
from .applog import logs_dir
from .config import Config


def _client_state():
    """Current model and its advertised cap from the generated OMP catalog."""
    f = Path("~/.omp/agent/models.json").expanduser()
    try:
        models = json.loads(f.read_text())["providers"]["ollmlx"]["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return "local", 0
    model = next((m["id"] for m in models
                  if m.get("id") not in ("local", "small")), "local")
    local = next((m for m in models if m.get("id") == "local"), {})
    return model, int(local.get("contextWindow", 0))


def _backend():
    """Backend of the latest stack start, or mlx for an old/empty app.log."""
    path = logs_dir() / "app.log"
    for line in reversed(monitor.tail_lines(path, 200)):
        if "stack start:" in line and " backend=llama" in line:
            return "llama"
        if "stack start:" in line:
            return "mlx"
    return "mlx"


def _loaded_model(procs, main_up, port):
    """Actual main process model, not the last client-catalog selection."""
    for name, p in procs:
        if name not in ("mlx_lm.server", "llama"):
            continue
        try:
            cmd = p.cmdline()
            flag = "-a" if name == "llama" else "--model"
            if flag in cmd:
                return cmd[cmd.index(flag) + 1]
        except (psutil.NoSuchProcess, psutil.AccessDenied, IndexError):
            pass
    if main_up:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=0.5) as r:
                data = json.load(r).get("data", [])
            if data and isinstance(data[0].get("id"), str):
                return data[0]["id"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None


def _perf(path):
    keys = ("live_seq", "live_rate", "pre_seq", "pre_rate", "summary",
            "context", "active")
    return dict(zip(keys, monitor.tail_tokps(path)))


class StackController:
    """Models launched by the tray remain its foreground children.

    A stack started elsewhere is never taken over: its watchdog would restart
    the old model if the tray merely killed the children.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = [rid for rid, _size, downloading
                       in hf.scan_installed(cfg.hf_hub) if not downloading]
        self.proc = None
        self.current = None
        self.lock = threading.Lock()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def can_switch(self):
        return self.running or not monitor.port_ok(self.cfg.mlx_port)

    def start(self, model):
        with self.lock:
            if model not in self.models:
                return 400, {"error": "model is not fully downloaded"}
            if self.running and self.current == model:
                return 202, {"model": model}
            if monitor.port_ok(self.cfg.mlx_port) and not self.running:
                return 409, {"error": "stop the terminal stack before switching"}
            if not self._stop():
                return 409, {"error": "previous model is still stopping"}
            python = (str(self.cfg.venv_python) if self.cfg.venv_python.exists()
                      else sys.executable)
            self.proc = subprocess.Popen(
                [python, "-m", "local_llm", "serve", model])
            self.current = model
            return 202, {"model": model}

    def _stop(self):
        if not self.running:
            self.proc = None
            self.current = None
            return True
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return False
        self.proc = None
        self.current = None
        return True

    def stop(self):
        with self.lock:
            self._stop()

def snapshot(cfg, controller=None) -> dict:
    """One UI-independent monitor sample; all values are JSON primitives."""
    _, max_input = _client_state()
    backend = _backend()
    main_log = servers.LLAMA_LOG if backend == "llama" else servers.MLX_LOG
    main_up = monitor.port_ok(cfg.mlx_port)
    small_up = bool(cfg.mlx_small_port and monitor.port_ok(cfg.mlx_small_port))
    percpu = psutil.cpu_percent(percpu=True)
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    procs = monitor.watch_procs(cfg.mlx_small_port, cfg.mlx_port)
    rows = []
    ours = 0
    uptime = 0
    for name, p in procs:
        try:
            rss = p.memory_info().rss
            ours += rss
            if name in ("mlx_lm.server", "llama") and main_up:
                uptime = max(uptime, int(time.time() - p.create_time()))
            rows.append({"name": name, "pid": p.pid,
                         "cpu": p.cpu_percent(), "rss": rss,
                         "rss_percent": rss / vm.total * 100,
                         "threads": p.num_threads()})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    model = _loaded_model(procs, main_up, cfg.mlx_port) or "no model loaded"
    eaters = monitor.top_eaters(3, {p.pid for _, p in procs})
    logs = [(tag, line) for tag, path in (
                (backend, main_log), ("litellm", servers.LITELLM_LOG),
                ("mlx-small", servers.MLX_SMALL_LOG),
                ("watchdog", servers.WATCHDOG_LOG))
            for line in monitor.tail_errors(path)][-3:]
    log_rows = [{"tag": tag, "text": text, "error": True}
                for tag, text in logs]
    log_rows += [{"tag": tag, "text": text, "error": False}
                 for tag, path in ((backend, main_log),
                                   ("litellm", servers.LITELLM_LOG))
                 for text in monitor.tail_lines(path, 100)]
    home = str(Path.home())
    clients = [
        {"name": "pi", "command": "pi --model ollmlx/local"},
        {"name": "omp", "command": "omp --model ollmlx/local"},
        {"name": "opencode", "command": "opencode --model ollmlx/local"},
        {"name": "claude", "command":
            f"claude --settings {str(cfg.claude_cfg).replace(home, '~')}"},
    ]
    return {
        "model": model,
        "backend": backend,
        "uptime_seconds": uptime,
        "max_input": max_input,
        "available_models": controller.models if controller else [],
        "can_switch": controller.can_switch if controller else False,
        "ports": {
            "main": {"label": backend, "port": cfg.mlx_port, "up": main_up},
            "small": ({"label": "small", "port": cfg.mlx_small_port,
                       "up": small_up} if cfg.mlx_small_port else None),
            "litellm": {"label": "litellm", "port": cfg.litellm_port,
                        "up": monitor.port_ok(cfg.litellm_port)},
        },
        "cpu": {"total": sum(percpu) / len(percpu) if percpu else 0,
                "cores": percpu},
        "memory": {"used": vm.used, "total": vm.total,
                   "available": vm.available, "percent": vm.percent},
        "swap": {"used": sw.used, "total": sw.total, "percent": sw.percent},
        "processes": rows,
        "process_summary": {
            "ours": ours, "others": max(0, vm.used - ours),
            "ours_percent": ours / vm.total * 100,
        },
        "top_eaters": [{"name": name, "pid": pid, "rss": rss}
                       for rss, pid, name in eaters],
        "main": _perf(main_log),
        "small": _perf(servers.MLX_SMALL_LOG) if small_up else None,
        "clients": clients,
        "logs": log_rows,
    }


def _make_server(cfg, controller=None):
    """Bind the snapshot API before launching the native window."""
    controller = controller or StackController(cfg)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/snapshot":
                self.send_error(404)
                return
            self.reply(200, snapshot(cfg, controller))

        def do_POST(self):
            if self.path != "/run":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 4096:
                    raise ValueError
                payload = json.loads(self.rfile.read(size))
                model = payload["model"]
                if not isinstance(model, str):
                    raise ValueError
            except (ValueError, KeyError, json.JSONDecodeError):
                self.reply(400, {"error": "expected a model name"})
                return
            self.reply(*controller.start(model))

        def log_message(self, _format, *args):
            pass

    try:
        return ThreadingHTTPServer(("127.0.0.1", cfg.widget_port), Handler)
    except OSError:
        raise SystemExit(f"   widget port :{cfg.widget_port} is in use"
                         " — set WIDGET_PORT to a free port") from None


def run_api(cfg) -> int:
    """Foreground API child of MLX Stack.app; SIGTERM also stops its model."""
    applog.setup()
    controller = StackController(cfg)
    server = _make_server(cfg, controller)
    parent = os.getppid()

    def watch_parent():
        # A killed/crashed app cannot run applicationWillTerminate. Once
        # launchd adopts us, keeping the API alive would leave a headless
        # model and a mystery Python process with no tray icon.
        while os.getppid() == parent:
            time.sleep(1)
        server.shutdown()

    threading.Thread(target=watch_parent, name="app-parent-watchdog",
                     daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
        return 0
    finally:
        server.server_close()
        controller.stop()


def _build(cfg) -> Path:
    from . import __version__

    root = Path(__file__).resolve().parent.parent / "macos"
    packaged = Path(__file__).resolve().parent / "macos"
    source_dir = packaged if (packaged / "widget.swift").is_file() else root
    source = source_dir / "widget.swift"
    template = source_dir / "Info.plist"
    icon = source_dir / "MLX Stack.icns"
    app = Path.home() / "Applications" / "MLX Stack.app"
    binary = app / "Contents" / "MacOS" / "MLXStack"
    plist = app / "Contents" / "Info.plist"
    app_icon = app / "Contents" / "Resources" / "MLX Stack.icns"
    if not source.is_file() or not template.is_file() or not icon.is_file():
        raise SystemExit(f"   widget sources are missing: {source_dir}")
    newest = max(source.stat().st_mtime, template.stat().st_mtime,
                 icon.stat().st_mtime)
    if (binary.exists() and binary.stat().st_mtime >= newest
            and plist.exists() and app_icon.exists()
            and f"<string>{__version__}</string>" in plist.read_text()):
        return app
    if not shutil.which("swiftc"):
        raise SystemExit("   Swift is required — run: xcode-select --install")
    ui.info("Building ~/Applications/MLX Stack.app")
    binary.parent.mkdir(parents=True, exist_ok=True)
    tmp = binary.with_suffix(".new")
    try:
        subprocess.run(["swiftc", "-O", "-parse-as-library", str(source),
                        "-o", str(tmp), "-framework", "SwiftUI",
                        "-framework", "AppKit"], check=True)
        os.replace(tmp, binary)
        plist.write_text(template.read_text().replace("__VERSION__", __version__))
        app_icon.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon, app_icon)
        if shutil.which("codesign"):
            subprocess.run(["codesign", "--force", "--deep", "--sign", "-",
                            str(app)], check=True)
    except subprocess.CalledProcessError:
        tmp.unlink(missing_ok=True)
        raise SystemExit("   native app build failed") from None
    return app


def run(cfg) -> int:
    """Build and open the standalone macOS app; it owns its API child."""
    app = _build(cfg)
    try:
        subprocess.run(["open", str(app), "--args", "--port",
                        str(cfg.widget_port)], check=True)
    except subprocess.CalledProcessError:
        raise SystemExit("   failed to open MLX Stack.app") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(run_api(Config.from_env()))
