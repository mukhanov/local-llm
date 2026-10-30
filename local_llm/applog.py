"""Persistent application log: ~/.ollmlx/logs/app.log (rotating).

The app used to leave nothing behind on a crash: its own exceptions went
to the terminal (gone), and the server logs lived in /tmp (wiped by
macOS on reboot). Now every run opens with a header line, lifecycle
events are logged (server starts, readiness, watchdog restarts, monitor
exit), and anything that would kill the process — an uncaught exception
in the main thread or in a background thread (watchdog, stack-start), or
SIGTERM/SIGHUP — is written here with a full traceback before the
process dies. The server children keep their own logs next to this file
(servers.py moved them out of /tmp for the same reason)."""
import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path

from . import __version__

log = logging.getLogger("local-llm")


def logs_dir() -> Path:
    """~/.ollmlx/logs (OLLMLX_HOME-respecting), created on demand."""
    d = (Path(os.environ.get("OLLMLX_HOME", "~/.ollmlx")).expanduser()
         / "logs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _signal_shutdown(signum, _frame):
    # die via an exception instead of vanishing: run_stack's finally then
    # stops the children (a hard kill orphans them) and the exit is logged
    log.info("%s — shutting down", signal.Signals(signum).name)
    raise SystemExit(128 + signum)


def setup() -> Path | None:
    """Attach the rotating file handler and the crash hooks. Returns the
    log path (None if the file can't be opened — the app runs on).
    Call once at process entry (cli.main), before anything else;
    safe to call again — the second call is a no-op."""
    path = logs_dir() / "app.log"
    root = logging.getLogger()
    if not any(isinstance(h, logging.handlers.RotatingFileHandler)
               and getattr(h, "baseFilename", "").endswith("app.log")
               for h in root.handlers):
        try:
            handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            root.setLevel(logging.INFO)
            root.addHandler(handler)
        except OSError:
            # an unwritable OLLMLX_HOME must not brick the app — crash
            # hooks below still fire (via logging's lastResort -> stderr)
            path = None

        prev_hook = sys.excepthook

        def _excepthook(tp, exc, tb):
            # KeyboardInterrupt is the normal q/Ctrl-C exit — no drama
            if tp is not KeyboardInterrupt:
                log.critical("uncaught exception — app dies",
                             exc_info=(tp, exc, tb))
            prev_hook(tp, exc, tb)   # keep the terminal traceback
        sys.excepthook = _excepthook

        prev_thread_hook = threading.excepthook

        def _thread_hook(args):
            # SystemExit is a worker's orderly bail-out (run_stack raises
            # it on failure paths) — the interesting crashes are the rest
            if args.exc_type not in (SystemExit, KeyboardInterrupt):
                log.error("uncaught exception in thread %s",
                          args.thread.name,
                          exc_info=(args.exc_type, args.exc_value,
                                    args.exc_traceback))
            prev_thread_hook(args)
        threading.excepthook = _thread_hook

        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, _signal_shutdown)
            except (ValueError, OSError, AttributeError):
                pass   # not the main thread / unsupported on this platform

    if path:
        log.info("local-llm %s · python %s · pid %d · args: %s",
                 __version__, sys.version.split()[0], os.getpid(),
                 " ".join(sys.argv[1:]) or "(model picker)")
    return path
