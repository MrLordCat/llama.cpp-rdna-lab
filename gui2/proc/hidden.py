"""Windows console handling for child processes.

A web/GUI process owns no console, so a child either flashes a window of its
own or, with CREATE_NO_WINDOW, gets no console at all -- and a process without
a console cannot be reached by GenerateConsoleCtrlEvent. llama-server needs
that signal: CTRL_BREAK runs its cleanup handler, while a hard kill tears the
process down mid-GPU-work. So children get their own *hidden* console and are
stopped by attaching a short-lived helper to it.

This mirrors gui/bench_runner.py from the old GUI, which is the version that
proved itself against ROCm and Vulkan builds.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = os.name == "nt"

CTRL_BREAK_EVENT = 1

#: Attach to the child's isolated console and broadcast CTRL_BREAK there.
_BREAK_HELPER = (
    "import ctypes,sys,time;"
    "k=ctypes.WinDLL('kernel32',use_last_error=True);"
    "k.FreeConsole();"
    "ok=k.AttachConsole(int(sys.argv[1]));"
    "ok or (_ for _ in ()).throw(OSError(ctypes.get_last_error(),'AttachConsole'));"
    "k.SetConsoleCtrlHandler(None,True);"
    "ok=k.GenerateConsoleCtrlEvent(1,int(sys.argv[1]));"
    "ok or (_ for _ in ()).throw(OSError(ctypes.get_last_error(),'GenerateConsoleCtrlEvent'));"
    "time.sleep(0.25)"
)


def console_python() -> str:
    """python.exe even when this process runs under pythonw.exe."""
    executable = Path(sys.executable)
    if IS_WINDOWS and executable.name.lower() == "pythonw.exe":
        console = executable.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(executable)


def spawn_options() -> dict[str, object]:
    """Popen keywords for a hidden but signalable child."""
    if not IS_WINDOWS:
        # Own process group, so a stop reaches the whole tree.
        return {"start_new_session": True}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,
        "startupinfo": startupinfo,
    }


def suppress_error_dialogs() -> None:
    """Stop Windows popping modal boxes for children with missing DLLs.

    A ROCm build launched without its runtime in PATH otherwise shows the
    "code execution cannot proceed" dialog and the child hangs until someone
    clicks it. The error mode is inherited, so one call at startup covers all
    children.
    """
    if not IS_WINDOWS:
        return
    import ctypes

    sem_failcriticalerrors = 0x0001
    sem_noopenfileerrorbox = 0x8000
    try:
        current = ctypes.windll.kernel32.SetErrorMode(0)
        ctypes.windll.kernel32.SetErrorMode(current | sem_failcriticalerrors | sem_noopenfileerrorbox)
    except Exception:  # pragma: no cover - best effort, never fatal
        pass


def send_break(pid: int) -> None:
    """Ask the process to shut down gracefully. Raises on failure."""
    if not IS_WINDOWS:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        return

    result = subprocess.run(
        [console_python(), "-c", _BREAK_HELPER, str(pid)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        detail = (result.stdout or "").strip()
        raise RuntimeError(detail or f"console-break helper exited with {result.returncode}")


def kill_tree(pid: int) -> None:
    """Last resort. Hard-killing GPU work can wedge the driver; ask first."""
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    os.killpg(os.getpgid(pid), signal.SIGKILL)
