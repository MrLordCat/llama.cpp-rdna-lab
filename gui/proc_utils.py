"""Subprocess helpers for the GUI.

Every console child spawned from a windowless GUI (pythonw / packaged exe)
opens a visible console window unless CREATE_NO_WINDOW is set. run_hidden()
is a drop-in for subprocess.run for capture-style calls made anywhere in the
GUI process.
"""

from __future__ import annotations

import os
import subprocess

HIDDEN_CREATIONFLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def run_hidden(command, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run that never flashes a console window on Windows."""
    if os.name == "nt" and "creationflags" not in kwargs:
        kwargs["creationflags"] = HIDDEN_CREATIONFLAGS
    return subprocess.run(command, **kwargs)


def suppress_windows_error_dialogs() -> None:
    """Stop Windows from showing modal error boxes for child processes.

    Without this, probing an exe with missing DLLs (e.g. llama-server from a
    ROCm build without its runtime in PATH) pops the system "code execution
    cannot proceed" dialog and blocks the GUI. The error mode is inherited by
    child processes, so calling once at startup covers every probe.
    """
    if os.name != "nt":
        return
    import ctypes

    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOOPENFILEERRORBOX = 0x8000
    try:
        current = ctypes.windll.kernel32.SetErrorMode(0)
        ctypes.windll.kernel32.SetErrorMode(current | SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX)
    except Exception:
        pass
