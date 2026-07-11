"""Background process runner for benchmark/autotune commands.

Extracted from benchmark_tab.py: runs a command in a QThread, streams output
lines, and asks the benchmark harness to stop gracefully on request.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


class BenchCommandThread(QThread):
    """Run benchmark command in background so UI remains responsive."""

    output = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, bool)

    def __init__(self, command: list[str], working_dir: Path, env: dict[str, str] | None = None):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.env = env or {}
        self._process: subprocess.Popen[str] | None = None
        self._stop_requested = False
        self._stop_signal_sent = False

    def _terminate_process_tree(self, proc: subprocess.Popen[str], wait: bool = True) -> None:
        if proc.poll() is not None:
            return
        soft_timeout = float(os.environ.get("LLAMA_GUI_BENCH_STOP_TIMEOUT", "240"))
        if not self._stop_signal_sent:
            self._stop_signal_sent = True
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception as exc:
                self.output.emit(f"Stop warning: graceful signal failed: {exc}")
                try:
                    proc.terminate()
                except Exception:
                    pass

        if not wait:
            return

        try:
            proc.wait(timeout=soft_timeout)
            return
        except subprocess.TimeoutExpired:
            pass

        hard_kill = os.environ.get("LLAMA_GUI_BENCH_ALLOW_HARD_KILL", "").strip().lower()
        if hard_kill not in {"1", "true", "yes", "on"}:
            self.output.emit(
                f"Stop warning: benchmark process pid={proc.pid} did not exit after {soft_timeout:.0f}s; "
                "leaving it alive to avoid hard-killing GPU work. "
                "Set LLAMA_GUI_BENCH_ALLOW_HARD_KILL=1 to force the old behavior."
            )
            return

        self.output.emit(f"Stop warning: force-killing benchmark process pid={proc.pid}")
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._process is not None:
            self._terminate_process_tree(self._process, wait=False)

    def run(self):
        try:
            process_env = os.environ.copy()
            process_env.update(self.env)

            process = subprocess.Popen(
                self.command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=process_env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )
            self._process = process

            if self._stop_requested:
                self._terminate_process_tree(process, wait=False)

            for line in process.stdout:
                self.output.emit(line.rstrip())

                if self._stop_requested and process.poll() is None:
                    self._terminate_process_tree(process, wait=False)

            process.wait()
            stopped = self._stop_requested
            self.finished_signal.emit(process.returncode == 0 and not stopped, stopped)
        except Exception as exc:
            self.output.emit(f"Bench error: {exc}")
            self.finished_signal.emit(False, self._stop_requested)
        finally:
            self._process = None
