"""Background benchmark worker utilities."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import QTableWidgetItem


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

    @staticmethod
    def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._process is not None:
            self._terminate_process_tree(self._process)

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
                self._terminate_process_tree(process)

            for line in process.stdout:
                self.output.emit(line.rstrip())

                if self._stop_requested and process.poll() is None:
                    self._terminate_process_tree(process)

            process.wait()
            stopped = self._stop_requested
            self.finished_signal.emit(process.returncode == 0 and not stopped, stopped)
        except Exception as exc:
            self.output.emit(f"Bench error: {exc}")
            self.finished_signal.emit(False, self._stop_requested)
        finally:
            self._process = None


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem with numeric sort semantics via UserRole."""

    def __init__(self, text: str, numeric_value: float):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, numeric_value)

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left_value = self.data(Qt.ItemDataRole.UserRole)
            right_value = other.data(Qt.ItemDataRole.UserRole)
            if left_value is not None and right_value is not None:
                try:
                    return float(left_value) < float(right_value)
                except (TypeError, ValueError):
                    pass
        return super().__lt__(other)