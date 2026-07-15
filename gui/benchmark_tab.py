"""Benchmark tab - dedicated benchmark and autotune workflows."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend_names import backend_key_from_display, display_backend_from_key
from bench_history import BenchHistoryMixin
from bench_runner import BenchCommandThread, console_python_executable, send_windows_console_break
from server_backend_panels import (
    ROCM_DEVICE_CHOICES,
    VULKAN_DEVICE_CHOICES,
    device_choice_args,
)
from bench_widgets import (
    NumericTableWidgetItem,
    configure_combo,
    configure_compact_table,
    configure_spinbox,
    create_scroll_panel,
)
from model_capabilities import model_supports_mtp
from proc_utils import run_hidden
from ui_widgets import FlowLayout, LogView, StatusPill, make_chip


class _ServerHelpProbeThread(QThread):
    """Warm the server --help cache off the GUI thread (loading a server
    binary pulls in every backend DLL and can take seconds)."""

    def __init__(self, tab, server_bin: Path):
        super().__init__(tab)
        self._tab = tab
        self._server_bin = server_bin

    def run(self):
        self._tab._server_help_output(self._server_bin)


class LlamaServerStopThread(QThread):
    """Soft-stop leftover llama-server processes without force-killing GPU work."""

    output = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int)

    def __init__(self, wait_seconds: float = 90.0):
        super().__init__()
        self.wait_seconds = wait_seconds

    @staticmethod
    def _creationflags() -> int:
        return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    @staticmethod
    def _list_llama_servers() -> list[dict[str, object]]:
        if os.name == "nt":
            ps_script = (
                "$procs = Get-CimInstance Win32_Process -Filter \"name = 'llama-server.exe'\" | "
                "Select-Object ProcessId,ParentProcessId,CommandLine; "
                "if ($procs) { $procs | ConvertTo-Json -Compress }"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                creationflags=LlamaServerStopThread._creationflags(),
            )
            payload = result.stdout.strip()
            if not payload:
                return []
            data = json.loads(payload)
            if isinstance(data, dict):
                data = [data]
            procs = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    pid = int(item.get("ProcessId", 0))
                except (TypeError, ValueError):
                    continue
                if pid <= 0:
                    continue
                procs.append(
                    {
                        "pid": pid,
                        "parent_pid": item.get("ParentProcessId"),
                        "command": str(item.get("CommandLine") or ""),
                    }
                )
            return procs

        result = subprocess.run(
            ["pgrep", "-af", "llama-server"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        procs = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            procs.append({"pid": pid, "parent_pid": None, "command": parts[1] if len(parts) > 1 else ""})
        return procs

    @staticmethod
    def _send_windows_ctrl_break(pid: int) -> tuple[bool, str]:
        try:
            send_windows_console_break(pid)
            return True, "CTRL_BREAK sent"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _request_soft_terminate(pid: int) -> tuple[bool, str]:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=LlamaServerStopThread._creationflags(),
            )
            message = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            return result.returncode == 0, message

        os.kill(pid, signal.SIGTERM)
        return True, "SIGTERM sent"

    @staticmethod
    def _remaining_pids(target_pids: set[int]) -> set[int]:
        try:
            current = {int(proc["pid"]) for proc in LlamaServerStopThread._list_llama_servers()}
        except Exception:
            return target_pids
        return target_pids & current

    def run(self):
        try:
            procs = self._list_llama_servers()
        except Exception as exc:
            self.output.emit(f"[WARN] Failed to list llama-server processes: {exc}")
            self.finished_signal.emit(0, 0)
            return

        total = len(procs)
        if total == 0:
            self.output.emit("[INFO] No leftover llama-server processes found.")
            self.finished_signal.emit(0, 0)
            return

        target_pids = {int(proc["pid"]) for proc in procs}
        for proc in procs:
            pid = int(proc["pid"])
            command = str(proc.get("command") or "")
            short_command = command if len(command) <= 180 else command[:177] + "..."
            self.output.emit(f"[INFO] Soft-stopping llama-server pid={pid}: {short_command}")
            try:
                if os.name == "nt":
                    ok, message = self._send_windows_ctrl_break(pid)
                else:
                    ok, message = self._request_soft_terminate(pid)
                level = "INFO" if ok else "WARN"
                self.output.emit(f"[{level}] pid={pid}: {message}")
            except Exception as exc:
                self.output.emit(f"[WARN] pid={pid}: graceful stop failed: {exc}")

        first_wait = min(5.0, max(0.5, self.wait_seconds / 4.0))
        time.sleep(first_wait)

        remaining = self._remaining_pids(target_pids)
        for pid in sorted(remaining):
            try:
                ok, message = self._request_soft_terminate(pid)
                level = "INFO" if ok else "WARN"
                self.output.emit(f"[{level}] pid={pid}: soft task terminate: {message}")
            except Exception as exc:
                self.output.emit(f"[WARN] pid={pid}: soft task terminate failed: {exc}")

        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            remaining = self._remaining_pids(target_pids)
            if not remaining:
                break
            time.sleep(1.0)

        remaining = self._remaining_pids(target_pids)
        stopped = total - len(remaining)
        if remaining:
            self.output.emit(
                "[WARN] Still running after soft stop: "
                + ", ".join(str(pid) for pid in sorted(remaining))
                + ". No /F force-kill was used."
            )
        else:
            self.output.emit("[INFO] All leftover llama-server processes stopped softly.")
        self.finished_signal.emit(stopped, len(remaining))


class BenchmarkTabWidget(BenchHistoryMixin, QWidget):
    """Dedicated Bench & Autotune tab.

    History-table and preset persistence methods live in BenchHistoryMixin
    (bench_history.py); process running in bench_runner.py; shared widget
    helpers in bench_widgets.py.
    """

    NGRAM_MOD_N_MIN = 12
    NGRAM_MOD_N_MATCH = 16
    NGRAM_MOD_N_MAX = 32
    MTP_DRAFT_N_MAX = 2
    REAL_CONTEXT_SAFE_FILL = 0.88
    REAL_CONTEXT_RESERVE_TOKENS = 2048
    REAL_CONTEXT_CHARS_PER_TOKEN = 2.6

    # Autotune context lanes: (display, ctx, repo-snapshot chars).
    # Screen lane = fast preset hunt (short prompt, cheap prefill); Long lanes
    # request a large prompt but are capped by the benchmark script so they fit
    # the selected ctx; Max 130K keeps the legacy short-prompt KV-stress semantics.
    AUTOTUNE_LANES = [
        ("Screen 12K — fast preset hunt",             12288,  24576),
        ("Long ctx 49K — ~32K actual prompt",         49152,  147456),
        ("Long ctx 98K — safety-capped prompt",       98304,  294912),
        ("Max 130K — KV stress (short prompt)",       131072, 24576),
        ("Custom",                                    0,      0),
    ]

    VALIDATE_CTX_CHOICES = [("50K", 49152), ("100K", 98304), ("130K", 131072)]

    @staticmethod
    def _request_timeout_for_ctx(ctx_size: int) -> int:
        if ctx_size >= 131072:
            return 1800
        if ctx_size >= 98304:
            return 1500
        if ctx_size >= 49152:
            return 900
        return 300

    @staticmethod
    def _server_extra_arg(server_extra: str) -> str:
        # argparse treats a separate value that starts with '-' as a new option.
        # Use --flag=value form so server flags such as "--no-mmap" are preserved.
        return f"--server-extra={server_extra}"

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.settings = getattr(parent, "settings", None)
        self.models_dir = parent.models_dir if hasattr(parent, "models_dir") else Path("models")
        self.project_root = parent.project_root if hasattr(parent, "project_root") else Path.cwd()
        self.history_csv = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.csv"
        self.history_csv_v2 = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY_V2.csv"
        self.best_presets_path = self.project_root / "gui" / "model_autotune_best.json"
        self.bench_thread = None
        self._version_payloads: dict[str, dict[str, object]] = {}
        self._current_mode = "single"
        self._last_selected_model = ""
        self._current_autotune_profile = "ctx130k-only"
        self._current_build_id = ""
        self._summary_sweep_cache: dict[str, tuple[str, str]] = {}
        self._server_help_cache: dict[str, str] = {}
        self._bench_help_cache: dict[str, str] = {}
        self._autotune_result = {"best": "", "summary_json": "", "summary_csv": ""}
        self._autotune_active_run: str | None = None
        self.stop_all_thread: LlamaServerStopThread | None = None
        self.create_ui()
        self.refresh_models_list()
        self.refresh_build_choices()
        self.load_settings()
        self.refresh_saved_presets_table()

    def load_settings(self) -> None:
        if self.settings is None:
            return

        # widget-change signals fire save_settings while values are being
        # restored, overwriting not-yet-loaded keys with stale widget state
        self._loading_settings = True
        try:
            self.at_batch_min_spin.setValue(self.settings.value("benchmark/autotune/batch_min", self.at_batch_min_spin.value(), type=int))
            self.at_batch_max_spin.setValue(self.settings.value("benchmark/autotune/batch_max", self.at_batch_max_spin.value(), type=int))
            self.at_batch_step_spin.setValue(self.settings.value("benchmark/autotune/batch_step", self.at_batch_step_spin.value(), type=int))

            self.at_ubatch_min_spin.setValue(self.settings.value("benchmark/autotune/ubatch_min", self.at_ubatch_min_spin.value(), type=int))
            self.at_ubatch_max_spin.setValue(self.settings.value("benchmark/autotune/ubatch_max", self.at_ubatch_max_spin.value(), type=int))
            self.at_ubatch_step_spin.setValue(self.settings.value("benchmark/autotune/ubatch_step", self.at_ubatch_step_spin.value(), type=int))

            for name, checkbox in self.autotune_kv_checks.items():
                checkbox.setChecked(self.settings.value(f"benchmark/autotune/kv/{name}", checkbox.isChecked(), type=bool))

            for name, checkbox in self.autotune_spec_checks.items():
                checkbox.setChecked(self.settings.value(f"benchmark/autotune/spec/{name}", checkbox.isChecked(), type=bool))

            for name, checkbox in self.autotune_extra_checks.items():
                checkbox.setChecked(self.settings.value(f"benchmark/autotune/extra/{name}", checkbox.isChecked(), type=bool))

            self.autotune_custom_extra_input.setText(
                self.settings.value("benchmark/autotune/custom_extra", self.autotune_custom_extra_input.text())
            )
            self.autotune_resume_checkbox.setChecked(
                self.settings.value("benchmark/autotune/resume_session", self.autotune_resume_checkbox.isChecked(), type=bool)
            )
            self.autotune_reset_session_checkbox.setChecked(
                self.settings.value("benchmark/autotune/reset_session", self.autotune_reset_session_checkbox.isChecked(), type=bool)
            )

            mode_index = self.settings.value("benchmark/mode_tab", 0, type=int)
            if 0 <= mode_index < self.mode_tabs.count():
                self.mode_tabs.setCurrentIndex(mode_index)

            lane_index = self.settings.value("benchmark/autotune/lane", 0, type=int)
            if 0 <= lane_index < self.lane_combo.count():
                self.lane_combo.setCurrentIndex(lane_index)
            self.lane_custom_ctx_spin.setValue(
                self.settings.value("benchmark/autotune/lane_custom_ctx", self.lane_custom_ctx_spin.value(), type=int)
            )
            self.lane_custom_ctx_spin.setEnabled(self.AUTOTUNE_LANES[self.lane_combo.currentIndex()][1] == 0)
            self.autotune_device_sweep_check.setChecked(
                self.settings.value("benchmark/autotune/device_sweep", False, type=bool)
            )
            device_text = self.settings.value("benchmark/devices", "")
            if device_text:
                idx = self.device_combo.findText(device_text)
                if idx < 0:
                    # Labels evolve as measured recommendations change. Restore
                    # the actual device order from the stable prefix.
                    device_prefix = str(device_text).split(" — ", 1)[0]
                    for choice_idx in range(self.device_combo.count()):
                        if self.device_combo.itemText(choice_idx).split(" — ", 1)[0] == device_prefix:
                            idx = choice_idx
                            break
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
            self.scale_prompt_check.setChecked(
                self.settings.value("benchmark/scale_prompt", False, type=bool)
            )
            mtp_draft_n = self.settings.value("benchmark/mtp_draft_n", self.MTP_DRAFT_N_MAX, type=int)
            at_mtp_draft_n = self.settings.value("benchmark/autotune/mtp_draft_n", self.MTP_DRAFT_N_MAX, type=int)
            if not self.settings.value("benchmark/mtp_draft_n_default_migrated_v2", False, type=bool):
                if mtp_draft_n == 8:
                    mtp_draft_n = self.MTP_DRAFT_N_MAX
                    self.settings.setValue("benchmark/mtp_draft_n", mtp_draft_n)
                if at_mtp_draft_n == 8:
                    at_mtp_draft_n = self.MTP_DRAFT_N_MAX
                    self.settings.setValue("benchmark/autotune/mtp_draft_n", at_mtp_draft_n)
                self.settings.setValue("benchmark/mtp_draft_n_default_migrated_v2", True)
            self.mtp_draft_spin.setValue(mtp_draft_n)
            # CSV field superseded the autotune spinbox; legacy int is the fallback
            self.at_mtp_draft_input.setText(
                str(self.settings.value("benchmark/autotune/mtp_draft_values", str(at_mtp_draft_n)))
            )

            spec_filter = self.settings.value("benchmark/history/spec_filter", "All")
            idx = self.history_spec_filter_combo.findText(spec_filter)
            if idx >= 0:
                self.history_spec_filter_combo.setCurrentIndex(idx)
            # the lane combo is populated during table refresh; stash the wish
            self._history_lane_filter_saved = self.settings.value("benchmark/history/lane_filter", "All lanes")
            self._history_backend_filter_saved = self.settings.value("benchmark/history/backend_filter", "All backends")

            self._update_autotune_grid_preview()
        except Exception as exc:
            self.log_output.append(f"[WARN] Failed to load autotune settings: {exc}")
        finally:
            self._loading_settings = False

    def save_settings(self) -> None:
        if self.settings is None or getattr(self, "_loading_settings", False):
            return

        try:
            self.settings.setValue("benchmark/autotune/batch_min", self.at_batch_min_spin.value())
            self.settings.setValue("benchmark/autotune/batch_max", self.at_batch_max_spin.value())
            self.settings.setValue("benchmark/autotune/batch_step", self.at_batch_step_spin.value())

            self.settings.setValue("benchmark/autotune/ubatch_min", self.at_ubatch_min_spin.value())
            self.settings.setValue("benchmark/autotune/ubatch_max", self.at_ubatch_max_spin.value())
            self.settings.setValue("benchmark/autotune/ubatch_step", self.at_ubatch_step_spin.value())

            for name, checkbox in self.autotune_kv_checks.items():
                self.settings.setValue(f"benchmark/autotune/kv/{name}", checkbox.isChecked())

            for name, checkbox in self.autotune_spec_checks.items():
                self.settings.setValue(f"benchmark/autotune/spec/{name}", checkbox.isChecked())

            for name, checkbox in self.autotune_extra_checks.items():
                self.settings.setValue(f"benchmark/autotune/extra/{name}", checkbox.isChecked())

            self.settings.setValue("benchmark/autotune/custom_extra", self.autotune_custom_extra_input.text())
            self.settings.setValue("benchmark/autotune/resume_session", self.autotune_resume_checkbox.isChecked())
            self.settings.setValue("benchmark/autotune/reset_session", self.autotune_reset_session_checkbox.isChecked())
            self.settings.setValue("benchmark/mode_tab", self.mode_tabs.currentIndex())
            self.settings.setValue("benchmark/autotune/lane", self.lane_combo.currentIndex())
            self.settings.setValue("benchmark/autotune/lane_custom_ctx", self.lane_custom_ctx_spin.value())
            self.settings.setValue("benchmark/autotune/device_sweep", self.autotune_device_sweep_check.isChecked())
            self.settings.setValue("benchmark/devices", self.device_combo.currentText())
            self.settings.setValue("benchmark/scale_prompt", self.scale_prompt_check.isChecked())
            self.settings.setValue("benchmark/mtp_draft_n", self.mtp_draft_spin.value())
            self.settings.setValue("benchmark/autotune/mtp_draft_values", self.at_mtp_draft_input.text().strip())
            self.settings.setValue("benchmark/history/spec_filter", self.history_spec_filter_combo.currentText())
            self.settings.setValue("benchmark/history/lane_filter", self.history_lane_filter_combo.currentText())
            self.settings.setValue("benchmark/history/backend_filter", self.history_backend_filter_combo.currentText())
        except Exception as exc:
            self.log_output.append(f"[WARN] Failed to save autotune settings: {exc}")

    # thin delegates to bench_widgets (kept so the many call sites stay unchanged)
    _create_scroll_panel    = staticmethod(create_scroll_panel)
    _configure_combo        = staticmethod(configure_combo)
    _configure_spinbox      = staticmethod(configure_spinbox)
    _configure_compact_table = staticmethod(configure_compact_table)

    def create_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        info_label = QLabel("📈 Bench & Autotune - dedicated benchmark workflows")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        info_label.setWordWrap(True)
        root_layout.addWidget(info_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root_layout.addWidget(splitter, 1)

        left_widget = QWidget()
        left_widget.setMinimumWidth(0)
        left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)
        right_widget = QWidget()
        right_widget.setMinimumWidth(0)
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(8)

        splitter.addWidget(self._create_scroll_panel(left_widget))
        splitter.addWidget(self._create_scroll_panel(right_widget))
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 520])

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()

        model_path_row = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to GGUF model...")
        self.model_path_input.setMinimumWidth(0)
        model_path_row.addWidget(self.model_path_input, 1)

        self.model_browse_btn = QPushButton("Browse")
        self.model_browse_btn.setToolTip("Select a GGUF model file")
        self.model_browse_btn.clicked.connect(self.browse_model)
        model_path_row.addWidget(self.model_browse_btn)
        model_layout.addLayout(model_path_row)

        model_list_row = QHBoxLayout()
        model_list_row.addWidget(QLabel("Detected:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_selected)
        model_list_row.addWidget(self.model_combo, 1)

        self.model_refresh_btn = QPushButton("Refresh")
        self.model_refresh_btn.setToolTip("Refresh detected GGUF models")
        self.model_refresh_btn.clicked.connect(self.refresh_models_list)
        model_list_row.addWidget(self.model_refresh_btn)
        model_layout.addLayout(model_list_row)

        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        build_group = QGroupBox("Build Target")
        build_layout = QGridLayout()
        build_layout.setHorizontalSpacing(8)
        build_layout.setVerticalSpacing(6)
        build_layout.addWidget(QLabel("Backend:"), 0, 0)
        self.build_backend_combo = QComboBox()
        self.build_backend_combo.currentTextChanged.connect(self._on_backend_changed)
        build_layout.addWidget(self.build_backend_combo, 0, 1)

        build_layout.addWidget(QLabel("Version:"), 1, 0)
        self.build_version_combo = QComboBox()
        build_layout.addWidget(self.build_version_combo, 1, 1)

        build_layout.addWidget(QLabel("Devices:"), 2, 0)
        self.device_combo = QComboBox()
        self.device_combo.setToolTip(
            "GPU selection for bench/autotune server runs (-dev/-sm/-ts).\n"
            "Auto = backend default. Applies to both Single Bench and Auto-tune;\n"
            "ignored by Auto-tune when the single-vs-dual sweep is enabled."
        )
        build_layout.addWidget(self.device_combo, 2, 1)
        build_layout.setColumnStretch(1, 1)
        build_group.setLayout(build_layout)
        left_layout.addWidget(build_group)
        self._device_choices: list[tuple[str, list[str]]] = [("Auto — backend default", [])]
        self._refresh_device_choices()

        # Mode sub-tabs: Single Bench and Auto-tune each carry their own
        # parameters and run button, so settings sit next to the action.
        self.mode_tabs = QTabWidget()

        single_page = QWidget()
        single_page_layout = QVBoxLayout(single_page)
        single_page_layout.setContentsMargins(6, 6, 6, 6)
        single_page_layout.setSpacing(8)

        single_layout = QGridLayout()
        single_layout.setHorizontalSpacing(8)
        single_layout.setVerticalSpacing(6)

        single_layout.addWidget(QLabel("Tasks:"), 0, 0)
        self.tasks_combo = QComboBox()
        self.tasks_combo.addItems(["v2-mini", "v2", "quick", "full"])
        self.tasks_combo.setCurrentText("quick")
        single_layout.addWidget(self.tasks_combo, 0, 1)

        single_layout.addWidget(QLabel("Runs:"), 1, 0)
        self.runs_spin = QSpinBox()
        self.runs_spin.setMinimum(1)
        self.runs_spin.setMaximum(10)
        self.runs_spin.setValue(1)
        single_layout.addWidget(self.runs_spin, 1, 1)

        single_layout.addWidget(QLabel("Spec:"), 2, 0)
        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["none", "ngram-mod", "mtp", "ngram-mtp"])
        self.spec_combo.setCurrentText("none")
        single_layout.addWidget(self.spec_combo, 2, 1)

        single_layout.addWidget(QLabel("Ctx:"), 3, 0)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setMinimum(8192)
        self.ctx_spin.setMaximum(131072)
        self.ctx_spin.setValue(131072)
        self.ctx_spin.setSingleStep(8192)
        single_layout.addWidget(self.ctx_spin, 3, 1)

        single_layout.addWidget(QLabel("Batch:"), 4, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(32)
        self.batch_spin.setMaximum(8192)
        self.batch_spin.setValue(512)
        self.batch_spin.setSingleStep(32)
        single_layout.addWidget(self.batch_spin, 4, 1)

        single_layout.addWidget(QLabel("UBatch:"), 5, 0)
        self.ubatch_spin = QSpinBox()
        self.ubatch_spin.setMinimum(32)
        self.ubatch_spin.setMaximum(8192)
        self.ubatch_spin.setValue(128)
        self.ubatch_spin.setSingleStep(32)
        single_layout.addWidget(self.ubatch_spin, 5, 1)

        single_layout.addWidget(QLabel("KV K/V:"), 6, 0)
        self.kv_combo = QComboBox()
        self.kv_combo.addItems(["q8_0", "q4_0", "f16", "bf16", "f32"])
        self.kv_combo.setCurrentText("q4_0")
        single_layout.addWidget(self.kv_combo, 6, 1)

        single_layout.addWidget(QLabel("Max tokens:"), 7, 0)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setMinimum(8)
        self.max_tokens_spin.setMaximum(1024)
        self.max_tokens_spin.setValue(16)
        single_layout.addWidget(self.max_tokens_spin, 7, 1)

        single_layout.addWidget(QLabel("MTP draft N:"), 8, 0)
        self.mtp_draft_spin = QSpinBox()
        self.mtp_draft_spin.setRange(1, 20)
        self.mtp_draft_spin.setValue(self.MTP_DRAFT_N_MAX)
        self.mtp_draft_spin.setToolTip(
            "--spec-draft-n-max, used when Spec is mtp/ngram-mtp.\n"
            "2 is the safer default for big prompts; try 4/8 only when tuning short-context decode."
        )
        single_layout.addWidget(self.mtp_draft_spin, 8, 1)
        single_layout.setColumnStretch(1, 1)
        single_page_layout.addLayout(single_layout)

        self.scale_prompt_check = QCheckBox("Long-prompt mode: scale prompt to ctx (~80% fill)")
        self.scale_prompt_check.setChecked(False)
        self.scale_prompt_check.setToolTip(
            "OFF: fixed 24576-char repo snapshot (~7K-token prompt), comparable\n"
            "with existing history. ON: prompt scales to ~80% of the context —\n"
            "use this to actually test long-prompt behavior at 50-100K."
        )
        single_page_layout.addWidget(self.scale_prompt_check)

        self.apply_best_btn = QPushButton("⭐ Apply Best Known")
        self.apply_best_btn.setToolTip(
            "Fill ctx/batch/ubatch/KV/spec from the best autotune result recorded for the selected model"
        )
        self.apply_best_btn.clicked.connect(self.apply_best_known_config)
        single_page_layout.addWidget(self.apply_best_btn)

        self.run_bench_btn = QPushButton("▶ Run Benchmark")
        self.run_bench_btn.setToolTip("Run a single benchmark with the parameters above")
        self.run_bench_btn.clicked.connect(self.run_benchmark)
        single_page_layout.addWidget(self.run_bench_btn)

        validate_row = QHBoxLayout()
        validate_row.addWidget(QLabel("Validate best at:"))
        self.validate_ctx_combo = QComboBox()
        for display, _ctx in self.VALIDATE_CTX_CHOICES:
            self.validate_ctx_combo.addItem(display)
        validate_row.addWidget(self.validate_ctx_combo)

        self.validate_best_btn = QPushButton("🔬 Validate Best (long prompt)")
        self.validate_best_btn.setToolTip(
            "Stage 2 of the screen→validate flow: apply the model's best known\n"
            "config, set the chosen long context, enable long-prompt mode and\n"
            "run a single benchmark — one click."
        )
        self.validate_best_btn.clicked.connect(self.validate_best_at_long_ctx)
        validate_row.addWidget(self.validate_best_btn, 1)
        single_page_layout.addLayout(validate_row)
        single_page_layout.addStretch(1)

        self.mode_tabs.addTab(single_page, "▶ Single Bench")

        autotune_page = QWidget()
        autotune_layout = QVBoxLayout(autotune_page)
        autotune_layout.setContentsMargins(6, 6, 6, 6)
        autotune_layout.setSpacing(8)

        lane_row = QHBoxLayout()
        lane_row.addWidget(QLabel("Lane:"))
        self.lane_combo = QComboBox()
        for display, _ctx, _chars in self.AUTOTUNE_LANES:
            self.lane_combo.addItem(display)
        self.lane_combo.setToolTip(
            "Context lane for this autotune run.\n"
            "Screen 12K: fast preset hunt (short prompt, cheap prefill).\n"
            "Long 50K/100K: prompt scales to ~80% of ctx — real long-prompt test.\n"
            "Max 130K: legacy short-prompt KV-stress lane (comparable with history)."
        )
        lane_row.addWidget(self.lane_combo, 1)

        lane_row.addWidget(QLabel("ctx:"))
        self.lane_custom_ctx_spin = QSpinBox()
        self.lane_custom_ctx_spin.setRange(8192, 131072)
        self.lane_custom_ctx_spin.setSingleStep(4096)
        self.lane_custom_ctx_spin.setValue(32768)
        self.lane_custom_ctx_spin.setEnabled(False)
        self.lane_custom_ctx_spin.setToolTip("Context size for the Custom lane")
        lane_row.addWidget(self.lane_custom_ctx_spin)
        autotune_layout.addLayout(lane_row)

        self.autotune_mode_info = QLabel("")
        self.autotune_mode_info.setWordWrap(True)
        self.autotune_mode_info.setStyleSheet("color: #b0b0b0;")
        autotune_layout.addWidget(self.autotune_mode_info)

        batch_grid = QGridLayout()
        batch_grid.setHorizontalSpacing(8)
        batch_grid.setVerticalSpacing(6)

        batch_grid.addWidget(QLabel("Batch:"), 0, 0)
        self.at_batch_min_spin = QSpinBox()
        self.at_batch_min_spin.setMinimum(32)
        self.at_batch_min_spin.setMaximum(8192)
        self.at_batch_min_spin.setValue(256)
        self.at_batch_min_spin.setSingleStep(32)
        self.at_batch_min_spin.setToolTip("Minimal batch value in sweep (>= 32)")
        batch_grid.addWidget(self.at_batch_min_spin, 0, 1)
        batch_grid.addWidget(QLabel("–"), 0, 2)
        self.at_batch_max_spin = QSpinBox()
        self.at_batch_max_spin.setMinimum(32)
        self.at_batch_max_spin.setMaximum(8192)
        self.at_batch_max_spin.setValue(1024)
        self.at_batch_max_spin.setSingleStep(32)
        self.at_batch_max_spin.setToolTip("Maximal batch value in sweep")
        batch_grid.addWidget(self.at_batch_max_spin, 0, 3)
        batch_grid.addWidget(QLabel("step"), 0, 4)
        self.at_batch_step_spin = QSpinBox()
        self.at_batch_step_spin.setMinimum(1)
        self.at_batch_step_spin.setMaximum(8192)
        self.at_batch_step_spin.setValue(256)
        self.at_batch_step_spin.setSingleStep(1)
        self.at_batch_step_spin.setToolTip("Increment for batch range")
        batch_grid.addWidget(self.at_batch_step_spin, 0, 5)

        batch_grid.addWidget(QLabel("UBatch:"), 1, 0)
        self.at_ubatch_min_spin = QSpinBox()
        self.at_ubatch_min_spin.setMinimum(32)
        self.at_ubatch_min_spin.setMaximum(8192)
        self.at_ubatch_min_spin.setValue(64)
        self.at_ubatch_min_spin.setSingleStep(32)
        self.at_ubatch_min_spin.setToolTip("Minimal ubatch value in sweep (>= 32)")
        batch_grid.addWidget(self.at_ubatch_min_spin, 1, 1)
        batch_grid.addWidget(QLabel("–"), 1, 2)
        self.at_ubatch_max_spin = QSpinBox()
        self.at_ubatch_max_spin.setMinimum(32)
        self.at_ubatch_max_spin.setMaximum(8192)
        self.at_ubatch_max_spin.setValue(256)
        self.at_ubatch_max_spin.setSingleStep(32)
        self.at_ubatch_max_spin.setToolTip("Maximal ubatch value in sweep")
        batch_grid.addWidget(self.at_ubatch_max_spin, 1, 3)
        batch_grid.addWidget(QLabel("step"), 1, 4)
        self.at_ubatch_step_spin = QSpinBox()
        self.at_ubatch_step_spin.setMinimum(1)
        self.at_ubatch_step_spin.setMaximum(8192)
        self.at_ubatch_step_spin.setValue(64)
        self.at_ubatch_step_spin.setSingleStep(1)
        self.at_ubatch_step_spin.setToolTip("Increment for ubatch range")
        batch_grid.addWidget(self.at_ubatch_step_spin, 1, 5)
        batch_grid.setColumnStretch(6, 1)
        autotune_layout.addLayout(batch_grid)

        kv_row = QHBoxLayout()
        kv_label = QLabel("KV sweep:")
        kv_label.setFixedWidth(86)
        kv_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        kv_row.addWidget(kv_label)
        kv_chip_host = QWidget()
        kv_flow = FlowLayout(kv_chip_host)
        self.autotune_kv_checks: dict[str, QPushButton] = {}
        for kv_name, enabled, hint in [
            ("q4_0", True, "Main KV cache for the current 130K target"),
            ("q8_0", False, "Higher-quality KV cache opt-in"),
            ("turbo4", False, "TurboKV 4-bit cache (128-block WHT, correctness path)"),
            ("turbo3", False, "TurboKV 3-bit cache (128-block WHT, correctness path)"),
            ("turbo2", False, "TurboKV 2-bit cache (128-block WHT, correctness path)"),
            ("f16", False, "FP16 KV (usually slower/heavier)"),
            ("bf16", False, "BF16 KV (usually slower/heavier)"),
            ("f32", False, "FP32 KV (debug/reference only)"),
        ]:
            chip = make_chip(kv_name, hint, enabled)
            self.autotune_kv_checks[kv_name] = chip
            kv_flow.addWidget(chip)
        kv_row.addWidget(kv_chip_host, 1)
        autotune_layout.addLayout(kv_row)

        spec_row = QHBoxLayout()
        spec_label = QLabel("Spec sweep:")
        spec_label.setFixedWidth(86)
        spec_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        spec_row.addWidget(spec_label)
        spec_chip_host = QWidget()
        spec_flow = FlowLayout(spec_chip_host)
        self.autotune_spec_checks: dict[str, QPushButton] = {}
        for mode, enabled, hint in [
            ("none", True, "Always keep plain decoding baseline in sweep"),
            ("ngram-mod", False, "Ngram speculative mode for explicit repeated/session probes"),
            ("draft", False, "Draft speculative mode when supported"),
            ("eagle3", False, "Eagle3 speculative mode when supported"),
            ("mtp", False, "MTP mode; requires server + MTP model support"),
            ("ngram-mtp", False, "Experimental ngram first, MTP fallback mode"),
        ]:
            chip = make_chip(mode, hint, enabled)
            self.autotune_spec_checks[mode] = chip
            spec_flow.addWidget(chip)
        spec_row.addWidget(spec_chip_host, 1)
        autotune_layout.addLayout(spec_row)

        mtp_draft_row = QHBoxLayout()
        mtp_draft_row.addWidget(QLabel("MTP draft N max:"))
        self.at_mtp_draft_input = QLineEdit()
        self.at_mtp_draft_input.setText(str(self.MTP_DRAFT_N_MAX))
        self.at_mtp_draft_input.setMaximumWidth(120)
        self.at_mtp_draft_input.setToolTip(
            "--spec-draft-n-max for mtp/ngram-mtp sweep configs.\n"
            "One value (e.g. 8) or a comma list (e.g. 2,4,8) — a list sweeps\n"
            "every draft budget in the same autotune run (multiplies the grid).\n"
            "2 is the safer default for big prompts; 4/8 for short-ctx decode."
        )
        mtp_draft_row.addWidget(self.at_mtp_draft_input)
        mtp_draft_row.addStretch()
        autotune_layout.addLayout(mtp_draft_row)

        extra_row = QHBoxLayout()
        extra_label = QLabel("Extra presets:")
        extra_label.setFixedWidth(86)
        extra_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        extra_row.addWidget(extra_label)
        self._autotune_extra_presets_map: dict[str, str] = {
            "base": "base",
            "ngram-balanced": (
                "ngram-balanced::"
                f"--spec-ngram-mod-n-min {self.NGRAM_MOD_N_MIN} "
                f"--spec-ngram-mod-n-match {self.NGRAM_MOD_N_MATCH} "
                f"--spec-ngram-mod-n-max {self.NGRAM_MOD_N_MAX}"
            ),
            "ngram-wide": "ngram-wide::--spec-ngram-mod-n-min 64 --spec-ngram-mod-n-match 32 --spec-ngram-mod-n-max 96",
        }
        extra_chip_host = QWidget()
        extra_flow = FlowLayout(extra_chip_host)
        self.autotune_extra_checks: dict[str, QPushButton] = {}
        for key, enabled, hint in [
            ("base", True, "No extra server arguments"),
            ("ngram-balanced", False, "Measured ngram-mod profile: 12/16/32"),
            ("ngram-wide", False, "Wider ngram window for ngram-mod"),
        ]:
            chip = make_chip(key, hint, enabled)
            self.autotune_extra_checks[key] = chip
            extra_flow.addWidget(chip)
        extra_row.addWidget(extra_chip_host, 1)
        autotune_layout.addLayout(extra_row)

        custom_extra_row = QHBoxLayout()
        custom_extra_row.addWidget(QLabel("Custom extras:"))
        self.autotune_custom_extra_input = QLineEdit()
        self.autotune_custom_extra_input.setPlaceholderText("mtp-n2::--spec-draft-n-max 2||mtp-n4::--spec-draft-n-max 4")
        self.autotune_custom_extra_input.setToolTip(
            "Optional extra presets; separate presets with ||.\n"
            "A preset's --spec-draft-n-max overrides the MTP draft N default,\n"
            "so several presets sweep draft budgets in one run."
        )
        self.autotune_custom_extra_input.setMinimumWidth(0)
        custom_extra_row.addWidget(self.autotune_custom_extra_input)
        autotune_layout.addLayout(custom_extra_row)

        self.autotune_device_sweep_check = QCheckBox("Sweep GPU order + single GPUs")
        self.autotune_device_sweep_check.setChecked(False)
        self.autotune_device_sweep_check.setToolTip(
            "Cross-multiplies every extra preset with four device configurations:\n"
            "both dual-GPU orders and each GPU by itself. This overrides the\n"
            "Devices selection above and multiplies the autotune grid by four."
        )
        autotune_layout.addWidget(self.autotune_device_sweep_check)

        quickset_row = QHBoxLayout()
        quickset_row.addWidget(QLabel("Grid quick-set:"))
        self.screen_grid_btn = QPushButton("Screen grid")
        self.screen_grid_btn.setToolTip(
            "Minimal stage-1 grid: batch=512, ubatch 64–256, kv=q4_0, spec=none, extras=base"
        )
        self.screen_grid_btn.clicked.connect(self._apply_screen_grid)
        quickset_row.addWidget(self.screen_grid_btn)

        self.full_grid_btn = QPushButton("Full grid")
        self.full_grid_btn.setToolTip("Wide sweep: batch 256–1024, ubatch 64–256, kv q4_0+q8_0")
        self.full_grid_btn.clicked.connect(self._apply_full_grid)
        quickset_row.addWidget(self.full_grid_btn)
        quickset_row.addStretch()
        autotune_layout.addLayout(quickset_row)

        session_grid = QGridLayout()
        session_grid.setHorizontalSpacing(14)
        session_grid.setVerticalSpacing(4)
        self.autotune_resume_checkbox = QCheckBox("Resume unfinished session")
        self.autotune_resume_checkbox.setChecked(True)
        self.autotune_resume_checkbox.setToolTip("Continue from saved autotune progress if a previous run was interrupted")
        session_grid.addWidget(self.autotune_resume_checkbox, 0, 0)

        self.autotune_reset_session_checkbox = QCheckBox("Reset saved session before run")
        self.autotune_reset_session_checkbox.setChecked(False)
        self.autotune_reset_session_checkbox.setToolTip("Ignore and overwrite previous session checkpoint for this model/profile")
        session_grid.addWidget(self.autotune_reset_session_checkbox, 1, 0)
        session_grid.setColumnStretch(1, 1)
        autotune_layout.addLayout(session_grid)

        self.autotune_grid_preview_label = QLabel("")
        self.autotune_grid_preview_label.setWordWrap(True)
        autotune_layout.addWidget(self.autotune_grid_preview_label)

        self.run_autotune_btn = QPushButton("🔁 Run Auto-tune")
        self.run_autotune_btn.setToolTip("Run the autotune grid on the selected context lane")
        self.run_autotune_btn.clicked.connect(self.run_autotune)
        autotune_layout.addWidget(self.run_autotune_btn)
        autotune_layout.addStretch(1)

        self.mode_tabs.addTab(autotune_page, "🔁 Auto-tune")
        left_layout.addWidget(self.mode_tabs)

        for combo, minimum_contents_length in [
            (self.model_combo, 18),
            (self.build_backend_combo, 10),
            (self.build_version_combo, 18),
            (self.device_combo, 18),
            (self.tasks_combo, 8),
            (self.spec_combo, 10),
            (self.kv_combo, 8),
            (self.lane_combo, 18),
            (self.validate_ctx_combo, 6),
        ]:
            self._configure_combo(combo, minimum_contents_length)

        for spin_box in [
            self.runs_spin,
            self.ctx_spin,
            self.batch_spin,
            self.ubatch_spin,
            self.max_tokens_spin,
            self.at_batch_min_spin,
            self.at_batch_max_spin,
            self.at_batch_step_spin,
            self.at_ubatch_min_spin,
            self.at_ubatch_max_spin,
            self.at_ubatch_step_spin,
            self.lane_custom_ctx_spin,
            self.mtp_draft_spin,
        ]:
            self._configure_spinbox(spin_box)

        for spin_box in [
            self.at_batch_min_spin,
            self.at_batch_max_spin,
            self.at_batch_step_spin,
            self.at_ubatch_min_spin,
            self.at_ubatch_max_spin,
            self.at_ubatch_step_spin,
        ]:
            spin_box.valueChanged.connect(self._update_autotune_grid_preview)
            spin_box.valueChanged.connect(self.save_settings)

        for checkbox in list(self.autotune_kv_checks.values()) + list(self.autotune_spec_checks.values()) + list(self.autotune_extra_checks.values()):
            checkbox.toggled.connect(self._update_autotune_grid_preview)
            checkbox.toggled.connect(self.save_settings)
        self.autotune_custom_extra_input.textChanged.connect(self._update_autotune_grid_preview)
        self.autotune_custom_extra_input.textChanged.connect(self.save_settings)
        self.autotune_resume_checkbox.toggled.connect(self.save_settings)
        self.autotune_reset_session_checkbox.toggled.connect(self.save_settings)
        self.mode_tabs.currentChanged.connect(lambda _index: self.save_settings())

        self.lane_combo.currentIndexChanged.connect(self._on_lane_changed)
        self.lane_custom_ctx_spin.valueChanged.connect(self._update_autotune_grid_preview)
        self.lane_custom_ctx_spin.valueChanged.connect(self.save_settings)
        self.autotune_device_sweep_check.toggled.connect(self._update_autotune_grid_preview)
        self.autotune_device_sweep_check.toggled.connect(self.save_settings)
        self.device_combo.currentIndexChanged.connect(lambda _index: self.save_settings())
        self.scale_prompt_check.toggled.connect(self.save_settings)
        self.mtp_draft_spin.valueChanged.connect(self.save_settings)
        self.at_mtp_draft_input.textChanged.connect(self._update_autotune_grid_preview)
        self.at_mtp_draft_input.textChanged.connect(self.save_settings)
        self._update_autotune_grid_preview()

        shared_btn_row = QHBoxLayout()
        shared_btn_row.setSpacing(8)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setToolTip("Stop the current benchmark or autotune run")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_current_run)
        shared_btn_row.addWidget(self.stop_btn, 1)

        self.stop_all_servers_btn = QPushButton("Stop All Servers")
        self.stop_all_servers_btn.setToolTip(
            "Soft-stop leftover llama-server processes with CTRL_BREAK/taskkill without /F"
        )
        self.stop_all_servers_btn.clicked.connect(self.stop_all_llama_servers)
        shared_btn_row.addWidget(self.stop_all_servers_btn, 1)

        self.open_history_btn = QPushButton("Open History")
        self.open_history_btn.setToolTip("Open build_logs/agent-workload/BENCH_HISTORY.md")
        self.open_history_btn.clicked.connect(self.open_history_md)
        shared_btn_row.addWidget(self.open_history_btn, 1)
        left_layout.addLayout(shared_btn_row)

        status_row = QHBoxLayout()
        self.status_label = StatusPill("● Ready")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        left_layout.addLayout(status_row)
        left_layout.addStretch(1)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout()
        self.log_output = LogView()
        self.log_output.setMinimumHeight(150)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, 2)

        presets_group = QGroupBox("Autotune Runs History (Best Result Per Run)")
        presets_layout = QVBoxLayout()

        history_filter_row = QHBoxLayout()
        history_filter_row.addWidget(QLabel("Spec:"))
        self.history_spec_filter_combo = QComboBox()
        self.history_spec_filter_combo.addItems(["All", "MTP only", "Non-MTP"])
        self.history_spec_filter_combo.setToolTip("Show only runs whose best spec was MTP-based (mtp/ngram-mtp) or not")
        history_filter_row.addWidget(self.history_spec_filter_combo)

        history_filter_row.addWidget(QLabel("Backend:"))
        self.history_backend_filter_combo = QComboBox()
        self.history_backend_filter_combo.addItem("All backends")
        self.history_backend_filter_combo.setToolTip("Filter autotune history by backend")
        history_filter_row.addWidget(self.history_backend_filter_combo)

        history_filter_row.addWidget(QLabel("Lane:"))
        self.history_lane_filter_combo = QComboBox()
        self.history_lane_filter_combo.addItem("All lanes")
        self.history_lane_filter_combo.setToolTip("Filter runs by context lane (ctx)")
        history_filter_row.addWidget(self.history_lane_filter_combo)

        history_legend = QLabel("■ best per backend / lane / spec")
        history_legend.setStyleSheet("color: #4CAF50;")
        history_filter_row.addWidget(history_legend)
        history_filter_row.addStretch()
        presets_layout.addLayout(history_filter_row)

        self.history_spec_filter_combo.currentIndexChanged.connect(self._on_history_filter_changed)
        self.history_backend_filter_combo.currentIndexChanged.connect(self._on_history_filter_changed)
        self.history_lane_filter_combo.currentIndexChanged.connect(self._on_history_filter_changed)

        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(18)
        self.presets_table.setHorizontalHeaderLabels([
            "Backend",
            "Run Time",
            "Model",
            "Best TPS",
            "Prompt TPS",
            "Decode TPS",
            "Ctx",
            "Batch/UBatch",
            "KV",
            "Best Spec",
            "Draft N",
            "Best Extra",
            "Extra args",
            "Swept Specs",
            "Swept Extras",
            "Build ID",
            "Run ID",
            "Label",
        ])
        self.presets_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.presets_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.presets_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.presets_table.setMinimumHeight(190)
        self._configure_compact_table(
            self.presets_table,
            [76, 140, 180, 82, 84, 84, 70, 105, 72, 96, 66, 110, 180, 120, 120, 120, 150, 160],
        )
        presets_layout.addWidget(self.presets_table)

        presets_actions = QHBoxLayout()
        presets_actions.setSpacing(8)
        self.apply_history_preset_btn = QPushButton("Apply Default")
        self.apply_history_preset_btn.setToolTip(
            "Apply the selected run as the default model preset (double-click a row does the same)"
        )
        self.apply_history_preset_btn.clicked.connect(self.apply_selected_run_as_default_preset)
        presets_actions.addWidget(self.apply_history_preset_btn)

        self.refresh_history_btn = QPushButton("Refresh")
        self.refresh_history_btn.setToolTip("Refresh autotune run history")
        self.refresh_history_btn.clicked.connect(self.refresh_saved_presets_table)
        presets_actions.addWidget(self.refresh_history_btn)

        history_hint = QLabel("Right-click a row: open/copy log, copy row, delete")
        history_hint.setStyleSheet("color: #7f8a97;")
        presets_actions.addWidget(history_hint)
        presets_actions.addStretch()
        presets_layout.addLayout(presets_actions)

        # row-level operations live in the context menu; keep QAction refs so
        # _set_running_state can disable them during a run
        self.open_history_log_action = QAction("Open Log", self)
        self.open_history_log_action.triggered.connect(self.open_selected_history_log)
        self.copy_history_log_action = QAction("Copy Log", self)
        self.copy_history_log_action.triggered.connect(self.copy_selected_history_log_to_clipboard)
        self.copy_history_row_action = QAction("Copy Row", self)
        self.copy_history_row_action.triggered.connect(self.copy_selected_history_row_to_clipboard)
        self.delete_history_run_action = QAction("Delete Run", self)
        self.delete_history_run_action.triggered.connect(self.delete_selected_preset)

        self.presets_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.presets_table.customContextMenuRequested.connect(self._show_history_context_menu)
        self.presets_table.itemDoubleClicked.connect(
            lambda _item: self.apply_selected_run_as_default_preset()
        )

        presets_group.setLayout(presets_layout)
        right_layout.addWidget(presets_group, 3)

        history_group = QGroupBox("Current Autotune Run History (Live)")
        history_layout = QVBoxLayout()
        self.autotune_history_table = QTableWidget()
        self.autotune_history_table.setColumnCount(12)
        self.autotune_history_table.setHorizontalHeaderLabels([
            "Run",
            "Ctx",
            "Batch",
            "UBatch",
            "KV",
            "Spec",
            "Draft N",
            "Extra",
            "Agg TPS",
            "Prompt TPS",
            "Decode TPS",
            "Status",
        ])
        self.autotune_history_table.setSortingEnabled(True)
        self.autotune_history_table.setMinimumHeight(190)
        self._configure_compact_table(
            self.autotune_history_table,
            [62, 70, 76, 76, 70, 96, 62, 120, 70, 84, 84, 110],
        )
        history_layout.addWidget(self.autotune_history_table)
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group, 4)

    _display_backend_from_key = staticmethod(display_backend_from_key)
    _backend_key_from_display = staticmethod(backend_key_from_display)

    def refresh_build_choices(self):
        previous_backend = self.build_backend_combo.currentText() if self.build_backend_combo.count() else "ROCm/HIP"
        self.build_backend_combo.blockSignals(True)
        self.build_backend_combo.clear()

        records = self.parent.get_registered_builds() if hasattr(self.parent, "get_registered_builds") else []
        keys = sorted({str(r.get("backend", "")).lower() for r in records if str(r.get("backend", "")).strip()})

        for key in keys:
            self.build_backend_combo.addItem(self._display_backend_from_key(key))

        for legacy in ["ROCm/HIP", "CPU", "Vulkan"]:
            if self.build_backend_combo.findText(legacy) < 0:
                self.build_backend_combo.addItem(legacy)

        idx = self.build_backend_combo.findText(previous_backend)
        if idx < 0:
            idx = self.build_backend_combo.findText("ROCm/HIP")
        self.build_backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.build_backend_combo.blockSignals(False)
        self.refresh_versions_for_backend(select_latest=True)
        self._refresh_device_choices()

    def _on_backend_changed(self, *_args):
        self.refresh_versions_for_backend(select_latest=True)
        self._refresh_device_choices()

    def refresh_versions_for_backend(self, select_latest: bool):
        backend_display = self.build_backend_combo.currentText().strip()
        backend_key = self._backend_key_from_display(backend_display)

        prev_id = ""
        prev_payload = self._version_payloads.get(self.build_version_combo.currentText().strip())
        if isinstance(prev_payload, dict):
            prev_id = str(prev_payload.get("build_id", ""))

        self.build_version_combo.clear()
        self._version_payloads = {}

        records = self.parent.get_registered_builds() if hasattr(self.parent, "get_registered_builds") else []
        registry = getattr(self.parent, "build_registry", None)
        usable = [
            r for r in records
            if str(r.get("status", "")) == "ready"
            and str(r.get("backend", "")).lower() == backend_key
            and str(r.get("build_dir", "")).strip()
            and Path(str(r.get("build_dir", "")).strip()).exists()
        ]
        usable.sort(
            key=lambda r: (
                registry.get_effective_build_timestamp(r) if registry is not None else str(r.get("created_at", "") or r.get("updated_at", "")),
                str(r.get("name", "")).lower(),
            ),
            reverse=True,
        )

        for rec in usable:
            source = str(rec.get("source_type", "fork"))
            source_ref = str(rec.get("source_ref", ""))
            short_ref = source_ref[:10] if source_ref else "-"
            build_id = str(rec.get("id", ""))
            short_id = build_id[-8:] if len(build_id) >= 8 else (build_id or "-")
            build_date = registry.get_effective_build_timestamp(rec) if registry is not None else (str(rec.get("created_at", "")).strip() or str(rec.get("updated_at", "")).strip() or "-")
            label = (
                f"{rec.get('name', Path(str(rec.get('build_dir'))).name)} "
                f"[{source}/{short_ref}] | id:{short_id} | built:{build_date}"
            )
            self.build_version_combo.addItem(label)
            self._version_payloads[label] = {
                "build_id": str(rec.get("id", "")),
                "build_dir": Path(str(rec.get("build_dir", ""))),
                "server_bin": str(rec.get("server_bin", "")),
            }

        if self.build_version_combo.count() == 0:
            self.build_version_combo.addItem("Auto")
            return

        if select_latest:
            self.build_version_combo.setCurrentIndex(0)
            return

        if prev_id:
            for idx in range(self.build_version_combo.count()):
                label = self.build_version_combo.itemText(idx)
                payload = self._version_payloads.get(label)
                if isinstance(payload, dict) and str(payload.get("build_id", "")) == prev_id:
                    self.build_version_combo.setCurrentIndex(idx)
                    return

        self.build_version_combo.setCurrentIndex(0)

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            str(self.models_dir),
            "GGUF Files (*.gguf);;All Files (*.*)",
        )
        if file_path:
            self.model_path_input.setText(file_path)

    def refresh_models_list(self):
        self.model_combo.clear()
        self.model_combo.addItem("-- Select Model --")
        if not self.models_dir.exists():
            self.model_combo.addItem("-- No .gguf models found --")
            return

        model_files = sorted(
            [p.name for p in self.models_dir.glob("*.gguf") if "mmproj" not in p.name.lower()],
            key=str.lower,
        )
        if not model_files:
            self.model_combo.addItem("-- No .gguf models found --")
            return

        self.model_combo.addItems(model_files)

    def on_model_selected(self, model_name: str):
        if model_name and not model_name.startswith("--"):
            self.model_path_input.setText(str(self.models_dir / model_name))
            self._last_selected_model = model_name

    def _resolve_selected_model(self) -> Path | None:
        model_path = self.model_path_input.text().strip()
        if model_path and Path(model_path).exists():
            return Path(model_path)

        if self._last_selected_model:
            fallback = self.models_dir / self._last_selected_model
            if fallback.exists():
                return fallback

        return None

    def _best_known_config_for_model(self, model_name: str) -> dict[str, str] | None:
        """Best recorded config for a model: saved best-preset first, then history."""
        presets = self._load_best_presets()
        candidates = [
            record for key, record in presets.items()
            if key == model_name or key.startswith(f"{model_name}::")
        ]
        if candidates:
            def tps_of(record: dict) -> float:
                try:
                    return float(record.get("best_tps", "0") or "0")
                except ValueError:
                    return 0.0
            return max(candidates, key=tps_of)

        best_row = None
        best_tps = -1.0
        for row in self._load_autotune_history_rows():
            if Path(str(row.get("model", ""))).name.lower() != model_name.lower():
                continue
            try:
                tps = float(str(row.get("aggregate_tps", "0") or "0"))
            except ValueError:
                continue
            if tps > best_tps:
                best_tps = tps
                best_row = row
        if best_row is None:
            return None

        parsed = self._parse_best_config_text(str(best_row.get("best_config", "")))
        return {
            "best_tps": f"{best_tps:.4f}",
            "ctx": parsed["ctx"] if parsed["ctx"] != "-" else str(best_row.get("ctx", "")),
            "batch": parsed["batch"] if parsed["batch"] != "-" else str(best_row.get("batch", "")),
            "ubatch": parsed["ubatch"] if parsed["ubatch"] != "-" else str(best_row.get("ubatch", "")),
            "kv_k": parsed["kv"] if parsed["kv"] != "-" else str(best_row.get("kv_k", "")),
            "spec_mode": parsed["spec"] if parsed["spec"] != "-" else str(best_row.get("spec_mode", "")),
        }

    def apply_best_known_config(self) -> None:
        """Fill the Single Bench parameters from the model's best autotune result."""
        model = self._resolve_selected_model()
        if model is None:
            self.status_label.setText("Select a model first to apply its best known config")
            return

        record = self._best_known_config_for_model(model.name)
        if record is None:
            self.status_label.setText(f"No autotune history found for {model.name}")
            return

        applied: list[str] = []
        for field, spin in (("ctx", self.ctx_spin), ("batch", self.batch_spin), ("ubatch", self.ubatch_spin)):
            raw = str(record.get(field, "")).strip()
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            spin.setValue(value)
            applied.append(f"{field}={spin.value()}")

        kv = str(record.get("kv_k", "")).strip().lower()
        if kv and self.kv_combo.findText(kv) >= 0:
            self.kv_combo.setCurrentText(kv)
            applied.append(f"kv={kv}")

        spec = str(record.get("spec_mode", "")).strip().lower()
        if spec and spec not in {"-", "mixed"} and self.spec_combo.findText(spec) >= 0:
            self.spec_combo.setCurrentText(spec)
            applied.append(f"spec={spec}")

        tps = str(record.get("best_tps", "")).strip()
        summary = ", ".join(applied) if applied else "nothing applicable"
        self.status_label.setText(f"Applied best known ({tps} TPS): {summary}")
        self.log_output.append(f"[INFO] Best known config for {model.name}: {summary} (recorded {tps} TPS)")

    def _resolve_selected_server(self) -> tuple[Path | None, str]:
        label = self.build_version_combo.currentText().strip()
        payload = self._version_payloads.get(label)
        if isinstance(payload, dict):
            server_bin = str(payload.get("server_bin", "")).strip()
            if server_bin and Path(server_bin).exists():
                return Path(server_bin), str(payload.get("build_id", ""))
            build_dir = payload.get("build_dir")
            if isinstance(build_dir, Path):
                candidates = [
                    build_dir / "bin" / "llama-server.exe",
                    build_dir / "bin" / "Release" / "llama-server.exe",
                    build_dir / "bin" / "Debug" / "llama-server.exe",
                    build_dir / "bin" / "llama-server",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate, str(payload.get("build_id", ""))

        selected_backend = self.build_backend_combo.currentText().strip()
        if hasattr(self.parent, "get_build_dir_for_backend"):
            fallback_dir = Path(self.parent.get_build_dir_for_backend(selected_backend))
            candidates = [
                fallback_dir / "bin" / "llama-server.exe",
                fallback_dir / "bin" / "Release" / "llama-server.exe",
                fallback_dir / "bin" / "Debug" / "llama-server.exe",
                fallback_dir / "bin" / "llama-server",
            ]
            for candidate in candidates:
                if candidate.exists():
                    build_id = ""
                    registry = getattr(self.parent, "build_registry", None)
                    if registry is not None:
                        build_id = registry.detect_build_id_from_server_bin(str(candidate))
                    return candidate, build_id

        return None, ""

    def run_benchmark(self):
        if self.bench_thread and self.bench_thread.isRunning():
            QMessageBox.information(self, "Benchmark", "Benchmark/autotune is already running")
            return

        model = self._resolve_selected_model()
        if model is None:
            QMessageBox.warning(self, "Benchmark", "Select valid model first")
            return

        server_bin, build_id = self._resolve_selected_server()
        if server_bin is None:
            QMessageBox.warning(self, "Benchmark", "llama-server not found for selected build version")
            return

        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        label = f"gui-bench-{model.stem}-{stamp}"
        spec_mode = self.spec_combo.currentText()
        spec_fallback_note = ""
        if spec_mode in {"mtp", "ngram-mtp"} and not model_supports_mtp(model):
            spec_fallback_note = (
                f"Selected spec mode '{spec_mode}' is incompatible with {model.name}; "
                "using 'none'."
            )
            spec_mode = "none"
        server_extra = self._active_lane_base_server_extra(self.ctx_spin.value())
        device_args = self._selected_device_args()
        if device_args:
            server_extra.extend(device_args)
        spec_cli_mode = "draft-mtp" if spec_mode == "mtp" else spec_mode
        spec_extra = [f"--spec-type {spec_cli_mode}"]
        if spec_mode in {"ngram-mod", "ngram-mtp"}:
            spec_extra.append(f"--spec-ngram-mod-n-min {self.NGRAM_MOD_N_MIN}")
            spec_extra.append(f"--spec-ngram-mod-n-match {self.NGRAM_MOD_N_MATCH}")
            spec_extra.append(f"--spec-ngram-mod-n-max {self.NGRAM_MOD_N_MAX}")
        if spec_mode in {"mtp", "ngram-mtp"}:
            spec_extra.append(f"--spec-draft-n-max {self.mtp_draft_spin.value()}")
        server_extra.extend(spec_extra)

        request_timeout = self._request_timeout_for_ctx(self.ctx_spin.value())

        command = [
            console_python_executable(),
            "scripts/agent_workload_bench.py",
            "--label",
            label,
            "--tasks",
            self.tasks_combo.currentText(),
            "--runs",
            str(self.runs_spin.value()),
            "--server-bin",
            str(server_bin),
            "--model",
            str(model),
            "--build-id",
            build_id,
            "--artifact-mode",
            "unified",
            "--ctx-size",
            str(self.ctx_spin.value()),
            "--batch-size",
            str(self.batch_spin.value()),
            "--ubatch-size",
            str(self.ubatch_spin.value()),
            "--cache-type-k",
            self.kv_combo.currentText(),
            "--cache-type-v",
            self.kv_combo.currentText(),
            "--gpu-layers",
            "999",
            "--parallel",
            "1",
            "--max-tokens",
            str(self.max_tokens_spin.value()),
            "--startup-timeout",
            "900",
            "--request-timeout",
            str(request_timeout),
            "--task-hard-timeout",
            "0",
            "--background-server-policy",
            "fail",
            "--real-context-mode",
            "repo-snapshot",
            "--real-context-chars",
            str(self.ctx_spin.value() * 3 if self.scale_prompt_check.isChecked() else 24576),
            "--real-context-safe-fill",
            f"{self.REAL_CONTEXT_SAFE_FILL:g}",
            "--real-context-reserve-tokens",
            str(self.REAL_CONTEXT_RESERVE_TOKENS),
            "--real-context-chars-per-token",
            str(self.REAL_CONTEXT_CHARS_PER_TOKEN),
            "--no-reuse",
            "--no-v2-prime-pass",
            "--no-disable-thinking",
            self._server_extra_arg(" ".join(server_extra)),
        ]

        if self.tasks_combo.currentText() == "quick":
            command.extend(["--task-ids", "triage_diff"])

        if self.ctx_spin.value() > 16384 and self._bench_supports_flag("--allow-ctx-above-16k"):
            command.append("--allow-ctx-above-16k")

        bench_env = self._bench_env_overrides()

        self._current_mode = "single"
        self._last_selected_model = model.name
        self._current_build_id = build_id
        self._set_running_state(True)
        self.log_output.clear()
        self.log_output.append(f"[INFO] Starting benchmark for {model.name}")
        self.log_output.append(f"[INFO] Build ID: {build_id or '-'}")
        if spec_fallback_note:
            self.log_output.append(f"[WARN] {spec_fallback_note}")
        if bench_env:
            env_summary = ", ".join(f"{key}={value}" for key, value in sorted(bench_env.items()))
            self.log_output.append(f"[INFO] Env overrides: {env_summary}")
        if server_extra:
            self.log_output.append(f"[INFO] Server extra: {' '.join(server_extra)}")
        self.bench_thread = BenchCommandThread(command=command, working_dir=self.project_root, env=bench_env)
        self.bench_thread.output.connect(self._on_bench_output)
        self.bench_thread.finished_signal.connect(self._on_bench_finished)
        self.bench_thread.start()

    def run_autotune(self):
        if self.bench_thread and self.bench_thread.isRunning():
            QMessageBox.information(self, "Auto-tune", "Benchmark/autotune is already running")
            return

        model = self._resolve_selected_model()
        if model is None:
            QMessageBox.warning(self, "Auto-tune", "Select valid model first")
            return

        server_bin, build_id = self._resolve_selected_server()
        if server_bin is None:
            QMessageBox.warning(self, "Auto-tune", "llama-server not found for selected build version")
            return

        lane_ctx, lane_chars = self._selected_lane()
        profile_key = f"ctx{lane_ctx // 1024}k"
        autotune_min_ctx = lane_ctx
        autotune_ctx_values = str(lane_ctx)
        batch_values = self._build_autotune_range_values(
            self.at_batch_min_spin.value(),
            self.at_batch_max_spin.value(),
            self.at_batch_step_spin.value(),
        )
        ubatch_values = self._build_autotune_range_values(
            self.at_ubatch_min_spin.value(),
            self.at_ubatch_max_spin.value(),
            self.at_ubatch_step_spin.value(),
        )
        if not batch_values:
            QMessageBox.warning(self, "Auto-tune", "Invalid AT Batch range. Check min/max/step.")
            return
        if not ubatch_values:
            QMessageBox.warning(self, "Auto-tune", "Invalid AT UBatch range. Check min/max/step.")
            return

        autotune_batch_values = ",".join(str(v) for v in batch_values)
        autotune_ubatch_values = ",".join(str(v) for v in ubatch_values)

        kv_values = self._selected_autotune_kv_values()
        if not kv_values:
            QMessageBox.warning(self, "Auto-tune", "Select at least one KV type for autotune sweep.")
            return

        requested_spec_values = self._selected_autotune_spec_values()
        if not requested_spec_values:
            QMessageBox.warning(self, "Auto-tune", "Select at least one Spec mode for autotune sweep.")
            return

        spec_values = self._resolve_autotune_spec_values(server_bin, model, ",".join(requested_spec_values))
        skipped_spec_values = [mode for mode in requested_spec_values if mode not in spec_values]
        if not spec_values:
            QMessageBox.warning(
                self,
                "Auto-tune",
                "No valid spec modes resolved for autotune.\n"
                f"Selected: {','.join(requested_spec_values)}\n"
                "Check llama-server --help and model compatibility.",
            )
            return

        extra_presets = self._selected_autotune_extra_presets()
        if not extra_presets:
            QMessageBox.warning(self, "Auto-tune", "No valid extra presets resolved for autotune.")
            return
        extra_presets = self._apply_draft_sweep_to_extra_presets(extra_presets)
        extra_presets = self._apply_device_sweep_to_extra_presets(extra_presets)

        autotune_extra_presets = "||".join(extra_presets)
        autotune_kv_values = ",".join(kv_values)
        autotune_tasks = "quick"
        autotune_task_ids = "triage_diff"
        # Sixteen-token runs are dominated by the first backend graph and make
        # speculative and non-speculative decode rates incomparable.
        autotune_max_tokens = "128"
        autotune_real_context_chars = str(lane_chars)

        ctx_count = len([v for v in autotune_ctx_values.split(",") if v.strip()])
        batch_count = len(batch_values)
        ubatch_count = len(ubatch_values)
        kv_count = len([v for v in autotune_kv_values.split(",") if v.strip()])
        spec_count = len(spec_values)
        extra_count = len(extra_presets)
        config_count = ctx_count * batch_count * ubatch_count * kv_count * spec_count * extra_count

        run_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        label = f"gui-autotune-{model.stem}-{run_stamp}"
        # session checkpoint is per model AND lane: resuming a 12K session into
        # a 100K run would mix incompatible config lists
        session_label = f"gui-autotune-{model.stem}-{profile_key}"
        autotune_session_file = self.project_root / "build_logs" / "agent-workload" / f"{session_label}-autotune-session.json"
        compatibility_notes: list[str] = []
        base_server_extra = self._active_lane_base_server_extra(autotune_min_ctx)
        # global device selection applies unless the sweep provides per-config devices
        if not self.autotune_device_sweep_check.isChecked():
            base_server_extra = base_server_extra + self._selected_device_args()

        request_timeout = self._request_timeout_for_ctx(autotune_min_ctx)

        command = [
            console_python_executable(),
            "scripts/agent_workload_bench.py",
            "--autotune",
            "--label",
            label,
            "--tasks",
            autotune_tasks,
            "--runs",
            "1",
            "--server-bin",
            str(server_bin),
            "--model",
            str(model),
            "--build-id",
            build_id,
            "--artifact-mode",
            "unified",
            "--gpu-layers",
            "-1",
            "--parallel",
            "1",
            "--max-tokens",
            autotune_max_tokens,
            "--startup-timeout",
            "900",
            "--request-timeout",
            str(request_timeout),
            "--task-hard-timeout",
            "0",
            "--background-server-policy",
            "fail",
            "--allow-ctx-above-16k",
            "--real-context-mode",
            "repo-snapshot",
            "--real-context-chars",
            autotune_real_context_chars,
            "--real-context-safe-fill",
            f"{self.REAL_CONTEXT_SAFE_FILL:g}",
            "--real-context-reserve-tokens",
            str(self.REAL_CONTEXT_RESERVE_TOKENS),
            "--real-context-chars-per-token",
            str(self.REAL_CONTEXT_CHARS_PER_TOKEN),
            "--no-reuse",
            "--no-v2-prime-pass",
            "--no-disable-thinking",
            "--autotune-min-ctx",
            str(autotune_min_ctx),
            "--autotune-ctx-values",
            autotune_ctx_values,
            "--autotune-batch-values",
            autotune_batch_values,
            "--autotune-ubatch-values",
            autotune_ubatch_values,
            "--autotune-kv-values",
            autotune_kv_values,
            "--autotune-spec-values",
            ",".join(spec_values),
            "--autotune-ngram-min",
            str(self.NGRAM_MOD_N_MIN),
            "--autotune-ngram-match",
            str(self.NGRAM_MOD_N_MATCH),
            "--autotune-ngram-max",
            str(self.NGRAM_MOD_N_MAX),
            "--autotune-mtp-draft-n-max",
            str(self._selected_mtp_draft_values()[0]),
            "--autotune-max-configs",
            str(max(64, config_count + 8)),
            "--autotune-update-preset",
            "--autotune-preset-file",
            "gui/model_presets.json",
        ]

        if base_server_extra:
            command.append(self._server_extra_arg(" ".join(base_server_extra)))

        if autotune_task_ids:
            command.extend(["--task-ids", autotune_task_ids])

        if self._bench_supports_flag("--task-fail-timeout"):
            command.extend(["--task-fail-timeout", "0"])
        else:
            compatibility_notes.append("--task-fail-timeout not supported by this benchmark script; using timeout policy: off")

        if self._bench_supports_flag("--autotune-extra-presets"):
            command.extend(["--autotune-extra-presets", autotune_extra_presets])
        elif any(item.lower() != "base" for item in extra_presets):
            compatibility_notes.append("--autotune-extra-presets not supported; running without extra preset sweep")

        if self._bench_supports_flag("--autotune-session-file"):
            command.extend(["--autotune-session-file", str(autotune_session_file)])
        else:
            compatibility_notes.append(
                "--autotune-session-file not supported; session checkpoint path will use script default"
            )

        if self._bench_supports_flag("--autotune-resume"):
            if self.autotune_resume_checkbox.isChecked():
                command.append("--autotune-resume")
            else:
                command.append("--no-autotune-resume")
        else:
            compatibility_notes.append(
                "--autotune-resume not supported; benchmark script default resume behavior will be used"
            )

        if self.autotune_reset_session_checkbox.isChecked():
            if self._bench_supports_flag("--autotune-reset-session"):
                command.append("--autotune-reset-session")
            else:
                compatibility_notes.append(
                    "--autotune-reset-session requested but not supported by benchmark script"
                )

        bench_env = self._bench_env_overrides()

        self._current_mode = "autotune"
        self._current_autotune_profile = profile_key
        self._current_build_id = build_id
        self._autotune_result = {
            "best": "",
            "summary_json": "",
            "summary_csv": "",
            "label": label,
            "model": model.name,
            "build_id": build_id,
            "started_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._reset_autotune_live_history()
        self._last_selected_model = model.name
        self._set_running_state(True)
        self.log_output.clear()
        self.log_output.append(f"[INFO] Starting autotune for {model.name}")
        self.log_output.append(f"[INFO] Build ID: {build_id or '-'}")
        self.log_output.append(
            f"[INFO] Lane: ctx={lane_ctx}, prompt chars={lane_chars}, configs: {config_count}"
        )
        if self.autotune_device_sweep_check.isChecked() and self._device_sweep_presets():
            self.log_output.append("[INFO] Device sweep: both dual orders plus both single GPUs")
        if bench_env:
            env_summary = ", ".join(f"{key}={value}" for key, value in sorted(bench_env.items()))
            self.log_output.append(f"[INFO] Env overrides: {env_summary}")
        if base_server_extra:
            self.log_output.append(f"[INFO] Base server extra: {' '.join(base_server_extra)}")
        self.log_output.append(f"[INFO] Workload: {autotune_tasks}, max_tokens: {autotune_max_tokens}")
        self.log_output.append(f"[INFO] Spec selected: {','.join(requested_spec_values)}")
        if skipped_spec_values:
            self.log_output.append(
                "[INFO] Spec skipped (unsupported by server/model): "
                + ",".join(skipped_spec_values)
            )
        self.log_output.append(f"[INFO] Spec effective: {','.join(spec_values)}")
        self.log_output.append(
            "[INFO] Lane: prompt-heavy repo-snapshot "
            f"(chars={autotune_real_context_chars}), no-reuse, no-prime, thinking on"
        )
        self.log_output.append(
            "[INFO] Sweep: "
            f"batch={autotune_batch_values} | "
            f"ubatch={autotune_ubatch_values} | "
            f"kv={autotune_kv_values} | "
            f"spec={','.join(spec_values)} | "
            f"extra={','.join(extra_presets)}"
        )
        self.log_output.append(f"[INFO] Run label: {label}")
        self.log_output.append(f"[INFO] Session file: {autotune_session_file}")
        self.log_output.append(
            "[INFO] Session mode: "
            f"resume={'on' if self.autotune_resume_checkbox.isChecked() else 'off'}, "
            f"reset={'on' if self.autotune_reset_session_checkbox.isChecked() else 'off'}"
        )

        self.log_output.append(
            f"[INFO] Task timeout policy: request {request_timeout}s, hard timeout off, fail timeout off"
        )

        for note in compatibility_notes:
            self.log_output.append(f"[INFO] Compatibility: {note}")

        self.bench_thread = BenchCommandThread(command=command, working_dir=self.project_root, env=bench_env)
        self.bench_thread.output.connect(self._on_bench_output)
        self.bench_thread.finished_signal.connect(self._on_bench_finished)
        self.bench_thread.start()

    def stop_current_run(self):
        if not self.bench_thread or not self.bench_thread.isRunning():
            QMessageBox.information(self, "Stop", "No benchmark/autotune is running")
            return

        answer = QMessageBox.question(
            self,
            "Stop",
            "Stop current benchmark/autotune now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.status_label.setText("Stopping benchmark/autotune...")
        self.log_output.append("[INFO] Stop requested by user. Terminating benchmark process...")
        self.bench_thread.request_stop()

    def stop_all_llama_servers(self):
        if self.stop_all_thread and self.stop_all_thread.isRunning():
            QMessageBox.information(self, "Stop All Servers", "A soft cleanup is already running")
            return

        answer = QMessageBox.question(
            self,
            "Stop All Servers",
            "Soft-stop all leftover llama-server processes now?\n\n"
            "This sends CTRL_BREAK/SIGTERM and taskkill without /F. It will not force-kill GPU work.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        timeout = 90.0
        try:
            timeout = float(os.environ.get("LLAMA_GUI_STOP_ALL_TIMEOUT", "90"))
        except ValueError:
            timeout = 90.0

        self.status_label.setText("Soft-stopping leftover llama-server processes...")
        self.log_output.append("[INFO] Stop All Servers requested. No force-kill will be used.")
        self.stop_all_servers_btn.setEnabled(False)
        self.stop_all_thread = LlamaServerStopThread(wait_seconds=timeout)
        self.stop_all_thread.output.connect(self._on_stop_all_servers_output)
        self.stop_all_thread.finished_signal.connect(self._on_stop_all_servers_finished)
        self.stop_all_thread.start()

    def _on_stop_all_servers_output(self, line: str):
        self.log_output.append(line)

    def _on_stop_all_servers_finished(self, stopped: int, remaining: int):
        self.stop_all_servers_btn.setEnabled(True)
        self.stop_all_thread = None
        if stopped == 0 and remaining == 0:
            self.status_label.setText("No leftover llama-server processes")
            QMessageBox.information(self, "Stop All Servers", "No leftover llama-server processes found.")
            return

        if remaining:
            self.status_label.setText(f"Soft stop incomplete: {remaining} server(s) still running")
            QMessageBox.warning(
                self,
                "Stop All Servers",
                f"Soft-stopped {stopped} server(s), but {remaining} still remain.\n"
                "No /F force-kill was used.",
            )
            return

        self.status_label.setText(f"Soft-stopped {stopped} llama-server process(es)")
        QMessageBox.information(self, "Stop All Servers", f"Soft-stopped {stopped} llama-server process(es).")

    @staticmethod
    def _build_autotune_range_values(min_value: int, max_value: int, step: int) -> list[int]:
        if step <= 0 or min_value <= 0 or max_value <= 0 or min_value > max_value:
            return []

        values: list[int] = []
        current = min_value
        while current <= max_value:
            values.append(current)
            current += step

        if not values:
            return []
        return values

    @staticmethod
    def _format_values_preview(values: list[int], limit: int = 8) -> str:
        if not values:
            return "-"
        if len(values) <= limit:
            return ",".join(str(v) for v in values)
        head = ",".join(str(v) for v in values[:limit])
        return f"{head}, ... ({len(values)} total)"

    def _selected_autotune_kv_values(self) -> list[str]:
        ordered = ["q4_0", "q8_0", "turbo4", "turbo3", "turbo2", "f16", "bf16", "f32"]
        return [name for name in ordered if name in self.autotune_kv_checks and self.autotune_kv_checks[name].isChecked()]

    def _selected_autotune_spec_values(self) -> list[str]:
        ordered = ["none", "ngram-mod", "draft", "eagle3", "mtp", "ngram-mtp"]
        return [name for name in ordered if name in self.autotune_spec_checks and self.autotune_spec_checks[name].isChecked()]

    @staticmethod
    def _parse_custom_extra_presets(values: str) -> list[str]:
        normalized = values.strip()
        if not normalized:
            return []

        chunks = [chunk.strip() for chunk in normalized.split("||") if chunk.strip()]
        parsed: list[str] = []
        seen: set[str] = set()

        for idx, chunk in enumerate(chunks, start=1):
            item = chunk
            if chunk.lower() in {"base", "default", "none", "off", "-"}:
                item = "base"
            elif "::" not in chunk and chunk.startswith("--"):
                item = f"custom-{idx}::{chunk}"

            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            parsed.append(item)

        return parsed

    def _selected_autotune_extra_presets(self) -> list[str]:
        selected: list[str] = []
        for key, checkbox in self.autotune_extra_checks.items():
            if checkbox.isChecked():
                selected.append(self._autotune_extra_presets_map.get(key, key))

        selected.extend(self._parse_custom_extra_presets(self.autotune_custom_extra_input.text()))

        unique: list[str] = []
        seen: set[str] = set()
        for item in selected:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(item)
        return unique

    def _active_lane_base_server_extra(self, ctx_size: int) -> list[str]:
        backend_key = self._backend_key_from_display(self.build_backend_combo.currentText().strip()).lower()
        if backend_key == "vulkan" and ctx_size >= 131072:
            return ["--no-mmap"]
        return []

    # -- device selection -------------------------------------------------------
    def _backend_device_choices(self) -> list[tuple]:
        backend_key = self._backend_key_from_display(self.build_backend_combo.currentText().strip()).lower()
        if backend_key == "rocm":
            return ROCM_DEVICE_CHOICES
        if backend_key == "vulkan":
            return VULKAN_DEVICE_CHOICES
        return []

    def _refresh_device_choices(self) -> None:
        choices: list[tuple[str, list[str]]] = [("Auto — backend default", [])]
        for choice in self._backend_device_choices():
            choices.append((choice[0], device_choice_args(choice)))

        previous = self.device_combo.currentText()
        self._device_choices = choices
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for display, _args in choices:
            self.device_combo.addItem(display)
        idx = self.device_combo.findText(previous)
        self.device_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.device_combo.blockSignals(False)

    def _selected_device_args(self) -> list[str]:
        idx = self.device_combo.currentIndex()
        if 0 <= idx < len(self._device_choices):
            return list(self._device_choices[idx][1])
        return []

    def _device_sweep_presets(self) -> list[tuple[str, str]]:
        """(name, server-args) pairs for the explicit device-placement sweep."""
        backend_key = self._backend_key_from_display(self.build_backend_combo.currentText().strip()).lower()
        if backend_key == "rocm":
            return [
                ("dual-1-0", "-dev ROCm1,ROCm0 -sm layer -ts 1,1"),
                ("dual-0-1", "-dev ROCm0,ROCm1 -sm layer -ts 1,1"),
                ("single-1", "-dev ROCm1 -sm none"),
                ("single-0", "-dev ROCm0 -sm none"),
            ]
        if backend_key == "vulkan":
            return [
                ("dual-0-1", "-dev Vulkan0,Vulkan1 -sm layer -ts 1,1"),
                ("dual-1-0", "-dev Vulkan1,Vulkan0 -sm layer -ts 1,1"),
                ("single-1", "-dev Vulkan1 -sm none"),
                ("single-0", "-dev Vulkan0 -sm none"),
            ]
        return []

    def _selected_mtp_draft_values(self) -> list[int]:
        """Draft-N budgets from the CSV field; invalid chunks are dropped."""
        values: list[int] = []
        for chunk in self.at_mtp_draft_input.text().split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = int(chunk)
            except ValueError:
                continue
            if 1 <= value <= 20 and value not in values:
                values.append(value)
        return values or [self.MTP_DRAFT_N_MAX]

    def _apply_draft_sweep_to_extra_presets(self, extra_presets: list[str]) -> list[str]:
        """Cross-multiply extra presets with draft-N budgets when several are given.

        Relies on the bench script honoring a preset-pinned --spec-draft-n-max
        over the sweep default. Only meaningful for mtp/ngram-mtp specs; other
        spec modes ignore the flag.
        """
        values = self._selected_mtp_draft_values()
        if len(values) <= 1:
            return extra_presets

        combined: list[str] = []
        for item in extra_presets:
            name, sep, args = item.partition("::")
            for value in values:
                combo_name = f"n{value}" if name == "base" else f"{name}+n{value}"
                combo_args = f"{args} --spec-draft-n-max {value}".strip() if sep else f"--spec-draft-n-max {value}"
                combined.append(f"{combo_name}::{combo_args}")
        return combined

    def _apply_device_sweep_to_extra_presets(self, extra_presets: list[str]) -> list[str]:
        """Cross-multiply extra presets with device sweep configs when enabled."""
        if not self.autotune_device_sweep_check.isChecked():
            return extra_presets
        device_presets = self._device_sweep_presets()
        if not device_presets:
            return extra_presets

        combined: list[str] = []
        for item in extra_presets:
            name, sep, args = item.partition("::")
            for device_name, device_args in device_presets:
                combo_name = device_name if name == "base" else f"{name}+{device_name}"
                combo_args = f"{args} {device_args}".strip() if sep else device_args
                combined.append(f"{combo_name}::{combo_args}")
        return combined

    # -- context lanes ----------------------------------------------------------
    def _selected_lane(self) -> tuple[int, int]:
        """(ctx, repo-snapshot chars) of the active autotune lane."""
        idx = self.lane_combo.currentIndex()
        _display, ctx, chars = self.AUTOTUNE_LANES[idx]
        if ctx == 0:  # Custom
            ctx = self.lane_custom_ctx_spin.value()
            chars = 24576 if ctx <= 16384 else ctx * 3
        return ctx, chars

    def _on_history_filter_changed(self, *_args) -> None:
        self.refresh_saved_presets_table()
        self.save_settings()

    def _on_lane_changed(self, *_args) -> None:
        is_custom = self.AUTOTUNE_LANES[self.lane_combo.currentIndex()][1] == 0
        self.lane_custom_ctx_spin.setEnabled(is_custom)
        self._update_autotune_grid_preview()
        self.save_settings()

    def _apply_screen_grid(self) -> None:
        """Stage-1 minimal grid for a fast preset hunt."""
        self.at_batch_min_spin.setValue(512)
        self.at_batch_max_spin.setValue(512)
        self.at_batch_step_spin.setValue(256)
        self.at_ubatch_min_spin.setValue(64)
        self.at_ubatch_max_spin.setValue(256)
        self.at_ubatch_step_spin.setValue(64)
        for name, checkbox in self.autotune_kv_checks.items():
            checkbox.setChecked(name == "q4_0")
        for name, checkbox in self.autotune_spec_checks.items():
            checkbox.setChecked(name == "none")
        for name, checkbox in self.autotune_extra_checks.items():
            checkbox.setChecked(name == "base")
        self.status_label.setText("Screen grid applied: 4 configs (b=512, ub 64–256, q4_0, spec=none)")

    def _apply_full_grid(self) -> None:
        """Wider stage-1 grid when the screen grid is too coarse."""
        self.at_batch_min_spin.setValue(256)
        self.at_batch_max_spin.setValue(1024)
        self.at_batch_step_spin.setValue(256)
        self.at_ubatch_min_spin.setValue(64)
        self.at_ubatch_max_spin.setValue(256)
        self.at_ubatch_step_spin.setValue(64)
        for name, checkbox in self.autotune_kv_checks.items():
            checkbox.setChecked(name in ("q4_0", "q8_0"))
        self.status_label.setText("Full grid applied: batch 256–1024, ubatch 64–256, kv q4_0+q8_0")

    def validate_best_at_long_ctx(self) -> None:
        """Stage 2 of screen→validate: run the best known config at long ctx."""
        model = self._resolve_selected_model()
        if model is None:
            QMessageBox.warning(self, "Validate", "Select a model first")
            return
        record = self._best_known_config_for_model(model.name)
        if record is None:
            QMessageBox.warning(
                self,
                "Validate",
                f"No autotune history for {model.name}.\nRun a screening autotune first (Screen 12K lane).",
            )
            return

        self.apply_best_known_config()
        _display, ctx = self.VALIDATE_CTX_CHOICES[self.validate_ctx_combo.currentIndex()]
        self.ctx_spin.setValue(ctx)
        self.scale_prompt_check.setChecked(True)
        self.log_output.append(
            f"[INFO] Validate stage: best known config at ctx={ctx}, long-prompt mode on"
        )
        self.run_benchmark()

    @staticmethod
    def _vulkan_runtime_env() -> dict[str, str]:
        return {
            "GGML_VK_FORCE_AMD_LARGE_MATMUL": "1",
        }

    def _bench_env_overrides(self) -> dict[str, str]:
        backend_key = self._backend_key_from_display(self.build_backend_combo.currentText().strip()).lower()
        if backend_key == "vulkan":
            return self._vulkan_runtime_env()
        return {}

    def _update_autotune_grid_preview(self) -> None:
        batch_values = self._build_autotune_range_values(
            self.at_batch_min_spin.value(),
            self.at_batch_max_spin.value(),
            self.at_batch_step_spin.value(),
        )
        ubatch_values = self._build_autotune_range_values(
            self.at_ubatch_min_spin.value(),
            self.at_ubatch_max_spin.value(),
            self.at_ubatch_step_spin.value(),
        )

        kv_values = self._selected_autotune_kv_values()
        selected_spec_values = self._selected_autotune_spec_values()
        extra_presets = self._apply_device_sweep_to_extra_presets(
            self._apply_draft_sweep_to_extra_presets(self._selected_autotune_extra_presets())
        )

        kv_text = ",".join(kv_values) if kv_values else "-"
        spec_text = ",".join(selected_spec_values) if selected_spec_values else "-"

        effective_spec_values = list(selected_spec_values)
        skipped_spec_values: list[str] = []
        has_runtime_context = False

        probing_help = False
        preview_model = self._resolve_selected_model()
        preview_server, _ = self._resolve_selected_server()
        if preview_model is not None and preview_server is not None and selected_spec_values:
            # never run server --help on the GUI thread: use the cache when
            # warm, otherwise probe in the background and refresh on finish
            if self._server_help_cached(preview_server):
                has_runtime_context = True
                effective_spec_values = self._resolve_autotune_spec_values(
                    preview_server,
                    preview_model,
                    ",".join(selected_spec_values),
                )
                skipped_spec_values = [mode for mode in selected_spec_values if mode not in effective_spec_values]
            else:
                probing_help = True
                self._start_server_help_probe(preview_server)

        effective_spec_text = ",".join(effective_spec_values) if effective_spec_values else "-"
        extra_count = len(extra_presets)
        extra_preview = ", ".join(extra_presets[:4])
        if len(extra_presets) > 4:
            extra_preview += f", ... ({len(extra_presets)} total)"

        valid_ranges = bool(batch_values) and bool(ubatch_values)
        valid_modes = bool(kv_values) and bool(effective_spec_values) and bool(extra_presets)
        total_configs = (
            len(batch_values) * len(ubatch_values) * len(kv_values) * len(effective_spec_values) * extra_count
            if valid_ranges and valid_modes
            else 0
        )

        lane_ctx, lane_chars = self._selected_lane()
        # rough per-config cost: server load + prefill (~500 tok/s aggregate)
        usable_prompt_tokens = (
            int(lane_ctx * self.REAL_CONTEXT_SAFE_FILL)
            - 16
            - int(self.REAL_CONTEXT_RESERVE_TOKENS)
        )
        usable_prompt_tokens = max(1024, usable_prompt_tokens)
        safe_chars = int(usable_prompt_tokens * self.REAL_CONTEXT_CHARS_PER_TOKEN)
        effective_chars = safe_chars if lane_chars <= 0 else min(lane_chars, safe_chars)
        prompt_tokens = min(int(effective_chars / self.REAL_CONTEXT_CHARS_PER_TOKEN), usable_prompt_tokens)
        per_config_sec = 75 + prompt_tokens / 500
        total_min = total_configs * per_config_sec / 60.0
        if total_min >= 90:
            eta_text = f"~{total_min / 60.0:.1f} h"
        else:
            eta_text = f"~{max(1, int(round(total_min)))} min"

        self.autotune_mode_info.setText(
            f"Lane: ctx={lane_ctx} · ~{prompt_tokens} prompt tok · quick:triage_diff · runs=1 · max_tokens=128"
        )
        self.autotune_mode_info.setToolTip(
            f"repo-snapshot chars={effective_chars}/{lane_chars} (~{prompt_tokens} prompt tokens)\n"
            "tasks=quick:triage_diff, runs=1, max_tokens=128, no-reuse, no-prime, thinking on."
        )
        lane_short = self.AUTOTUNE_LANES[self.lane_combo.currentIndex()][0].split(" — ")[0]
        if lane_short == "Custom":
            lane_short = f"Custom {lane_ctx}"
        self.run_autotune_btn.setText(f"🔁 Run Auto-tune — {lane_short}")

        detail_lines = [
            f"Batch values: {self._format_values_preview(batch_values)}",
            f"UBatch values: {self._format_values_preview(ubatch_values)}",
            f"KV modes: {kv_text}",
            f"Spec modes (selected): {spec_text}",
            f"Extra presets: {extra_preview or '-'}",
            "Configs = ctx=1 x kv x batch x ubatch x spec x extra",
        ]
        if has_runtime_context:
            detail_lines.insert(4, f"Spec modes (effective): {effective_spec_text}")
            if skipped_spec_values:
                detail_lines.append(
                    "Spec auto-skipped (unsupported by server/model): " + ",".join(skipped_spec_values)
                )
        else:
            detail_lines.append("Spec effective set is resolved at run start (needs model + server).")

        errors: list[str] = []
        if not kv_values:
            errors.append("choose at least one KV mode")
        if not selected_spec_values:
            errors.append("choose at least one Spec mode")
        if not extra_presets:
            errors.append("choose at least one extra preset")
        if not valid_ranges:
            errors.append("check that min <= max and step > 0")

        summary = (
            f"Grid: {total_configs} configs · {eta_text} (~{int(per_config_sec)}s/config)   "
            f"—  {len(batch_values)}b × {len(ubatch_values)}ub × kv {kv_text} × spec {effective_spec_text} × {extra_count} extra"
        )
        lines = [summary]
        if probing_help:
            lines.append("Spec support: probing server --help in background…")
        if skipped_spec_values:
            lines.append("Spec auto-skipped: " + ",".join(skipped_spec_values))
        if errors:
            lines.append("Fix selection: " + "; ".join(errors))

        if valid_ranges and valid_modes:
            self.autotune_grid_preview_label.setStyleSheet("color: #b0b0b0;")
        else:
            self.autotune_grid_preview_label.setStyleSheet("color: #ff6b6b;")

        self.autotune_grid_preview_label.setText("\n".join(lines))
        self.autotune_grid_preview_label.setToolTip("\n".join(detail_lines))

    @staticmethod
    def _parse_csv_values(values: str) -> list[str]:
        return [v.strip().lower() for v in values.split(",") if v.strip()]

    def _bench_help_output(self) -> str:
        script_path = self.project_root / "scripts" / "agent_workload_bench.py"
        python_executable = console_python_executable()
        cache_key = f"{python_executable}|{script_path}"

        if cache_key in self._bench_help_cache:
            return self._bench_help_cache[cache_key]

        try:
            result = run_hidden(
                [python_executable, str(script_path), "--help"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                cwd=self.project_root,
            )
            output = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
            self._bench_help_cache[cache_key] = output
            return output
        except Exception:
            self._bench_help_cache[cache_key] = ""
            return ""

    def _bench_supports_flag(self, flag: str) -> bool:
        normalized = flag.strip().lower()
        if not normalized:
            return False
        if not normalized.startswith("--"):
            normalized = f"--{normalized}"
        return normalized in self._bench_help_output()

    @staticmethod
    def _server_help_cache_key(server_bin: Path) -> str:
        try:
            return str(server_bin.resolve())
        except Exception:
            return str(server_bin)

    def _server_help_cached(self, server_bin: Path) -> bool:
        return self._server_help_cache_key(server_bin) in self._server_help_cache

    def _start_server_help_probe(self, server_bin: Path) -> None:
        thread = getattr(self, "_help_probe_thread", None)
        if thread is not None and thread.isRunning():
            return
        thread = _ServerHelpProbeThread(self, server_bin)
        thread.finished.connect(self._update_autotune_grid_preview)
        self._help_probe_thread = thread
        thread.start()

    def _server_help_output(self, server_bin: Path) -> str:
        cache_key = self._server_help_cache_key(server_bin)

        if cache_key in self._server_help_cache:
            return self._server_help_cache[cache_key]

        try:
            result = run_hidden(
                [str(server_bin), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            output = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
            self._server_help_cache[cache_key] = output
            return output
        except Exception:
            self._server_help_cache[cache_key] = ""
            return ""

    def _resolve_autotune_spec_values(self, server_bin: Path, model: Path, raw_values: str) -> list[str]:
        requested = raw_values.strip().lower()
        output = self._server_help_output(server_bin)

        # Parse only explicit --spec-type enum options to avoid false positives
        # from unrelated flags like --spec-draft-*.
        spec_type_modes: list[str] = []
        match = re.search(r"--spec-type\s*\[([^\]]+)\]", output)
        if match:
            for raw in match.group(1).split("|"):
                mode = raw.strip().lower()
                if mode:
                    spec_type_modes.append("mtp" if mode == "draft-mtp" else mode)

        allowed_modes = set(spec_type_modes)
        mtp_compatible = model_supports_mtp(model)

        def is_mode_model_compatible(mode: str) -> bool:
            if mode not in {"mtp", "ngram-mtp"}:
                return True
            return mtp_compatible

        if requested and requested not in {"auto", "all"}:
            ordered: list[str] = []
            seen: set[str] = set()
            requested_values = self._parse_csv_values(requested)
            for value in requested_values:
                if allowed_modes and value not in allowed_modes:
                    continue
                if not is_mode_model_compatible(value):
                    continue
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)

            # A model switch should not leave autotune with an empty grid just
            # because the previous MTP-capable model had only MTP selected.
            # Keep malformed/unsupported selections invalid, but degrade an
            # explicitly selected MTP-only set to the universal baseline mode.
            requested_mtp_only = bool(requested_values) and all(
                value in {"mtp", "ngram-mtp"} for value in requested_values
            )
            if (
                not ordered
                and not mtp_compatible
                and requested_mtp_only
                and (not allowed_modes or "none" in allowed_modes)
            ):
                return ["none"]
            return ordered

        if not spec_type_modes:
            spec_type_modes = ["none", "ngram-mod"]

        supported_order = ["none", "ngram-mod", "mtp", "ngram-mtp", "eagle3", "eagle"]
        resolved: list[str] = []
        for mode in supported_order:
            if mode in spec_type_modes:
                resolved.append(mode)

        if "none" not in resolved:
            resolved.insert(0, "none")

        if "ngram-mod" not in resolved:
            ngram_candidates = [mode for mode in spec_type_modes if mode.startswith("ngram")]
            if ngram_candidates:
                fallback_ngram = "ngram-mod" if "ngram-mod" in ngram_candidates else ngram_candidates[0]
                if fallback_ngram not in resolved:
                    resolved.append(fallback_ngram)

        if not mtp_compatible:
            resolved = [mode for mode in resolved if mode not in {"mtp", "ngram-mtp"}]

        unique: list[str] = []
        seen: set[str] = set()
        for mode in resolved:
            if mode in seen:
                continue
            seen.add(mode)
            unique.append(mode)
        return unique

    def _server_supports_mtp(self, server_bin: Path) -> bool:
        return "mtp" in self._server_help_output(server_bin)

    def _on_bench_output(self, line: str):
        self.log_output.append(line)
        if self._current_mode == "autotune":
            self._update_autotune_live_history_from_line(line)
        if line.startswith("BEST:"):
            self._autotune_result["best"] = line
        if line.startswith("CURRENT BEST:"):
            self._autotune_result["best"] = line
        if line.endswith("-autotune-summary.json"):
            self._autotune_result["summary_json"] = line.split("Wrote ", 1)[-1].strip()
        if line.endswith("-autotune-summary.csv"):
            self._autotune_result["summary_csv"] = line.split("Wrote ", 1)[-1].strip()
        if (
            "Aggregate completion TPS" in line
            or line.startswith(("BEST:", "CURRENT BEST:", "RUN RESULT:", "CONFIG RESULT:"))
        ):
            self.status_label.setText(line)
        if line.startswith("CONFIG FAILED ("):
            self.status_label.setText(line)

    def _on_bench_finished(self, success: bool, stopped: bool = False):
        self._set_running_state(False)

        if stopped:
            if self._current_mode == "autotune" and self._autotune_active_run is not None:
                stopped_row = self._find_autotune_history_row(self._autotune_active_run)
                if stopped_row is not None:
                    self.autotune_history_table.setItem(stopped_row, 11, QTableWidgetItem("stopped"))
                self._autotune_active_run = None
            self.status_label.set_state("neutral", "Benchmark stopped")
            self.log_output.append("[INFO] Benchmark/autotune stopped by user")
            if self._current_mode == "autotune":
                self.refresh_saved_presets_table()
            QMessageBox.information(self, "Bench", "Run stopped. Completed autotune configs can be resumed.")
            self.bench_thread = None
            return

        if success:
            self.status_label.set_state("ok", "Benchmark completed")
            if self._current_mode == "autotune":
                self.refresh_saved_presets_table()
            if hasattr(self.parent, "refresh_build_registry"):
                self.parent.refresh_build_registry()
            if hasattr(self.parent, "builds_info_tab") and hasattr(self.parent.builds_info_tab, "refresh_builds_info"):
                self.parent.builds_info_tab.refresh_builds_info()

            # Auto-apply best preset to server tab
            if hasattr(self.parent, "server_tab") and hasattr(self.parent.server_tab, "apply_model_file_preset"):
                self.parent.server_tab.apply_model_file_preset()

            message = "Benchmark finished."
            if self._current_mode == "autotune":
                message = "Auto-tune finished. Run history updated and best preset applied to Launch Server tab."
            QMessageBox.information(self, "Bench", message)
        else:
            if self._current_mode == "autotune" and self._autotune_active_run is not None:
                failed_row = self._find_autotune_history_row(self._autotune_active_run)
                if failed_row is not None:
                    self.autotune_history_table.setItem(failed_row, 11, QTableWidgetItem("failed"))
                self._autotune_active_run = None
            if self._current_mode == "autotune":
                self.refresh_saved_presets_table()
            self.status_label.set_state("error", "Benchmark failed")
            QMessageBox.warning(self, "Bench", "Benchmark/autotune failed. Check log output.")

        self.bench_thread = None

    def _show_history_context_menu(self, pos):
        if self.presets_table.rowCount() == 0:
            return
        index = self.presets_table.indexAt(pos)
        if index.isValid():
            self.presets_table.selectRow(index.row())
        menu = QMenu(self.presets_table)
        apply_action = menu.addAction("Apply Default")
        apply_action.triggered.connect(self.apply_selected_run_as_default_preset)
        apply_action.setEnabled(self.apply_history_preset_btn.isEnabled())
        menu.addSeparator()
        menu.addAction(self.open_history_log_action)
        menu.addAction(self.copy_history_log_action)
        menu.addAction(self.copy_history_row_action)
        menu.addSeparator()
        menu.addAction(self.delete_history_run_action)
        menu.exec(self.presets_table.viewport().mapToGlobal(pos))

    def _set_running_state(self, running: bool):
        if running:
            self.status_label.set_state("busy")
        self.run_bench_btn.setEnabled(not running)
        self.run_autotune_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.model_browse_btn.setEnabled(not running)
        self.model_refresh_btn.setEnabled(not running)
        self.apply_history_preset_btn.setEnabled(not running)
        self.refresh_history_btn.setEnabled(not running)
        self.open_history_log_action.setEnabled(not running)
        self.copy_history_log_action.setEnabled(not running)
        self.copy_history_row_action.setEnabled(not running)
        self.delete_history_run_action.setEnabled(not running)
        self.at_batch_min_spin.setEnabled(not running)
        self.at_batch_max_spin.setEnabled(not running)
        self.at_batch_step_spin.setEnabled(not running)
        self.at_ubatch_min_spin.setEnabled(not running)
        self.at_ubatch_max_spin.setEnabled(not running)
        self.at_ubatch_step_spin.setEnabled(not running)
        for checkbox in self.autotune_kv_checks.values():
            checkbox.setEnabled(not running)
        for checkbox in self.autotune_spec_checks.values():
            checkbox.setEnabled(not running)
        for checkbox in self.autotune_extra_checks.values():
            checkbox.setEnabled(not running)
        self.autotune_custom_extra_input.setEnabled(not running)
        self.autotune_resume_checkbox.setEnabled(not running)
        self.autotune_reset_session_checkbox.setEnabled(not running)


    def _reset_autotune_live_history(self) -> None:
        self._autotune_active_run = None
        self.autotune_history_table.setRowCount(0)

    def _find_autotune_history_row(self, run_label: str) -> int | None:
        for row in range(self.autotune_history_table.rowCount()):
            item = self.autotune_history_table.item(row, 0)
            if item is not None and item.text() == run_label:
                return row
        return None

    @staticmethod
    def _sanitize_compact_token(value: str, fallback: str = "-") -> str:
        token = str(value or "").strip().rstrip(",;")
        return token if token else fallback

    def _update_autotune_live_history_from_line(self, line: str) -> None:
        start_match = re.search(r"Autotune \[(\d+)/(\d+)\]:\s*(.+)$", line)
        if start_match:
            run_idx_text, total_text, payload = start_match.groups()

            fields: dict[str, str] = {}
            for chunk in payload.split(","):
                if "=" not in chunk:
                    continue
                key, value = chunk.split("=", 1)
                fields[key.strip().lower()] = value.strip()

            try:
                run_idx = int(run_idx_text)
                total = int(total_text)
                ctx = int(fields.get("ctx", ""))
                batch = int(fields.get("b", ""))
                ubatch = int(fields.get("ub", ""))
            except ValueError:
                return

            kv = self._sanitize_compact_token(fields.get("kv", ""))
            spec_mode = self._sanitize_compact_token(fields.get("spec", ""))
            extra_value = self._sanitize_compact_token(fields.get("extra", "base"), fallback="base")
            draft_n = self._sanitize_compact_token(fields.get("draftn", "-"))

            run_label = f"{run_idx}/{total}"
            row = self.autotune_history_table.rowCount()
            self.autotune_history_table.insertRow(row)
            self.autotune_history_table.setItem(row, 0, NumericTableWidgetItem(run_label, run_idx))
            self.autotune_history_table.setItem(row, 1, NumericTableWidgetItem(str(ctx), ctx))
            self.autotune_history_table.setItem(row, 2, NumericTableWidgetItem(str(batch), batch))
            self.autotune_history_table.setItem(row, 3, NumericTableWidgetItem(str(ubatch), ubatch))
            self.autotune_history_table.setItem(row, 4, QTableWidgetItem(kv))
            self.autotune_history_table.setItem(row, 5, QTableWidgetItem(spec_mode))
            self.autotune_history_table.setItem(row, 6, QTableWidgetItem(draft_n))
            self.autotune_history_table.setItem(row, 7, QTableWidgetItem(extra_value))
            self.autotune_history_table.setItem(row, 8, QTableWidgetItem("-"))
            self.autotune_history_table.setItem(row, 9, QTableWidgetItem("-"))
            self.autotune_history_table.setItem(row, 10, QTableWidgetItem("-"))
            self.autotune_history_table.setItem(row, 11, QTableWidgetItem("running"))
            self._autotune_active_run = run_label
            self.autotune_history_table.scrollToBottom()
            self.status_label.setText(line)
            return

        if self._autotune_active_run is None:
            return

        active_row = self._find_autotune_history_row(self._autotune_active_run)
        if active_row is None:
            return

        progress_match = re.search(
            r"PROMPT PROGRESS:\s*task=(\S+)\s+pct=([0-9.]+)\s+tokens=(\d+)/(\d+|\?)\s+status=(\w+)",
            line,
        )
        if progress_match:
            _task_id, pct_text, done_tokens, total_tokens, progress_status = progress_match.groups()
            pct_value = float(pct_text)
            if total_tokens == "?":
                detail_text = f"prompt {pct_value:.1f}%"
            else:
                detail_text = f"prompt {pct_value:.1f}% ({done_tokens}/{total_tokens})"
            table_text = f"prompt {pct_value:.1f}%"
            self.autotune_history_table.setItem(active_row, 11, QTableWidgetItem(table_text))
            self.status_label.setText(detail_text if progress_status == "running" else "prompt done, decoding")
            return

        tps_match = re.search(r"Aggregate completion TPS by wall time:\s*([0-9]+(?:\.[0-9]+)?)", line)
        if tps_match:
            tps_value = float(tps_match.group(1))
            self.autotune_history_table.setItem(active_row, 8, NumericTableWidgetItem(f"{tps_value:.2f}", tps_value))
            self.autotune_history_table.setItem(active_row, 11, QTableWidgetItem("done"))
            # keep the run active: CONFIG RESULT (prompt/decode TPS) prints next
            return

        result_match = re.search(
            r"CONFIG RESULT: aggregate_tps=([0-9.]+) prompt_tps=([0-9.]+) decode_tps=([0-9.]+)",
            line,
        )
        if result_match:
            agg_value, prompt_value, decode_value = (float(g) for g in result_match.groups())
            self.autotune_history_table.setItem(active_row, 8, NumericTableWidgetItem(f"{agg_value:.2f}", agg_value))
            self.autotune_history_table.setItem(active_row, 9, NumericTableWidgetItem(f"{prompt_value:.1f}", prompt_value))
            self.autotune_history_table.setItem(active_row, 10, NumericTableWidgetItem(f"{decode_value:.2f}", decode_value))
            self.autotune_history_table.setItem(active_row, 11, QTableWidgetItem("done"))
            self._autotune_active_run = None
            return

        if line.strip().startswith("error:"):
            status_item = self.autotune_history_table.item(active_row, 11)
            if status_item is None or status_item.text() == "running":
                self.autotune_history_table.setItem(active_row, 11, QTableWidgetItem("error"))
            self._autotune_active_run = None
            return

        if line.startswith("CONFIG FAILED ("):
            self.autotune_history_table.setItem(active_row, 11, QTableWidgetItem("failed"))
            self._autotune_active_run = None
