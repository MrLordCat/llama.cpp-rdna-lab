"""Benchmark tab - dedicated benchmark and autotune workflows."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QFileDialog,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from model_capabilities import model_supports_mtp


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


class BenchmarkTabWidget(QWidget):
    """Dedicated Bench & Autotune tab."""

    NGRAM_MOD_N_MIN = 12
    NGRAM_MOD_N_MATCH = 16
    NGRAM_MOD_N_MAX = 32
    MTP_DRAFT_N_MAX = 8

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
        self._live_best_by_key: dict[str, dict[str, str]] = {}
        self._summary_sweep_cache: dict[str, tuple[str, str]] = {}
        self._server_help_cache: dict[str, str] = {}
        self._bench_help_cache: dict[str, str] = {}
        self._autotune_result = {"best": "", "summary_json": "", "summary_csv": ""}
        self._autotune_active_run: str | None = None
        self.create_ui()
        self.refresh_models_list()
        self.refresh_build_choices()
        self.load_settings()
        self.refresh_saved_presets_table()

    def load_settings(self) -> None:
        if self.settings is None:
            return

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

            self._update_autotune_grid_preview()
        except Exception as exc:
            self.log_output.append(f"[WARN] Failed to load autotune settings: {exc}")

    def save_settings(self) -> None:
        if self.settings is None:
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
        except Exception as exc:
            self.log_output.append(f"[WARN] Failed to save autotune settings: {exc}")

    @staticmethod
    def _create_scroll_panel(widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumSize(0, 0)
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_area.setWidget(widget)
        return scroll_area

    @staticmethod
    def _configure_combo(combo: QComboBox, minimum_contents_length: int = 12) -> None:
        combo.setMinimumContentsLength(minimum_contents_length)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumWidth(80)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _configure_spinbox(spin_box: QSpinBox) -> None:
        spin_box.setMinimumWidth(76)
        spin_box.setMaximumWidth(118)
        spin_box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _configure_compact_table(table: QTableWidget, column_widths: list[int]) -> None:
        table.setAlternatingRowColors(True)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.setMinimumWidth(0)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        header = table.horizontalHeader()
        header.setMinimumSectionSize(48)
        header.setStretchLastSection(False)
        for column, width in enumerate(column_widths):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            table.setColumnWidth(column, width)

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
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to GGUF model...")
        self.model_path_input.setMinimumWidth(0)
        model_layout.addWidget(self.model_path_input)

        self.model_browse_btn = QPushButton("Browse")
        self.model_browse_btn.setToolTip("Select a GGUF model file")
        self.model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_browse_btn)

        model_layout.addWidget(QLabel("Detected models:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_selected)
        model_layout.addWidget(self.model_combo)

        self.model_refresh_btn = QPushButton("Refresh")
        self.model_refresh_btn.setToolTip("Refresh detected GGUF models")
        self.model_refresh_btn.clicked.connect(self.refresh_models_list)
        model_layout.addWidget(self.model_refresh_btn)

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
        build_layout.setColumnStretch(1, 1)
        build_group.setLayout(build_layout)
        left_layout.addWidget(build_group)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        single_group = QGroupBox("Single Benchmark (used by Run Benchmark)")
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
        single_layout.setColumnStretch(1, 1)

        single_group.setLayout(single_layout)
        params_layout.addWidget(single_group)

        autotune_group = QGroupBox("Auto-tune Grid (used by Run Auto-tune 130K)")
        autotune_layout = QVBoxLayout()

        autotune_mode_info = QLabel(
            "Fixed mode: ctx=131072, tasks=quick:triage_diff, runs=1, repo-snapshot chars=24576, "
            "max_tokens=16, no-reuse, no-prime, thinking on. 130K may spill KV/context into system RAM."
        )
        autotune_mode_info.setWordWrap(True)
        autotune_mode_info.setStyleSheet("color: #b0b0b0;")
        autotune_layout.addWidget(autotune_mode_info)

        batch_grid = QGridLayout()
        batch_grid.setHorizontalSpacing(8)
        batch_grid.setVerticalSpacing(6)
        batch_grid.addWidget(QLabel("Batch min:"), 0, 0)
        self.at_batch_min_spin = QSpinBox()
        self.at_batch_min_spin.setMinimum(32)
        self.at_batch_min_spin.setMaximum(8192)
        self.at_batch_min_spin.setValue(256)
        self.at_batch_min_spin.setSingleStep(32)
        self.at_batch_min_spin.setToolTip("Minimal batch value in sweep (>= 32)")
        batch_grid.addWidget(self.at_batch_min_spin, 0, 1)

        batch_grid.addWidget(QLabel("Batch max:"), 1, 0)
        self.at_batch_max_spin = QSpinBox()
        self.at_batch_max_spin.setMinimum(32)
        self.at_batch_max_spin.setMaximum(8192)
        self.at_batch_max_spin.setValue(1024)
        self.at_batch_max_spin.setSingleStep(32)
        self.at_batch_max_spin.setToolTip("Maximal batch value in sweep")
        batch_grid.addWidget(self.at_batch_max_spin, 1, 1)

        batch_grid.addWidget(QLabel("Batch step:"), 2, 0)
        self.at_batch_step_spin = QSpinBox()
        self.at_batch_step_spin.setMinimum(1)
        self.at_batch_step_spin.setMaximum(8192)
        self.at_batch_step_spin.setValue(256)
        self.at_batch_step_spin.setSingleStep(1)
        self.at_batch_step_spin.setToolTip("Increment for batch range")
        batch_grid.addWidget(self.at_batch_step_spin, 2, 1)
        batch_grid.setColumnStretch(1, 1)
        autotune_layout.addLayout(batch_grid)

        ubatch_grid = QGridLayout()
        ubatch_grid.setHorizontalSpacing(8)
        ubatch_grid.setVerticalSpacing(6)
        ubatch_grid.addWidget(QLabel("UBatch min:"), 0, 0)
        self.at_ubatch_min_spin = QSpinBox()
        self.at_ubatch_min_spin.setMinimum(32)
        self.at_ubatch_min_spin.setMaximum(8192)
        self.at_ubatch_min_spin.setValue(64)
        self.at_ubatch_min_spin.setSingleStep(32)
        self.at_ubatch_min_spin.setToolTip("Minimal ubatch value in sweep (>= 32)")
        ubatch_grid.addWidget(self.at_ubatch_min_spin, 0, 1)

        ubatch_grid.addWidget(QLabel("UBatch max:"), 1, 0)
        self.at_ubatch_max_spin = QSpinBox()
        self.at_ubatch_max_spin.setMinimum(32)
        self.at_ubatch_max_spin.setMaximum(8192)
        self.at_ubatch_max_spin.setValue(256)
        self.at_ubatch_max_spin.setSingleStep(32)
        self.at_ubatch_max_spin.setToolTip("Maximal ubatch value in sweep")
        ubatch_grid.addWidget(self.at_ubatch_max_spin, 1, 1)

        ubatch_grid.addWidget(QLabel("UBatch step:"), 2, 0)
        self.at_ubatch_step_spin = QSpinBox()
        self.at_ubatch_step_spin.setMinimum(1)
        self.at_ubatch_step_spin.setMaximum(8192)
        self.at_ubatch_step_spin.setValue(64)
        self.at_ubatch_step_spin.setSingleStep(1)
        self.at_ubatch_step_spin.setToolTip("Increment for ubatch range")
        ubatch_grid.addWidget(self.at_ubatch_step_spin, 2, 1)
        ubatch_grid.setColumnStretch(1, 1)
        autotune_layout.addLayout(ubatch_grid)

        kv_grid = QGridLayout()
        kv_grid.setHorizontalSpacing(14)
        kv_grid.setVerticalSpacing(4)
        kv_grid.addWidget(QLabel("KV sweep:"), 0, 0)
        self.autotune_kv_checks: dict[str, QCheckBox] = {}
        for index, (kv_name, enabled, hint) in enumerate([
            ("q4_0", True, "Main KV cache for the current 130K target"),
            ("q8_0", False, "Higher-quality KV cache opt-in"),
            ("turbo4", False, "TurboKV 4-bit cache (128-block WHT, correctness path)"),
            ("turbo3", False, "TurboKV 3-bit cache (128-block WHT, correctness path)"),
            ("turbo2", False, "TurboKV 2-bit cache (128-block WHT, correctness path)"),
            ("f16", False, "FP16 KV (usually slower/heavier)"),
            ("bf16", False, "BF16 KV (usually slower/heavier)"),
            ("f32", False, "FP32 KV (debug/reference only)"),
        ]):
            checkbox = QCheckBox(kv_name)
            checkbox.setChecked(enabled)
            checkbox.setToolTip(hint)
            self.autotune_kv_checks[kv_name] = checkbox
            kv_grid.addWidget(checkbox, index // 2, (index % 2) + 1)
        kv_grid.setColumnStretch(3, 1)
        autotune_layout.addLayout(kv_grid)

        spec_grid = QGridLayout()
        spec_grid.setHorizontalSpacing(14)
        spec_grid.setVerticalSpacing(4)
        spec_grid.addWidget(QLabel("Spec sweep:"), 0, 0)
        self.autotune_spec_checks: dict[str, QCheckBox] = {}
        for index, (mode, enabled, hint) in enumerate([
            ("none", True, "Always keep plain decoding baseline in sweep"),
            ("ngram-mod", False, "Ngram speculative mode for explicit repeated/session probes"),
            ("draft", False, "Draft speculative mode when supported"),
            ("eagle3", False, "Eagle3 speculative mode when supported"),
            ("mtp", False, "MTP mode; requires server + MTP model support"),
            ("ngram-mtp", False, "Experimental ngram first, MTP fallback mode"),
        ]):
            checkbox = QCheckBox(mode)
            checkbox.setChecked(enabled)
            checkbox.setToolTip(hint)
            self.autotune_spec_checks[mode] = checkbox
            spec_grid.addWidget(checkbox, index // 2, (index % 2) + 1)
        spec_grid.setColumnStretch(3, 1)
        autotune_layout.addLayout(spec_grid)

        extra_grid = QGridLayout()
        extra_grid.setHorizontalSpacing(14)
        extra_grid.setVerticalSpacing(4)
        extra_grid.addWidget(QLabel("Extra presets:"), 0, 0)
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
        self.autotune_extra_checks: dict[str, QCheckBox] = {}
        for index, (key, enabled, hint) in enumerate([
            ("base", True, "No extra server arguments"),
            ("ngram-balanced", False, "Measured ngram-mod profile: 12/16/32"),
            ("ngram-wide", False, "Wider ngram window for ngram-mod"),
        ]):
            checkbox = QCheckBox(key)
            checkbox.setChecked(enabled)
            checkbox.setToolTip(hint)
            self.autotune_extra_checks[key] = checkbox
            extra_grid.addWidget(checkbox, index // 2, (index % 2) + 1)
        extra_grid.setColumnStretch(3, 1)
        autotune_layout.addLayout(extra_grid)

        custom_extra_row = QHBoxLayout()
        custom_extra_row.addWidget(QLabel("Custom extras:"))
        self.autotune_custom_extra_input = QLineEdit()
        self.autotune_custom_extra_input.setPlaceholderText("name::--arg value||name2::--arg2 value")
        self.autotune_custom_extra_input.setToolTip("Optional extra presets; separate presets with ||")
        self.autotune_custom_extra_input.setMinimumWidth(0)
        custom_extra_row.addWidget(self.autotune_custom_extra_input)
        autotune_layout.addLayout(custom_extra_row)

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

        autotune_group.setLayout(autotune_layout)
        params_layout.addWidget(autotune_group)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        for combo, minimum_contents_length in [
            (self.model_combo, 18),
            (self.build_backend_combo, 10),
            (self.build_version_combo, 18),
            (self.tasks_combo, 8),
            (self.spec_combo, 10),
            (self.kv_combo, 8),
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
        self._update_autotune_grid_preview()

        btn_grid = QGridLayout()
        btn_grid.setHorizontalSpacing(8)
        btn_grid.setVerticalSpacing(6)
        self.run_bench_btn = QPushButton("Run Benchmark")
        self.run_bench_btn.setToolTip("Run the single benchmark profile above")
        self.run_bench_btn.clicked.connect(self.run_benchmark)
        btn_grid.addWidget(self.run_bench_btn, 0, 0)

        self.run_autotune_btn = QPushButton("Run Auto-tune 130K")
        self.run_autotune_btn.setToolTip("Run the 130K cold repo-snapshot autotune grid")
        self.run_autotune_btn.clicked.connect(self.run_autotune)
        btn_grid.addWidget(self.run_autotune_btn, 1, 0)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the current benchmark or autotune run")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_current_run)
        btn_grid.addWidget(self.stop_btn, 2, 0)

        self.open_history_btn = QPushButton("Open History")
        self.open_history_btn.setToolTip("Open build_logs/agent-workload/BENCH_HISTORY.md")
        self.open_history_btn.clicked.connect(self.open_history_md)
        btn_grid.addWidget(self.open_history_btn, 3, 0)

        btn_grid.setColumnStretch(0, 1)
        left_layout.addLayout(btn_grid)

        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        log_group = QGroupBox("Run Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(150)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, 2)

        presets_group = QGroupBox("Autotune Runs History (Best Result Per Run)")
        presets_layout = QVBoxLayout()
        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(14)
        self.presets_table.setHorizontalHeaderLabels([
            "Run Time",
            "Model",
            "Best TPS",
            "Ctx",
            "Batch/UBatch",
            "KV",
            "Best Spec",
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
            [140, 180, 82, 70, 105, 72, 96, 110, 180, 120, 120, 120, 150, 160],
        )
        presets_layout.addWidget(self.presets_table)

        presets_actions = QGridLayout()
        presets_actions.setHorizontalSpacing(8)
        presets_actions.setVerticalSpacing(6)
        self.apply_history_preset_btn = QPushButton("Apply Default")
        self.apply_history_preset_btn.setToolTip("Apply the selected run as the default model preset")
        self.apply_history_preset_btn.clicked.connect(self.apply_selected_run_as_default_preset)
        presets_actions.addWidget(self.apply_history_preset_btn, 0, 0)

        self.delete_history_run_btn = QPushButton("Delete")
        self.delete_history_run_btn.setToolTip("Delete the selected autotune run from history")
        self.delete_history_run_btn.clicked.connect(self.delete_selected_preset)
        presets_actions.addWidget(self.delete_history_run_btn, 1, 0)

        self.refresh_history_btn = QPushButton("Refresh")
        self.refresh_history_btn.setToolTip("Refresh autotune run history")
        self.refresh_history_btn.clicked.connect(self.refresh_saved_presets_table)
        presets_actions.addWidget(self.refresh_history_btn, 2, 0)

        self.open_history_log_btn = QPushButton("Open Log")
        self.open_history_log_btn.setToolTip("Open the log for the selected run")
        self.open_history_log_btn.clicked.connect(self.open_selected_history_log)
        presets_actions.addWidget(self.open_history_log_btn, 3, 0)

        self.copy_history_log_btn = QPushButton("Copy Log")
        self.copy_history_log_btn.setToolTip("Copy the selected run log content")
        self.copy_history_log_btn.clicked.connect(self.copy_selected_history_log_to_clipboard)
        presets_actions.addWidget(self.copy_history_log_btn, 4, 0)

        self.copy_history_row_btn = QPushButton("Copy Row")
        self.copy_history_row_btn.setToolTip("Copy the selected run data")
        self.copy_history_row_btn.clicked.connect(self.copy_selected_history_row_to_clipboard)
        presets_actions.addWidget(self.copy_history_row_btn, 5, 0)

        presets_layout.addLayout(presets_actions)

        presets_group.setLayout(presets_layout)
        right_layout.addWidget(presets_group, 3)

        history_group = QGroupBox("Current Autotune Run History (Live)")
        history_layout = QVBoxLayout()
        self.autotune_history_table = QTableWidget()
        self.autotune_history_table.setColumnCount(9)
        self.autotune_history_table.setHorizontalHeaderLabels([
            "Run",
            "Ctx",
            "Batch",
            "UBatch",
            "KV",
            "Spec",
            "Extra",
            "TPS",
            "Status",
        ])
        self.autotune_history_table.setSortingEnabled(True)
        self.autotune_history_table.setMinimumHeight(190)
        self._configure_compact_table(
            self.autotune_history_table,
            [62, 70, 76, 76, 70, 96, 120, 70, 78],
        )
        history_layout.addWidget(self.autotune_history_table)
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group, 4)

    @staticmethod
    def _display_backend_from_key(key: str) -> str:
        mapping = {
            "rocm": "ROCm/HIP",
            "cpu": "CPU",
            "cuda": "CUDA",
            "vulkan": "Vulkan",
            "metal": "Metal",
            "sycl": "SYCL",
            "opencl": "OpenCL",
        }
        return mapping.get(key.lower(), key)

    @staticmethod
    def _backend_key_from_display(display: str) -> str:
        mapping = {
            "ROCm/HIP": "rocm",
            "CPU": "cpu",
            "CUDA": "cuda",
            "Vulkan": "vulkan",
            "Metal": "metal",
            "SYCL": "sycl",
            "OpenCL": "opencl",
        }
        return mapping.get(display, display.lower())

    def refresh_build_choices(self):
        previous_backend = self.build_backend_combo.currentText() if self.build_backend_combo.count() else "ROCm/HIP"
        self.build_backend_combo.blockSignals(True)
        self.build_backend_combo.clear()

        records = self.parent.get_registered_builds() if hasattr(self.parent, "get_registered_builds") else []
        keys = sorted({str(r.get("backend", "")).lower() for r in records if str(r.get("backend", "")).strip()})

        for key in keys:
            self.build_backend_combo.addItem(self._display_backend_from_key(key))

        for legacy in ["ROCm/HIP", "CPU", "CUDA", "Vulkan", "Metal", "SYCL", "OpenCL"]:
            if self.build_backend_combo.findText(legacy) < 0:
                self.build_backend_combo.addItem(legacy)

        idx = self.build_backend_combo.findText(previous_backend)
        if idx < 0:
            idx = self.build_backend_combo.findText("ROCm/HIP")
        self.build_backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.build_backend_combo.blockSignals(False)
        self.refresh_versions_for_backend(select_latest=True)

    def _on_backend_changed(self, *_args):
        self.refresh_versions_for_backend(select_latest=True)

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
        server_extra = self._active_lane_base_server_extra(self.ctx_spin.value())
        spec_extra = [f"--spec-type {spec_mode}"]
        if spec_mode in {"ngram-mod", "ngram-mtp"}:
            spec_extra.append(f"--spec-ngram-mod-n-min {self.NGRAM_MOD_N_MIN}")
            spec_extra.append(f"--spec-ngram-mod-n-match {self.NGRAM_MOD_N_MATCH}")
            spec_extra.append(f"--spec-ngram-mod-n-max {self.NGRAM_MOD_N_MAX}")
        if spec_mode in {"mtp", "ngram-mtp"}:
            spec_extra.append(f"--spec-draft-n-max {self.MTP_DRAFT_N_MAX}")
        server_extra.extend(spec_extra)

        command = [
            sys.executable,
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
            "180",
            "--task-hard-timeout",
            "45",
            "--background-server-policy",
            "fail",
            "--real-context-mode",
            "repo-snapshot",
            "--real-context-chars",
            "24576",
            "--real-context-safe-fill",
            "0.88",
            "--no-reuse",
            "--no-v2-prime-pass",
            "--no-disable-thinking",
            "--server-extra",
            " ".join(server_extra),
        ]

        if self.tasks_combo.currentText() == "quick":
            command.extend(["--task-ids", "triage_diff"])

        bench_env = self._bench_env_overrides()

        self._current_mode = "single"
        self._last_selected_model = model.name
        self._current_build_id = build_id
        self._set_running_state(True)
        self.log_output.clear()
        self.log_output.append(f"[INFO] Starting benchmark for {model.name}")
        self.log_output.append(f"[INFO] Build ID: {build_id or '-'}")
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

        profile_key = "ctx130k-only"
        autotune_min_ctx = 131072
        autotune_ctx_values = "131072"
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

        autotune_extra_presets = "||".join(extra_presets)
        autotune_kv_values = ",".join(kv_values)
        autotune_tasks = "quick"
        autotune_task_ids = "triage_diff"
        autotune_max_tokens = "16"
        autotune_real_context_chars = "24576"

        ctx_count = len([v for v in autotune_ctx_values.split(",") if v.strip()])
        batch_count = len(batch_values)
        ubatch_count = len(ubatch_values)
        kv_count = len([v for v in autotune_kv_values.split(",") if v.strip()])
        spec_count = len(spec_values)
        extra_count = len(extra_presets)
        config_count = ctx_count * batch_count * ubatch_count * kv_count * spec_count * extra_count

        run_stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        label = f"gui-autotune-{model.stem}-{run_stamp}"
        session_label = f"gui-autotune-{model.stem}"
        autotune_session_file = self.project_root / "build_logs" / "agent-workload" / f"{session_label}-autotune-session.json"
        compatibility_notes: list[str] = []
        base_server_extra = self._active_lane_base_server_extra(autotune_min_ctx)

        command = [
            sys.executable,
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
            "180",
            "--task-hard-timeout",
            "45",
            "--background-server-policy",
            "fail",
            "--allow-ctx-above-16k",
            "--real-context-mode",
            "repo-snapshot",
            "--real-context-chars",
            autotune_real_context_chars,
            "--real-context-safe-fill",
            "0.88",
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
            "--autotune-max-configs",
            str(max(64, config_count + 8)),
            "--autotune-update-preset",
            "--autotune-preset-file",
            "gui/model_presets.json",
        ]

        if base_server_extra:
            command.extend(["--server-extra", " ".join(base_server_extra)])

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
        self.log_output.append(f"[INFO] Autotune profile: 130K fixed, configs: {config_count}")
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

        self.log_output.append("[INFO] Task timeout policy: 45s hard, fail timeout off")

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
        extra_presets = self._selected_autotune_extra_presets()

        kv_text = ",".join(kv_values) if kv_values else "-"
        spec_text = ",".join(selected_spec_values) if selected_spec_values else "-"

        effective_spec_values = list(selected_spec_values)
        skipped_spec_values: list[str] = []
        has_runtime_context = False

        preview_model = self._resolve_selected_model()
        preview_server, _ = self._resolve_selected_server()
        if preview_model is not None and preview_server is not None and selected_spec_values:
            has_runtime_context = True
            effective_spec_values = self._resolve_autotune_spec_values(
                preview_server,
                preview_model,
                ",".join(selected_spec_values),
            )
            skipped_spec_values = [mode for mode in selected_spec_values if mode not in effective_spec_values]

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

        lines = [
            f"Batch values: {self._format_values_preview(batch_values)}",
            f"UBatch values: {self._format_values_preview(ubatch_values)}",
            f"KV modes: {kv_text}",
            f"Spec modes (selected): {spec_text}",
            f"Extra presets: {extra_preview or '-'}",
            f"Estimated configs: {total_configs} (ctx=1 x kv x batch x ubatch x spec x extra)",
            "Runtime lane: ctx=131072 repo-snapshot chars=24576, no-reuse, no-prime, thinking on",
            "Hint: for the active 130K quick lane use batch 256..1024 and ubatch 64..256",
        ]

        if has_runtime_context:
            lines.insert(4, f"Spec modes (effective): {effective_spec_text}")
            if skipped_spec_values:
                lines.append(
                    "Spec auto-skipped (unsupported by server/model): " + ",".join(skipped_spec_values)
                )
        else:
            lines.append("Spec effective set is resolved at run start (needs model + server).")

        if not kv_values:
            lines.append("Selection error: choose at least one KV mode")
        if not selected_spec_values:
            lines.append("Selection error: choose at least one Spec mode")
        if not extra_presets:
            lines.append("Selection error: choose at least one extra preset")

        if valid_ranges and valid_modes:
            self.autotune_grid_preview_label.setStyleSheet("color: #b0b0b0;")
        else:
            self.autotune_grid_preview_label.setStyleSheet("color: #ff6b6b;")
            if not valid_ranges:
                lines.append("Range error: check that min <= max and step > 0")

        self.autotune_grid_preview_label.setText("\n".join(lines))

    @staticmethod
    def _parse_csv_values(values: str) -> list[str]:
        return [v.strip().lower() for v in values.split(",") if v.strip()]

    def _bench_help_output(self) -> str:
        script_path = self.project_root / "scripts" / "agent_workload_bench.py"
        cache_key = f"{sys.executable}|{script_path}"

        if cache_key in self._bench_help_cache:
            return self._bench_help_cache[cache_key]

        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
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

    def _server_help_output(self, server_bin: Path) -> str:
        try:
            cache_key = str(server_bin.resolve())
        except Exception:
            cache_key = str(server_bin)

        if cache_key in self._server_help_cache:
            return self._server_help_cache[cache_key]

        try:
            result = subprocess.run(
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
                    spec_type_modes.append(mode)

        allowed_modes = set(spec_type_modes)
        mtp_compatible = model_supports_mtp(model)

        def is_mode_model_compatible(mode: str) -> bool:
            if mode not in {"mtp", "ngram-mtp"}:
                return True
            return mtp_compatible

        if requested and requested not in {"auto", "all"}:
            ordered: list[str] = []
            seen: set[str] = set()
            for value in self._parse_csv_values(requested):
                if allowed_modes and value not in allowed_modes:
                    continue
                if not is_mode_model_compatible(value):
                    continue
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
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
        if "Aggregate completion TPS" in line or line.startswith("BEST:") or line.startswith("CURRENT BEST:"):
            self.status_label.setText(line)
        if line.startswith("CONFIG FAILED ("):
            self.status_label.setText(line)

    def _on_bench_finished(self, success: bool, stopped: bool = False):
        self._set_running_state(False)

        if stopped:
            if self._current_mode == "autotune" and self._autotune_active_run is not None:
                stopped_row = self._find_autotune_history_row(self._autotune_active_run)
                if stopped_row is not None:
                    self.autotune_history_table.setItem(stopped_row, 8, QTableWidgetItem("stopped"))
                self._autotune_active_run = None
            self.status_label.setText("Benchmark stopped")
            self.log_output.append("[INFO] Benchmark/autotune stopped by user")
            if self._current_mode == "autotune":
                self.refresh_saved_presets_table()
            QMessageBox.information(self, "Bench", "Run stopped. Completed autotune configs can be resumed.")
            self.bench_thread = None
            return

        if success:
            self.status_label.setText("Benchmark completed")
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
                    self.autotune_history_table.setItem(failed_row, 8, QTableWidgetItem("failed"))
                self._autotune_active_run = None
            if self._current_mode == "autotune":
                self.refresh_saved_presets_table()
            self.status_label.setText("Benchmark failed")
            QMessageBox.warning(self, "Bench", "Benchmark/autotune failed. Check log output.")

        self.bench_thread = None

    def _set_running_state(self, running: bool):
        self.run_bench_btn.setEnabled(not running)
        self.run_autotune_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.model_browse_btn.setEnabled(not running)
        self.model_refresh_btn.setEnabled(not running)
        self.apply_history_preset_btn.setEnabled(not running)
        self.delete_history_run_btn.setEnabled(not running)
        self.refresh_history_btn.setEnabled(not running)
        self.open_history_log_btn.setEnabled(not running)
        self.copy_history_log_btn.setEnabled(not running)
        self.copy_history_row_btn.setEnabled(not running)
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

    def _live_key_for_current_profile(self) -> str:
        profile_bucket = "ctx130k-only"
        model_name = self._last_selected_model or Path(self.model_path_input.text().strip() or "model.gguf").name
        return f"{model_name}::{profile_bucket}"

    def _update_live_best_from_line(self, line: str) -> None:
        # Expected format:
        # CURRENT BEST: ctx=131072 b=2048 ub=512 kv=q4_0 spec=none extra=base aggregate_tps=1.23
        match = re.search(
            r"ctx=(\d+)\s+b=(\d+)\s+ub=(\d+)\s+kv=([^\s,]+)\s+spec=([^\s,]+)(?:\s+extra=([^\s,]+))?\s+aggregate_tps=([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if not match:
            return

        ctx, batch, ubatch, kv, spec_mode, extra_preset, tps = match.groups()
        kv = kv.strip().rstrip(",;")
        spec_mode = spec_mode.strip().rstrip(",;")
        extra_preset = (extra_preset or "base").strip().rstrip(",;")
        model_name = self._last_selected_model or Path(self.model_path_input.text().strip() or "model.gguf").name
        profile_bucket = "ctx130k-only"
        key = f"{model_name}::{profile_bucket}"
        self._live_best_by_key[key] = {
            "model": model_name,
            "profile": profile_bucket,
            "best_tps": tps,
            "ctx": ctx,
            "batch": batch,
            "ubatch": ubatch,
            "kv_k": kv,
            "kv_v": kv,
            "spec_mode": spec_mode,
            "extra_preset": extra_preset or "base",
            "extra_args": "",
            "build_id": self._current_build_id,
            "run_id": "",
            "label": "",
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.refresh_saved_presets_table()

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

            run_label = f"{run_idx}/{total}"
            row = self.autotune_history_table.rowCount()
            self.autotune_history_table.insertRow(row)
            self.autotune_history_table.setItem(row, 0, NumericTableWidgetItem(run_label, run_idx))
            self.autotune_history_table.setItem(row, 1, NumericTableWidgetItem(str(ctx), ctx))
            self.autotune_history_table.setItem(row, 2, NumericTableWidgetItem(str(batch), batch))
            self.autotune_history_table.setItem(row, 3, NumericTableWidgetItem(str(ubatch), ubatch))
            self.autotune_history_table.setItem(row, 4, QTableWidgetItem(kv))
            self.autotune_history_table.setItem(row, 5, QTableWidgetItem(spec_mode))
            self.autotune_history_table.setItem(row, 6, QTableWidgetItem(extra_value))
            self.autotune_history_table.setItem(row, 7, QTableWidgetItem("-"))
            self.autotune_history_table.setItem(row, 8, QTableWidgetItem("running"))
            self._autotune_active_run = run_label
            self.autotune_history_table.scrollToBottom()
            self.status_label.setText(line)
            return

        if self._autotune_active_run is None:
            return

        active_row = self._find_autotune_history_row(self._autotune_active_run)
        if active_row is None:
            return

        tps_match = re.search(r"Aggregate completion TPS by wall time:\s*([0-9]+(?:\.[0-9]+)?)", line)
        if tps_match:
            tps_value = float(tps_match.group(1))
            self.autotune_history_table.setItem(active_row, 7, NumericTableWidgetItem(f"{tps_value:.2f}", tps_value))
            self.autotune_history_table.setItem(active_row, 8, QTableWidgetItem("done"))
            self._autotune_active_run = None
            return

        if line.strip().startswith("error:"):
            self.autotune_history_table.setItem(active_row, 8, QTableWidgetItem("error"))
            self._autotune_active_run = None
            return

        if line.startswith("CONFIG FAILED ("):
            self.autotune_history_table.setItem(active_row, 8, QTableWidgetItem("failed"))
            self._autotune_active_run = None

    def _load_best_presets(self) -> dict[str, dict[str, str]]:
        if not self.best_presets_path.exists():
            return {}
        try:
            data = json.loads(self.best_presets_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            return {}
        return {}

    def _save_best_presets(self, presets: dict[str, dict[str, str]]) -> None:
        self.best_presets_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_presets_path.write_text(json.dumps(presets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _update_best_preset_for_model(self, model_name: str, profile_key: str = "model-best", min_ctx: int | None = None) -> None:
        if not model_name or not self.history_csv.exists():
            return

        best_row = None
        best_tps = -1.0
        try:
            with self.history_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("errors", "0")) not in ("", "0"):
                        continue
                    row_model = Path(str(row.get("model", ""))).name
                    if row_model.lower() != model_name.lower():
                        continue
                    if min_ctx is not None:
                        try:
                            row_ctx = int(str(row.get("ctx", "0") or "0"))
                        except ValueError:
                            row_ctx = 0
                        if row_ctx < min_ctx:
                            continue
                    try:
                        tps = float(str(row.get("aggregate_tps", "0") or "0"))
                    except ValueError:
                        tps = 0.0
                    if tps > best_tps:
                        best_tps = tps
                        best_row = row
        except Exception:
            return

        if not best_row:
            return

        presets = self._load_best_presets()
        store_key = model_name if profile_key == "model-best" else f"{model_name}::{profile_key}"
        presets[store_key] = {
            "model": model_name,
            "profile": profile_key,
            "best_tps": f"{best_tps:.4f}",
            "ctx": str(best_row.get("ctx", "")),
            "batch": str(best_row.get("batch", "")),
            "ubatch": str(best_row.get("ubatch", "")),
            "kv_k": str(best_row.get("kv_k", "")),
            "kv_v": str(best_row.get("kv_v", "")),
            "spec_mode": str(best_row.get("spec_mode", "")),
            "extra_preset": str(best_row.get("extra_preset", "")),
            "extra_args": str(best_row.get("extra_args", "")),
            "build_id": str(best_row.get("build_id", "")),
            "run_id": str(best_row.get("run_id", "")),
            "label": str(best_row.get("label", "")),
            "timestamp": str(best_row.get("timestamp", "")),
        }
        self._save_best_presets(presets)
        self.refresh_saved_presets_table()

    @staticmethod
    def _parse_best_config_text(best_config: str) -> dict[str, str]:
        parsed = {
            "ctx": "-",
            "batch": "-",
            "ubatch": "-",
            "kv": "-",
            "spec": "-",
            "extra_preset": "-",
            "extra_args": "-",
        }

        text = best_config.strip()
        if not text:
            return parsed

        match = re.search(
            r"ctx=(\d+)\s+b=(\d+)\s+ub=(\d+)\s+kv=([^\s,]+)\s+spec=([^\s,]+)(?:\s+extra=([^\s,]+))?(?:\s+extra_args=(.*))?$",
            text,
        )
        if not match:
            return parsed

        ctx, batch, ubatch, kv, spec_mode, extra_preset, extra_args = match.groups()
        parsed["ctx"] = ctx
        parsed["batch"] = batch
        parsed["ubatch"] = ubatch
        parsed["kv"] = kv.strip().rstrip(",;")
        parsed["spec"] = spec_mode.strip().rstrip(",;")
        parsed["extra_preset"] = (extra_preset or "base").strip().rstrip(",;")

        extra_args_text = (extra_args or "").strip()
        parsed["extra_args"] = "-" if not extra_args_text or extra_args_text == "<none>" else extra_args_text
        return parsed

    def _load_autotune_history_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen_keys: set[str] = set()
        history_candidates = [self.history_csv_v2, self.history_csv]

        for history_csv in history_candidates:
            if not history_csv.exists():
                continue
            try:
                with history_csv.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if str(row.get("mode", "")).strip().lower() != "autotune":
                            continue
                        run_id = str(row.get("run_id", "")).strip()
                        timestamp = str(row.get("timestamp", "")).strip()
                        label = str(row.get("label", "")).strip()
                        key = f"{timestamp}::{run_id}::{label}"
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        rows.append({str(k): str(v) for k, v in row.items()})
            except Exception:
                continue

        rows.sort(key=lambda item: (item.get("timestamp", ""), item.get("run_id", "")), reverse=True)
        return rows

    def _selected_history_row_data(self) -> dict[str, str] | None:
        row = self.presets_table.currentRow()
        if row < 0:
            return None

        run_time_item = self.presets_table.item(row, 0)
        model_item = self.presets_table.item(row, 1)
        run_id_item = self.presets_table.item(row, 12)
        label_item = self.presets_table.item(row, 13)

        run_time = run_time_item.text().strip() if run_time_item is not None else ""
        model_name = model_item.text().strip() if model_item is not None else ""
        run_id = run_id_item.text().strip() if run_id_item is not None else ""
        label = label_item.text().strip() if label_item is not None else ""

        for history_row in self._load_autotune_history_rows():
            history_run_id = str(history_row.get("run_id", "")).strip()
            history_time = str(history_row.get("timestamp", "")).strip()
            history_label = str(history_row.get("label", "")).strip()
            history_model = Path(str(history_row.get("model", "")).strip()).name

            if run_id and run_id != "-" and history_run_id and history_run_id == run_id:
                return history_row
            if (
                history_time == run_time
                and history_label == label
                and history_model.lower() == model_name.lower()
            ):
                return history_row

        return None

    def _extract_sweep_sets_from_summary(self, row_data: dict[str, str]) -> tuple[str, str]:
        summary_file = str(row_data.get("summary_file", "")).strip()
        if not summary_file:
            return "-", "-"

        if summary_file in self._summary_sweep_cache:
            return self._summary_sweep_cache[summary_file]

        summary_path = self.project_root / "build_logs" / "agent-workload" / summary_file
        if not summary_path.exists():
            self._summary_sweep_cache[summary_file] = ("-", "-")
            return "-", "-"

        spec_values: list[str] = []
        extra_values: list[str] = []
        seen_specs: set[str] = set()
        seen_extras: set[str] = set()

        try:
            with summary_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    spec = self._sanitize_compact_token(str(row.get("spec_mode", "")), fallback="")
                    extra = self._sanitize_compact_token(str(row.get("extra_preset", "")), fallback="")

                    if spec and spec not in seen_specs:
                        seen_specs.add(spec)
                        spec_values.append(spec)
                    if extra and extra not in seen_extras:
                        seen_extras.add(extra)
                        extra_values.append(extra)
        except Exception:
            self._summary_sweep_cache[summary_file] = ("-", "-")
            return "-", "-"

        specs_text = ",".join(spec_values) if spec_values else "-"
        extras_text = ",".join(extra_values) if extra_values else "-"
        self._summary_sweep_cache[summary_file] = (specs_text, extras_text)
        return specs_text, extras_text

    @staticmethod
    def _kv_cache_index_from_name(kv_name: str) -> int:
        kv_map = {
            "f16": 0,
            "bf16": 1,
            "f32": 2,
            "q8_0": 3,
            "q5_1": 4,
            "q5_0": 5,
            "q4_1": 6,
            "q4_0": 7,
            "iq4_nl": 8,
            "tbq4_0": 9,
            "tbq3_0": 10,
            "tq3_0": 11,
            "turbo4": 12,
            "turbo4_0": 12,
            "turbo3": 13,
            "turbo3_0": 13,
            "turbo2": 14,
            "turbo2_0": 14,
        }
        return kv_map.get(kv_name.strip().lower(), 3)

    @staticmethod
    def _bool_from_history(value: str, default: bool = True) -> bool:
        text = str(value or "").strip().lower()
        if text in {"on", "true", "1", "yes", "y"}:
            return True
        if text in {"off", "false", "0", "no", "n"}:
            return False
        return default

    @staticmethod
    def _int_from_history(value: str, default: int) -> int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _delete_run_from_history_file(self, history_path: Path, target_row: dict[str, str]) -> int:
        if not history_path.exists():
            return 0

        target_run_id = str(target_row.get("run_id", "")).strip()
        target_time = str(target_row.get("timestamp", "")).strip()
        target_label = str(target_row.get("label", "")).strip()
        target_model = Path(str(target_row.get("model", "")).strip()).name.lower()

        try:
            with history_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                rows = [dict(row) for row in reader]
        except Exception:
            return 0

        if not fieldnames:
            return 0

        filtered_rows: list[dict[str, str]] = []
        removed = 0
        for row in rows:
            row_run_id = str(row.get("run_id", "")).strip()
            row_time = str(row.get("timestamp", "")).strip()
            row_label = str(row.get("label", "")).strip()
            row_model = Path(str(row.get("model", "")).strip()).name.lower()

            matched = False
            if target_run_id and row_run_id and row_run_id == target_run_id:
                matched = True
            elif (
                row_time == target_time
                and row_label == target_label
                and row_model == target_model
                and str(row.get("mode", "")).strip().lower() == "autotune"
            ):
                matched = True

            if matched:
                removed += 1
                continue
            filtered_rows.append(row)

        if removed <= 0:
            return 0

        try:
            with history_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
        except Exception:
            return 0

        return removed

    def _resolve_history_model_path(self, row_data: dict[str, str]) -> Path | None:
        model_raw = str(row_data.get("model", "")).strip()
        if model_raw:
            direct_path = Path(model_raw)
            if direct_path.exists():
                return direct_path

            relative_path = self.project_root / model_raw
            if relative_path.exists():
                return relative_path

        model_name = Path(model_raw).name.strip()
        if model_name:
            fallback = self.models_dir / model_name
            if fallback.exists():
                return fallback

        return None

    def _resolve_history_artifact_path(self, artifact_value: str) -> Path | None:
        value = str(artifact_value or "").strip()
        if not value:
            return None

        candidate = Path(value)
        if candidate.exists():
            return candidate

        fallback_paths = [
            self.project_root / value,
            self.project_root / "build_logs" / "agent-workload" / value,
        ]
        for path in fallback_paths:
            if path.exists():
                return path

        return None

    def _history_log_candidates(self, row_data: dict[str, str]) -> list[str]:
        candidates: list[str] = []

        def add_candidate(value: str) -> None:
            text = str(value or "").strip()
            if not text or text in {"-", "<none>"}:
                return
            if text not in candidates:
                candidates.append(text)

        add_candidate(str(row_data.get("server_log_file", "")))
        add_candidate(str(row_data.get("server_log", "")))

        summary_file = str(row_data.get("summary_file", "")).strip()
        summary_suffix = "-autotune-summary.csv"
        if summary_file.endswith(summary_suffix):
            add_candidate(summary_file[: -len(summary_suffix)] + ".server.log")

        csv_file = str(row_data.get("csv_file", "")).strip()
        if csv_file.lower().endswith(".csv"):
            add_candidate(str(Path(csv_file).with_suffix(".server.log")))

        jsonl_file = str(row_data.get("jsonl_file", "")).strip()
        if jsonl_file.lower().endswith(".jsonl"):
            add_candidate(str(Path(jsonl_file).with_suffix(".server.log")))

        label = str(row_data.get("label", "")).strip()
        if label:
            add_candidate(f"{label}.server.log")

        return candidates

    def _discover_history_log_variants(self, log_value: str) -> list[Path]:
        value = str(log_value or "").strip()
        if not value.lower().endswith(".server.log"):
            return []

        file_name = Path(value).name
        base_name = file_name[: -len(".server.log")]
        if not base_name:
            return []

        roots = [
            self.project_root / "build_logs" / "agent-workload",
            self.project_root,
        ]
        patterns = [
            f"{base_name}.server.log",
            f"{base_name}-cfg*.server.log",
            f"{base_name}*.server.log",
        ]

        matches: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for pattern in patterns:
                for found in root.glob(pattern):
                    if not found.is_file():
                        continue
                    key = str(found)
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(found)

        matches.sort(
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        return matches

    def _resolve_history_log_paths(self, row_data: dict[str, str]) -> tuple[list[Path], list[str]]:
        candidates = self._history_log_candidates(row_data)
        resolved: list[Path] = []
        seen_paths: set[str] = set()

        for log_value in candidates:
            log_path = self._resolve_history_artifact_path(log_value)
            if log_path is not None:
                path_key = str(log_path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    resolved.append(log_path)

            for variant in self._discover_history_log_variants(log_value):
                path_key = str(variant)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)
                resolved.append(variant)

        return resolved, candidates

    @staticmethod
    def _clipboard_set_text(text: str) -> bool:
        clipboard = QApplication.clipboard()
        if clipboard is None:
            return False
        clipboard.setText(text)
        return True

    @staticmethod
    def _preview_candidates(candidates: list[str], limit: int = 8) -> str:
        if not candidates:
            return "(none)"
        preview = candidates[:limit]
        if len(candidates) > limit:
            preview.append("...")
        return "\n".join(preview)

    def open_selected_history_log(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Open Log", "Select a run row in history first.")
            return

        log_paths, candidates = self._resolve_history_log_paths(row_data)
        if not log_paths:
            QMessageBox.warning(
                self,
                "Open Log",
                "Log file not found for selected run.\n\nTried:\n"
                + self._preview_candidates(candidates),
            )
            return

        log_path = log_paths[0]

        try:
            if os.name == "nt":
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(log_path)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Log", f"Failed to open log file:\n{exc}")
            return

        self.status_label.setText(f"Opened log: {log_path.name}")

    def copy_selected_history_log_to_clipboard(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Copy Log", "Select a run row in history first.")
            return

        log_paths, candidates = self._resolve_history_log_paths(row_data)
        if not log_paths:
            QMessageBox.warning(
                self,
                "Copy Log",
                "Log file not found for selected run.\n\nTried:\n"
                + self._preview_candidates(candidates),
            )
            return

        text_chunks: list[str] = []
        if len(log_paths) == 1:
            try:
                log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                QMessageBox.warning(self, "Copy Log", f"Failed to read log file:\n{exc}")
                return
        else:
            for path in log_paths:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                text_chunks.append(f"===== {path.name} =====\n{content}")
            if not text_chunks:
                QMessageBox.warning(self, "Copy Log", "No readable log files found for selected run.")
                return
            log_text = "\n\n".join(text_chunks)

        if not self._clipboard_set_text(log_text):
            QMessageBox.warning(self, "Copy Log", "Clipboard is not available.")
            return

        if len(log_paths) == 1:
            self.status_label.setText(f"Log copied to clipboard: {log_paths[0].name}")
        else:
            self.status_label.setText(f"Copied {len(log_paths)} logs to clipboard")

    def copy_selected_history_row_to_clipboard(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Copy Run Data", "Select a run row in history first.")
            return

        log_paths, candidates = self._resolve_history_log_paths(row_data)

        preferred_keys = [
            "timestamp",
            "run_id",
            "label",
            "mode",
            "model",
            "aggregate_tps",
            "best_config",
            "ctx",
            "batch",
            "ubatch",
            "kv_k",
            "kv_v",
            "spec_mode",
            "extra_preset",
            "extra_args",
            "tasks",
            "runs",
            "max_tokens",
            "errors",
            "build_id",
            "build_name",
            "build_backend",
            "summary_file",
            "server_log_file",
            "csv_file",
            "jsonl_file",
        ]

        lines: list[str] = ["# Autotune selected run"]
        seen_keys: set[str] = set()
        for key in preferred_keys:
            if key in row_data:
                value = str(row_data.get(key, "")).strip()
                lines.append(f"{key}: {value or '-'}")
                seen_keys.add(key)

        for key in sorted(row_data.keys()):
            if key in seen_keys:
                continue
            value = str(row_data.get(key, "")).strip()
            if value:
                lines.append(f"{key}: {value}")

        lines.append("")
        lines.append("resolved_server_logs:")
        if log_paths:
            for path in log_paths:
                lines.append(f"- {path}")
        else:
            lines.append("- -")
        lines.append("log_candidates:")
        for candidate in candidates:
            lines.append(f"- {candidate}")

        payload = "\n".join(lines).strip() + "\n"
        if not self._clipboard_set_text(payload):
            QMessageBox.warning(self, "Copy Run Data", "Clipboard is not available.")
            return

        self.status_label.setText("Selected run data copied to clipboard")

    def apply_selected_run_as_default_preset(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Apply Preset", "Select a run row in history first.")
            return

        model_raw = str(row_data.get("model", "")).strip()
        model_name = Path(model_raw).name
        if not model_name:
            QMessageBox.warning(self, "Apply Preset", "Selected run does not contain a model name.")
            return

        parsed_cfg = self._parse_best_config_text(str(row_data.get("best_config", "")))

        ctx = self._int_from_history(parsed_cfg["ctx"] if parsed_cfg["ctx"] != "-" else row_data.get("ctx", ""), 0)
        batch = self._int_from_history(parsed_cfg["batch"] if parsed_cfg["batch"] != "-" else row_data.get("batch", ""), 0)
        ubatch = self._int_from_history(parsed_cfg["ubatch"] if parsed_cfg["ubatch"] != "-" else row_data.get("ubatch", ""), 0)

        if ctx <= 0 or batch <= 0 or ubatch <= 0:
            QMessageBox.warning(
                self,
                "Apply Preset",
                "Selected run does not have a valid best configuration (ctx/batch/ubatch).",
            )
            return

        kv_name = parsed_cfg["kv"] if parsed_cfg["kv"] != "-" else str(row_data.get("kv_k", "")).strip()
        if not kv_name:
            kv_name = str(row_data.get("kv_v", "")).strip()

        spec_mode = parsed_cfg["spec"] if parsed_cfg["spec"] != "-" else str(row_data.get("spec_mode", "")).strip()
        extra_preset = parsed_cfg["extra_preset"] if parsed_cfg["extra_preset"] != "-" else str(row_data.get("extra_preset", "")).strip()
        extra_args = parsed_cfg["extra_args"]
        if extra_args == "-":
            extra_args = str(row_data.get("extra_args", "")).strip()
        if extra_args == "<none>":
            extra_args = ""

        spec_mode = spec_mode.strip().lower()
        if spec_mode and spec_mode not in {"-", "none", "mixed"} and "--spec-type" not in extra_args:
            extra_args = f"--spec-type {spec_mode}" if not extra_args else f"--spec-type {spec_mode}\n{extra_args}"

        preset_path = self.project_root / "gui" / "model_presets.json"
        if not preset_path.exists():
            QMessageBox.warning(self, "Apply Preset", f"Preset file not found:\n{preset_path}")
            return

        try:
            data = json.loads(preset_path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "Apply Preset", f"Failed to read preset file:\n{exc}")
            return

        presets = data.get("presets")
        if not isinstance(presets, list):
            QMessageBox.warning(self, "Apply Preset", "Invalid model_presets.json format (missing presets array).")
            return

        run_id = str(row_data.get("run_id", "")).strip() or "-"
        run_time = str(row_data.get("timestamp", "")).strip() or "-"
        aggregate_tps = str(row_data.get("aggregate_tps", "")).strip() or "0"

        default_name = f"History Default {model_name}"
        default_preset = {
            "pattern": re.escape(model_name),
            "name": default_name,
            "ctx": ctx,
            "batch_size": batch,
            "ubatch_size": ubatch,
            "gpu_layers": self._int_from_history(row_data.get("gpu_layers", ""), 99),
            "parallel": self._int_from_history(row_data.get("parallel", ""), 1),
            "flash_attn": self._bool_from_history(row_data.get("flash_attn", ""), default=True),
            "kv_cache": self._kv_cache_index_from_name(kv_name),
            "notes": (
                "Applied from Autotune Runs History: "
                f"run_id={run_id}, tps={aggregate_tps}, time={run_time}, "
                f"spec={spec_mode or '-'}, extra={extra_preset or '-'}"
            ),
        }
        if extra_args:
            default_preset["extra_args"] = extra_args

        filtered_presets = [
            item
            for item in presets
            if not (isinstance(item, dict) and str(item.get("name", "")) == default_name)
        ]
        filtered_presets.insert(0, default_preset)
        data["presets"] = filtered_presets

        try:
            preset_path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Apply Preset", f"Failed to write preset file:\n{exc}")
            return

        applied_live = False
        if hasattr(self.parent, "server_tab") and hasattr(self.parent.server_tab, "apply_model_file_preset"):
            model_path = self._resolve_history_model_path(row_data)
            if model_path is not None and hasattr(self.parent.server_tab, "server_model_path"):
                self.parent.server_tab.server_model_path.setText(str(model_path))
            apply_result = self.parent.server_tab.apply_model_file_preset()
            applied_live = bool(isinstance(apply_result, dict) and apply_result.get("matched"))

        self.status_label.setText(f"Default preset applied from run: {run_id}")
        QMessageBox.information(
            self,
            "Apply Preset",
            "Selected run preset saved as default in gui/model_presets.json"
            + (" and applied to Launch Server tab." if applied_live else "."),
        )

    def refresh_saved_presets_table(self):
        rows = self._load_autotune_history_rows()
        self._summary_sweep_cache.clear()
        self.presets_table.setRowCount(0)

        for row_data in rows:
            run_time = str(row_data.get("timestamp", "") or "-")
            model_raw = str(row_data.get("model", "") or "-")
            model_name = Path(model_raw).name if model_raw not in {"", "-"} else "-"
            run_id = str(row_data.get("run_id", "") or "-")
            build_id = str(row_data.get("build_id", "") or "-")
            label = str(row_data.get("label", "") or "-")

            aggregate_text = str(row_data.get("aggregate_tps", "0") or "0")
            try:
                aggregate_value = float(aggregate_text)
            except ValueError:
                aggregate_value = 0.0

            parsed_cfg = self._parse_best_config_text(str(row_data.get("best_config", "")))
            if parsed_cfg["ctx"] == "-":
                parsed_cfg["ctx"] = str(row_data.get("ctx", "") or "-")
            if parsed_cfg["spec"] == "-":
                parsed_cfg["spec"] = str(row_data.get("spec_mode", "") or "-")
            if parsed_cfg["extra_preset"] == "-":
                parsed_cfg["extra_preset"] = str(row_data.get("extra_preset", "") or "-")
            if parsed_cfg["extra_args"] == "-":
                fallback_args = str(row_data.get("extra_args", "") or "").strip()
                if fallback_args:
                    parsed_cfg["extra_args"] = fallback_args

            parsed_cfg["spec"] = self._sanitize_compact_token(parsed_cfg["spec"])
            parsed_cfg["extra_preset"] = self._sanitize_compact_token(parsed_cfg["extra_preset"], fallback="base")

            swept_specs, swept_extras = self._extract_sweep_sets_from_summary(row_data)

            row = self.presets_table.rowCount()
            self.presets_table.insertRow(row)

            run_item = QTableWidgetItem(run_time)
            run_item.setData(Qt.ItemDataRole.UserRole, run_id)
            self.presets_table.setItem(row, 0, run_item)
            self.presets_table.setItem(row, 1, QTableWidgetItem(model_name or "-"))
            self.presets_table.setItem(row, 2, NumericTableWidgetItem(f"{aggregate_value:.4f}", aggregate_value))
            self.presets_table.setItem(row, 3, QTableWidgetItem(parsed_cfg["ctx"]))
            self.presets_table.setItem(row, 4, QTableWidgetItem(f"{parsed_cfg['batch']}/{parsed_cfg['ubatch']}"))
            self.presets_table.setItem(row, 5, QTableWidgetItem(parsed_cfg["kv"]))
            self.presets_table.setItem(row, 6, QTableWidgetItem(parsed_cfg["spec"]))
            self.presets_table.setItem(row, 7, QTableWidgetItem(parsed_cfg["extra_preset"]))
            self.presets_table.setItem(row, 8, QTableWidgetItem(parsed_cfg["extra_args"]))
            self.presets_table.setItem(row, 9, QTableWidgetItem(swept_specs))
            self.presets_table.setItem(row, 10, QTableWidgetItem(swept_extras))
            self.presets_table.setItem(row, 11, QTableWidgetItem(build_id or "-"))
            self.presets_table.setItem(row, 12, QTableWidgetItem(run_id or "-"))
            self.presets_table.setItem(row, 13, QTableWidgetItem(label or "-"))

    def delete_selected_preset(self) -> None:
        row_data = self._selected_history_row_data()
        if row_data is None:
            QMessageBox.warning(self, "Delete Run", "Select a run row in history first.")
            return

        model_name = Path(str(row_data.get("model", "")).strip()).name or "-"
        run_id = str(row_data.get("run_id", "")).strip() or "-"
        run_time = str(row_data.get("timestamp", "")).strip() or "-"
        label = str(row_data.get("label", "")).strip() or "-"

        confirm = QMessageBox.question(
            self,
            "Delete Run",
            f"Delete selected autotune run?\n\nModel: {model_name}\nRun ID: {run_id}\nTime: {run_time}\nLabel: {label}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        removed_total = 0
        removed_total += self._delete_run_from_history_file(self.history_csv_v2, row_data)
        removed_total += self._delete_run_from_history_file(self.history_csv, row_data)

        self.refresh_saved_presets_table()
        if removed_total > 0:
            self.status_label.setText(f"Run deleted from history: {run_id}")
        else:
            self.status_label.setText("Selected run was not found in history files")
            QMessageBox.warning(self, "Delete Run", "Selected run was not found in BENCH_HISTORY files.")

    def open_history_md(self):
        history_md = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.md"
        if not history_md.exists():
            QMessageBox.warning(self, "History", "BENCH_HISTORY.md not found yet")
            return

        if os.name == "nt":
            os.startfile(str(history_md))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(history_md)])
