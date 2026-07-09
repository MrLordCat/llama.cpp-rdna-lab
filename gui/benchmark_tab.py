"""Benchmark tab - dedicated benchmark and autotune workflows."""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
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
from bench_runner import BenchCommandThread
from bench_widgets import (
    NumericTableWidgetItem,
    configure_combo,
    configure_compact_table,
    configure_spinbox,
    create_scroll_panel,
)
from model_capabilities import model_supports_mtp


class BenchmarkTabWidget(BenchHistoryMixin, QWidget):
    """Dedicated Bench & Autotune tab.

    History-table and preset persistence methods live in BenchHistoryMixin
    (bench_history.py); process running in bench_runner.py; shared widget
    helpers in bench_widgets.py.
    """

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

            mode_index = self.settings.value("benchmark/mode_tab", 0, type=int)
            if 0 <= mode_index < self.mode_tabs.count():
                self.mode_tabs.setCurrentIndex(mode_index)

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
            self.settings.setValue("benchmark/mode_tab", self.mode_tabs.currentIndex())
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
        single_layout.setColumnStretch(1, 1)
        single_page_layout.addLayout(single_layout)

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
        single_page_layout.addStretch(1)

        self.mode_tabs.addTab(single_page, "▶ Single Bench")

        autotune_page = QWidget()
        autotune_layout = QVBoxLayout(autotune_page)
        autotune_layout.setContentsMargins(6, 6, 6, 6)
        autotune_layout.setSpacing(8)

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

        self.run_autotune_btn = QPushButton("🔁 Run Auto-tune 130K")
        self.run_autotune_btn.setToolTip("Run the 130K cold repo-snapshot autotune grid")
        self.run_autotune_btn.clicked.connect(self.run_autotune)
        autotune_layout.addWidget(self.run_autotune_btn)
        autotune_layout.addStretch(1)

        self.mode_tabs.addTab(autotune_page, "🔁 Auto-tune 130K")
        left_layout.addWidget(self.mode_tabs)

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
        self.mode_tabs.currentChanged.connect(lambda _index: self.save_settings())
        self._update_autotune_grid_preview()

        shared_btn_row = QHBoxLayout()
        shared_btn_row.setSpacing(8)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setToolTip("Stop the current benchmark or autotune run")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_current_run)
        shared_btn_row.addWidget(self.stop_btn, 1)

        self.open_history_btn = QPushButton("Open History")
        self.open_history_btn.setToolTip("Open build_logs/agent-workload/BENCH_HISTORY.md")
        self.open_history_btn.clicked.connect(self.open_history_md)
        shared_btn_row.addWidget(self.open_history_btn, 1)
        left_layout.addLayout(shared_btn_row)

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
        server_extra = self._active_lane_base_server_extra(self.ctx_spin.value())
        spec_cli_mode = "draft-mtp" if spec_mode == "mtp" else spec_mode
        spec_extra = [f"--spec-type {spec_cli_mode}"]
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

