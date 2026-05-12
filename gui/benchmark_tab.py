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
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class BenchCommandThread(QThread):
    """Run benchmark command in background so UI remains responsive."""

    output = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, bool)

    def __init__(self, command: list[str], working_dir: Path):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
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
            process = subprocess.Popen(
                self.command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
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

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.models_dir = parent.models_dir if hasattr(parent, "models_dir") else Path("models")
        self.project_root = parent.project_root if hasattr(parent, "project_root") else Path.cwd()
        self.history_csv = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.csv"
        self.history_csv_v2 = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY_V2.csv"
        self.best_presets_path = self.project_root / "gui" / "model_autotune_best.json"
        self.bench_thread = None
        self._version_payloads: dict[str, dict[str, object]] = {}
        self._current_mode = "single"
        self._last_selected_model = ""
        self._current_autotune_profile = "ctx32k-only"
        self._current_build_id = ""
        self._live_best_by_key: dict[str, dict[str, str]] = {}
        self._autotune_result = {"best": "", "summary_json": "", "summary_csv": ""}
        self._autotune_active_run: str | None = None
        self.create_ui()
        self.refresh_models_list()
        self.refresh_build_choices()
        self.refresh_saved_presets_table()

    def create_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel("📈 Bench & Autotune - dedicated benchmark workflows")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        model_row = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to GGUF model...")
        model_row.addWidget(self.model_path_input)

        self.model_browse_btn = QPushButton("📂 Browse")
        self.model_browse_btn.clicked.connect(self.browse_model)
        model_row.addWidget(self.model_browse_btn)
        model_layout.addLayout(model_row)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Detected models:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_selected)
        combo_row.addWidget(self.model_combo)

        self.model_refresh_btn = QPushButton("🔄 Refresh")
        self.model_refresh_btn.clicked.connect(self.refresh_models_list)
        combo_row.addWidget(self.model_refresh_btn)
        model_layout.addLayout(combo_row)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        build_group = QGroupBox("Build Target")
        build_layout = QHBoxLayout()
        build_layout.addWidget(QLabel("Backend:"))
        self.build_backend_combo = QComboBox()
        self.build_backend_combo.currentTextChanged.connect(self._on_backend_changed)
        build_layout.addWidget(self.build_backend_combo)

        build_layout.addWidget(QLabel("Version:"))
        self.build_version_combo = QComboBox()
        build_layout.addWidget(self.build_version_combo)
        build_group.setLayout(build_layout)
        layout.addWidget(build_group)

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()

        single_group = QGroupBox("Single Benchmark (used by Run Benchmark)")
        single_layout = QVBoxLayout()

        single_row1 = QHBoxLayout()
        single_row1.addWidget(QLabel("Tasks:"))
        self.tasks_combo = QComboBox()
        self.tasks_combo.addItems(["v2-mini", "v2", "quick", "full"])
        self.tasks_combo.setCurrentText("v2-mini")
        single_row1.addWidget(self.tasks_combo)

        single_row1.addWidget(QLabel("Runs:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setMinimum(1)
        self.runs_spin.setMaximum(10)
        self.runs_spin.setValue(1)
        single_row1.addWidget(self.runs_spin)

        single_row1.addWidget(QLabel("Spec:"))
        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["none", "ngram-mod", "mtp"])
        self.spec_combo.setCurrentText("none")
        single_row1.addWidget(self.spec_combo)
        single_layout.addLayout(single_row1)

        single_row2 = QHBoxLayout()
        single_row2.addWidget(QLabel("Ctx:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setMinimum(8192)
        self.ctx_spin.setMaximum(131072)
        self.ctx_spin.setValue(65536)
        self.ctx_spin.setSingleStep(8192)
        single_row2.addWidget(self.ctx_spin)

        single_row2.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(32)
        self.batch_spin.setMaximum(8192)
        self.batch_spin.setValue(2048)
        self.batch_spin.setSingleStep(32)
        single_row2.addWidget(self.batch_spin)

        single_row2.addWidget(QLabel("UBatch:"))
        self.ubatch_spin = QSpinBox()
        self.ubatch_spin.setMinimum(32)
        self.ubatch_spin.setMaximum(8192)
        self.ubatch_spin.setValue(512)
        self.ubatch_spin.setSingleStep(32)
        single_row2.addWidget(self.ubatch_spin)
        single_layout.addLayout(single_row2)

        single_row3 = QHBoxLayout()
        single_row3.addWidget(QLabel("KV K/V:"))
        self.kv_combo = QComboBox()
        self.kv_combo.addItems(["q8_0", "q4_0", "f16", "bf16", "f32"])
        self.kv_combo.setCurrentText("q4_0")
        single_row3.addWidget(self.kv_combo)

        single_row3.addWidget(QLabel("Max tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setMinimum(8)
        self.max_tokens_spin.setMaximum(1024)
        self.max_tokens_spin.setValue(80)
        single_row3.addWidget(self.max_tokens_spin)
        single_row3.addStretch(1)
        single_layout.addLayout(single_row3)

        single_group.setLayout(single_layout)
        params_layout.addWidget(single_group)

        autotune_group = QGroupBox("Auto-tune Grid (used by Run Auto-tune 32K)")
        autotune_layout = QVBoxLayout()

        autotune_mode_info = QLabel(
            "Fixed mode: ctx=32768, tasks=v2-mini, runs=1, prompt-heavy repo-snapshot, "
            "no-reuse, no-prime, thinking on, request/task timeout=20s"
        )
        autotune_mode_info.setStyleSheet("color: #b0b0b0;")
        autotune_layout.addWidget(autotune_mode_info)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Batch min:"))
        self.at_batch_min_spin = QSpinBox()
        self.at_batch_min_spin.setMinimum(32)
        self.at_batch_min_spin.setMaximum(8192)
        self.at_batch_min_spin.setValue(2048)
        self.at_batch_min_spin.setSingleStep(32)
        self.at_batch_min_spin.setToolTip("Minimal batch value in sweep (>= 32)")
        row4.addWidget(self.at_batch_min_spin)

        row4.addWidget(QLabel("Batch max:"))
        self.at_batch_max_spin = QSpinBox()
        self.at_batch_max_spin.setMinimum(32)
        self.at_batch_max_spin.setMaximum(8192)
        self.at_batch_max_spin.setValue(8192)
        self.at_batch_max_spin.setSingleStep(32)
        self.at_batch_max_spin.setToolTip("Maximal batch value in sweep")
        row4.addWidget(self.at_batch_max_spin)

        row4.addWidget(QLabel("Batch step:"))
        self.at_batch_step_spin = QSpinBox()
        self.at_batch_step_spin.setMinimum(1)
        self.at_batch_step_spin.setMaximum(8192)
        self.at_batch_step_spin.setValue(2048)
        self.at_batch_step_spin.setSingleStep(1)
        self.at_batch_step_spin.setToolTip("Increment for batch range")
        row4.addWidget(self.at_batch_step_spin)
        row4.addStretch(1)
        autotune_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("UBatch min:"))
        self.at_ubatch_min_spin = QSpinBox()
        self.at_ubatch_min_spin.setMinimum(32)
        self.at_ubatch_min_spin.setMaximum(8192)
        self.at_ubatch_min_spin.setValue(128)
        self.at_ubatch_min_spin.setSingleStep(32)
        self.at_ubatch_min_spin.setToolTip("Minimal ubatch value in sweep (>= 32)")
        row5.addWidget(self.at_ubatch_min_spin)

        row5.addWidget(QLabel("UBatch max:"))
        self.at_ubatch_max_spin = QSpinBox()
        self.at_ubatch_max_spin.setMinimum(32)
        self.at_ubatch_max_spin.setMaximum(8192)
        self.at_ubatch_max_spin.setValue(512)
        self.at_ubatch_max_spin.setSingleStep(32)
        self.at_ubatch_max_spin.setToolTip("Maximal ubatch value in sweep")
        row5.addWidget(self.at_ubatch_max_spin)

        row5.addWidget(QLabel("UBatch step:"))
        self.at_ubatch_step_spin = QSpinBox()
        self.at_ubatch_step_spin.setMinimum(1)
        self.at_ubatch_step_spin.setMaximum(8192)
        self.at_ubatch_step_spin.setValue(64)
        self.at_ubatch_step_spin.setSingleStep(1)
        self.at_ubatch_step_spin.setToolTip("Increment for ubatch range")
        row5.addWidget(self.at_ubatch_step_spin)
        row5.addStretch(1)
        autotune_layout.addLayout(row5)

        row6 = QHBoxLayout()
        row6.addWidget(QLabel("Spec modes:"))
        self.autotune_spec_values_input = QLineEdit()
        self.autotune_spec_values_input.setPlaceholderText("auto or comma list: none,ngram-mod,draft,eagle3")
        self.autotune_spec_values_input.setText("auto")
        self.autotune_spec_values_input.setToolTip("auto = detect supported modes from llama-server --help")
        row6.addWidget(self.autotune_spec_values_input)

        row6.addWidget(QLabel("Extra presets:"))
        self.autotune_extra_presets_input = QLineEdit()
        self.autotune_extra_presets_input.setPlaceholderText("base||shape::--foo bar||--threads 8")
        self.autotune_extra_presets_input.setText("base")
        self.autotune_extra_presets_input.setToolTip("Split presets with || ; format name::args or base")
        row6.addWidget(self.autotune_extra_presets_input)
        autotune_layout.addLayout(row6)

        row7 = QHBoxLayout()
        self.autotune_resume_checkbox = QCheckBox("Resume unfinished session")
        self.autotune_resume_checkbox.setChecked(True)
        self.autotune_resume_checkbox.setToolTip("Continue from saved autotune progress if a previous run was interrupted")
        row7.addWidget(self.autotune_resume_checkbox)

        self.autotune_reset_session_checkbox = QCheckBox("Reset saved session before run")
        self.autotune_reset_session_checkbox.setChecked(False)
        self.autotune_reset_session_checkbox.setToolTip("Ignore and overwrite previous session checkpoint for this model/profile")
        row7.addWidget(self.autotune_reset_session_checkbox)
        row7.addStretch(1)
        autotune_layout.addLayout(row7)

        self.autotune_grid_preview_label = QLabel("")
        self.autotune_grid_preview_label.setWordWrap(True)
        autotune_layout.addWidget(self.autotune_grid_preview_label)

        autotune_group.setLayout(autotune_layout)
        params_layout.addWidget(autotune_group)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        for spin_box in [
            self.at_batch_min_spin,
            self.at_batch_max_spin,
            self.at_batch_step_spin,
            self.at_ubatch_min_spin,
            self.at_ubatch_max_spin,
            self.at_ubatch_step_spin,
        ]:
            spin_box.valueChanged.connect(self._update_autotune_grid_preview)
        self.autotune_spec_values_input.textChanged.connect(self._update_autotune_grid_preview)
        self.autotune_extra_presets_input.textChanged.connect(self._update_autotune_grid_preview)
        self._update_autotune_grid_preview()

        btn_row = QHBoxLayout()
        self.run_bench_btn = QPushButton("⚡ Run Benchmark")
        self.run_bench_btn.clicked.connect(self.run_benchmark)
        btn_row.addWidget(self.run_bench_btn)

        self.run_autotune_btn = QPushButton("🎯 Run Auto-tune 32K (v2-mini x1)")
        self.run_autotune_btn.clicked.connect(self.run_autotune)
        btn_row.addWidget(self.run_autotune_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_current_run)
        btn_row.addWidget(self.stop_btn)

        self.open_history_btn = QPushButton("📄 Open BENCH_HISTORY.md")
        self.open_history_btn.clicked.connect(self.open_history_md)
        btn_row.addWidget(self.open_history_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(140)
        layout.addWidget(self.log_output)

        presets_group = QGroupBox("Autotune Runs History (Best Result Per Run)")
        presets_layout = QVBoxLayout()
        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(12)
        self.presets_table.setHorizontalHeaderLabels([
            "Run Time",
            "Model",
            "Best TPS",
            "Ctx",
            "Batch/UBatch",
            "KV",
            "Spec",
            "Extra preset",
            "Extra args",
            "Build ID",
            "Run ID",
            "Label",
        ])
        self.presets_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.presets_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.presets_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.presets_table.setFixedHeight(200)
        presets_layout.addWidget(self.presets_table)

        presets_actions = QHBoxLayout()
        self.refresh_history_btn = QPushButton("🔄 Refresh Run History")
        self.refresh_history_btn.clicked.connect(self.refresh_saved_presets_table)
        presets_actions.addWidget(self.refresh_history_btn)
        presets_actions.addStretch(1)
        presets_layout.addLayout(presets_actions)

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

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
        self.autotune_history_table.setMinimumHeight(300)
        history_layout.addWidget(self.autotune_history_table)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

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
            "99",
            "--parallel",
            "1",
            "--max-tokens",
            str(self.max_tokens_spin.value()),
            "--startup-timeout",
            "180",
            "--request-timeout",
            "180",
            "--background-server-policy",
            "fail",
            "--server-extra",
            f"--spec-type {self.spec_combo.currentText()}",
        ]

        self._current_mode = "single"
        self._last_selected_model = model.name
        self._current_build_id = build_id
        self._set_running_state(True)
        self.log_output.clear()
        self.log_output.append(f"[INFO] Starting benchmark for {model.name}")
        self.log_output.append(f"[INFO] Build ID: {build_id or '-'}")
        self.bench_thread = BenchCommandThread(command=command, working_dir=self.project_root)
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

        profile_key = "ctx32k-only"
        autotune_min_ctx = 32768
        autotune_ctx_values = "32768"
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
        spec_values = self._resolve_autotune_spec_values(server_bin, model, self.autotune_spec_values_input.text().strip())
        if not spec_values:
            QMessageBox.warning(self, "Auto-tune", "No valid spec modes resolved for autotune.")
            return

        extra_presets = self._parse_autotune_extra_presets(self.autotune_extra_presets_input.text().strip())
        if not extra_presets:
            QMessageBox.warning(self, "Auto-tune", "No valid extra presets resolved for autotune.")
            return

        autotune_extra_presets = "||".join(extra_presets)
        autotune_kv_values = "q8_0,q4_0"
        autotune_tasks = "v2-mini"
        autotune_max_tokens = "120"
        autotune_real_context_chars = "21872"

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
            "99",
            "--parallel",
            "1",
            "--max-tokens",
            autotune_max_tokens,
            "--startup-timeout",
            "180",
            "--request-timeout",
            "20",
            "--task-fail-timeout",
            "20",
            "--background-server-policy",
            "fail",
            "--allow-ctx-above-16k",
            "--real-context-mode",
            "repo-snapshot",
            "--real-context-chars",
            autotune_real_context_chars,
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
            "--autotune-extra-presets",
            autotune_extra_presets,
            "--autotune-max-configs",
            str(max(64, config_count + 8)),
            "--autotune-update-preset",
            "--autotune-preset-file",
            "gui/model_presets.json",
            "--autotune-session-file",
            str(autotune_session_file),
        ]

        if self.autotune_resume_checkbox.isChecked():
            command.append("--autotune-resume")
        else:
            command.append("--no-autotune-resume")

        if self.autotune_reset_session_checkbox.isChecked():
            command.append("--autotune-reset-session")

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
        self.log_output.append(f"[INFO] Autotune profile: 32K fixed, configs: {config_count}")
        self.log_output.append(f"[INFO] Workload: {autotune_tasks}, max_tokens: {autotune_max_tokens}")
        self.log_output.append(
            "[INFO] Lane: prompt-heavy repo-snapshot "
            f"(chars={autotune_real_context_chars}), no-reuse, no-prime, thinking on"
        )
        self.log_output.append(
            "[INFO] Sweep: "
            f"batch={autotune_batch_values} | "
            f"ubatch={autotune_ubatch_values} | "
            f"spec={','.join(spec_values)} | "
            f"extra_presets={len(extra_presets)}"
        )
        self.log_output.append(f"[INFO] Run label: {label}")
        self.log_output.append(f"[INFO] Session file: {autotune_session_file}")
        self.log_output.append(
            "[INFO] Session mode: "
            f"resume={'on' if self.autotune_resume_checkbox.isChecked() else 'off'}, "
            f"reset={'on' if self.autotune_reset_session_checkbox.isChecked() else 'off'}"
        )
        self.log_output.append("[INFO] Task timeout policy: 20s hard cutoff (slow task => config fail)")
        self.bench_thread = BenchCommandThread(command=command, working_dir=self.project_root)
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

        spec_raw = self.autotune_spec_values_input.text().strip().lower()
        if spec_raw in {"", "auto", "all"}:
            spec_values: list[str] = []
            spec_factor: int | None = None
            spec_text = "auto-detect"
        else:
            spec_values = []
            seen: set[str] = set()
            for value in self._parse_csv_values(spec_raw):
                if value in seen:
                    continue
                seen.add(value)
                spec_values.append(value)
            spec_factor = len(spec_values)
            spec_text = ",".join(spec_values) if spec_values else "-"

        extra_presets = self._parse_autotune_extra_presets(self.autotune_extra_presets_input.text().strip())
        extra_count = len(extra_presets)
        extra_preview = ", ".join(extra_presets[:4])
        if len(extra_presets) > 4:
            extra_preview += f", ... ({len(extra_presets)} total)"

        kv_factor = 2  # fixed: q8_0,q4_0 in current autotune flow
        valid_ranges = bool(batch_values) and bool(ubatch_values)
        base_factor = len(batch_values) * len(ubatch_values) * kv_factor * extra_count if valid_ranges else 0
        if spec_factor is None:
            total_text = f"{base_factor} * spec(auto)"
        else:
            total_text = str(base_factor * spec_factor)

        lines = [
            f"Batch values: {self._format_values_preview(batch_values)}",
            f"UBatch values: {self._format_values_preview(ubatch_values)}",
            f"Spec modes: {spec_text}",
            f"Extra presets: {extra_preview or '-'}",
            f"Estimated configs: {total_text} (ctx=1 x kv=2 x batch x ubatch x spec x extra)",
            "Runtime lane: repo-snapshot chars=21872, no-reuse, no-prime, thinking on",
            "Hint: for 32..128 set min=32, max=128, step=32",
        ]

        if valid_ranges:
            self.autotune_grid_preview_label.setStyleSheet("color: #b0b0b0;")
        else:
            self.autotune_grid_preview_label.setStyleSheet("color: #ff6b6b;")
            lines.append("Range error: check that min <= max and step > 0")

        self.autotune_grid_preview_label.setText("\n".join(lines))

    @staticmethod
    def _parse_csv_values(values: str) -> list[str]:
        return [v.strip().lower() for v in values.split(",") if v.strip()]

    @staticmethod
    def _parse_autotune_extra_presets(values: str) -> list[str]:
        normalized = values.strip()
        if not normalized:
            return ["base"]

        chunks = [chunk.strip() for chunk in normalized.split("||") if chunk.strip()]
        if not chunks:
            return ["base"]

        result: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            item = "base" if chunk.lower() in {"base", "default", "none", "off", "-"} else chunk
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result or ["base"]

    @staticmethod
    def _server_help_output(server_bin: Path) -> str:
        try:
            result = subprocess.run(
                [str(server_bin), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
        except Exception:
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

        if requested and requested not in {"auto", "all"}:
            ordered: list[str] = []
            seen: set[str] = set()
            for value in self._parse_csv_values(requested):
                if allowed_modes and value not in allowed_modes:
                    continue
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
            return ordered or ["none"]

        if not spec_type_modes:
            spec_type_modes = ["none", "ngram-mod"]

        supported_order = ["none", "ngram-mod", "mtp", "eagle3", "eagle"]
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

        model_name = model.name.lower()
        if "mtp" in resolved and "mtp" not in model_name and "nextn" not in model_name:
            resolved = [mode for mode in resolved if mode != "mtp"]

        unique: list[str] = []
        seen: set[str] = set()
        for mode in resolved:
            if mode in seen:
                continue
            seen.add(mode)
            unique.append(mode)
        return unique

    @staticmethod
    def _server_supports_mtp(server_bin: Path) -> bool:
        return "mtp" in BenchmarkTabWidget._server_help_output(server_bin)

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
        self.refresh_history_btn.setEnabled(not running)
        self.at_batch_min_spin.setEnabled(not running)
        self.at_batch_max_spin.setEnabled(not running)
        self.at_batch_step_spin.setEnabled(not running)
        self.at_ubatch_min_spin.setEnabled(not running)
        self.at_ubatch_max_spin.setEnabled(not running)
        self.at_ubatch_step_spin.setEnabled(not running)
        self.autotune_spec_values_input.setEnabled(not running)
        self.autotune_extra_presets_input.setEnabled(not running)
        self.autotune_resume_checkbox.setEnabled(not running)
        self.autotune_reset_session_checkbox.setEnabled(not running)

    def _live_key_for_current_profile(self) -> str:
        profile_bucket = "ctx32k-only"
        model_name = self._last_selected_model or Path(self.model_path_input.text().strip() or "model.gguf").name
        return f"{model_name}::{profile_bucket}"

    def _update_live_best_from_line(self, line: str) -> None:
        # Expected format:
        # CURRENT BEST: ctx=65536 b=1024 ub=1024 kv=q8_0 spec=none extra=base aggregate_tps=48.27
        match = re.search(
            r"ctx=(\d+)\s+b=(\d+)\s+ub=(\d+)\s+kv=([^\s]+)\s+spec=([^\s]+)(?:\s+extra=([^\s]+))?\s+aggregate_tps=([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if not match:
            return

        ctx, batch, ubatch, kv, spec_mode, extra_preset, tps = match.groups()
        model_name = self._last_selected_model or Path(self.model_path_input.text().strip() or "model.gguf").name
        profile_bucket = "ctx32k-only"
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

    def _update_autotune_live_history_from_line(self, line: str) -> None:
        start_match = re.search(
            r"Autotune \[(\d+)/(\d+)\]: ctx=(\d+), b=(\d+), ub=(\d+), kv=([^,]+), spec=([^\s]+)(?:, extra=([^\s]+))?",
            line,
        )
        if start_match:
            run_idx, total, ctx, batch, ubatch, kv, spec_mode, extra_preset = start_match.groups()
            extra_value = extra_preset or "base"
            run_label = f"{run_idx}/{total}"
            row = self.autotune_history_table.rowCount()
            self.autotune_history_table.insertRow(row)
            self.autotune_history_table.setItem(row, 0, NumericTableWidgetItem(run_label, int(run_idx)))
            self.autotune_history_table.setItem(row, 1, NumericTableWidgetItem(ctx, int(ctx)))
            self.autotune_history_table.setItem(row, 2, NumericTableWidgetItem(batch, int(batch)))
            self.autotune_history_table.setItem(row, 3, NumericTableWidgetItem(ubatch, int(ubatch)))
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
            r"ctx=(\d+)\s+b=(\d+)\s+ub=(\d+)\s+kv=([^\s]+)\s+spec=([^\s]+)(?:\s+extra=([^\s]+))?(?:\s+extra_args=(.*))?$",
            text,
        )
        if not match:
            return parsed

        ctx, batch, ubatch, kv, spec_mode, extra_preset, extra_args = match.groups()
        parsed["ctx"] = ctx
        parsed["batch"] = batch
        parsed["ubatch"] = ubatch
        parsed["kv"] = kv
        parsed["spec"] = spec_mode
        parsed["extra_preset"] = extra_preset or "base"

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

    def refresh_saved_presets_table(self):
        rows = self._load_autotune_history_rows()
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
            self.presets_table.setItem(row, 9, QTableWidgetItem(build_id or "-"))
            self.presets_table.setItem(row, 10, QTableWidgetItem(run_id or "-"))
            self.presets_table.setItem(row, 11, QTableWidgetItem(label or "-"))

    def delete_selected_preset(self) -> None:
        row = self.presets_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Delete Preset", "Select a preset row first.")
            return

        model_item = self.presets_table.item(row, 0)
        if model_item is None:
            QMessageBox.warning(self, "Delete Preset", "Cannot resolve selected preset.")
            return

        preset_key = str(model_item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not preset_key:
            QMessageBox.warning(self, "Delete Preset", "Cannot resolve preset key.")
            return

        presets = self._load_best_presets()
        target = presets.get(preset_key)
        if target is None:
            QMessageBox.warning(self, "Delete Preset", "Preset is already missing in storage.")
            self._live_best_by_key.pop(preset_key, None)
            self.refresh_saved_presets_table()
            return

        model_name = str(target.get("model", preset_key))
        profile = str(target.get("profile", "model-best"))
        confirm = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset for {model_name} ({profile})?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        presets.pop(preset_key, None)
        self._live_best_by_key.pop(preset_key, None)
        self._save_best_presets(presets)
        self.refresh_saved_presets_table()
        self.status_label.setText(f"Preset deleted: {model_name} ({profile})")

    def open_history_md(self):
        history_md = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.md"
        if not history_md.exists():
            QMessageBox.warning(self, "History", "BENCH_HISTORY.md not found yet")
            return

        if os.name == "nt":
            os.startfile(str(history_md))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(history_md)])
