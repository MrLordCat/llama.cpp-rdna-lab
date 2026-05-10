"""Benchmark tab - dedicated benchmark and autotune workflows."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
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
    finished_signal = pyqtSignal(bool)

    def __init__(self, command: list[str], working_dir: Path):
        super().__init__()
        self.command = command
        self.working_dir = working_dir

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
            )
            for line in process.stdout:
                self.output.emit(line.rstrip())
            process.wait()
            self.finished_signal.emit(process.returncode == 0)
        except Exception as exc:
            self.output.emit(f"Bench error: {exc}")
            self.finished_signal.emit(False)


class BenchmarkTabWidget(QWidget):
    """Dedicated Bench & Autotune tab."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.models_dir = parent.models_dir if hasattr(parent, "models_dir") else Path("models")
        self.project_root = parent.project_root if hasattr(parent, "project_root") else Path.cwd()
        self.history_csv = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.csv"
        self.best_presets_path = self.project_root / "gui" / "model_autotune_best.json"
        self.bench_thread = None
        self._version_payloads: dict[str, dict[str, object]] = {}
        self._current_mode = "single"
        self._last_selected_model = ""
        self._current_autotune_profile = "model-best"
        self._current_build_id = ""
        self._live_best_by_key: dict[str, dict[str, str]] = {}
        self._autotune_result = {"best": "", "summary_json": "", "summary_csv": ""}
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

        params_group = QGroupBox("Benchmark Parameters")
        params_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Tasks:"))
        self.tasks_combo = QComboBox()
        self.tasks_combo.addItems(["quick", "full"])
        row1.addWidget(self.tasks_combo)

        row1.addWidget(QLabel("Runs:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setMinimum(1)
        self.runs_spin.setMaximum(10)
        self.runs_spin.setValue(1)
        row1.addWidget(self.runs_spin)

        row1.addWidget(QLabel("Spec:"))
        self.spec_combo = QComboBox()
        self.spec_combo.addItems(["none", "ngram-mod", "mtp"])
        self.spec_combo.setCurrentText("none")
        row1.addWidget(self.spec_combo)
        params_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Ctx:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setMinimum(8192)
        self.ctx_spin.setMaximum(131072)
        self.ctx_spin.setValue(65536)
        self.ctx_spin.setSingleStep(8192)
        row2.addWidget(self.ctx_spin)

        row2.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(32)
        self.batch_spin.setMaximum(8192)
        self.batch_spin.setValue(2048)
        self.batch_spin.setSingleStep(32)
        row2.addWidget(self.batch_spin)

        row2.addWidget(QLabel("UBatch:"))
        self.ubatch_spin = QSpinBox()
        self.ubatch_spin.setMinimum(32)
        self.ubatch_spin.setMaximum(8192)
        self.ubatch_spin.setValue(512)
        self.ubatch_spin.setSingleStep(32)
        row2.addWidget(self.ubatch_spin)
        params_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("KV K/V:"))
        self.kv_combo = QComboBox()
        self.kv_combo.addItems(["q8_0", "q4_0", "f16", "bf16", "f32"])
        self.kv_combo.setCurrentText("q4_0")
        row3.addWidget(self.kv_combo)

        row3.addWidget(QLabel("Max tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setMinimum(8)
        self.max_tokens_spin.setMaximum(1024)
        self.max_tokens_spin.setValue(80)
        row3.addWidget(self.max_tokens_spin)

        row3.addWidget(QLabel("Autotune profile:"))
        self.autotune_profile_combo = QComboBox()
        self.autotune_profile_combo.addItems(["32K+", "64K+"])
        self.autotune_profile_combo.setCurrentText("64K+")
        self.autotune_profile_combo.currentTextChanged.connect(self._on_autotune_profile_changed)
        row3.addWidget(self.autotune_profile_combo)

        row3.addWidget(QLabel("Depth:"))
        self.autotune_depth_combo = QComboBox()
        self.autotune_depth_combo.addItems([
            "Quick (~16)",
            "Standard (~48)",
            "Full (~108)",
        ])
        self.autotune_depth_combo.setCurrentIndex(1)
        row3.addWidget(self.autotune_depth_combo)
        params_layout.addLayout(row3)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        btn_row = QHBoxLayout()
        self.run_bench_btn = QPushButton("⚡ Run Benchmark")
        self.run_bench_btn.clicked.connect(self.run_benchmark)
        btn_row.addWidget(self.run_bench_btn)

        self.run_autotune_btn = QPushButton("🎯 Run Auto-tune")
        self.run_autotune_btn.clicked.connect(self.run_autotune)
        btn_row.addWidget(self.run_autotune_btn)

        self.open_history_btn = QPushButton("📄 Open BENCH_HISTORY.md")
        self.open_history_btn.clicked.connect(self.open_history_md)
        btn_row.addWidget(self.open_history_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        layout.addWidget(self.log_output)

        presets_group = QGroupBox("Best Presets By Model")
        presets_layout = QVBoxLayout()
        self.presets_table = QTableWidget()
        self.presets_table.setColumnCount(8)
        self.presets_table.setHorizontalHeaderLabels([
            "Model",
            "Best TPS",
            "Ctx",
            "Batch/UBatch",
            "KV",
            "Spec",
            "Build ID",
            "Updated",
        ])
        presets_layout.addWidget(self.presets_table)
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        self._on_autotune_profile_changed(self.autotune_profile_combo.currentText())

    def _on_autotune_profile_changed(self, profile_text: str):
        profile = (profile_text or "").strip() or "32K+"
        self.run_autotune_btn.setText(f"🎯 Run Auto-tune {profile}")

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

        spec_values = ["none", "ngram-mod"]
        if ("mtp" in model.name.lower() or "nextn" in model.name.lower()) and self._server_supports_mtp(server_bin):
            spec_values.append("mtp")

        profile_text = self.autotune_profile_combo.currentText().strip()
        if profile_text == "64K+":
            autotune_min_ctx = 65536
            ctx_quick = "65536"
            ctx_standard = "65536"
            ctx_full = "65536"
            profile_key = "ctx64k-plus"
        else:
            autotune_min_ctx = 32768
            ctx_quick = "32768,65536"
            ctx_standard = "32768,49152,65536"
            ctx_full = "32768,49152,65536"
            profile_key = "ctx32k-plus"

        depth_text = self.autotune_depth_combo.currentText().strip()
        if depth_text.startswith("Quick"):
            autotune_ctx_values = ctx_quick
            autotune_batch_values = "1024,2048"
            autotune_ubatch_values = "1024,2048"
            autotune_kv_values = "q8_0"
            depth_key = "quick"
        elif depth_text.startswith("Full"):
            autotune_ctx_values = ctx_full
            autotune_batch_values = "1024,2048,4096"
            autotune_ubatch_values = "1024,2048,4096"
            autotune_kv_values = "q8_0,q4_0"
            depth_key = "full"
        else:
            autotune_ctx_values = ctx_standard
            autotune_batch_values = "1024,2048"
            autotune_ubatch_values = "1024,2048"
            autotune_kv_values = "q8_0,q4_0"
            depth_key = "standard"

        ctx_count = len([v for v in autotune_ctx_values.split(",") if v.strip()])
        batch_count = len([v for v in autotune_batch_values.split(",") if v.strip()])
        ubatch_count = len([v for v in autotune_ubatch_values.split(",") if v.strip()])
        kv_count = len([v for v in autotune_kv_values.split(",") if v.strip()])
        spec_count = len(spec_values)
        config_count = ctx_count * batch_count * ubatch_count * kv_count * spec_count

        label = f"gui-autotune-{model.stem}"
        command = [
            sys.executable,
            "scripts/agent_workload_bench.py",
            "--autotune",
            "--label",
            label,
            "--tasks",
            "quick",
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
            "160",
            "--startup-timeout",
            "180",
            "--request-timeout",
            "180",
            "--background-server-policy",
            "fail",
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
            "--autotune-max-configs",
            str(max(64, config_count + 8)),
            "--autotune-update-preset",
            "--autotune-preset-file",
            "gui/model_presets.json",
        ]

        self._current_mode = "autotune"
        self._current_autotune_profile = f"{profile_key}-{depth_key}"
        self._current_build_id = build_id
        self._autotune_result = {"best": "", "summary_json": "", "summary_csv": ""}
        self._last_selected_model = model.name
        self._set_running_state(True)
        self.log_output.clear()
        self.log_output.append(f"[INFO] Starting autotune for {model.name}")
        self.log_output.append(f"[INFO] Build ID: {build_id or '-'}")
        self.log_output.append(f"[INFO] Autotune profile: {profile_text}, depth: {depth_text}, configs: {config_count}")
        self.bench_thread = BenchCommandThread(command=command, working_dir=self.project_root)
        self.bench_thread.output.connect(self._on_bench_output)
        self.bench_thread.finished_signal.connect(self._on_bench_finished)
        self.bench_thread.start()

    @staticmethod
    def _server_supports_mtp(server_bin: Path) -> bool:
        try:
            result = subprocess.run(
                [str(server_bin), "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            return "mtp" in output.lower()
        except Exception:
            return False

    def _on_bench_output(self, line: str):
        self.log_output.append(line)
        if line.startswith("BEST:"):
            self._autotune_result["best"] = line
        if line.startswith("CURRENT BEST:"):
            self._autotune_result["best"] = line
            self._update_live_best_from_line(line)
        if line.endswith("-autotune-summary.json"):
            self._autotune_result["summary_json"] = line.split("Wrote ", 1)[-1].strip()
        if line.endswith("-autotune-summary.csv"):
            self._autotune_result["summary_csv"] = line.split("Wrote ", 1)[-1].strip()
        if "Aggregate completion TPS" in line or line.startswith("BEST:") or line.startswith("CURRENT BEST:"):
            self.status_label.setText(line)

    def _on_bench_finished(self, success: bool):
        self._set_running_state(False)

        if success:
            self.status_label.setText("Benchmark completed")
            if self._current_mode == "autotune":
                if self._current_autotune_profile.startswith("ctx64k-plus"):
                    self._update_best_preset_for_model(self._last_selected_model, profile_key="ctx64k-plus", min_ctx=65536)
                else:
                    self._update_best_preset_for_model(self._last_selected_model, profile_key="ctx32k-plus", min_ctx=32768)
                self._live_best_by_key.pop(self._live_key_for_current_profile(), None)
                self.refresh_saved_presets_table()
            else:
                self._update_best_preset_for_model(self._last_selected_model)
            if hasattr(self.parent, "refresh_build_registry"):
                self.parent.refresh_build_registry()
            if hasattr(self.parent, "builds_info_tab") and hasattr(self.parent.builds_info_tab, "refresh_builds_info"):
                self.parent.builds_info_tab.refresh_builds_info()

            message = "Benchmark finished and best model preset updated."
            if self._current_mode == "autotune":
                message = "Auto-tune finished. Best preset saved for this model."
            QMessageBox.information(self, "Bench", message)
        else:
            self.status_label.setText("Benchmark failed")
            QMessageBox.warning(self, "Bench", "Benchmark/autotune failed. Check log output.")

    def _set_running_state(self, running: bool):
        self.run_bench_btn.setEnabled(not running)
        self.run_autotune_btn.setEnabled(not running)
        self.model_browse_btn.setEnabled(not running)
        self.model_refresh_btn.setEnabled(not running)

    def _live_key_for_current_profile(self) -> str:
        profile_bucket = "ctx64k-plus" if self._current_autotune_profile.startswith("ctx64k-plus") else "ctx32k-plus"
        model_name = self._last_selected_model or Path(self.model_path_input.text().strip() or "model.gguf").name
        return f"{model_name}::{profile_bucket}"

    def _update_live_best_from_line(self, line: str) -> None:
        # Expected format:
        # CURRENT BEST: ctx=65536 b=1024 ub=1024 kv=q8_0 spec=none aggregate_tps=48.27
        match = re.search(
            r"ctx=(\d+)\s+b=(\d+)\s+ub=(\d+)\s+kv=([^\s]+)\s+spec=([^\s]+)\s+aggregate_tps=([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if not match:
            return

        ctx, batch, ubatch, kv, spec_mode, tps = match.groups()
        model_name = self._last_selected_model or Path(self.model_path_input.text().strip() or "model.gguf").name
        profile_bucket = "ctx64k-plus" if self._current_autotune_profile.startswith("ctx64k-plus") else "ctx32k-plus"
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
            "build_id": self._current_build_id,
            "run_id": "",
            "label": "",
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.refresh_saved_presets_table()

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
            "build_id": str(best_row.get("build_id", "")),
            "run_id": str(best_row.get("run_id", "")),
            "label": str(best_row.get("label", "")),
            "timestamp": str(best_row.get("timestamp", "")),
        }
        self._save_best_presets(presets)
        self.refresh_saved_presets_table()

    def refresh_saved_presets_table(self):
        presets = self._load_best_presets()
        for key, value in self._live_best_by_key.items():
            if key not in presets:
                presets[key] = dict(value)
                continue
            try:
                old_tps = float(str(presets[key].get("best_tps", "0") or "0"))
            except ValueError:
                old_tps = 0.0
            try:
                new_tps = float(str(value.get("best_tps", "0") or "0"))
            except ValueError:
                new_tps = 0.0
            if new_tps >= old_tps:
                presets[key] = dict(value)
        rows = sorted(presets.items(), key=lambda item: item[0].lower())
        self.presets_table.setRowCount(0)
        for model_name, data in rows:
            display_name = str(data.get("model", model_name))
            profile = str(data.get("profile", "model-best"))
            if profile != "model-best":
                display_name = f"{display_name} ({profile})"
            row = self.presets_table.rowCount()
            self.presets_table.insertRow(row)
            self.presets_table.setItem(row, 0, QTableWidgetItem(display_name))
            self.presets_table.setItem(row, 1, QTableWidgetItem(str(data.get("best_tps", "-"))))
            self.presets_table.setItem(row, 2, QTableWidgetItem(str(data.get("ctx", "-"))))
            self.presets_table.setItem(
                row,
                3,
                QTableWidgetItem(f"{data.get('batch', '-')}/{data.get('ubatch', '-')}")
            )
            self.presets_table.setItem(
                row,
                4,
                QTableWidgetItem(f"{data.get('kv_k', '-')}/{data.get('kv_v', '-')}")
            )
            self.presets_table.setItem(row, 5, QTableWidgetItem(str(data.get("spec_mode", "-"))))
            self.presets_table.setItem(row, 6, QTableWidgetItem(str(data.get("build_id", "-"))))
            self.presets_table.setItem(row, 7, QTableWidgetItem(str(data.get("timestamp", "-"))))

    def open_history_md(self):
        history_md = self.project_root / "build_logs" / "agent-workload" / "BENCH_HISTORY.md"
        if not history_md.exists():
            QMessageBox.warning(self, "History", "BENCH_HISTORY.md not found yet")
            return

        if os.name == "nt":
            os.startfile(str(history_md))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(history_md)])
