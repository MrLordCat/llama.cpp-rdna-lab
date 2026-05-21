"""Server tab - Launch llama-server"""

import json
import re
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTextEdit, QFileDialog, QMessageBox, QScrollArea, QSplitter, QSizePolicy
)
import os
import shlex

from PyQt6.QtCore import Qt

from threads import ServerThread


class ServerTabWidget(QWidget):
    """Tab for launching llama-server"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.models_dir = parent.models_dir if hasattr(parent, "models_dir") else Path("models")
        self.server_thread = None
        self.server_process = None
        self._memory_fit_warning_shown = False
        self._registered_build_map: dict[str, dict[str, object]] = {}
        self._model_presets = self._load_model_presets()
        self.create_ui()
        self.refresh_server_build_choices()
        self.refresh_server_models_list()
        self.load_settings()

    def create_ui(self):
        """Create server tab UI with QSplitter layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info_label = QLabel("🚀 Launch Server - Start llama-server with OpenAI compatible API")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)

        # Create splitter for left (settings) and right (log) panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # LEFT PANEL: All settings
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        model_row = QHBoxLayout()
        self.server_model_path = QLineEdit()
        self.server_model_path.setPlaceholderText("Path to model file...")
        model_row.addWidget(self.server_model_path)

        self.browse_server_model_btn = QPushButton("📂 Browse")
        self.browse_server_model_btn.clicked.connect(self.browse_server_model)
        model_row.addWidget(self.browse_server_model_btn)

        model_layout.addLayout(model_row)

        # Model list
        model_list_label = QLabel("Available Models:")
        model_layout.addWidget(model_list_label)

        self.server_models_combo = QComboBox()
        self.server_models_combo.currentTextChanged.connect(self.on_server_model_selected)
        model_layout.addWidget(self.server_models_combo)

        model_actions_row = QHBoxLayout()
        self.server_models_refresh_btn = QPushButton("🔄 Refresh")
        self.server_models_refresh_btn.clicked.connect(self.refresh_server_models_list)
        model_actions_row.addWidget(self.server_models_refresh_btn)
        model_actions_row.addStretch()
        model_layout.addLayout(model_actions_row)

        # Presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.server_preset_combo = QComboBox()
        self.server_preset_combo.addItems(["Default", "Fast", "Quality", "Balanced", "VRAM Limited"])
        self.server_preset_combo.currentIndexChanged.connect(self.apply_server_preset)
        preset_layout.addWidget(self.server_preset_combo)

        self.server_apply_model_preset_btn = QPushButton("✨ Apply Model Preset")
        self.server_apply_model_preset_btn.clicked.connect(self.apply_model_file_preset)
        preset_layout.addWidget(self.server_apply_model_preset_btn)
        model_layout.addLayout(preset_layout)

        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)

        # Server settings
        server_group = QGroupBox("Server Configuration")
        server_layout = QVBoxLayout()

        # Host and port
        host_layout = QHBoxLayout()
        host_layout.addWidget(QLabel("Host:Port:"))
        self.server_host_input = QLineEdit()
        self.server_host_input.setText("0.0.0.0")
        self.server_host_input.setMaximumWidth(120)
        host_layout.addWidget(self.server_host_input)

        host_layout.addWidget(QLabel("Port:"))
        self.server_port_spinbox = QSpinBox()
        self.server_port_spinbox.setMinimum(1024)
        self.server_port_spinbox.setMaximum(65535)
        self.server_port_spinbox.setValue(8000)
        self.server_port_spinbox.setMaximumWidth(80)
        host_layout.addWidget(self.server_port_spinbox)
        host_layout.addStretch()
        server_layout.addLayout(host_layout)

        # Backend/Mode
        backend_layout = QHBoxLayout()
        backend_layout.addWidget(QLabel("Backend:"))
        self.server_backend_combo = QComboBox()
        self.server_backend_combo.addItems(["GPU", "CPU"])
        backend_layout.addWidget(self.server_backend_combo)

        backend_layout.addWidget(QLabel("Mode:"))
        self.server_mode_combo = QComboBox()
        self.server_mode_combo.addItems(["Inference", "Embedding"])
        backend_layout.addWidget(self.server_mode_combo)
        backend_layout.addStretch()
        server_layout.addLayout(backend_layout)

        build_layout = QHBoxLayout()
        build_layout.addWidget(QLabel("Build Backend:"))
        self.server_build_backend_combo = QComboBox()
        self.server_build_backend_combo.currentTextChanged.connect(self._on_build_backend_changed)
        build_layout.addWidget(self.server_build_backend_combo)

        build_layout.addWidget(QLabel("Build Version:"))
        self.server_build_version_combo = QComboBox()
        build_layout.addWidget(self.server_build_version_combo)
        build_layout.addStretch()
        server_layout.addLayout(build_layout)

        # Thread settings
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("Threads:"))
        self.server_threads_spinbox = QSpinBox()
        self.server_threads_spinbox.setMinimum(1)
        self.server_threads_spinbox.setMaximum(64)
        self.server_threads_spinbox.setValue(8)
        thread_layout.addWidget(self.server_threads_spinbox)

        thread_layout.addWidget(QLabel("Batch:"))
        self.server_batch_spinbox = QSpinBox()
        self.server_batch_spinbox.setMinimum(32)
        self.server_batch_spinbox.setMaximum(8192)
        self.server_batch_spinbox.setValue(2048)
        self.server_batch_spinbox.setSingleStep(32)
        thread_layout.addWidget(self.server_batch_spinbox)

        thread_layout.addWidget(QLabel("UBatch:"))
        self.server_ubatch_spinbox = QSpinBox()
        self.server_ubatch_spinbox.setMinimum(32)
        self.server_ubatch_spinbox.setMaximum(8192)
        self.server_ubatch_spinbox.setValue(512)
        self.server_ubatch_spinbox.setSingleStep(32)
        thread_layout.addWidget(self.server_ubatch_spinbox)

        thread_layout.addWidget(QLabel("HTTP Threads:"))
        self.server_http_threads_spinbox = QSpinBox()
        self.server_http_threads_spinbox.setMinimum(1)
        self.server_http_threads_spinbox.setMaximum(64)
        self.server_http_threads_spinbox.setValue(max(1, (os.cpu_count() or 8) // 2))
        thread_layout.addWidget(self.server_http_threads_spinbox)
        thread_layout.addStretch()
        server_layout.addLayout(thread_layout)

        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(QLabel("Parallel:"))
        self.server_parallel_spinbox = QSpinBox()
        self.server_parallel_spinbox.setMinimum(1)
        self.server_parallel_spinbox.setMaximum(8)
        self.server_parallel_spinbox.setValue(1)
        parallel_layout.addWidget(self.server_parallel_spinbox)
        parallel_layout.addStretch()
        server_layout.addLayout(parallel_layout)

        server_group.setLayout(server_layout)
        scroll_layout.addWidget(server_group)

        # Resources
        resources_group = QGroupBox("Resources")
        resources_layout = QVBoxLayout()

        # GPU layers
        gpu_layout = QHBoxLayout()
        gpu_layout.addWidget(QLabel("GPU Layers:"))
        self.server_gpu_layers_spinbox = QSpinBox()
        self.server_gpu_layers_spinbox.setMinimum(0)
        self.server_gpu_layers_spinbox.setMaximum(999)
        self.server_gpu_layers_spinbox.setValue(99)
        gpu_layout.addWidget(self.server_gpu_layers_spinbox)

        gpu_layout.addWidget(QLabel("Context:"))
        self.server_context_spinbox = QSpinBox()
        self.server_context_spinbox.setMinimum(8192)
        self.server_context_spinbox.setMaximum(131072)
        self.server_context_spinbox.setValue(32768)
        self.server_context_spinbox.setSingleStep(8192)
        gpu_layout.addWidget(self.server_context_spinbox)
        gpu_layout.addStretch()
        resources_layout.addLayout(gpu_layout)

        # KV Cache
        kv_layout = QHBoxLayout()
        kv_layout.addWidget(QLabel("KV Cache Type:"))
        self.server_kv_type_combo = QComboBox()
        self.server_kv_type_combo.addItems(["f16", "bf16", "f32", "q8_0", "q4_0"])
        self.server_kv_type_combo.setCurrentText("f16")
        kv_layout.addWidget(self.server_kv_type_combo)

        kv_layout.addWidget(QLabel("KV Quantize:"))
        self.server_kv_quant_check = QCheckBox()
        kv_layout.addWidget(self.server_kv_quant_check)
        kv_layout.addStretch()
        resources_layout.addLayout(kv_layout)

        resources_group.setLayout(resources_layout)
        scroll_layout.addWidget(resources_group)

        # Speculative Decoding
        spec_group = QGroupBox("Speculative Decoding")
        spec_layout = QVBoxLayout()

        spec_type_layout = QHBoxLayout()
        spec_type_layout.addWidget(QLabel("Type:"))
        self.server_spec_type_combo = QComboBox()
        self.server_spec_type_combo.addItems(["None", "draft", "ngram-mod", "mtp"])
        self.server_spec_type_combo.currentTextChanged.connect(self.on_spec_type_changed)
        spec_type_layout.addWidget(self.server_spec_type_combo)
        spec_type_layout.addStretch()
        spec_layout.addLayout(spec_type_layout)

        # Speculative params
        draft_layout = QHBoxLayout()
        draft_layout.addWidget(QLabel("Draft N:"))
        self.server_spec_draft_n = QSpinBox()
        self.server_spec_draft_n.setMinimum(1)
        self.server_spec_draft_n.setMaximum(20)
        self.server_spec_draft_n.setValue(5)
        draft_layout.addWidget(self.server_spec_draft_n)

        draft_layout.addWidget(QLabel("Max N:"))
        self.server_spec_draft_n_max = QSpinBox()
        self.server_spec_draft_n_max.setMinimum(1)
        self.server_spec_draft_n_max.setMaximum(20)
        self.server_spec_draft_n_max.setValue(3)
        draft_layout.addWidget(self.server_spec_draft_n_max)
        draft_layout.addStretch()
        spec_layout.addLayout(draft_layout)

        # NGram parameters
        ngram_layout = QHBoxLayout()
        ngram_layout.addWidget(QLabel("NGram Min:"))
        self.server_ngram_min = QSpinBox()
        self.server_ngram_min.setMinimum(1)
        self.server_ngram_min.setMaximum(512)
        self.server_ngram_min.setValue(1)
        ngram_layout.addWidget(self.server_ngram_min)

        ngram_layout.addWidget(QLabel("Match:"))
        self.server_ngram_match = QSpinBox()
        self.server_ngram_match.setMinimum(1)
        self.server_ngram_match.setMaximum(512)
        self.server_ngram_match.setValue(80)
        ngram_layout.addWidget(self.server_ngram_match)

        ngram_layout.addWidget(QLabel("Max:"))
        self.server_ngram_max = QSpinBox()
        self.server_ngram_max.setMinimum(1)
        self.server_ngram_max.setMaximum(512)
        self.server_ngram_max.setValue(128)
        ngram_layout.addWidget(self.server_ngram_max)
        ngram_layout.addStretch()
        spec_layout.addLayout(ngram_layout)

        self.ngram_layout_group = ngram_layout
        self.ngram_layout_group.setEnabled(False)

        spec_group.setLayout(spec_layout)
        scroll_layout.addWidget(spec_group)

        # Sampling
        sampling_group = QGroupBox("Sampling Parameters")
        sampling_layout = QVBoxLayout()

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.server_temperature_spinbox = QDoubleSpinBox()
        self.server_temperature_spinbox.setMinimum(0.0)
        self.server_temperature_spinbox.setMaximum(2.0)
        self.server_temperature_spinbox.setValue(0.7)
        self.server_temperature_spinbox.setSingleStep(0.1)
        temp_layout.addWidget(self.server_temperature_spinbox)

        temp_layout.addWidget(QLabel("Top-P:"))
        self.server_top_p_spinbox = QDoubleSpinBox()
        self.server_top_p_spinbox.setMinimum(0.0)
        self.server_top_p_spinbox.setMaximum(1.0)
        self.server_top_p_spinbox.setValue(0.95)
        self.server_top_p_spinbox.setSingleStep(0.05)
        temp_layout.addWidget(self.server_top_p_spinbox)

        temp_layout.addWidget(QLabel("Top-K:"))
        self.server_top_k_spinbox = QSpinBox()
        self.server_top_k_spinbox.setMinimum(0)
        self.server_top_k_spinbox.setMaximum(500)
        self.server_top_k_spinbox.setValue(40)
        temp_layout.addWidget(self.server_top_k_spinbox)
        temp_layout.addStretch()
        sampling_layout.addLayout(temp_layout)

        sampling_group.setLayout(sampling_layout)
        scroll_layout.addWidget(sampling_group)

        perf_group = QGroupBox("Performance Options")
        perf_layout = QVBoxLayout()

        self.server_flash_attn_check = QCheckBox("Enable Flash Attention")
        self.server_flash_attn_check.setChecked(True)
        perf_layout.addWidget(self.server_flash_attn_check)

        self.server_no_warmup_check = QCheckBox("Skip warmup (--no-warmup)")
        self.server_no_warmup_check.setChecked(True)
        perf_layout.addWidget(self.server_no_warmup_check)

        self.server_no_mmap_check = QCheckBox("Load model into RAM (--no-mmap)")
        self.server_no_mmap_check.setChecked(False)
        self.server_no_mmap_check.setToolTip("Useful for CPU fallback / -ngl 0 routes when mmap paging limits throughput")
        perf_layout.addWidget(self.server_no_mmap_check)

        self.server_disable_thinking_check = QCheckBox(
            "Disable thinking for benchmark-like throughput (--chat-template-kwargs)"
        )
        self.server_disable_thinking_check.setChecked(False)
        perf_layout.addWidget(self.server_disable_thinking_check)

        self.server_auto_fit_check = QCheckBox("Auto-fit params to free memory (-fit on)")
        self.server_auto_fit_check.setChecked(True)
        perf_layout.addWidget(self.server_auto_fit_check)

        perf_group.setLayout(perf_layout)
        scroll_layout.addWidget(perf_group)

        # CORS and API key
        cors_group = QGroupBox("Security & API")
        cors_layout = QVBoxLayout()

        self.server_cors_check = QCheckBox("Enable CORS")
        self.server_cors_check.setChecked(True)
        cors_layout.addWidget(self.server_cors_check)

        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        self.server_api_key_input = QLineEdit()
        self.server_api_key_input.setPlaceholderText("Leave empty for no key")
        self.server_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.server_api_key_input)
        cors_layout.addLayout(api_key_layout)

        cors_group.setLayout(cors_layout)
        scroll_layout.addWidget(cors_group)

        # Extra arguments
        extra_args_group = QGroupBox("Extra Arguments")
        extra_args_layout = QVBoxLayout()

        self.server_extra_args = QTextEdit()
        self.server_extra_args.setPlaceholderText("Additional llama-server arguments (one per line)")
        self.server_extra_args.setMaximumHeight(80)
        extra_args_layout.addWidget(self.server_extra_args)

        extra_args_group.setLayout(extra_args_layout)
        scroll_layout.addWidget(extra_args_group)
        scroll_layout.addStretch(1)

        scroll_area.setWidget(scroll_widget)
        left_layout.addWidget(scroll_area, 1)

        # RIGHT PANEL: Log and control buttons
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Control buttons at top
        buttons_layout = QHBoxLayout()

        self.server_start_btn = QPushButton("▶️ Start Server")
        self.server_start_btn.clicked.connect(self.start_server)
        self.server_start_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; background-color: #4CAF50; color: white; }")
        buttons_layout.addWidget(self.server_start_btn)

        self.server_stop_btn = QPushButton("⏹️ Stop Server")
        self.server_stop_btn.clicked.connect(self.stop_server)
        self.server_stop_btn.setEnabled(False)
        self.server_stop_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; background-color: #f44336; color: white; }")
        buttons_layout.addWidget(self.server_stop_btn)

        self.server_web_btn = QPushButton("🌐 Open Web")
        self.server_web_btn.clicked.connect(self.open_web_ui)
        self.server_web_btn.setEnabled(False)
        buttons_layout.addWidget(self.server_web_btn)

        self.server_clear_log_btn = QPushButton("🧹 Clear Log")
        self.server_clear_log_btn.clicked.connect(self.clear_server_log)
        buttons_layout.addWidget(self.server_clear_log_btn)

        buttons_layout.addStretch()
        right_layout.addLayout(buttons_layout)

        # Status
        self.server_status_label = QLabel("Status: Stopped")
        self.server_status_label.setStyleSheet("font-weight: bold; color: red;")
        right_layout.addWidget(self.server_status_label)

        # Log output
        log_label = QLabel("Server Output Log:")
        right_layout.addWidget(log_label)

        self.server_log = QTextEdit()
        self.server_log.setReadOnly(True)
        self.server_log.setMinimumHeight(220)
        self.server_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.server_log, 1)

        # Add panels to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter, 1)

    def on_spec_type_changed(self):
        """Handle speculative type change"""
        spec_type = self.server_spec_type_combo.currentText()
        is_ngram = spec_type == "ngram-mod"
        if hasattr(self, "ngram_layout_group"):
            # Enable/disable ngram widgets
            for i in range(self.ngram_layout_group.count()):
                widget = self.ngram_layout_group.itemAt(i).widget()
                if widget:
                    widget.setEnabled(is_ngram)

    def browse_server_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            str(self.models_dir),
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path:
            self.server_model_path.setText(file_path)

    def on_server_model_selected(self, model_name: str):
        """Handle model selection from combo"""
        if model_name and model_name != "-- Select Model --" and model_name != "-- No .gguf models found --":
            model_path = self.models_dir / model_name
            self.server_model_path.setText(str(model_path))
            self.apply_model_file_preset()

    def _load_model_presets(self) -> list[dict]:
        """Load model presets from JSON file used by legacy GUI."""
        preset_path = self.parent.project_root / "gui" / "model_presets.json"
        if not preset_path.exists():
            return []
        try:
            data = json.loads(preset_path.read_text(encoding="utf-8"))
            presets = data.get("presets", [])
            return [p for p in presets if isinstance(p, dict)]
        except Exception:
            return []

    def apply_model_file_preset(self):
        """Apply first regex-matching model preset from model_presets.json."""
        self._model_presets = self._load_model_presets()
        model_path = self.server_model_path.text().strip()
        if not model_path:
            return {"matched": False, "reason": "empty-model-path"}
        model_name = Path(model_path).name

        match = None
        for preset in self._model_presets:
            pattern = preset.get("pattern")
            if not pattern:
                continue
            try:
                if re.search(pattern, model_name, flags=re.IGNORECASE):
                    match = preset
                    break
            except re.error:
                continue

        if not match:
            return {"matched": False, "reason": "no-preset-match", "model": model_name}

        # Reset spec type to None before applying preset to avoid stale MTP/spec state
        self.server_spec_type_combo.setCurrentText("None")
        self.on_spec_type_changed()

        if "ctx" in match:
            self.server_context_spinbox.setValue(int(match["ctx"]))
        if "batch_size" in match:
            self.server_batch_spinbox.setValue(int(match["batch_size"]))
        if "ubatch_size" in match:
            self.server_ubatch_spinbox.setValue(int(match["ubatch_size"]))
        if "gpu_layers" in match:
            self.server_gpu_layers_spinbox.setValue(int(match["gpu_layers"]))
        if "parallel" in match:
            self.server_parallel_spinbox.setValue(int(match["parallel"]))
        if "flash_attn" in match:
            self.server_flash_attn_check.setChecked(bool(match["flash_attn"]))
        if "no_mmap" in match:
            self.server_no_mmap_check.setChecked(bool(match["no_mmap"]))
        if "disable_thinking" in match:
            self.server_disable_thinking_check.setChecked(bool(match["disable_thinking"]))
        if "extra_args" in match:
            self.server_extra_args.setPlainText(str(match["extra_args"]).strip())
            self._apply_spec_controls_from_extra_args(str(match["extra_args"]).strip())

        kv_map = {
            0: "f16",
            1: "bf16",
            2: "f32",
            3: "q8_0",
            7: "q4_0",
        }
        kv_value = match.get("kv_cache")
        if isinstance(kv_value, int) and kv_value in kv_map:
            kv_text = kv_map[kv_value]
            idx = self.server_kv_type_combo.findText(kv_text)
            if idx >= 0:
                self.server_kv_type_combo.setCurrentIndex(idx)

        if "notes" in match:
            self.server_log.append(f"[INFO] Applied model preset: {match.get('name', 'Unnamed')} - {match['notes']}")

        return {
            "matched": True,
            "model": model_name,
            "preset_name": match.get("name", "Unnamed"),
            "context": self.server_context_spinbox.value(),
            "batch": self.server_batch_spinbox.value(),
            "ubatch": self.server_ubatch_spinbox.value(),
            "parallel": self.server_parallel_spinbox.value(),
            "kv": self.server_kv_type_combo.currentText(),
            "flash_attn": self.server_flash_attn_check.isChecked(),
            "extra_args": self.server_extra_args.toPlainText().strip(),
        }

    def refresh_server_models_list(self):
        """Populate model list from models directory"""
        self.server_models_combo.clear()
        self.server_models_combo.addItem("-- Select Model --")

        if not self.models_dir.exists():
            self.server_models_combo.addItem("-- No .gguf models found --")
            return

        model_files = sorted(
            [
                p.name
                for p in self.models_dir.glob("*.gguf")
                if "mmproj" not in p.name.lower()
            ],
            key=str.lower,
        )

        if not model_files:
            self.server_models_combo.addItem("-- No .gguf models found --")
            return

        self.server_models_combo.addItems(model_files)

    def _apply_spec_controls_from_extra_args(self, extra_args: str):
        """Parse known speculative flags from preset extra args and sync visible controls."""
        if not extra_args:
            return

        try:
            tokens = shlex.split(extra_args, posix=(os.name != "nt"))
        except ValueError:
            tokens = extra_args.split()

        spec_type = None
        ngram_min = None
        ngram_max = None
        ngram_match = None
        mtp_draft_n_max = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if tok == "--spec-type" and nxt is not None:
                spec_type = nxt.strip().lower()
                i += 2
                continue
            if tok == "--spec-ngram-mod-n-min" and nxt is not None:
                ngram_min = nxt
                i += 2
                continue
            if tok == "--spec-ngram-mod-n-max" and nxt is not None:
                ngram_max = nxt
                i += 2
                continue
            if tok == "--spec-ngram-mod-n-match" and nxt is not None:
                ngram_match = nxt
                i += 2
                continue
            if tok == "--spec-draft-n-max" and nxt is not None:
                mtp_draft_n_max = nxt
                i += 2
                continue
            i += 1

        if spec_type == "ngram-mod":
            self.server_spec_type_combo.setCurrentText("ngram-mod")
        elif spec_type == "mtp":
            self.server_spec_type_combo.setCurrentText("mtp")
        elif spec_type == "draft":
            self.server_spec_type_combo.setCurrentText("draft")
        elif spec_type == "none":
            self.server_spec_type_combo.setCurrentText("None")

        if ngram_min is not None:
            try:
                self.server_ngram_min.setValue(max(1, min(512, int(ngram_min))))
            except ValueError:
                pass
        if ngram_max is not None:
            try:
                self.server_ngram_max.setValue(max(1, min(512, int(ngram_max))))
            except ValueError:
                pass
        if ngram_match is not None:
            try:
                self.server_ngram_match.setValue(max(1, min(512, int(ngram_match))))
            except ValueError:
                pass
        if mtp_draft_n_max is not None:
            try:
                self.server_spec_draft_n_max.setValue(max(1, min(20, int(mtp_draft_n_max))))
            except ValueError:
                pass

        self.on_spec_type_changed()

    def apply_server_preset(self):
        """Apply preset configuration"""
        preset = self.server_preset_combo.currentText()

        presets = {
            "Default": {"gpu_layers": 99, "context": 32768, "batch": 2048, "threads": 8},
            "Fast": {"gpu_layers": 99, "context": 4096, "batch": 4096, "threads": 4},
            "Quality": {"gpu_layers": 99, "context": 16384, "batch": 1024, "threads": 16},
            "Balanced": {"gpu_layers": 50, "context": 8192, "batch": 2048, "threads": 8},
            "VRAM Limited": {"gpu_layers": 20, "context": 4096, "batch": 512, "threads": 4}
        }

        if preset in presets:
            cfg = presets[preset]
            self.server_gpu_layers_spinbox.setValue(cfg["gpu_layers"])
            self.server_context_spinbox.setValue(cfg["context"])
            self.server_batch_spinbox.setValue(cfg["batch"])
            self.server_threads_spinbox.setValue(cfg["threads"])

    @staticmethod
    def _vulkan_runtime_env() -> dict[str, str]:
        return {
            "GGML_VK_FORCE_AMD_LARGE_MATMUL": "1",
        }

    def start_server(self):
        """Start llama-server"""
        if self.server_thread and self.server_thread.isRunning():
            QMessageBox.information(self, "Server", "Server is already running")
            return

        model_path = self.server_model_path.text().strip()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Error", "Select valid model file")
            return

        build_dir = self._resolve_selected_build_dir()
        llama_server = self._find_llama_server_binary(build_dir)
        if not llama_server:
            QMessageBox.warning(
                self,
                "Server binary not found",
                f"llama-server executable not found in build directory:\n{build_dir}\n\n"
                "Build the selected backend first in Build tab."
            )
            return

        port = self.server_port_spinbox.value()
        command = [
            str(llama_server),
            "-m", model_path,
            "--host", self.server_host_input.text().strip() or "0.0.0.0",
            "--port", str(port),
            "-c", str(self.server_context_spinbox.value()),
            "-t", str(self.server_threads_spinbox.value()),
            "--threads-http", str(self.server_http_threads_spinbox.value()),
            "--batch-size", str(self.server_batch_spinbox.value()),
            "--ubatch-size", str(self.server_ubatch_spinbox.value()),
            "--parallel", str(self.server_parallel_spinbox.value()),
        ]

        if self.server_backend_combo.currentText() == "GPU":
            command.extend(["-ngl", str(self.server_gpu_layers_spinbox.value())])
        else:
            command.extend(["-ngl", "0"])

        kv_type = self.server_kv_type_combo.currentText().strip()
        if kv_type and kv_type != "f16":
            command.extend(["--cache-type-k", kv_type, "--cache-type-v", kv_type])

        extra_args = self.server_extra_args.toPlainText().strip()
        extra_tokens = []
        if extra_args:
            try:
                extra_tokens = shlex.split(extra_args, posix=(os.name != "nt"))
            except ValueError:
                extra_tokens = extra_args.split()

        # Avoid duplicate speculative flags when preset already supplies them via Extra Arguments.
        has_spec_in_extra = any(tok.startswith("--spec-") or tok == "--spec-type" for tok in extra_tokens)
        has_no_mmap_in_extra = "--no-mmap" in extra_tokens

        spec_type = self.server_spec_type_combo.currentText().strip().lower()
        if not has_spec_in_extra:
            if spec_type == "mtp":
                command.extend(["--spec-type", "mtp", "--spec-draft-n-max", str(self.server_spec_draft_n_max.value())])
                # MTP currently requires single parallel sequence in llama-server.
                if self.server_parallel_spinbox.value() != 1:
                    self.server_log.append("[INFO] MTP requires --parallel 1, overriding selected value")
                    command.extend(["--parallel", "1"])
            elif spec_type == "ngram-mod":
                command.extend([
                    "--spec-type", "ngram-mod",
                    "--spec-ngram-mod-n-match", str(self.server_ngram_match.value()),
                    "--spec-ngram-mod-n-min", str(self.server_ngram_min.value()),
                    "--spec-ngram-mod-n-max", str(self.server_ngram_max.value()),
                ])

        if self.server_flash_attn_check.isChecked():
            command.extend(["--flash-attn", "on"])

        if self.server_no_warmup_check.isChecked():
            command.append("--no-warmup")

        if self.server_no_mmap_check.isChecked() and not has_no_mmap_in_extra:
            command.append("--no-mmap")

        if self.server_disable_thinking_check.isChecked():
            command.extend([
                "--chat-template-kwargs",
                '{"enable_thinking":false,"preserve_thinking":false}',
            ])

        if not self.server_auto_fit_check.isChecked():
            command.extend(["-fit", "off"])

        if extra_tokens:
            command.extend(extra_tokens)

        api_key = self.server_api_key_input.text().strip()
        if api_key:
            command.extend(["--api-key", api_key])

        self.server_start_btn.setEnabled(False)
        self.server_stop_btn.setEnabled(True)
        self.server_web_btn.setEnabled(False)
        self.server_status_label.setText("Status: Starting...")
        self.server_status_label.setStyleSheet("font-weight: bold; color: orange;")

        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Starting server...")

        server_env = None
        if "rocm" in build_dir.name.lower() and hasattr(self.parent, "build_manager"):
            server_env = self.parent.build_manager.get_rocm_env()
        elif "vulkan" in build_dir.name.lower():
            server_env = self._vulkan_runtime_env()

        self.server_log.clear()
        self._memory_fit_warning_shown = False
        self.server_log.append(f"[INFO] Build: {build_dir}")
        self.server_log.append(f"[INFO] Command: {' '.join(command)}")
        if server_env and "GGML_VK_FORCE_AMD_LARGE_MATMUL" in server_env:
            env_summary = ", ".join(f"{key}={value}" for key, value in sorted(server_env.items()))
            self.server_log.append(f"[INFO] Env overrides: {env_summary}")

        self.server_thread = ServerThread(command, str(self.parent.project_root), port=port, env=server_env)
        self.server_thread.output_ready.connect(self.on_server_status)
        self.server_thread.server_ready.connect(self.on_server_ready)
        self.server_thread.finished_signal.connect(self.on_server_finished)
        self.server_thread.error_signal.connect(self.on_server_error)
        self.server_thread.start()

    def _resolve_selected_build_dir(self) -> Path:
        """Resolve build directory based on selected build mode"""
        selected_version = self.server_build_version_combo.currentText().strip()
        payload = self._registered_build_map.get(selected_version)
        if isinstance(payload, dict):
            candidate = payload.get("build_dir")
            if isinstance(candidate, Path) and candidate.exists():
                return candidate

        selected_backend = self.server_build_backend_combo.currentText().strip()
        if selected_backend != "Auto" and hasattr(self.parent, "get_build_dir_for_backend"):
            return Path(self.parent.get_build_dir_for_backend(selected_backend))

        candidates = [
            self.parent.project_root / "build-rocm",
            self.parent.project_root / "build-cpu",
            self.parent.project_root / "build",
            self.parent.project_root / "build-cuda",
            self.parent.project_root / "build-vulkan",
        ]
        for candidate in candidates:
            if self._find_llama_server_binary(candidate):
                return candidate
        return self.parent.project_root / "build"

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

    def _on_build_backend_changed(self, *_args):
        self.refresh_server_build_versions_for_backend(select_latest=True)

    def refresh_server_build_versions_for_backend(self, select_latest: bool):
        selected_backend_display = self.server_build_backend_combo.currentText().strip() if self.server_build_backend_combo.count() else "Auto"
        selected_backend_key = self._backend_key_from_display(selected_backend_display) if selected_backend_display != "Auto" else ""
        previous_version_id = ""
        prev_payload = self._registered_build_map.get(self.server_build_version_combo.currentText().strip())
        if isinstance(prev_payload, dict):
            previous_version_id = str(prev_payload.get("build_id", ""))

        self.server_build_version_combo.clear()
        self._registered_build_map = {}

        records = []
        if hasattr(self.parent, "get_registered_builds"):
            records = self.parent.get_registered_builds()

        usable = []
        for rec in records:
            if str(rec.get("status", "")) != "ready":
                continue
            build_dir_text = str(rec.get("build_dir", "")).strip()
            if not build_dir_text:
                continue
            build_dir = Path(build_dir_text)
            if not build_dir.exists():
                continue
            if selected_backend_key and str(rec.get("backend", "")).lower() != selected_backend_key:
                continue
            # Only show versions that are actually runnable from this tab.
            if not self._find_llama_server_binary(build_dir):
                continue
            usable.append(rec)

        registry = getattr(self.parent, "build_registry", None)

        # Newest real binary build first, then by name.
        usable.sort(
            key=lambda r: (
                registry.get_effective_build_timestamp(r) if registry is not None else str(r.get("created_at", "") or r.get("updated_at", "")),
                str(r.get("name", "")).lower(),
            ),
            reverse=True,
        )
        for rec in usable:
            build_dir = Path(str(rec.get("build_dir")))
            source = str(rec.get("source_type", "fork"))
            source_ref = str(rec.get("source_ref", ""))
            short_ref = source_ref[:10] if source_ref else "-"
            name = str(rec.get("name", build_dir.name))
            build_id = str(rec.get("id", ""))
            short_id = build_id[-8:] if len(build_id) >= 8 else (build_id or "-")
            build_date = registry.get_effective_build_timestamp(rec) if registry is not None else (str(rec.get("created_at", "")).strip() or str(rec.get("updated_at", "")).strip() or "-")
            label = f"{name} [{source}/{short_ref}] | id:{short_id} | built:{build_date}"
            self.server_build_version_combo.addItem(label)
            self._registered_build_map[label] = {
                "build_dir": build_dir,
                "build_id": str(rec.get("id", "")),
            }

        if self.server_build_version_combo.count() == 0:
            self.server_build_version_combo.addItem("Auto")
            return

        if select_latest:
            self.server_build_version_combo.setCurrentIndex(0)
            return

        # Try preserving previous selected build id.
        if previous_version_id:
            for idx in range(self.server_build_version_combo.count()):
                label = self.server_build_version_combo.itemText(idx)
                payload = self._registered_build_map.get(label)
                if isinstance(payload, dict) and str(payload.get("build_id", "")) == previous_version_id:
                    self.server_build_version_combo.setCurrentIndex(idx)
                    return

        self.server_build_version_combo.setCurrentIndex(0)

    def refresh_server_build_choices(self):
        """Populate backend selector and version selector from registry with latest version default."""
        previous_backend = self.server_build_backend_combo.currentText() if hasattr(self, "server_build_backend_combo") else "Auto"
        self.server_build_backend_combo.blockSignals(True)
        self.server_build_backend_combo.clear()
        self.server_build_backend_combo.addItem("Auto")

        records = []
        if hasattr(self.parent, "get_registered_builds"):
            records = self.parent.get_registered_builds()

        backend_keys = sorted({str(r.get("backend", "")).lower() for r in records if str(r.get("backend", "")).strip()})
        for key in backend_keys:
            self.server_build_backend_combo.addItem(self._display_backend_from_key(key))

        # Keep legacy backend quick-select options.
        for legacy in ["ROCm/HIP", "CPU", "CUDA", "Vulkan", "Metal", "SYCL", "OpenCL"]:
            if self.server_build_backend_combo.findText(legacy) < 0:
                self.server_build_backend_combo.addItem(legacy)

        idx = self.server_build_backend_combo.findText(previous_backend)
        self.server_build_backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.server_build_backend_combo.blockSignals(False)
        self.refresh_server_build_versions_for_backend(select_latest=True)

    @staticmethod
    def _find_llama_server_binary(build_dir: Path) -> Path | None:
        """Find llama-server executable in known build output locations"""
        candidates = [
            build_dir / "bin" / "llama-server.exe",
            build_dir / "bin" / "Release" / "llama-server.exe",
            build_dir / "bin" / "Debug" / "llama-server.exe",
            build_dir / "bin" / "llama-server",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def on_server_ready(self, url: str):
        """Handle server ready event"""
        self.server_web_btn.setEnabled(True)
        self.server_status_label.setText(f"Status: Running ({url})")
        self.server_status_label.setStyleSheet("font-weight: bold; color: green;")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(f"Server running: {url}")

    def on_server_finished(self, exit_code: int):
        """Handle server process exit"""
        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_web_btn.setEnabled(False)
        if exit_code == 0:
            self.server_status_label.setText("Status: Stopped")
            self.server_status_label.setStyleSheet("font-weight: bold; color: red;")
            self.server_log.append("[INFO] Server stopped")
        else:
            self.server_status_label.setText(f"Status: Crashed (exit {exit_code})")
            self.server_status_label.setStyleSheet("font-weight: bold; color: red;")
            self.server_log.append(f"[ERROR] Server crashed with exit code {exit_code}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Server stopped")

    def on_server_status(self, message: str):
        """Handle server status update"""
        self.server_log.append(message)

        if (
            not self._memory_fit_warning_shown
            and "cannot meet free memory target" in message.lower()
        ):
            self._memory_fit_warning_shown = True
            self.server_log.append(
                "[HINT] VRAM pressure detected. Try lower GPU Layers (e.g. -10),"
                " smaller context, or disable auto-fit (-fit off) for debugging."
            )

        if "listening" in message.lower() or "started" in message.lower():
            self.server_status_label.setText("Status: Running ✓")
            self.server_status_label.setStyleSheet("font-weight: bold; color: green;")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Server running on port " + str(self.server_port_spinbox.value()))
        elif "error" in message.lower():
            self.server_status_label.setStyleSheet("font-weight: bold; color: red;")

    def on_server_error(self, error: str):
        """Handle server error"""
        self.server_log.append(f"[ERROR] {error}")
        self.server_status_label.setText("Status: Error ✗")
        self.server_status_label.setStyleSheet("font-weight: bold; color: red;")
        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_web_btn.setEnabled(False)
        QMessageBox.critical(self, "Server Error", f"Server failed to start:\n{error}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Server error")

    def stop_server(self):
        """Stop llama-server"""
        if self.server_thread and self.server_thread.isRunning():
            self.server_thread.stop()
            self.server_thread.wait(2000)

        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_web_btn.setEnabled(False)
        self.server_status_label.setText("Status: Stopped")
        self.server_status_label.setStyleSheet("font-weight: bold; color: red;")
        self.server_log.append("[INFO] Server stopped")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Server stopped")

    def open_web_ui(self):
        """Open web UI in browser"""
        port = self.server_port_spinbox.value()
        url = f"http://localhost:{port}"

        import webbrowser
        webbrowser.open(url)

        self.server_log.append(f"[INFO] Opening web UI at {url}")

    def clear_server_log(self):
        """Clear server log"""
        self.server_log.clear()
        self.server_log.append("[INFO] Log cleared")

    def load_settings(self):
        """Load server settings"""
        if not hasattr(self.parent, "settings"):
            return

        settings = self.parent.settings
        self.server_host_input.setText(settings.value("server/host", "0.0.0.0"))
        self.server_port_spinbox.setValue(int(settings.value("server/port", 8000)))
        self.server_backend_combo.setCurrentText(settings.value("server/backend", "GPU"))
        self.server_mode_combo.setCurrentText(settings.value("server/mode", "Inference"))
        saved_backend = settings.value("server/build_backend", settings.value("server/build", "Auto"))
        saved_version_id = settings.value("server/build_version_id", "")
        idx = self.server_build_backend_combo.findText(saved_backend)
        self.server_build_backend_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.refresh_server_build_versions_for_backend(select_latest=False)
        if saved_version_id:
            for i in range(self.server_build_version_combo.count()):
                label = self.server_build_version_combo.itemText(i)
                payload = self._registered_build_map.get(label)
                if isinstance(payload, dict) and str(payload.get("build_id", "")) == str(saved_version_id):
                    self.server_build_version_combo.setCurrentIndex(i)
                    break

        self.server_gpu_layers_spinbox.setValue(int(settings.value("server/gpu_layers", 99)))
        self.server_context_spinbox.setValue(int(settings.value("server/context", 32768)))
        self.server_batch_spinbox.setValue(int(settings.value("server/batch", 2048)))
        self.server_ubatch_spinbox.setValue(int(settings.value("server/ubatch", 512)))
        self.server_threads_spinbox.setValue(int(settings.value("server/threads", 8)))
        self.server_http_threads_spinbox.setValue(int(settings.value("server/http_threads", max(1, (os.cpu_count() or 8) // 2))))
        self.server_parallel_spinbox.setValue(int(settings.value("server/parallel", 1)))
        self.server_kv_type_combo.setCurrentText(settings.value("server/kv_type", "f16"))

        self.server_temperature_spinbox.setValue(float(settings.value("server/temperature", 0.7)))
        self.server_top_p_spinbox.setValue(float(settings.value("server/top_p", 0.95)))
        self.server_top_k_spinbox.setValue(int(settings.value("server/top_k", 40)))

        self.server_spec_type_combo.setCurrentText(settings.value("server/spec_type", "None"))
        self.server_spec_draft_n.setValue(int(settings.value("server/spec_draft_n", 5)))
        self.server_spec_draft_n_max.setValue(int(settings.value("server/spec_draft_n_max", 3)))
        self.server_ngram_min.setValue(int(settings.value("server/spec_ngram_min", 1)))
        self.server_ngram_match.setValue(int(settings.value("server/spec_ngram_match", 80)))
        self.server_ngram_max.setValue(int(settings.value("server/spec_ngram_max", 128)))

        self.server_flash_attn_check.setChecked(settings.value("server/flash_attn", True, type=bool))
        self.server_no_warmup_check.setChecked(settings.value("server/no_warmup", True, type=bool))
        self.server_no_mmap_check.setChecked(settings.value("server/no_mmap", False, type=bool))
        self.server_disable_thinking_check.setChecked(settings.value("server/disable_thinking", False, type=bool))
        self.server_auto_fit_check.setChecked(settings.value("server/auto_fit", True, type=bool))

        self.server_cors_check.setChecked(settings.value("server/cors", True, type=bool))
        self.server_api_key_input.setText(settings.value("server/api_key", ""))
        self.server_extra_args.setPlainText(settings.value("server/extra_args", ""))

        model_path = settings.value("server/model_path", "")
        if model_path:
            self.server_model_path.setText(model_path)

        selected_model_name = settings.value("server/model_name", "")
        if selected_model_name:
            idx = self.server_models_combo.findText(selected_model_name)
            if idx >= 0:
                self.server_models_combo.setCurrentIndex(idx)

        self.on_spec_type_changed()

    def save_settings(self):
        """Save server settings"""
        if not hasattr(self.parent, "settings"):
            return

        settings = self.parent.settings
        settings.setValue("server/host", self.server_host_input.text().strip())
        settings.setValue("server/port", self.server_port_spinbox.value())
        settings.setValue("server/backend", self.server_backend_combo.currentText())
        settings.setValue("server/mode", self.server_mode_combo.currentText())
        settings.setValue("server/build_backend", self.server_build_backend_combo.currentText())
        settings.setValue("server/build", self.server_build_backend_combo.currentText())
        selected_label = self.server_build_version_combo.currentText().strip()
        payload = self._registered_build_map.get(selected_label)
        settings.setValue("server/build_version_id", str(payload.get("build_id", "")) if isinstance(payload, dict) else "")

        settings.setValue("server/gpu_layers", self.server_gpu_layers_spinbox.value())
        settings.setValue("server/context", self.server_context_spinbox.value())
        settings.setValue("server/batch", self.server_batch_spinbox.value())
        settings.setValue("server/ubatch", self.server_ubatch_spinbox.value())
        settings.setValue("server/threads", self.server_threads_spinbox.value())
        settings.setValue("server/http_threads", self.server_http_threads_spinbox.value())
        settings.setValue("server/parallel", self.server_parallel_spinbox.value())
        settings.setValue("server/kv_type", self.server_kv_type_combo.currentText())

        settings.setValue("server/temperature", self.server_temperature_spinbox.value())
        settings.setValue("server/top_p", self.server_top_p_spinbox.value())
        settings.setValue("server/top_k", self.server_top_k_spinbox.value())

        settings.setValue("server/spec_type", self.server_spec_type_combo.currentText())
        settings.setValue("server/spec_draft_n", self.server_spec_draft_n.value())
        settings.setValue("server/spec_draft_n_max", self.server_spec_draft_n_max.value())
        settings.setValue("server/spec_ngram_min", self.server_ngram_min.value())
        settings.setValue("server/spec_ngram_match", self.server_ngram_match.value())
        settings.setValue("server/spec_ngram_max", self.server_ngram_max.value())

        settings.setValue("server/flash_attn", self.server_flash_attn_check.isChecked())
        settings.setValue("server/no_warmup", self.server_no_warmup_check.isChecked())
        settings.setValue("server/no_mmap", self.server_no_mmap_check.isChecked())
        settings.setValue("server/disable_thinking", self.server_disable_thinking_check.isChecked())
        settings.setValue("server/auto_fit", self.server_auto_fit_check.isChecked())

        settings.setValue("server/cors", self.server_cors_check.isChecked())
        settings.setValue("server/api_key", self.server_api_key_input.text())
        settings.setValue("server/extra_args", self.server_extra_args.toPlainText())

        settings.setValue("server/model_path", self.server_model_path.text().strip())
        model_name = self.server_models_combo.currentText()
        if model_name and model_name not in ("-- Select Model --", "-- No .gguf models found --"):
            settings.setValue("server/model_name", model_name)
