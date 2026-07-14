"""Server tab - Launch llama-server"""

import json
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QTextEdit, QFileDialog, QMessageBox, QScrollArea, QSplitter,
    QSizePolicy
)
import os
import re
import shlex

from PyQt6.QtCore import Qt, QTimer

from backend_names import backend_key_from_display, display_backend_from_key
from threads import ServerThread
from server_backend_panels import BackendPanels
from server_monitor import ServerMonitorPanel, ServerMonitorThread
from server_presets import ServerPresetsMixin
from ui_widgets import CollapsibleSection, LogView, StatusPill


class ServerTabWidget(ServerPresetsMixin, QWidget):
    """Tab for launching llama-server.

    Preset application and speculative-decoding profile logic live in
    ServerPresetsMixin (server_presets.py); backend-specific sub-tab panels
    in server_backend_panels.py.
    """

    NGRAM_MOD_N_MIN = 12
    NGRAM_MOD_N_MATCH = 16
    NGRAM_MOD_N_MAX = 32
    MTP_DRAFT_N_MAX = 2
    SERVER_CONTEXT_MAX = 262144
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.models_dir = parent.models_dir if hasattr(parent, "models_dir") else Path("models")
        self.server_thread = None
        self.server_process = None
        self._memory_fit_warning_shown = False
        self._registered_build_map: dict[str, dict[str, object]] = {}
        self._model_presets = self._load_model_presets()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._update_command_preview)
        self.create_ui()
        self._on_vision_toggled(False)
        self.refresh_server_build_choices()
        self.refresh_server_models_list()
        self.load_settings()
        self.on_spec_type_changed()  # sync spec-field enablement with loaded type
        self._wire_command_preview()

        self.monitor_thread = ServerMonitorThread(self)
        self.monitor_thread.stats_ready.connect(self.monitor_panel.update_stats)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.monitor_thread.stop)
        self.monitor_thread.start()

    @staticmethod
    def _grid_pair(grid, row: int, pair: int, text: str, widget) -> None:
        """Place a right-aligned label + uniformly sized field into a grid.

        Pair N occupies columns 2N (label) and 2N+1 (field) so every group
        using this helper gets the same visual rhythm.
        """
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label, row, pair * 2)
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setFixedWidth(96)
        elif isinstance(widget, QComboBox):
            widget.setMinimumWidth(110)
        grid.addWidget(widget, row, pair * 2 + 1)

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

        # Model list: label + combo + refresh on one row
        model_list_row = QHBoxLayout()
        models_label = QLabel("Available Models:")
        models_label.setFixedWidth(108)
        model_list_row.addWidget(models_label)
        self.server_models_combo = QComboBox()
        self.server_models_combo.currentTextChanged.connect(self.on_server_model_selected)
        model_list_row.addWidget(self.server_models_combo, 1)

        self.server_models_refresh_btn = QPushButton("🔄 Refresh")
        self.server_models_refresh_btn.clicked.connect(self.refresh_server_models_list)
        model_list_row.addWidget(self.server_models_refresh_btn)
        model_layout.addLayout(model_list_row)

        vision_row = QHBoxLayout()
        self.server_vision_check = QCheckBox("Vision")
        self.server_vision_check.setToolTip("Load a multimodal projector with --mmproj")
        self.server_vision_check.toggled.connect(self._on_vision_toggled)
        vision_row.addWidget(self.server_vision_check)

        self.server_mmproj_path = QLineEdit()
        self.server_mmproj_path.setPlaceholderText("Path to mmproj GGUF...")
        vision_row.addWidget(self.server_mmproj_path, 1)

        self.server_mmproj_browse_btn = QPushButton("Browse...")
        self.server_mmproj_browse_btn.clicked.connect(self.browse_server_mmproj)
        vision_row.addWidget(self.server_mmproj_browse_btn)

        self.server_mmproj_offload_check = QCheckBox("GPU offload")
        self.server_mmproj_offload_check.setChecked(True)
        self.server_mmproj_offload_check.setToolTip("Keep the vision encoder on GPU; disable to save VRAM")
        vision_row.addWidget(self.server_mmproj_offload_check)
        model_layout.addLayout(vision_row)

        # Presets: quick presets apply on selection; the model preset comes
        # from gui/model_presets.json and is re-applied automatically when a
        # model is picked — the button is a manual re-apply.
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Quick preset:")
        preset_label.setFixedWidth(108)
        preset_layout.addWidget(preset_label)
        self.server_preset_combo = QComboBox()
        self.server_preset_combo.addItems(["Default", "Fast", "Quality", "Balanced", "VRAM Limited", "CPU Fallback"])
        self.server_preset_combo.setToolTip("Generic starting points; applied immediately on selection")
        self.server_preset_combo.currentIndexChanged.connect(self.apply_server_preset)
        preset_layout.addWidget(self.server_preset_combo)
        preset_layout.addStretch()

        self.server_apply_model_preset_btn = QPushButton("✨ Re-apply Model Preset")
        self.server_apply_model_preset_btn.setToolTip(
            "Apply the matching preset for this model from gui/model_presets.json.\n"
            "Runs automatically when a model is selected — use this to re-apply\n"
            "after manual changes."
        )
        self.server_apply_model_preset_btn.clicked.connect(self.apply_model_file_preset)
        preset_layout.addWidget(self.server_apply_model_preset_btn)
        model_layout.addLayout(preset_layout)

        model_group.setLayout(model_layout)
        scroll_layout.addWidget(model_group)

        # Server settings — one aligned grid: label/field pairs in fixed columns
        server_group = QGroupBox("Server Configuration")
        server_grid = QGridLayout()
        server_grid.setHorizontalSpacing(10)
        server_grid.setVerticalSpacing(8)

        self.server_host_input = QLineEdit()
        self.server_host_input.setText("0.0.0.0")
        self.server_host_input.setFixedWidth(110)

        self.server_port_spinbox = QSpinBox()
        self.server_port_spinbox.setMinimum(1024)
        self.server_port_spinbox.setMaximum(65535)
        self.server_port_spinbox.setValue(8000)

        self.server_mode_combo = QComboBox()
        self.server_mode_combo.addItems(["Inference", "Embedding"])
        self.server_mode_combo.setToolTip("Embedding starts the server with --embeddings (embedding endpoint only)")

        self.server_parallel_spinbox = QSpinBox()
        self.server_parallel_spinbox.setMinimum(1)
        self.server_parallel_spinbox.setMaximum(8)
        self.server_parallel_spinbox.setValue(1)

        self.server_threads_spinbox = QSpinBox()
        self.server_threads_spinbox.setMinimum(1)
        self.server_threads_spinbox.setMaximum(64)
        self.server_threads_spinbox.setValue(8)

        self.server_batch_spinbox = QSpinBox()
        self.server_batch_spinbox.setMinimum(32)
        self.server_batch_spinbox.setMaximum(8192)
        self.server_batch_spinbox.setValue(2048)
        self.server_batch_spinbox.setSingleStep(32)

        self.server_ubatch_spinbox = QSpinBox()
        self.server_ubatch_spinbox.setMinimum(32)
        self.server_ubatch_spinbox.setMaximum(8192)
        self.server_ubatch_spinbox.setValue(512)
        self.server_ubatch_spinbox.setSingleStep(32)

        self.server_http_threads_spinbox = QSpinBox()
        self.server_http_threads_spinbox.setMinimum(1)
        self.server_http_threads_spinbox.setMaximum(64)
        self.server_http_threads_spinbox.setValue(max(1, (os.cpu_count() or 8) // 2))

        self.server_build_backend_combo = QComboBox()
        self.server_build_backend_combo.currentTextChanged.connect(self._on_build_backend_changed)
        self.server_build_version_combo = QComboBox()

        self._grid_pair(server_grid, 0, 0, "Host:", self.server_host_input)
        self._grid_pair(server_grid, 0, 1, "Port:", self.server_port_spinbox)
        self._grid_pair(server_grid, 0, 2, "Mode:", self.server_mode_combo)
        self._grid_pair(server_grid, 0, 3, "Parallel:", self.server_parallel_spinbox)
        self._grid_pair(server_grid, 1, 0, "Threads:", self.server_threads_spinbox)
        self._grid_pair(server_grid, 1, 1, "Batch:", self.server_batch_spinbox)
        self._grid_pair(server_grid, 1, 2, "UBatch:", self.server_ubatch_spinbox)
        self._grid_pair(server_grid, 1, 3, "HTTP Threads:", self.server_http_threads_spinbox)
        self._grid_pair(server_grid, 2, 0, "Backend:", self.server_build_backend_combo)
        backend_version_label = QLabel("Version:")
        backend_version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        server_grid.addWidget(backend_version_label, 2, 2)
        server_grid.addWidget(self.server_build_version_combo, 2, 3, 1, 5)
        server_grid.setColumnStretch(8, 1)

        server_group.setLayout(server_grid)
        scroll_layout.addWidget(server_group)

        # Backend-specific settings: sub-tabs per build backend (ROCm/Vulkan/CPU).
        # The active tab follows the Build Backend selection; only the active
        # tab's parameters are applied to the launch command.
        backend_panels_group = QGroupBox("Backend Settings")
        backend_panels_layout = QVBoxLayout()
        self.backend_panels = BackendPanels()
        self.backend_panels.setToolTip(
            "Parameters specific to the selected build backend. The active tab\n"
            "follows the Build Backend above and is applied on launch."
        )
        backend_panels_layout.addWidget(self.backend_panels)
        backend_panels_group.setLayout(backend_panels_layout)
        scroll_layout.addWidget(backend_panels_group)

        # Resources + Speculative Decoding share one row: params are narrow,
        # so pairing groups uses the panel width instead of growing downward
        res_spec_row = QHBoxLayout()
        res_spec_row.setSpacing(8)

        resources_group = QGroupBox("Resources")
        resources_grid = QGridLayout()
        resources_grid.setHorizontalSpacing(10)
        resources_grid.setVerticalSpacing(8)

        self.server_gpu_layers_spinbox = QSpinBox()
        self.server_gpu_layers_spinbox.setMinimum(0)
        self.server_gpu_layers_spinbox.setMaximum(999)
        self.server_gpu_layers_spinbox.setValue(99)

        self.server_context_spinbox = QSpinBox()
        self.server_context_spinbox.setMinimum(8192)
        self.server_context_spinbox.setMaximum(self.SERVER_CONTEXT_MAX)
        self.server_context_spinbox.setValue(32768)
        self.server_context_spinbox.setSingleStep(8192)
        self.server_context_spinbox.setToolTip(
            "Server context size in tokens. Qwen3.6 advertises 262144 tokens;\n"
            "larger values still require enough VRAM/RAM and may need quantized KV cache."
        )

        self.server_kv_type_combo = QComboBox()
        self.server_kv_type_combo.addItems([
            "f16", "bf16", "f32", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl",
            "tbq4_0", "tbq3_0", "tq3_0",
        ])
        self.server_kv_type_combo.setCurrentText("f16")
        self.server_kv_type_combo.setToolTip(
            "KV cache quantization types:\n"
            "f16: default half-precision\n"
            "bf16: bfloat16, recommended for Qwen3.5/3.6\n"
            "q8_0/q5_x/q4_x/iq4_nl: quantized KV cache\n"
            "tbq4_0/tbq3_0: TurboQuant CPU-only, forces -ngl 0\n"
            "tq3_0: TurboQuant GPU KV cache\n"
            "TurboQuant KV types force flash_attn=on."
        )

        self.server_prompt_cache_ram_spinbox = QSpinBox()
        self.server_prompt_cache_ram_spinbox.setRange(0, 65536)
        self.server_prompt_cache_ram_spinbox.setSingleStep(256)
        self.server_prompt_cache_ram_spinbox.setValue(0)
        self.server_prompt_cache_ram_spinbox.setToolTip(
            "RAM budget for saved idle prompts. Zero disables the separate prompt cache; active-slot prefix reuse remains available."
        )

        self.server_ctx_checkpoints_spinbox = QSpinBox()
        self.server_ctx_checkpoints_spinbox.setRange(0, 64)
        self.server_ctx_checkpoints_spinbox.setValue(4)
        self.server_ctx_checkpoints_spinbox.setToolTip(
            "Number of recurrent/SWA rollback checkpoints retained per slot. Qwen3.6 checkpoints are about 150 MiB each."
        )

        self.server_checkpoint_interval_spinbox = QSpinBox()
        self.server_checkpoint_interval_spinbox.setRange(-1, self.SERVER_CONTEXT_MAX)
        self.server_checkpoint_interval_spinbox.setSpecialValueText("Off")
        self.server_checkpoint_interval_spinbox.setSingleStep(1024)
        self.server_checkpoint_interval_spinbox.setValue(-1)
        self.server_checkpoint_interval_spinbox.setToolTip(
            "Checkpoint interval during long prefill. Off still permits the required near-end rollback checkpoint."
        )

        self._grid_pair(resources_grid, 0, 0, "GPU Layers:", self.server_gpu_layers_spinbox)
        self._grid_pair(resources_grid, 0, 1, "Context:", self.server_context_spinbox)
        self._grid_pair(resources_grid, 1, 0, "KV Cache:", self.server_kv_type_combo)
        self._grid_pair(resources_grid, 1, 1, "Prompt Cache MiB:", self.server_prompt_cache_ram_spinbox)
        self._grid_pair(resources_grid, 2, 0, "Checkpoints:", self.server_ctx_checkpoints_spinbox)
        self._grid_pair(resources_grid, 2, 1, "Interval:", self.server_checkpoint_interval_spinbox)
        resources_grid.setColumnStretch(4, 1)
        resources_grid.setRowStretch(3, 1)
        resources_group.setLayout(resources_grid)
        res_spec_row.addWidget(resources_group, 1)

        # Speculative Decoding — separate sub-grids per parameter family so
        # on_spec_type_changed can enable/disable them as before
        spec_group = QGroupBox("Speculative Decoding")
        spec_layout = QVBoxLayout()
        spec_layout.setSpacing(8)

        spec_type_grid = QGridLayout()
        spec_type_grid.setHorizontalSpacing(10)
        spec_type_grid.setColumnMinimumWidth(0, 84)
        self.server_spec_type_combo = QComboBox()
        self.server_spec_type_combo.addItems(["None", "draft", "ngram-mod", "mtp"])
        self.server_spec_type_combo.currentTextChanged.connect(self.on_spec_type_changed)
        self._grid_pair(spec_type_grid, 0, 0, "Type:", self.server_spec_type_combo)
        spec_type_grid.setColumnStretch(2, 1)
        spec_layout.addLayout(spec_type_grid)

        # Speculative params (enabled only for mtp/draft types)
        draft_grid = QGridLayout()
        draft_grid.setHorizontalSpacing(10)
        draft_grid.setColumnMinimumWidth(0, 84)
        self.server_spec_draft_n_max = QSpinBox()
        self.server_spec_draft_n_max.setMinimum(1)
        self.server_spec_draft_n_max.setMaximum(20)
        self.server_spec_draft_n_max.setValue(self.MTP_DRAFT_N_MAX)
        self.server_spec_draft_n_max.setToolTip(
            "--spec-draft-n-max: max draft tokens per verify step. "
            "2 is the safer default for long prompts; tune higher values manually."
        )
        self._grid_pair(draft_grid, 0, 0, "Draft Max N:", self.server_spec_draft_n_max)
        draft_grid.setColumnStretch(2, 1)
        spec_layout.addLayout(draft_grid)
        self.draft_layout_group = draft_grid

        # NGram parameters
        ngram_grid = QGridLayout()
        ngram_grid.setHorizontalSpacing(10)
        ngram_grid.setColumnMinimumWidth(0, 84)
        self.server_ngram_min = QSpinBox()
        self.server_ngram_min.setMinimum(1)
        self.server_ngram_min.setMaximum(512)
        self.server_ngram_min.setValue(self.NGRAM_MOD_N_MIN)

        self.server_ngram_match = QSpinBox()
        self.server_ngram_match.setMinimum(1)
        self.server_ngram_match.setMaximum(512)
        self.server_ngram_match.setValue(self.NGRAM_MOD_N_MATCH)

        self.server_ngram_max = QSpinBox()
        self.server_ngram_max.setMinimum(1)
        self.server_ngram_max.setMaximum(512)
        self.server_ngram_max.setValue(self.NGRAM_MOD_N_MAX)

        self._grid_pair(ngram_grid, 0, 0, "NGram Min:", self.server_ngram_min)
        self._grid_pair(ngram_grid, 0, 1, "Match:", self.server_ngram_match)
        self._grid_pair(ngram_grid, 0, 2, "Max:", self.server_ngram_max)
        ngram_grid.setColumnStretch(6, 1)
        spec_layout.addLayout(ngram_grid)

        self.ngram_layout_group = ngram_grid
        self.ngram_layout_group.setEnabled(False)

        spec_layout.addStretch(1)
        spec_group.setLayout(spec_layout)
        res_spec_row.addWidget(spec_group, 1)
        scroll_layout.addLayout(res_spec_row)

        # Sampling + Performance Options share the next row
        samp_perf_row = QHBoxLayout()
        samp_perf_row.setSpacing(8)

        sampling_group = QGroupBox("Sampling Parameters")
        sampling_grid = QGridLayout()
        sampling_grid.setHorizontalSpacing(10)
        sampling_grid.setVerticalSpacing(8)

        self.server_temperature_spinbox = QDoubleSpinBox()
        self.server_temperature_spinbox.setMinimum(0.0)
        self.server_temperature_spinbox.setMaximum(2.0)
        self.server_temperature_spinbox.setValue(0.7)
        self.server_temperature_spinbox.setSingleStep(0.1)

        self.server_top_p_spinbox = QDoubleSpinBox()
        self.server_top_p_spinbox.setMinimum(0.0)
        self.server_top_p_spinbox.setMaximum(1.0)
        self.server_top_p_spinbox.setValue(0.95)
        self.server_top_p_spinbox.setSingleStep(0.05)

        self.server_top_k_spinbox = QSpinBox()
        self.server_top_k_spinbox.setMinimum(0)
        self.server_top_k_spinbox.setMaximum(500)
        self.server_top_k_spinbox.setValue(40)

        self._grid_pair(sampling_grid, 0, 0, "Temperature:", self.server_temperature_spinbox)
        self._grid_pair(sampling_grid, 0, 1, "Top-P:", self.server_top_p_spinbox)
        self._grid_pair(sampling_grid, 1, 0, "Top-K:", self.server_top_k_spinbox)
        sampling_grid.setColumnStretch(4, 1)
        sampling_grid.setRowStretch(2, 1)
        sampling_group.setLayout(sampling_grid)
        samp_perf_row.addWidget(sampling_group, 1)

        perf_group = QGroupBox("Performance Options")
        perf_layout = QVBoxLayout()
        perf_layout.setSpacing(6)

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
            "Disable thinking for throughput (--chat-template-kwargs)"
        )
        self.server_disable_thinking_check.setChecked(False)
        perf_layout.addWidget(self.server_disable_thinking_check)

        self.server_auto_fit_check = QCheckBox("Auto-fit params to free memory (-fit on)")
        self.server_auto_fit_check.setChecked(True)
        perf_layout.addWidget(self.server_auto_fit_check)

        perf_layout.addStretch(1)
        perf_group.setLayout(perf_layout)
        samp_perf_row.addWidget(perf_group, 1)
        scroll_layout.addLayout(samp_perf_row)

        # Security & API + Extra Arguments share the last row
        sec_extra_row = QHBoxLayout()
        sec_extra_row.setSpacing(8)

        cors_group = QGroupBox("Security && API")
        cors_layout = QVBoxLayout()
        cors_layout.setSpacing(8)

        self.server_cors_check = QCheckBox("Enable CORS")
        self.server_cors_check.setChecked(True)
        cors_layout.addWidget(self.server_cors_check)

        api_key_layout = QHBoxLayout()
        api_key_label = QLabel("API Key:")
        api_key_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        api_key_layout.addWidget(api_key_label)
        self.server_api_key_input = QLineEdit()
        self.server_api_key_input.setPlaceholderText("Leave empty for no key")
        self.server_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.server_api_key_input, 1)
        cors_layout.addLayout(api_key_layout)

        cors_layout.addStretch(1)
        cors_group.setLayout(cors_layout)
        sec_extra_row.addWidget(cors_group, 1)

        extra_args_group = QGroupBox("Extra Arguments")
        extra_args_layout = QVBoxLayout()

        self.server_extra_args = QTextEdit()
        self.server_extra_args.setPlaceholderText("Additional llama-server arguments (one per line)")
        self.server_extra_args.setMaximumHeight(80)
        extra_args_layout.addWidget(self.server_extra_args)

        extra_args_group.setLayout(extra_args_layout)
        sec_extra_row.addWidget(extra_args_group, 1)
        scroll_layout.addLayout(sec_extra_row)
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
        self.server_start_btn.setStyleSheet(
            "QPushButton { font-size: 12px; padding: 8px; background-color: #4CAF50; color: white; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #808080; }"
        )
        buttons_layout.addWidget(self.server_start_btn)

        self.server_stop_btn = QPushButton("⏹️ Stop Server")
        self.server_stop_btn.clicked.connect(self.stop_server)
        self.server_stop_btn.setEnabled(False)
        self.server_stop_btn.setStyleSheet(
            "QPushButton { font-size: 12px; padding: 8px; background-color: #f44336; color: white; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #808080; }"
        )
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

        # Status pill
        status_row = QHBoxLayout()
        self.server_status_label = StatusPill("● Stopped")
        status_row.addWidget(self.server_status_label)
        status_row.addStretch()
        right_layout.addLayout(status_row)

        # Live monitor: GPU/RAM load + server throughput from /metrics
        self.monitor_panel = ServerMonitorPanel()
        right_layout.addWidget(self.monitor_panel)

        # Live command preview: exactly what Start Server will run
        self.server_cmd_preview = QTextEdit()
        self.server_cmd_preview.setReadOnly(True)
        self.server_cmd_preview.setMaximumHeight(190)
        self.server_cmd_preview.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 11px;")
        self.server_cmd_preview.setToolTip("Full llama-server command with env overrides, updated as you change settings")
        settings = self.parent.settings if hasattr(self.parent, "settings") else None
        preview_section = CollapsibleSection(
            "Command Preview",
            self.server_cmd_preview,
            settings=settings,
            settings_key="server/command_preview_expanded",
        )
        right_layout.addWidget(preview_section)

        # Log output
        log_label = QLabel("Server Output Log:")
        right_layout.addWidget(log_label)

        self.server_log = LogView()
        self.server_log.setMinimumHeight(220)
        self.server_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.server_log, 1)

        # Add panels to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        # params benefit from width (paired groups); monitor/logs don't
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([950, 450])

        layout.addWidget(splitter, 1)

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

    def browse_server_mmproj(self):
        """Browse for a multimodal projector GGUF."""
        current = self.server_mmproj_path.text().strip()
        start_dir = str(Path(current).parent) if current else str(self.models_dir)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Multimodal Projector",
            start_dir,
            "GGUF Files (*.gguf);;All Files (*.*)",
        )
        if file_path:
            self.server_mmproj_path.setText(file_path)
            self.server_vision_check.setChecked(True)

    def _on_vision_toggled(self, enabled: bool):
        self.server_mmproj_path.setEnabled(enabled)
        self.server_mmproj_browse_btn.setEnabled(enabled)
        self.server_mmproj_offload_check.setEnabled(enabled)
        if enabled and not self.server_mmproj_path.text().strip():
            self._autodetect_mmproj()

    def _autodetect_mmproj(self):
        model_path = Path(self.server_model_path.text().strip())
        search_dir = model_path.parent if model_path.parent.exists() else self.models_dir
        model_name = model_path.name.lower()
        projectors = sorted(search_dir.glob("*mmproj*.gguf"), key=lambda p: p.name.lower())

        family_match = re.search(r"qwen3[._-]?[56]", model_name)
        size_match = re.search(r"(\d+(?:\.\d+)?b(?:-a\d+b)?)", model_name)
        if family_match and size_match:
            family = family_match.group(0).replace("_", ".").replace("-", ".")
            size = size_match.group(1).replace("-", ".")
            for candidate in projectors:
                candidate_name = candidate.name.lower().replace("_", ".").replace("-", ".")
                if family in candidate_name and size in candidate_name:
                    self.server_mmproj_path.setText(str(candidate))
                    return

        preferred = ["mmproj-F16.gguf", "mmproj-BF16.gguf", "mmproj-F32.gguf"]
        candidates = [search_dir / name for name in preferred]
        candidates.extend(projectors)
        for candidate in candidates:
            if candidate.exists():
                self.server_mmproj_path.setText(str(candidate))
                return

    def on_server_model_selected(self, model_name: str):
        """Handle model selection from combo"""
        if model_name and model_name != "-- Select Model --" and model_name != "-- No .gguf models found --":
            model_path = self.models_dir / model_name
            self.server_model_path.setText(str(model_path))
            self.apply_model_file_preset()
            if self.server_vision_check.isChecked():
                self._autodetect_mmproj()

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

    @staticmethod
    def _vulkan_runtime_env() -> dict[str, str]:
        return {
            "GGML_VK_FORCE_AMD_LARGE_MATMUL": "1",
        }

    def _compose_server_command(self) -> tuple[list[str], dict[str, str] | None, Path, list[str], list[str]]:
        """Build the launch command and env from current UI state, side-effect free.

        Returns (command, env, build_dir, problems, notes). Non-empty problems
        means the command is not launchable as composed; notes are informational.
        """
        problems: list[str] = []
        notes: list[str] = []

        model_path = self.server_model_path.text().strip()
        if not model_path or not Path(model_path).exists():
            problems.append("Select a valid model file")

        build_dir = self._resolve_selected_build_dir()
        llama_server = self._find_llama_server_binary(build_dir)
        if not llama_server:
            problems.append(
                f"llama-server not found in {build_dir} — build the selected backend in the Build tab"
            )

        port = self.server_port_spinbox.value()
        command = [
            str(llama_server) if llama_server else "llama-server",
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

        kv_type = self.server_kv_type_combo.currentText().strip()
        turboq_cpu_only = kv_type.startswith("tbq")
        turboq_kv = turboq_cpu_only or kv_type.startswith("tq")

        if turboq_cpu_only:
            command.extend(["-ngl", "0"])
        else:
            command.extend(["-ngl", str(self.server_gpu_layers_spinbox.value())])

        if kv_type and kv_type != "f16":
            command.extend(["--cache-type-k", kv_type, "--cache-type-v", kv_type])

        extra_args = self.server_extra_args.toPlainText().strip()
        extra_tokens = []
        if extra_args:
            try:
                extra_tokens = shlex.split(extra_args, posix=(os.name != "nt"))
            except ValueError:
                extra_tokens = extra_args.split()

        if self._spec_type_from_tokens(extra_tokens) == "ngram-mod":
            extra_tokens = self._normalize_ngram_extra_tokens(extra_tokens)

        # Extra Arguments override the visible speculative mode only when they
        # explicitly provide --spec-type. Other --spec-* knobs may be combined
        # with the UI mode.
        has_spec_type_in_extra = self._spec_type_from_tokens(extra_tokens) is not None
        has_spec_draft_n_max_in_extra = any(
            tok == "--spec-draft-n-max" or tok.startswith("--spec-draft-n-max=")
            for tok in extra_tokens
        )
        has_no_mmap_in_extra = "--no-mmap" in extra_tokens

        def extra_has_any(*flags: str) -> bool:
            return any(tok in flags or any(tok.startswith(flag + "=") for flag in flags) for tok in extra_tokens)

        # Prometheus endpoint feeds the Live Monitor panel
        if "--metrics" not in extra_tokens:
            command.append("--metrics")

        if not extra_has_any("--cache-ram"):
            command.extend(["--cache-ram", str(self.server_prompt_cache_ram_spinbox.value())])
        if not extra_has_any("-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"):
            command.extend(["--ctx-checkpoints", str(self.server_ctx_checkpoints_spinbox.value())])
        if not extra_has_any("-cpent", "--checkpoint-every-n-tokens"):
            command.extend(["--checkpoint-every-n-tokens", str(self.server_checkpoint_interval_spinbox.value())])

        if self.server_vision_check.isChecked():
            mmproj_path = self.server_mmproj_path.text().strip()
            if not mmproj_path or not Path(mmproj_path).is_file():
                problems.append("Vision is enabled but the mmproj file is missing")
            elif not extra_has_any("-mm", "--mmproj"):
                command.extend(["--mmproj", mmproj_path])
            if not self.server_mmproj_offload_check.isChecked() and not extra_has_any("--no-mmproj-offload"):
                command.append("--no-mmproj-offload")

        spec_type = self.server_spec_type_combo.currentText().strip().lower()
        if not has_spec_type_in_extra:
            if spec_type == "mtp":
                command.extend(["--spec-type", "draft-mtp"])
                if not has_spec_draft_n_max_in_extra:
                    command.extend(["--spec-draft-n-max", str(self.server_spec_draft_n_max.value())])
                # MTP currently requires single parallel sequence in llama-server.
                if self.server_parallel_spinbox.value() != 1:
                    notes.append("MTP requires --parallel 1, overriding selected value")
                    command.extend(["--parallel", "1"])
            elif spec_type == "ngram-mod":
                command.extend(self._ngram_mod_args())

        if self.server_mode_combo.currentText() == "Embedding":
            command.append("--embeddings")

        if self.server_flash_attn_check.isChecked() or turboq_kv:
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

        if turboq_cpu_only or not self.server_auto_fit_check.isChecked():
            command.extend(["-fit", "off"])

        # Backend-specific args from the active Backend Settings sub-tab
        # (device selection / split mode etc.). Skipped when the user already
        # provided the same flag via Extra Arguments.
        backend_args = self.backend_panels.args()
        skip = False
        filtered_backend_args: list[str] = []
        for tok in backend_args:
            if skip:
                skip = False
                continue
            if tok in (
                "-dev", "--device",
                "-sm", "--split-mode",
                "-ts", "--tensor-split",
                "-ngl", "--gpu-layers", "--n-gpu-layers",
            ) and tok in extra_tokens:
                skip = True  # drop the flag and its value; Extra Arguments wins
                continue
            filtered_backend_args.append(tok)
        command.extend(filtered_backend_args)

        if extra_tokens:
            command.extend(extra_tokens)

        api_key = self.server_api_key_input.text().strip()
        if api_key:
            command.extend(["--api-key", api_key])

        server_env = None
        if "rocm" in build_dir.name.lower() and hasattr(self.parent, "build_manager"):
            server_env = self.parent.build_manager.get_rocm_env()
        elif "vulkan" in build_dir.name.lower():
            server_env = self._vulkan_runtime_env()

        # merge backend-panel env overrides (panel values win)
        panel_env = self.backend_panels.env()
        if panel_env:
            server_env = {**(server_env or {}), **panel_env}

        return command, server_env, build_dir, problems, notes

    @staticmethod
    def _effective_ngl(command: list[str]) -> int | None:
        """Last -ngl value in the command (later flags win in llama-server)."""
        value = None
        for i, tok in enumerate(command[:-1]):
            if tok in ("-ngl", "--gpu-layers", "--n-gpu-layers"):
                try:
                    value = int(command[i + 1])
                except ValueError:
                    pass
        return value

    # -- live command preview --------------------------------------------------
    def _wire_command_preview(self):
        """Debounce-connect every setting widget to the command preview."""
        for combo in self.findChildren(QComboBox):
            combo.currentIndexChanged.connect(self._schedule_command_preview)
        for spin in self.findChildren(QSpinBox):
            spin.valueChanged.connect(self._schedule_command_preview)
        for spin in self.findChildren(QDoubleSpinBox):
            spin.valueChanged.connect(self._schedule_command_preview)
        for check in self.findChildren(QCheckBox):
            check.toggled.connect(self._schedule_command_preview)
        for line in self.findChildren(QLineEdit):
            line.textChanged.connect(self._schedule_command_preview)
        self.server_extra_args.textChanged.connect(self._schedule_command_preview)
        self._update_command_preview()

    def _schedule_command_preview(self, *_args):
        self._preview_timer.start()

    def _update_command_preview(self):
        command, server_env, build_dir, problems, notes = self._compose_server_command()

        lines: list[str] = []
        if problems:
            lines.extend(f"⚠ {problem}" for problem in problems)
            lines.append("")
        lines.append(f"# build: {build_dir.name}")
        # env dicts may be full os.environ copies (rocm env); show only overrides
        for key, value in sorted((server_env or {}).items()):
            if os.environ.get(key) != value:
                if len(value) > 120:
                    value = value[:117] + "…"
                lines.append(f"ENV {key}={value}")

        # one flag (with its values) per line, binary first; mask the API key
        current = Path(command[0]).name
        for tok in command[1:]:
            if tok.startswith("-"):
                lines.append(current)
                current = "  " + tok
            else:
                value = "••••" if current.strip() == "--api-key" else tok
                current += f" {value}"
        lines.append(current)

        for note in notes:
            lines.append(f"# note: {note}")

        self.server_cmd_preview.setPlainText("\n".join(lines))

    def start_server(self):
        """Start llama-server"""
        if self.server_thread and self.server_thread.isRunning():
            QMessageBox.information(self, "Server", "Server is already running")
            return

        command, server_env, build_dir, problems, notes = self._compose_server_command()
        if problems:
            QMessageBox.warning(self, "Server", "\n".join(problems))
            return

        # ngl=0 on a GPU build silently runs the whole model on CPU
        if "cpu" not in build_dir.name.lower() and self._effective_ngl(command) == 0:
            answer = QMessageBox.question(
                self,
                "GPU layers = 0",
                "GPU Layers is 0 on a GPU build — the model will run entirely on CPU.\n\n"
                "Start anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.server_start_btn.setEnabled(False)
        self.server_stop_btn.setEnabled(True)
        self.server_web_btn.setEnabled(False)
        self.server_status_label.set_state("busy", "Starting…")

        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Starting server...")

        self.server_log.clear()
        self._memory_fit_warning_shown = False
        self.server_log.append(f"[INFO] Build: {build_dir}")
        self.server_log.append(f"[INFO] Command: {' '.join(command)}")
        for note in notes:
            self.server_log.append(f"[INFO] {note}")
        if server_env and "GGML_VK_FORCE_AMD_LARGE_MATMUL" in server_env:
            env_summary = ", ".join(f"{key}={value}" for key, value in sorted(server_env.items()))
            self.server_log.append(f"[INFO] Env overrides: {env_summary}")

        port = self.server_port_spinbox.value()
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
            self.parent.project_root / "build-vulkan",
        ]
        for candidate in candidates:
            if self._find_llama_server_binary(candidate):
                return candidate
        return self.parent.project_root / "build"

    _display_backend_from_key = staticmethod(display_backend_from_key)
    _backend_key_from_display = staticmethod(backend_key_from_display)

    def _on_build_backend_changed(self, *_args):
        self.refresh_server_build_versions_for_backend(select_latest=True)

        # keep the backend-specific settings sub-tab in sync with the selection
        if hasattr(self, "backend_panels"):
            display = self.server_build_backend_combo.currentText().strip()
            if display and display != "Auto":
                self.backend_panels.set_backend(self._backend_key_from_display(display))

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
        for legacy in ["ROCm/HIP", "CPU", "Vulkan"]:
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
        self.server_status_label.set_state("ok", f"Running ({url})")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(f"Server running: {url}")

        pid = None
        if self.server_thread is not None and getattr(self.server_thread, "process", None) is not None:
            pid = self.server_thread.process.pid
        self.monitor_thread.set_server(
            url,
            api_key=self.server_api_key_input.text().strip(),
            pid=pid,
            ctx_tokens=self.server_context_spinbox.value(),
        )

    def on_server_finished(self, exit_code: int):
        """Handle server process exit"""
        self.monitor_thread.clear_server()
        self.server_start_btn.setEnabled(True)
        self.server_stop_btn.setEnabled(False)
        self.server_web_btn.setEnabled(False)
        if exit_code == 0:
            self.server_status_label.set_state("neutral", "Stopped")
            self.server_log.append("[INFO] Server stopped")
        else:
            self.server_status_label.set_state("error", f"Crashed (exit {exit_code})")
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
            self.server_status_label.set_state("ok", "Running ✓")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Server running on port " + str(self.server_port_spinbox.value()))
        elif "error" in message.lower():
            self.server_status_label.set_state("error")

    def on_server_error(self, error: str):
        """Handle server error"""
        self.monitor_thread.clear_server()
        self.server_log.append(f"[ERROR] {error}")
        self.server_status_label.set_state("error", "Error ✗")
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
            self.server_stop_btn.setEnabled(False)
            self.server_web_btn.setEnabled(False)
            self.server_status_label.set_state("busy", "Stopping…")
            self.server_log.append("[INFO] Graceful server shutdown requested")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Stopping server gracefully")
            return

        self.on_server_finished(0)

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

        try:
            panels_raw = settings.value("server/backend_panels", "")
            if panels_raw:
                self.backend_panels.from_settings(json.loads(panels_raw))
        except (ValueError, TypeError):
            pass  # ignore malformed saved state
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
        self.server_prompt_cache_ram_spinbox.setValue(int(settings.value("server/prompt_cache_ram", 0)))
        self.server_ctx_checkpoints_spinbox.setValue(int(settings.value("server/ctx_checkpoints", 4)))
        self.server_checkpoint_interval_spinbox.setValue(int(settings.value("server/checkpoint_interval", -1)))

        self.server_vision_check.setChecked(settings.value("server/vision", False, type=bool))
        self.server_mmproj_path.setText(settings.value("server/mmproj_path", ""))
        self.server_mmproj_offload_check.setChecked(settings.value("server/mmproj_offload", True, type=bool))
        self._on_vision_toggled(self.server_vision_check.isChecked())

        self.server_temperature_spinbox.setValue(float(settings.value("server/temperature", 0.7)))
        self.server_top_p_spinbox.setValue(float(settings.value("server/top_p", 0.95)))
        self.server_top_k_spinbox.setValue(int(settings.value("server/top_k", 40)))

        saved_spec_type = str(settings.value("server/spec_type", "None")).strip()
        if saved_spec_type.lower() == "draft-mtp":
            saved_spec_type = "mtp"
        self.server_spec_type_combo.setCurrentText(saved_spec_type)
        spec_draft_n_max = int(settings.value("server/spec_draft_n_max", self.MTP_DRAFT_N_MAX))
        if not settings.value("server/spec_draft_n_max_default_migrated_v2", False, type=bool):
            if spec_draft_n_max == 8:
                spec_draft_n_max = self.MTP_DRAFT_N_MAX
                settings.setValue("server/spec_draft_n_max", spec_draft_n_max)
            settings.setValue("server/spec_draft_n_max_default_migrated_v2", True)
        self.server_spec_draft_n_max.setValue(spec_draft_n_max)
        self.server_ngram_min.setValue(int(settings.value("server/spec_ngram_min", self.NGRAM_MOD_N_MIN)))
        self.server_ngram_match.setValue(int(settings.value("server/spec_ngram_match", self.NGRAM_MOD_N_MATCH)))
        self.server_ngram_max.setValue(int(settings.value("server/spec_ngram_max", self.NGRAM_MOD_N_MAX)))

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
        settings.setValue("server/backend_panels", json.dumps(self.backend_panels.to_settings()))
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
        settings.setValue("server/prompt_cache_ram", self.server_prompt_cache_ram_spinbox.value())
        settings.setValue("server/ctx_checkpoints", self.server_ctx_checkpoints_spinbox.value())
        settings.setValue("server/checkpoint_interval", self.server_checkpoint_interval_spinbox.value())

        settings.setValue("server/vision", self.server_vision_check.isChecked())
        settings.setValue("server/mmproj_path", self.server_mmproj_path.text().strip())
        settings.setValue("server/mmproj_offload", self.server_mmproj_offload_check.isChecked())

        settings.setValue("server/temperature", self.server_temperature_spinbox.value())
        settings.setValue("server/top_p", self.server_top_p_spinbox.value())
        settings.setValue("server/top_k", self.server_top_k_spinbox.value())

        settings.setValue("server/spec_type", self.server_spec_type_combo.currentText())
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
