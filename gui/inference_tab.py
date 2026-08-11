"""
Inference tab for RDNA LLM Studio

Provides tab for running inference with llama-cli:
- Model selection and quick model list
- Generation parameters (tokens, temperature, top-p/k, context)
- GPU/CPU selection and optimization
- Output streaming
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QFont, QTextCursor

from threads import InferenceThread


class InferenceTabWidget(QWidget):
    """Tab for running inference with llama-cli"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.project_root = parent.project_root
        self.models_dir = parent.models_dir
        self.build_dir = parent.build_dir
        self.inference_thread = None
        self.create_ui()
        self.load_settings()

    def create_ui(self):
        """Create tab for running inference"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Model selection
        model_group = QGroupBox("Выбор модели")
        model_layout = QVBoxLayout()

        model_select_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Путь к модели .gguf")
        model_select_layout.addWidget(QLabel("Model:"))
        model_select_layout.addWidget(self.model_path_edit)

        browse_btn = QPushButton("Обзор")
        browse_btn.clicked.connect(self.browse_model)
        model_select_layout.addWidget(browse_btn)

        model_layout.addLayout(model_select_layout)

        # Список доступных models
        self.models_list = QComboBox()
        self.refresh_models_list()
        self.models_list.currentTextChanged.connect(self.on_model_selected)
        model_layout.addWidget(QLabel("Быстрый выбор:"))
        model_layout.addWidget(self.models_list)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Inference parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()

        # Prompt
        params_layout.addWidget(QLabel("Prompt:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt here...")
        self.prompt_edit.setMaximumHeight(100)
        params_layout.addWidget(self.prompt_edit)

        # Parameters in multiple columns
        params_grid = QHBoxLayout()

        # Column 1
        col1 = QVBoxLayout()

        n_predict_layout = QHBoxLayout()
        n_predict_layout.addWidget(QLabel("Tokens:"))
        self.n_predict_spin = QSpinBox()
        self.n_predict_spin.setRange(-1, 8192)
        # D096-N: thinking models (Qwen3.6) spend 150-600+ tokens on reasoning
        # before the answer; the old 128-token default produced empty answers.
        self.n_predict_spin.setValue(1024)
        n_predict_layout.addWidget(self.n_predict_spin)
        col1.addLayout(n_predict_layout)

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.8)
        temp_layout.addWidget(self.temp_spin)
        col1.addLayout(temp_layout)

        params_grid.addLayout(col1)

        # Column 2
        col2 = QVBoxLayout()

        top_p_layout = QHBoxLayout()
        top_p_layout.addWidget(QLabel("Top-P:"))
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.9)
        top_p_layout.addWidget(self.top_p_spin)
        col2.addLayout(top_p_layout)

        top_k_layout = QHBoxLayout()
        top_k_layout.addWidget(QLabel("Top-K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 200)
        self.top_k_spin.setValue(40)
        top_k_layout.addWidget(self.top_k_spin)
        col2.addLayout(top_k_layout)

        params_grid.addLayout(col2)

        # Column 3
        col3 = QVBoxLayout()

        ctx_size_layout = QHBoxLayout()
        ctx_size_layout.addWidget(QLabel("Context Size:"))
        self.ctx_size_spin = QSpinBox()
        self.ctx_size_spin.setRange(128, 262144)  # Up to 256K
        self.ctx_size_spin.setValue(2048)
        self.ctx_size_spin.setSingleStep(1024)  # Step by 1K
        ctx_size_layout.addWidget(self.ctx_size_spin)
        col3.addLayout(ctx_size_layout)

        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Threads:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(os.cpu_count() or 4)
        threads_layout.addWidget(self.threads_spin)
        col3.addLayout(threads_layout)

        params_grid.addLayout(col3)

        params_layout.addLayout(params_grid)

        # Additional Options
        self.gpu_layers_checkbox = QCheckBox("Use GPU")
        self.gpu_layers_checkbox.setChecked(True)
        params_layout.addWidget(self.gpu_layers_checkbox)

        gpu_layers_layout = QHBoxLayout()
        gpu_layers_layout.addWidget(QLabel("GPU Layers:"))
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(0, 100)
        self.gpu_layers_spin.setValue(33)
        gpu_layers_layout.addWidget(self.gpu_layers_spin)
        gpu_layers_layout.addStretch()
        params_layout.addLayout(gpu_layers_layout)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_inference)
        self.run_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.clicked.connect(lambda: self.output_text.clear())
        buttons_layout.addWidget(self.clear_btn)

        layout.addLayout(buttons_layout)

        # Вывод
        layout.addWidget(QLabel("Model Output:"))
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.output_text)

    def browse_model(self):
        """Select model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            str(self.models_dir),
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def refresh_models_list(self):
        """Update available models list"""
        self.models_list.clear()
        self.models_list.addItem("-- Select Model --")

        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*.gguf"):
                self.models_list.addItem(str(model_file.name))

    def on_model_selected(self, model_name: str):
        """Handle model selection from list"""
        if model_name and model_name != "-- Select Model --":
            model_path = self.models_dir / model_name
            self.model_path_edit.setText(str(model_path))

    def run_inference(self):
        """Run inference"""
        model_path = self.model_path_edit.text()

        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Error", "Please select an existing model file")
            return

        prompt = self.prompt_edit.toPlainText()
        if not prompt:
            QMessageBox.warning(self, "Error", "Please enter a prompt")
            return

        # Determining executable file - check multiple possible locations
        possible_paths = [
            self.build_dir / "bin" / "llama-cli.exe",
            self.build_dir / "bin" / "Release" / "llama-cli.exe",
            self.build_dir / "bin" / "Debug" / "llama-cli.exe",
            self.build_dir / "bin" / "llama-cli",  # Linux/Mac
        ]

        llama_cli = None
        for path in possible_paths:
            if path.exists():
                llama_cli = path
                break

        if not llama_cli:
            QMessageBox.critical(self, "Error", "llama-cli executable not found. Please build the project first.")
            return

        # Building command
        command = [
            str(llama_cli),
            "-m", model_path,
            "-p", prompt,
            "-n", str(self.n_predict_spin.value()),
            "--temp", str(self.temp_spin.value()),
            "--top-p", str(self.top_p_spin.value()),
            "--top-k", str(self.top_k_spin.value()),
            "-c", str(self.ctx_size_spin.value()),
            "-t", str(self.threads_spin.value()),
        ]

        if self.gpu_layers_checkbox.isChecked():
            command.extend(["-ngl", str(self.gpu_layers_spin.value())])

        self.output_text.clear()
        self.output_text.append(f"🚀 Running command:\n{' '.join(command)}\n\n")

        # Running in separate thread
        self.inference_thread = InferenceThread(command, str(self.project_root))
        self.inference_thread.output_ready.connect(self.append_output)
        self.inference_thread.finished_signal.connect(self.inference_finished)
        self.inference_thread.error_signal.connect(self.inference_error)
        self.inference_thread.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Running inference...")

    def stop_inference(self):
        """Stop inference"""
        if self.inference_thread:
            self.inference_thread.stop()
            self.output_text.append("\n\n⏹️ Stopped by user")

    def append_output(self, text: str):
        """Add text to output"""
        self.output_text.moveCursor(QTextCursor.MoveOperation.End)
        self.output_text.insertPlainText(text)
        self.output_text.moveCursor(QTextCursor.MoveOperation.End)

    def inference_finished(self):
        """Inference finished"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Inference completed")
        self.output_text.append("\n\n✅ Completed")

    def inference_error(self, error: str):
        """Handle inference error"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Error при выполнении")
        self.output_text.append(f"\n\n❌ Error: {error}")

    def load_settings(self):
        """Load inference settings from QSettings"""
        if not hasattr(self.parent, "settings"):
            return

        settings = self.parent.settings
        self.model_path_edit.setText(settings.value("inference/model_path", ""))
        self.prompt_edit.setPlainText(settings.value("inference/prompt", ""))

        self.n_predict_spin.setValue(int(settings.value("inference/n_predict", 1024)))
        self.temp_spin.setValue(float(settings.value("inference/temp", 0.8)))
        self.top_p_spin.setValue(float(settings.value("inference/top_p", 0.9)))
        self.top_k_spin.setValue(int(settings.value("inference/top_k", 40)))
        self.ctx_size_spin.setValue(int(settings.value("inference/ctx", 2048)))
        self.threads_spin.setValue(int(settings.value("inference/threads", os.cpu_count() or 4)))
        self.gpu_layers_checkbox.setChecked(settings.value("inference/use_gpu", True, type=bool))
        self.gpu_layers_spin.setValue(int(settings.value("inference/gpu_layers", 33)))

        selected_model_name = settings.value("inference/model_name", "")
        if selected_model_name:
            idx = self.models_list.findText(selected_model_name)
            if idx >= 0:
                self.models_list.setCurrentIndex(idx)

    def save_settings(self):
        """Save inference settings to QSettings"""
        if not hasattr(self.parent, "settings"):
            return

        settings = self.parent.settings
        settings.setValue("inference/model_path", self.model_path_edit.text().strip())
        settings.setValue("inference/prompt", self.prompt_edit.toPlainText())

        settings.setValue("inference/n_predict", self.n_predict_spin.value())
        settings.setValue("inference/temp", self.temp_spin.value())
        settings.setValue("inference/top_p", self.top_p_spin.value())
        settings.setValue("inference/top_k", self.top_k_spin.value())
        settings.setValue("inference/ctx", self.ctx_size_spin.value())
        settings.setValue("inference/threads", self.threads_spin.value())
        settings.setValue("inference/use_gpu", self.gpu_layers_checkbox.isChecked())
        settings.setValue("inference/gpu_layers", self.gpu_layers_spin.value())

        model_name = self.models_list.currentText()
        if model_name and model_name != "-- Select Model --":
            settings.setValue("inference/model_name", model_name)
