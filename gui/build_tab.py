"""Build tab - CMake configure and build interface"""

import subprocess
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QCheckBox,
    QSpinBox, QComboBox, QLineEdit, QTextEdit, QFileDialog, QMessageBox, QProgressBar, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from build_manager import BuildManager, ConfigureThread, BuildThread

try:
    from dependency_installer import (
        install_dependencies_auto,
        check_and_install_msvc,
        check_and_install_cmake,
        check_and_install_ninja,
    )
except ImportError:
    def install_dependencies_auto():
        return False

    def check_and_install_msvc():
        return False

    def check_and_install_cmake():
        return False

    def check_and_install_ninja():
        return False


class QuickBenchmarkThread(QThread):
    """Run quick benchmark in background so UI remains responsive."""
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
            self.output.emit(f"Benchmark error: {exc}")
            self.finished_signal.emit(False)


class BuildTabWidget(QWidget):
    """Tab for building the project"""
    
    status_updated = pyqtSignal(str)
    autotune_completed = pyqtSignal(bool, object)
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.build_manager = parent.build_manager
        self.configure_thread = None
        self.build_thread = None
        self.bench_thread = None
        self._autotune_silent = False
        self._autotune_callbacks = []
        self._autotune_result = {}
        self.build_dir = "build"
        self.cmake_preset = "default"
        self.create_ui()

    def create_ui(self):
        """Create build tab UI"""
        layout = QVBoxLayout(self)

        info_label = QLabel("🔧 Build & Setup - Configure and compile llama.cpp")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)

        # Create scroll area for build options
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Backend selection
        backend_group = QGroupBox("Build Configuration")
        backend_layout = QVBoxLayout()

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["CPU", "CUDA", "ROCm/HIP", "Metal", "Vulkan", "SYCL", "OpenCL"])
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        backend_row.addWidget(self.backend_combo)
        backend_row.addStretch()
        backend_layout.addLayout(backend_row)

        # Build type
        build_type_row = QHBoxLayout()
        build_type_row.addWidget(QLabel("Build Type:"))
        self.build_type_combo = QComboBox()
        self.build_type_combo.addItems(["Release", "Debug", "RelWithDebInfo", "MinSizeRel"])
        build_type_row.addWidget(self.build_type_combo)
        build_type_row.addStretch()
        backend_layout.addLayout(build_type_row)

        # Parallel jobs
        jobs_row = QHBoxLayout()
        jobs_row.addWidget(QLabel("Parallel Jobs (-j):"))
        self.jobs_spinbox = QSpinBox()
        self.jobs_spinbox.setMinimum(1)
        self.jobs_spinbox.setMaximum(32)
        self.jobs_spinbox.setValue(8)
        jobs_row.addWidget(self.jobs_spinbox)
        jobs_row.addStretch()
        backend_layout.addLayout(jobs_row)

        backend_group.setLayout(backend_layout)
        scroll_layout.addWidget(backend_group)

        # Dependencies
        deps_group = QGroupBox("Install Dependencies")
        deps_layout = QVBoxLayout()

        self.check_deps_btn = QPushButton("✓ Check Dependencies")
        self.check_deps_btn.clicked.connect(self.check_dependencies)
        deps_layout.addWidget(self.check_deps_btn)

        deps_install_layout = QHBoxLayout()
        
        self.install_msvc_btn = QPushButton("📦 MSVC")
        self.install_msvc_btn.clicked.connect(lambda: self._install_dependency("msvc"))
        deps_install_layout.addWidget(self.install_msvc_btn)

        self.install_cmake_btn = QPushButton("📦 CMake")
        self.install_cmake_btn.clicked.connect(lambda: self._install_dependency("cmake"))
        deps_install_layout.addWidget(self.install_cmake_btn)

        self.install_ninja_btn = QPushButton("📦 Ninja")
        self.install_ninja_btn.clicked.connect(lambda: self._install_dependency("ninja"))
        deps_install_layout.addWidget(self.install_ninja_btn)

        self.install_all_deps_btn = QPushButton("📦 All")
        self.install_all_deps_btn.clicked.connect(self.install_all_dependencies)
        deps_install_layout.addWidget(self.install_all_deps_btn)

        deps_layout.addLayout(deps_install_layout)
        deps_group.setLayout(deps_layout)
        scroll_layout.addWidget(deps_group)

        # CMake options
        cmake_group = QGroupBox("CMake Configure Options")
        cmake_layout = QVBoxLayout()

        # Generator selection
        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel("Generator:"))
        self.generator_combo = QComboBox()
        self.generator_combo.addItems(["Auto", "Ninja", "Visual Studio 17 2022"])
        self.generator_combo.setCurrentText("Auto")
        gen_row.addWidget(self.generator_combo)
        gen_row.addStretch()
        cmake_layout.addLayout(gen_row)

        # Build directory
        build_dir_row = QHBoxLayout()
        build_dir_row.addWidget(QLabel("Build Directory:"))
        self.build_dir_input = QLineEdit()
        self.build_dir_input.setText("build")
        self.build_dir_input.setReadOnly(True)
        build_dir_row.addWidget(self.build_dir_input)
        build_dir_row.addStretch()
        cmake_layout.addLayout(build_dir_row)

        # Platform options (for ROCm)
        self.rocm_amdgpu_label = QLabel("ROCm AMDGPU Targets:")
        rocm_amdgpu_row = QHBoxLayout()
        rocm_amdgpu_row.addWidget(self.rocm_amdgpu_label)
        self.rocm_amdgpu_input = QLineEdit()
        self.rocm_amdgpu_input.setText("gfx1100;gfx1101;gfx1201")
        self.rocm_amdgpu_input.setPlaceholderText("gfx1100;gfx1101;gfx1201")
        rocm_amdgpu_row.addWidget(self.rocm_amdgpu_input)
        rocm_amdgpu_row.addStretch()
        cmake_layout.addLayout(rocm_amdgpu_row)
        self.rocm_amdgpu_label.setVisible(False)
        self.rocm_amdgpu_input.setVisible(False)

        # Checkboxes for options
        self.use_ccache_check = QCheckBox("Use ccache (if available)")
        cmake_layout.addWidget(self.use_ccache_check)

        self.enable_lto_check = QCheckBox("Enable LTO (Link Time Optimization)")
        cmake_layout.addWidget(self.enable_lto_check)

        self.enable_ofast_check = QCheckBox("Enable -Ofast optimization")
        cmake_layout.addWidget(self.enable_ofast_check)

        cmake_group.setLayout(cmake_layout)
        scroll_layout.addWidget(cmake_group)

        # Extra CMake flags
        extra_flags_group = QGroupBox("Extra CMake Flags")
        extra_flags_layout = QVBoxLayout()

        extra_flags_label = QLabel("Additional CMake parameters (one per line):")
        extra_flags_layout.addWidget(extra_flags_label)

        self.extra_cmake_flags = QTextEdit()
        self.extra_cmake_flags.setPlaceholderText(
            "-DCMAKE_C_COMPILER=clang\n-DCMAKE_CXX_COMPILER=clang++"
        )
        self.extra_cmake_flags.setMaximumHeight(80)
        extra_flags_layout.addWidget(self.extra_cmake_flags)

        extra_flags_group.setLayout(extra_flags_layout)
        scroll_layout.addWidget(extra_flags_group)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Progress and status
        progress_group = QGroupBox("Build Progress")
        progress_layout = QVBoxLayout()

        self.build_progress = QProgressBar()
        self.build_progress.setVisible(False)
        progress_layout.addWidget(self.build_progress)

        self.build_status_label = QLabel("Ready to configure")
        progress_layout.addWidget(self.build_status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.configure_btn = QPushButton("🔧 Configure CMake")
        self.configure_btn.clicked.connect(self.configure_build)
        self.configure_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        buttons_layout.addWidget(self.configure_btn)

        self.build_btn = QPushButton("🔨 Build")
        self.build_btn.clicked.connect(self.build_project)
        self.build_btn.setEnabled(False)
        self.build_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        buttons_layout.addWidget(self.build_btn)

        self.rebuild_btn = QPushButton("🔄 Rebuild")
        self.rebuild_btn.clicked.connect(self.rebuild_project)
        self.rebuild_btn.setEnabled(False)
        self.rebuild_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        buttons_layout.addWidget(self.rebuild_btn)

        self.clean_btn = QPushButton("🧹 Clean")
        self.clean_btn.clicked.connect(self.clean_build)
        self.clean_btn.setEnabled(False)
        self.clean_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        buttons_layout.addWidget(self.clean_btn)

        self.quick_bench_btn = QPushButton("⚡ Quick ROCm Bench")
        self.quick_bench_btn.clicked.connect(self.run_quick_benchmark)
        self.quick_bench_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        buttons_layout.addWidget(self.quick_bench_btn)

        self.autotune_btn = QPushButton("🎯 Auto-tune 32K+")
        self.autotune_btn.clicked.connect(self.run_large_context_autotune)
        self.autotune_btn.setStyleSheet("QPushButton { font-size: 12px; padding: 8px; }")
        buttons_layout.addWidget(self.autotune_btn)

        self.cancel_build_btn = QPushButton("❌ Cancel")
        self.cancel_build_btn.setEnabled(False)
        self.cancel_build_btn.clicked.connect(self.cancel_build)
        buttons_layout.addWidget(self.cancel_build_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

    def on_backend_changed(self, backend: str):
        """Handle backend change"""
        is_rocm = backend == "ROCm/HIP"
        self.rocm_amdgpu_label.setVisible(is_rocm)
        self.rocm_amdgpu_input.setVisible(is_rocm)

        if backend == "ROCm/HIP":
            self.generator_combo.setCurrentText("Ninja")
            if hasattr(self.parent, "get_build_dir_for_backend"):
                self.build_dir = self.parent.get_build_dir_for_backend(backend)
            else:
                self.build_dir = "build-rocm"
        elif backend == "CUDA":
            if hasattr(self.parent, "get_build_dir_for_backend"):
                self.build_dir = self.parent.get_build_dir_for_backend(backend)
            else:
                self.build_dir = "build-cuda"
        else:
            self.build_dir = "build"

        self.build_dir_input.setText(self.build_dir)

    def check_dependencies(self):
        """Check if all dependencies are installed"""
        missing = []

        if not self.build_manager.find_cmake():
            missing.append("CMake")

        if self.backend_combo.currentText() in ["ROCm/HIP"] and not self.build_manager.find_hip_sdk():
            missing.append("HIP SDK")

        if not self.build_manager.find_ninja() and self.generator_combo.currentText() == "Ninja":
            missing.append("Ninja")

        if missing:
            msg = f"Missing dependencies:\n" + "\n".join(f"  • {dep}" for dep in missing)
            reply = QMessageBox.question(
                self,
                "Missing Dependencies",
                msg + "\n\nInstall now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.install_all_dependencies()
        else:
            QMessageBox.information(self, "Dependencies OK", "All required dependencies are installed!")

    def _install_dependency(self, dep_type: str):
        """Install specific dependency"""
        if dep_type == "msvc":
            if check_and_install_msvc():
                QMessageBox.information(self, "MSVC", "MSVC installed successfully!")
            else:
                QMessageBox.warning(self, "MSVC", "Failed to install MSVC")

        elif dep_type == "cmake":
            if check_and_install_cmake():
                QMessageBox.information(self, "CMake", "CMake installed successfully!")
            else:
                QMessageBox.warning(self, "CMake", "Failed to install CMake")

        elif dep_type == "ninja":
            if check_and_install_ninja():
                QMessageBox.information(self, "Ninja", "Ninja installed successfully!")
            else:
                QMessageBox.warning(self, "Ninja", "Failed to install Ninja")

    def install_all_dependencies(self):
        """Install all dependencies"""
        if install_dependencies_auto():
            QMessageBox.information(self, "Success", "Dependencies installed successfully!")
            self.check_dependencies()
        else:
            QMessageBox.warning(self, "Error", "Failed to install some dependencies")

    def configure_build(self):
        """Configure project with CMake"""
        if self.configure_thread and self.configure_thread.isRunning():
            QMessageBox.warning(self, "Already Running", "Configure is already in progress")
            return

        backend = self.backend_combo.currentText()
        generator = self.generator_combo.currentText()
        
        # Update build_dir based on backend
        if hasattr(self.parent, "get_build_dir_for_backend"):
            self.build_dir = self.parent.get_build_dir_for_backend(backend)
        else:
            self.build_dir = self._get_backend_build_dir(backend)
        
        self.build_dir_input.setText(self.build_dir)

        self.configure_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self.clean_btn.setEnabled(False)
        self.build_progress.setVisible(True)
        self.build_progress.setValue(0)
        self.build_status_label.setText(f"Configuring {backend} build in {self.build_dir}...")

        # Get extra flags
        extra_flags_text = self.extra_cmake_flags.toPlainText()
        extra_flags = [line.strip() for line in extra_flags_text.split('\n') if line.strip()]

        additional_options = {}
        backend_key = backend.upper()
        if backend == "ROCm/HIP":
            backend_key = "ROCM"
            rocm_targets = self.rocm_amdgpu_input.text().strip() or "gfx1201"
            additional_options["AMDGPU_TARGETS"] = rocm_targets
            additional_options["GGML_HIP_MMQ_MFMA"] = True
            additional_options["GGML_HIP_NO_VMM"] = True

        if self.enable_lto_check.isChecked():
            additional_options["CMAKE_INTERPROCEDURAL_OPTIMIZATION"] = True

        if self.enable_ofast_check.isChecked():
            additional_options["CMAKE_CXX_FLAGS_RELEASE"] = "-Ofast"

        self.build_manager.build_dir = Path(self.build_dir)
        command = self.build_manager.get_configure_command(
            backend=backend_key,
            additional_options=additional_options,
        )

        if generator != "Auto" and "-G" not in command:
            command.extend(["-G", generator])

        command.extend(extra_flags)

        env = self.build_manager.get_rocm_env() if backend == "ROCm/HIP" else None
        working_dir = Path(self.parent.project_root)

        self.configure_thread = ConfigureThread(
            command=command,
            working_dir=working_dir,
            env=env,
        )

        self.configure_thread.output.connect(self.on_configure_progress)
        self.configure_thread.finished_signal.connect(self.on_configure_finished)
        self.configure_thread.start()

    def on_configure_progress(self, message: str):
        """Handle configure progress"""
        self.build_status_label.setText(message)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(message)

    def on_configure_finished(self, success: bool):
        """Handle configure finished"""
        if success:
            self.build_progress.setValue(100)
            self.build_status_label.setText("✓ CMake configuration completed successfully")

            if self.backend_combo.currentText() == "ROCm/HIP":
                expected_target = (self.rocm_amdgpu_input.text().split(";") or ["gfx1201"])[0].strip()
                report = self.build_manager.validate_rocm_cache(Path(self.build_dir), expected_target=expected_target)
                if report["ok"]:
                    self.build_status_label.setText("✓ ROCm config validated (HIP/MMQ/VMM/targets)")
                else:
                    details = []
                    if report["missing"]:
                        details.append("Missing: " + ", ".join(report["missing"]))
                    if report["mismatch"]:
                        details.append("Mismatch: " + "; ".join(report["mismatch"]))
                    self.build_status_label.setText("⚠ ROCm configure completed with validation warnings")
                    QMessageBox.warning(self, "ROCm Validation", "\n".join(details))
        else:
            self.build_status_label.setText("✗ CMake configuration failed")
            QMessageBox.critical(self, "Configure Error", "CMake configuration failed. Check log output.")

        self.configure_btn.setEnabled(True)
        self.build_btn.setEnabled(success)
        self.rebuild_btn.setEnabled(success)
        self.clean_btn.setEnabled(success)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Configure completed" if success else "Configure failed")

    def on_configure_error(self, error: str):
        """Handle configure error"""
        self.build_progress.setVisible(False)
        self.build_status_label.setText(f"✗ Configure failed: {error[:100]}")
        self.configure_btn.setEnabled(True)
        QMessageBox.critical(self, "Configure Error", f"CMake configuration failed:\n{error}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Configure failed")

    def build_project(self):
        """Build project"""
        if self.build_thread and self.build_thread.isRunning():
            QMessageBox.warning(self, "Already Running", "Build is already in progress")
            return

        # Update build_dir to match selected backend
        if hasattr(self.parent, "get_build_dir_for_backend"):
            backend = self.backend_combo.currentText()
            self.build_dir = self.parent.get_build_dir_for_backend(backend)
        
        build_dir_path = Path(self.build_dir)
        if not build_dir_path.exists():
            QMessageBox.warning(
                self,
                "Build Directory Not Found",
                f"Build directory '{self.build_dir}' does not exist.\n\n"
                "Please run CMake configure first."
            )
            return

        self.build_btn.setEnabled(False)
        self.configure_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self.clean_btn.setEnabled(False)
        self.cancel_build_btn.setEnabled(True)
        self.build_progress.setVisible(True)
        self.build_progress.setValue(0)
        self.build_status_label.setText(f"Building in {self.build_dir}...")

        backend_key = self.backend_combo.currentText().upper()
        if backend_key == "ROCM/HIP":
            backend_key = "ROCM"

        self.build_manager.build_dir = build_dir_path
        build_command = self.build_manager.get_build_command(
            config=self.build_type_combo.currentText(),
            jobs=self.jobs_spinbox.value(),
            backend=backend_key,
        )

        env = self.build_manager.get_rocm_env() if backend_key == "ROCM" else None
        self.build_thread = BuildThread(
            commands=[build_command],
            working_dir=Path(self.parent.project_root),
            env=env,
        )

        self.build_thread.output.connect(self.on_build_output)
        self.build_thread.progress.connect(self.on_build_percent)
        self.build_thread.finished_signal.connect(self.on_build_finished)
        self.build_thread.start()

    def on_build_output(self, message: str):
        """Handle build output stream"""
        self.build_status_label.setText(message)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(message)

    def on_build_percent(self, progress: int):
        """Handle build progress percent."""
        self.build_progress.setValue(min(progress, 99))

    def on_build_finished(self, success: bool):
        """Handle build finished"""
        if success:
            self.build_progress.setValue(100)
            self.build_status_label.setText("✓ Build completed successfully")
        else:
            self.build_status_label.setText("✗ Build failed")
            QMessageBox.critical(self, "Build Error", "Build failed. Check output log.")

        self.build_btn.setEnabled(True)
        self.configure_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        self.clean_btn.setEnabled(True)
        self.cancel_build_btn.setEnabled(False)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Build completed" if success else "Build failed")

    def on_build_error(self, error: str):
        """Handle build error"""
        self.build_progress.setVisible(False)
        self.build_status_label.setText(f"✗ Build failed: {error[:100]}")
        self.build_btn.setEnabled(True)
        self.configure_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        self.clean_btn.setEnabled(True)
        self.cancel_build_btn.setEnabled(False)
        QMessageBox.critical(self, "Build Error", f"Build failed:\n{error}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Build failed")

    def rebuild_project(self):
        """Clean and build project"""
        self.clean_build()
        self.build_project()

    def clean_build(self):
        """Clean build directory"""
        build_dir_path = Path(self.build_dir)
        if not build_dir_path.exists():
            QMessageBox.warning(self, "Error", f"Build directory '{self.build_dir}' does not exist")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Clean",
            f"Delete all files in '{self.build_dir}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            try:
                shutil.rmtree(build_dir_path)
                self.build_status_label.setText(f"✓ Cleaned {self.build_dir}")
                self.build_btn.setEnabled(False)
                self.rebuild_btn.setEnabled(False)
                if self.parent.statusBar():
                    self.parent.statusBar().showMessage(f"Cleaned {self.build_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clean build directory:\n{str(e)}")

    def cancel_build(self):
        """Cancel build"""
        if self.build_thread and self.build_thread.isRunning():
            self.build_thread.stop()
            self.build_thread.wait(2000)
            self.build_status_label.setText("Build cancelled")
            self.build_btn.setEnabled(True)
            self.configure_btn.setEnabled(True)
            self.rebuild_btn.setEnabled(True)
            self.clean_btn.setEnabled(True)
            self.cancel_build_btn.setEnabled(False)
            self.build_progress.setVisible(False)

    def run_quick_benchmark(self):
        """Run quick benchmark profile from GUI."""
        if self.bench_thread and self.bench_thread.isRunning():
            QMessageBox.information(self, "Benchmark", "Quick benchmark is already running")
            return

        model_files = sorted((Path(self.parent.models_dir)).rglob("*.gguf"))
        if not model_files:
            QMessageBox.warning(self, "Benchmark", "No GGUF model found in models/")
            return

        server_bin = Path("build-rocm/bin/llama-server.exe")
        if not server_bin.exists():
            QMessageBox.warning(self, "Benchmark", "Missing build-rocm/bin/llama-server.exe")
            return

        model_path = model_files[0]
        command = [
            sys.executable,
            "scripts/agent_workload_bench.py",
            "--label", "gui-quick-bench",
            "--tasks", "quick",
            "--runs", "1",
            "--server-bin", str(server_bin),
            "--model", str(model_path),
            "--ctx-size", "32768",
            "--batch-size", "1024",
            "--ubatch-size", "1024",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--gpu-layers", "99",
            "--parallel", "1",
            "--max-tokens", "80",
            "--startup-timeout", "120",
            "--request-timeout", "120",
        ]

        self.quick_bench_btn.setEnabled(False)
        self.build_status_label.setText(f"Running quick benchmark with {model_path.name}...")
        self.bench_thread = QuickBenchmarkThread(command=command, working_dir=Path(self.parent.project_root))
        self.bench_thread.output.connect(self._on_quick_bench_output)
        self.bench_thread.finished_signal.connect(self._on_quick_bench_finished)
        self.bench_thread.start()

    def run_large_context_autotune(
        self,
        model_path: str | None = None,
        silent: bool = False,
        completion_callback=None,
        sweep_mode: str = "full",
    ) -> bool:
        """Run large-context autotune sweep and update model presets.

        Returns True when autotune process is started, else False.
        """
        if self.bench_thread and self.bench_thread.isRunning():
            if not silent:
                QMessageBox.information(self, "Auto-tune", "Benchmark/autotune is already running")
            return False

        resolved_model = self._resolve_benchmark_model(model_path)
        if not resolved_model:
            if not silent:
                QMessageBox.warning(self, "Auto-tune", "No GGUF model found or selected")
            return False

        server_bin = Path("build-rocm/bin/llama-server.exe")
        if not server_bin.exists():
            if not silent:
                QMessageBox.warning(self, "Auto-tune", "Missing build-rocm/bin/llama-server.exe")
            return False

        spec_values = ["none", "ngram-mod"]
        model_name = resolved_model.name.lower()
        if ("mtp" in model_name or "nextn" in model_name) and self._server_supports_mtp(server_bin):
            spec_values.append("mtp")

        command = [
            sys.executable,
            "scripts/agent_workload_bench.py",
            "--autotune",
            "--label", f"gui-autotune-{resolved_model.stem}",
            "--tasks", "quick",
            "--runs", "1",
            "--server-bin", str(server_bin),
            "--model", str(resolved_model),
            "--gpu-layers", "99",
            "--parallel", "1",
            "--max-tokens", "160",
            "--startup-timeout", "120",
            "--request-timeout", "120",
            "--background-server-policy", "fail",
            "--autotune-min-ctx", "32768",
            "--autotune-ctx-values", "32768,49152,65536",
            "--autotune-batch-values", "1024,2048,4096",
            "--autotune-ubatch-values", "1024,2048,4096",
            "--autotune-kv-values", "q8_0,q4_0",
            "--autotune-spec-values", ",".join(spec_values),
            "--autotune-update-preset",
            "--autotune-preset-file", "gui/model_presets.json",
        ]

        if sweep_mode == "smoke":
            command.extend([
                "--autotune-ctx-values", "32768",
                "--autotune-batch-values", "1024",
                "--autotune-ubatch-values", "1024",
                "--autotune-kv-values", "q8_0",
                "--autotune-spec-values", ",".join(spec_values),
                "--autotune-max-configs", "4",
            ])

        self._autotune_silent = silent
        self._autotune_result = {
            "model": str(resolved_model),
            "best": "",
            "summary_json": "",
            "summary_csv": "",
        }
        self._autotune_callbacks = []
        if completion_callback is not None:
            self._autotune_callbacks.append(completion_callback)

        self.autotune_btn.setEnabled(False)
        self.quick_bench_btn.setEnabled(False)
        self.build_status_label.setText(f"Running 32K+ autotune for {resolved_model.name}...")
        self.bench_thread = QuickBenchmarkThread(command=command, working_dir=Path(self.parent.project_root))
        self.bench_thread.output.connect(self._on_autotune_output)
        self.bench_thread.finished_signal.connect(self._on_autotune_finished)
        self.bench_thread.start()
        return True

    def _resolve_benchmark_model(self, preferred_model_path: str | None = None) -> Path | None:
        """Pick model for benchmark/autotune, preferring selected server model."""
        if preferred_model_path:
            preferred = Path(preferred_model_path)
            if preferred.exists():
                return preferred

        if hasattr(self.parent, "server_tab"):
            selected_path = self.parent.server_tab.server_model_path.text().strip()
            if selected_path and Path(selected_path).exists():
                return Path(selected_path)

        model_files = sorted((Path(self.parent.models_dir)).rglob("*.gguf"))
        filtered = [p for p in model_files if "mmproj" not in p.name.lower()]
        return filtered[0] if filtered else None

    @staticmethod
    def _server_supports_mtp(server_bin: Path) -> bool:
        """Best-effort capability probe for --spec-type mtp support in llama-server."""
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

    def _on_quick_bench_output(self, line: str):
        if "Aggregate completion TPS by wall time" in line:
            self.build_status_label.setText(line)
        elif "Wrote" in line and "agent-workload" in line:
            self.build_status_label.setText(line)

    def _on_quick_bench_finished(self, success: bool):
        self.quick_bench_btn.setEnabled(True)
        if success:
            QMessageBox.information(
                self,
                "Quick Benchmark",
                "Quick benchmark finished. Results are in build_logs/agent-workload/."
            )
        else:
            QMessageBox.warning(self, "Quick Benchmark", "Quick benchmark failed. Check build status/output.")

    def _on_autotune_output(self, line: str):
        if line.startswith("BEST:"):
            self._autotune_result["best"] = line
        elif line.endswith("-autotune-summary.json"):
            self._autotune_result["summary_json"] = line.split("Wrote ", 1)[-1].strip()
        elif line.endswith("-autotune-summary.csv"):
            self._autotune_result["summary_csv"] = line.split("Wrote ", 1)[-1].strip()

        if line.startswith("Autotune [") or line.startswith("BEST:"):
            self.build_status_label.setText(line)
        elif "-autotune-summary" in line or "Updated preset:" in line:
            self.build_status_label.setText(line)

    def _on_autotune_finished(self, success: bool):
        self.autotune_btn.setEnabled(True)
        self.quick_bench_btn.setEnabled(True)
        payload = dict(self._autotune_result)
        payload["success"] = success
        self.autotune_completed.emit(success, payload)

        for callback in self._autotune_callbacks:
            try:
                callback(success, payload)
            except Exception:
                pass
        self._autotune_callbacks = []

        if not self._autotune_silent:
            if success:
                QMessageBox.information(
                    self,
                    "Auto-tune 32K+",
                    "Autotune finished. Summary is in build_logs/agent-workload/ and preset was updated."
                )
            else:
                QMessageBox.warning(self, "Auto-tune 32K+", "Autotune failed. Check build status/output.")

    @staticmethod
    def _get_backend_build_dir(backend: str) -> str:
        """Get build directory based on backend"""
        backend_map = {
            "CPU": "build",
            "CUDA": "build-cuda",
            "ROCm/HIP": "build-rocm",
            "Metal": "build-metal",
            "Vulkan": "build-vulkan",
            "SYCL": "build-sycl",
            "OpenCL": "build-opencl"
        }
        return backend_map.get(backend, "build")
