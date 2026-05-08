"""Main window - Refactored LlamaCppGUI coordinator"""

import os
from pathlib import Path
import sys

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QMessageBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSettings, QSize

# Import utility modules
from project_utils import ProjectManager
from build_registry import BuildVersionRegistry

# Import tab widgets
from server_tab import ServerTabWidget
from inference_tab import InferenceTabWidget
from download_tab import DownloadTabWidget
from build_tab import BuildTabWidget
from benchmark_tab import BenchmarkTabWidget
from builds_info_tab import BuildsInfoTabWidget
from hardware_tab import HardwareTabWidget

# Import managers
from build_manager import BuildManager
from model_downloader import ModelDownloader
from hardware_detector import HardwareDetector
from gui_api import GuiApiServer


class LlamaCppGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("llama.cpp GUI - AMD Radeon RX 9070 XT Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize settings
        self.settings = QSettings("llama-cpp-gui", "llama-cpp-gui")
        self.project_manager = ProjectManager(settings=self.settings)
        
        # Find project root
        self.project_root = self.project_manager.project_root
        if not self.project_root:
            QMessageBox.critical(self, "Error", "Could not find llama.cpp project root")
            sys.exit(1)
        
        # Initialize directories
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.build_dir = "build"
        
        # Initialize managers
        self.build_manager = BuildManager(self.project_root)
        self.model_downloader = ModelDownloader(self.models_dir)
        self.hardware_detector = HardwareDetector()
        self.build_registry = BuildVersionRegistry(self.project_root)
        self.build_registry.sync_with_existing_builds()
        
        # Create UI
        self.create_ui()

        # Start local GUI API for automation/e2e tests.
        self.api_server = GuiApiServer(
            self,
            host="127.0.0.1",
            port=int(os.environ.get("LLAMA_GUI_API_PORT", "8765")),
        )
        self.api_server.start()
        
        # Set window icon if available
        self.set_window_icon()
    
    def set_window_icon(self):
        """Set window icon if available"""
        icon_paths = [
            self.project_root / "media" / "icon.png",
            self.project_root / "media" / "icon.ico",
            Path(__file__).parent / "icon.png",
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                try:
                    self.setWindowIcon(QIcon(str(icon_path)))
                    break
                except:
                    pass
    
    def create_ui(self):
        """Create main UI with tabs"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create all tabs
        self.server_tab = ServerTabWidget(self)
        self.inference_tab = InferenceTabWidget(self)
        self.download_tab = DownloadTabWidget(self)
        self.build_tab = BuildTabWidget(self)
        self.benchmark_tab = BenchmarkTabWidget(self)
        self.builds_info_tab = BuildsInfoTabWidget(self)
        self.hardware_tab = HardwareTabWidget(self)
        
        # Add tabs to tab widget
        self.tabs.addTab(self.server_tab, "🚀 Launch Server")
        self.tabs.addTab(self.inference_tab, "⚡ Inference")
        self.tabs.addTab(self.download_tab, "📥 Download Models")
        self.tabs.addTab(self.build_tab, "🔧 Build & Setup")
        self.tabs.addTab(self.benchmark_tab, "📈 Bench & Autotune")
        self.tabs.addTab(self.builds_info_tab, "📋 Installed Builds")
        self.tabs.addTab(self.hardware_tab, "💻 System Info")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Restore window geometry
        self.restore_geometry()
    
    def statusBar(self):
        """Get status bar"""
        if not hasattr(self, "_status_bar"):
            self._status_bar = super().statusBar()
        return self._status_bar
    
    def get_build_dir_for_backend(self, backend: str) -> str:
        """Get build directory based on backend (required by build_tab)"""
        backend_map = {
            "ROCm/HIP": "ROCm",
        }
        normalized = backend_map.get(backend, backend)
        return str(self.project_manager.get_build_dir_for_backend(normalized))

    def get_registered_builds(self) -> list[dict]:
        """Expose persistent build registry records for tabs/widgets."""
        if not hasattr(self, "build_registry"):
            return []
        return self.build_registry.list_builds()

    def refresh_build_registry(self) -> int:
        """Rescan filesystem and sync build registry with existing build directories."""
        if not hasattr(self, "build_registry"):
            return 0
        imported = self.build_registry.sync_with_existing_builds()
        self.build_registry.update_benchmark_stats_from_history()
        return imported
    
    def refresh_models_list(self):
        """Refresh models list in tabs"""
        if hasattr(self.download_tab, "load_popular_models"):
            self.download_tab.load_popular_models()
        if hasattr(self.inference_tab, "refresh_models_list"):
            self.inference_tab.refresh_models_list()

    def on_tab_changed(self, index: int):
        """Refresh dynamic tab data when user switches tabs"""
        if index == 0 and hasattr(self.server_tab, "refresh_server_models_list"):
            self.server_tab.refresh_server_models_list()
            if hasattr(self.server_tab, "refresh_server_build_choices"):
                self.server_tab.refresh_server_build_choices()
        elif index == 1 and hasattr(self.inference_tab, "refresh_models_list"):
            self.inference_tab.refresh_models_list()
        elif index == 4 and hasattr(self.benchmark_tab, "refresh_models_list"):
            self.benchmark_tab.refresh_models_list()
            self.benchmark_tab.refresh_build_choices()
            self.benchmark_tab.refresh_saved_presets_table()
        elif index == 5 and hasattr(self.builds_info_tab, "refresh_builds_info"):
            self.builds_info_tab.refresh_builds_info()
    
    def closeEvent(self, event):
        """Handle window close"""
        # Stop any running servers/inference
        if hasattr(self.server_tab, "server_thread") and self.server_tab.server_thread:
            if self.server_tab.server_thread.isRunning():
                self.server_tab.stop_server()
        
        if hasattr(self.inference_tab, "inference_thread") and self.inference_tab.inference_thread:
            if self.inference_tab.inference_thread.isRunning():
                self.inference_tab.stop_inference()
        
        # Save window geometry and settings
        self.save_geometry()

        if hasattr(self, "api_server"):
            self.api_server.stop()

        if hasattr(self.server_tab, "save_settings"):
            self.server_tab.save_settings()
        if hasattr(self.inference_tab, "save_settings"):
            self.inference_tab.save_settings()
        
        event.accept()
    
    def save_geometry(self):
        """Save window geometry to settings"""
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/windowState", self.saveState())
        self.settings.setValue("window/size", self.size())
        self.settings.setValue("window/pos", self.pos())
    
    def restore_geometry(self):
        """Restore window geometry from settings"""
        geometry = self.settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        windowState = self.settings.value("window/windowState")
        if windowState:
            self.restoreState(windowState)
        
        size = self.settings.value("window/size", QSize(1400, 900))
        if size:
            self.resize(size)
        
        pos = self.settings.value("window/pos")
        if pos:
            self.move(pos)
