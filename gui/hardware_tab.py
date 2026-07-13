"""
Hardware information tab for llama.cpp GUI

Shows:
- System hardware information (CPU, GPU, RAM)
- Available CPU, ROCm and Vulkan backends
- Backend recommendations
- HIP SDK version information
"""

import platform
import webbrowser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QMessageBox, QFileDialog
)
from PyQt6.QtGui import QFont


class HardwareTabWidget(QWidget):
    """Hardware information tab"""
    
    def __init__(self, parent):
        """Initialize hardware tab
        
        Args:
            parent: Parent LlamaCppGUI instance
        """
        super().__init__()
        self.parent = parent
        self.hardware_detector = parent.hardware_detector
        self.build_manager = parent.build_manager
        self.create_ui()
        
        # Populate hardware info on startup
        self.detect_hardware()
    
    def create_ui(self):
        """Create UI components"""
        layout = QVBoxLayout(self)
        
        info_label = QLabel("💻 Your System Information")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)
        
        self.hardware_info_text = QTextEdit()
        self.hardware_info_text.setReadOnly(True)
        self.hardware_info_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.hardware_info_text)
        
        # Buttons row
        buttons_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh Information")
        refresh_btn.clicked.connect(self.detect_hardware)
        buttons_layout.addWidget(refresh_btn)
        
        # ROCm update button (Windows only)
        if platform.system() == "Windows":
            self.rocm_update_btn = QPushButton("📦 Update/Install HIP SDK")
            self.rocm_update_btn.clicked.connect(self.update_rocm)
            self.rocm_update_btn.setToolTip("Open AMD website to download HIP SDK for ROCm support")
            buttons_layout.addWidget(self.rocm_update_btn)
        
        layout.addLayout(buttons_layout)
    
    def detect_hardware(self):
        """Detect hardware characteristics and show ROCm/Vulkan info"""
        info = self.hardware_detector.get_hardware_info()
        
        # Auto-select backend after hardware detection
        if hasattr(self.parent, '_auto_select_backend'):
            self.parent._auto_select_backend()
        
        text = "💻 SYSTEM INFORMATION\n"
        text += "=" * 60 + "\n\n"
        
        text += f"Operating System: {info['os']}\n"
        text += f"CPU: {info['cpu']['name']}\n"
        text += f"Cores: {info['cpu']['cores']} | Threads: {info['cpu']['threads']}\n"
        text += f"RAM: {info['memory']['total_gb']:.1f} GB ({info['memory']['percent_used']:.1f}% used)\n\n"
        
        text += "🎮 GPU INFORMATION:\n"
        text += "-" * 60 + "\n"
        
        if info['gpu']:
            for i, gpu in enumerate(info['gpu']):
                text += f"\nGPU {i+1}: {gpu['name']}\n"
                
                if gpu.get('type') == 'AMD':
                    text += f"  Type: AMD Radeon\n"
                    text += f"  Backend: {gpu.get('backend', 'ROCm or Vulkan')}\n"
                    
                    if gpu.get('is_9070xt'):
                        text += f"  🎯 AMD 9070XT DETECTED (RDNA 4)\n"
                        text += f"  Recommended: ROCm with gfx1201 support\n"
                else:
                    text += f"  Type: {gpu.get('type', 'Unknown')}\n"
                    text += f"  Backend: {gpu.get('backend', 'Unknown')}\n"
                
                if 'memory' in gpu:
                    text += f"  Memory: {gpu['memory']}\n"
        else:
            text += "  ❌ No GPU detected - will use CPU inference\n"
        
        text += "\n" + "=" * 60 + "\n"
        text += "📋 BACKEND RECOMMENDATION:\n"
        text += "-" * 60 + "\n"
        text += f"  Recommended: {info.get('recommended_backend', 'CPU')}\n"
        
        # Handle both old and new hardware detector versions
        if 'backend_reason' in info:
            text += f"  Reason: {info['backend_reason']}\n"
        
        text += "\n✅ DEPENDENCY STATUS:\n"
        text += "-" * 60 + "\n"
        
        if info.get('rocm_available', False):
            text += f"  ✅ ROCm: Installed\n"
            # Show ROCm version
            rocm_version = self.build_manager.detect_rocm_version()
            if rocm_version:
                text += f"     Version: {rocm_version}\n"
                # Check if upgrade recommended for RDNA4
                try:
                    major_minor = float(rocm_version.split('.')[0] + '.' + rocm_version.split('.')[1])
                    if major_minor < 6.4:
                        text += f"     ⚠️ Upgrade to 6.4+ recommended for RDNA4 native support\n"
                    else:
                        text += f"     ✅ Native RDNA4 support available\n"
                except:
                    pass
        else:
            text += f"  ❌ ROCm: NOT installed\n"
        
        if info.get('vulkan_available', False):
            text += f"  ✅ Vulkan: Installed\n"
        else:
            text += f"  ❌ Vulkan: NOT installed\n"
        
        text += "\n💡 For AMD 9070XT: ROCm is HIGHLY RECOMMENDED\n"
        text += "   If ROCm is not available, Vulkan can be used as fallback.\n"
        text += "\n📦 Latest HIP SDK: 7.1.1 (download button above)\n"
        
        self.hardware_info_text.setText(text)
    
    def update_rocm(self):
        """Download and install/update HIP SDK"""
        # Get current version
        current_version = self.build_manager.detect_rocm_version()
        latest_version = "7.1.1"
        
        # Build message
        if current_version:
            msg = f"Current HIP SDK version: {current_version}\n"
            msg += f"Latest available version: {latest_version}\n\n"
            try:
                current_major_minor = float(current_version.split('.')[0] + '.' + current_version.split('.')[1])
                if current_major_minor >= 6.4:
                    msg += "✅ You have a recent version with RDNA4 support.\n\n"
                else:
                    msg += "⚠️ Upgrade recommended for native RDNA4 support.\n\n"
            except:
                pass
        else:
            msg = "HIP SDK is not currently installed.\n\n"
        
        msg += "Open AMD HIP SDK download page?\n\n"
        msg += "Note: AMD requires accepting a license agreement before download."
        
        reply = QMessageBox.question(
            self,
            "HIP SDK Update",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Open AMD HIP SDK download page
            url = "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
            webbrowser.open(url)
