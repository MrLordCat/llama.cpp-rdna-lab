"""
Project utilities for llama.cpp GUI

Provides utilities for:
- Finding or selecting llama.cpp repository
- Getting build directories for different backends
- Detecting build information
"""

import os
import re
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Dict

from PyQt6.QtWidgets import QFileDialog, QMessageBox


class ProjectManager:
    """Manages llama.cpp project paths and build directories"""
    
    # Build directory names for different backends
    BUILD_DIRS = {
        "CPU": "build-cpu",
        "CUDA": "build-cuda",
        "Metal": "build-metal",
        "Vulkan": "build-vulkan",
        "SYCL": "build-sycl",
        "ROCm": "build-rocm",
    }
    
    def __init__(self, project_root: Optional[Path] = None, settings=None):
        """Initialize ProjectManager
        
        Args:
            project_root: Path to llama.cpp repository
            settings: QSettings object for saving/loading paths
        """
        self.project_root = project_root or self.find_or_select_project_root(settings)
        self.settings = settings
    
    def find_or_select_project_root(self, settings=None) -> Optional[Path]:
        """Find llama.cpp repository or ask user to select it"""
        
        # 1. Check saved path first
        if settings:
            saved_path = settings.value("project_root", "")
            if saved_path:
                saved_path = Path(saved_path)
                if self.is_valid_llama_cpp_repo(saved_path):
                    return saved_path
        
        # 2. Check if running from within the repo
        current_file = Path(__file__).resolve()
        possible_roots = [
            current_file.parent.parent,  # gui/ -> project root
            current_file.parent,  # if in root
            Path.cwd(),  # current working directory
            Path.cwd().parent,
        ]
        
        for root in possible_roots:
            if self.is_valid_llama_cpp_repo(root):
                return root
        
        # 3. Search common locations
        search_paths = self.get_common_repo_locations()
        for path in search_paths:
            if path.exists() and self.is_valid_llama_cpp_repo(path):
                return path
        
        # 4. Ask user to select the folder
        return self.ask_user_for_repo_path()
    
    @staticmethod
    def is_valid_llama_cpp_repo(path: Path) -> bool:
        """Check if path is a valid llama.cpp repository"""
        if not path.exists() or not path.is_dir():
            return False
        
        # Check for key files that should exist in llama.cpp
        required_files = [
            "CMakeLists.txt",
            "include/llama.h",
        ]
        
        # At least one of these should exist
        optional_indicators = [
            "src/llama.cpp",
            "ggml",
            "examples",
            "AGENTS.md",  # Our custom file
        ]
        
        # Check required files
        for req_file in required_files:
            if not (path / req_file).exists():
                return False
        
        # Check at least one optional indicator
        for opt_file in optional_indicators:
            if (path / opt_file).exists():
                return True
        
        return False
    
    @staticmethod
    def get_common_repo_locations() -> List[Path]:
        """Get common locations where llama.cpp might be cloned"""
        locations = []
        
        # User's home directory
        home = Path.home()
        
        # Common development folders (cross-platform)
        common_folders = [
            home / "Documents" / "GitHub" / "llama.cpp",
            home / "Documents" / "GitHub" / "llama.cpp-with-GUI",
            home / "source" / "repos" / "llama.cpp",
            home / "Projects" / "llama.cpp",
            home / "dev" / "llama.cpp",
            home / "code" / "llama.cpp",
            home / "git" / "llama.cpp",
            home / "llama.cpp",
        ]
        
        # Add platform-specific paths
        if platform.system() == "Windows":
            common_folders.extend([
                Path("C:/llama.cpp"),
                Path("D:/llama.cpp"),
                Path("C:/GitHub/llama.cpp"),
                Path("D:/GitHub/llama.cpp"),
            ])
        else:
            # Linux/macOS paths
            common_folders.extend([
                home / "src" / "llama.cpp",
                home / "repos" / "llama.cpp",
                Path("/opt/llama.cpp"),
            ])
        
        # Add variations with -with-GUI suffix
        for folder in common_folders.copy():
            if "llama.cpp" in str(folder) and "-with-GUI" not in str(folder):
                locations.append(Path(str(folder) + "-with-GUI"))
        
        locations.extend(common_folders)
        return locations
    
    @staticmethod
    def ask_user_for_repo_path() -> Optional[Path]:
        """Show dialog to ask user to select llama.cpp folder"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("llama.cpp Repository Not Found")
        msg.setText(
            "Could not find llama.cpp repository automatically.\n\n"
            "Please select the folder where you cloned llama.cpp.\n\n"
            "The folder should contain:\n"
            "• CMakeLists.txt\n"
            "• include/llama.h\n"
            "• ggml/ folder"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        
        result = msg.exec()
        
        if result == QMessageBox.StandardButton.Cancel:
            return None
        
        # Open folder selection dialog
        folder = QFileDialog.getExistingDirectory(
            None,
            "Select llama.cpp Repository Folder",
            str(Path.home() / "Documents"),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder:
            return None
        
        folder_path = Path(folder)
        
        if not ProjectManager.is_valid_llama_cpp_repo(folder_path):
            QMessageBox.warning(
                None,
                "Invalid Repository",
                f"The selected folder does not appear to be a valid llama.cpp repository.\n\n"
                f"Selected: {folder_path}\n\n"
                "Please clone llama.cpp first:\n"
                "git clone https://github.com/ggml-org/llama.cpp.git"
            )
            # Try again
            return ProjectManager.ask_user_for_repo_path()
        
        return folder_path
    
    def get_build_dir_for_backend(self, backend: Optional[str]) -> Path:
        """Get the build directory for a specific backend"""
        if backend is None:
            backend = "CPU"
        
        # First check if backend-specific directory exists
        backend_dir_name = self.BUILD_DIRS.get(backend, f"build-{backend.lower()}")
        backend_dir = self.project_root / backend_dir_name
        
        if backend_dir.exists():
            return backend_dir
        
        # Fallback to generic 'build' directory
        generic_build = self.project_root / "build"
        if generic_build.exists():
            return generic_build
        
        # Return the backend-specific path even if it doesn't exist yet
        return backend_dir
    
    def get_build_version_info(self, build_path: Path) -> Dict[str, str]:
        """Get version (git commit) and build date for a build directory"""
        info = {"version": "unknown", "date": "unknown"}
        cmake_cache = build_path / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                import datetime
                mtime = cmake_cache.stat().st_mtime
                info["date"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
        
        # Try to get git commit from build-info
        for candidate in [
            build_path / "llama-version.cmake",
            build_path / "build-info.h",
        ]:
            if candidate.exists():
                try:
                    content = candidate.read_text(errors='ignore')
                    m = re.search(r'LLAMA_BUILD_COMMIT["\s:=]+([a-f0-9]{7,40})', content)
                    if m:
                        info["version"] = m.group(1)[:8]
                        break
                except Exception:
                    pass
        
        if info["version"] == "unknown":
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, cwd=self.project_root, timeout=5
                )
                if result.returncode == 0:
                    info["version"] = result.stdout.strip()
            except Exception:
                pass
        
        return info

    def get_available_builds(self) -> Dict[str, Dict]:
        """Get all available builds with their backends"""
        builds = {}
        
        # Check generic build directory
        generic_build = self.project_root / "build"
        if generic_build.exists() and (generic_build / "CMakeCache.txt").exists():
            backend = self.detect_build_backend(generic_build)
            ver_info = self.get_build_version_info(generic_build)
            builds["build"] = {
                "path": generic_build,
                "backend": backend,
                "display_name": f"build ({backend})",
                "version": ver_info["version"],
                "date": ver_info["date"],
            }
        
        # Check backend-specific directories from BUILD_DIRS
        for backend_name, dir_name in self.BUILD_DIRS.items():
            dir_path = self.project_root / dir_name
            if dir_path.exists() and (dir_path / "CMakeCache.txt").exists():
                detected_backend = self.detect_build_backend(dir_path)
                ver_info = self.get_build_version_info(dir_path)
                builds[dir_name] = {
                    "path": dir_path,
                    "backend": detected_backend or backend_name,
                    "display_name": f"{dir_name} ({detected_backend or backend_name})",
                    "version": ver_info["version"],
                    "date": ver_info["date"],
                }
        
        # Also scan for any other build-* directories
        try:
            for item in self.project_root.iterdir():
                if item.is_dir() and item.name.startswith("build") and item.name not in builds:
                    if (item / "CMakeCache.txt").exists():
                        detected_backend = self.detect_build_backend(item)
                        ver_info = self.get_build_version_info(item)
                        builds[item.name] = {
                            "path": item,
                            "backend": detected_backend,
                            "display_name": f"{item.name} ({detected_backend})",
                            "version": ver_info["version"],
                            "date": ver_info["date"],
                        }
        except Exception:
            pass
        
        return builds
    
    @staticmethod
    def detect_build_backend(build_path: Path) -> str:
        """Detect backend from CMakeCache.txt"""
        cmake_cache = build_path / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                content = cmake_cache.read_text(errors='ignore')
                if "GGML_CUDA:BOOL=ON" in content:
                    return "CUDA"
                elif "GGML_HIP:BOOL=ON" in content or "GGML_ROCM:BOOL=ON" in content:
                    return "ROCm"
                elif "GGML_METAL:BOOL=ON" in content:
                    return "Metal"
                elif "GGML_VULKAN:BOOL=ON" in content:
                    return "Vulkan"
                elif "GGML_SYCL:BOOL=ON" in content:
                    return "SYCL"
            except:
                pass
        return "CPU"
