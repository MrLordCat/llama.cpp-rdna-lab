"""Builds Info tab - Display and manage installed builds"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt


class BuildsInfoTabWidget(QWidget):
    """Tab for displaying and managing installed builds"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.project_root = parent.project_root if hasattr(parent, "project_root") else Path.cwd()
        self.create_ui()
        self.refresh_builds_info()

    def create_ui(self):
        """Create builds info tab UI"""
        layout = QVBoxLayout(self)

        info_label = QLabel("📋 Installed Builds - View and manage compiled builds")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)

        # Builds list
        list_group = QGroupBox("Available Builds")
        list_layout = QVBoxLayout()

        self.builds_table = QTableWidget()
        self.builds_table.setColumnCount(5)
        self.builds_table.setHorizontalHeaderLabels([
            "Build Name", "Backend", "Git Commit", "Size", "Modified"
        ])
        self.builds_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.builds_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.builds_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.builds_table.setAlternatingRowColors(True)
        list_layout.addWidget(self.builds_table)

        list_group.setLayout(list_layout)
        layout.addWidget(list_group)

        # Executables group
        exec_group = QGroupBox("Build Executables")
        exec_layout = QVBoxLayout()

        self.executables_table = QTableWidget()
        self.executables_table.setColumnCount(3)
        self.executables_table.setHorizontalHeaderLabels([
            "Executable Name", "Size", "Modified"
        ])
        self.executables_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.executables_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.executables_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.executables_table.setMaximumHeight(150)
        exec_layout.addWidget(self.executables_table)

        exec_group.setLayout(exec_layout)
        layout.addWidget(exec_group)

        # Build details
        details_group = QGroupBox("Build Details")
        details_layout = QVBoxLayout()

        self.build_details_label = QLabel("Select a build to view details")
        details_layout.addWidget(self.build_details_label)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_builds_info)
        buttons_layout.addWidget(self.refresh_btn)

        self.open_build_btn = QPushButton("📂 Open Folder")
        self.open_build_btn.clicked.connect(self.open_build_folder)
        buttons_layout.addWidget(self.open_build_btn)

        self.rename_build_btn = QPushButton("✏️ Rename")
        self.rename_build_btn.clicked.connect(self.rename_build_folder)
        buttons_layout.addWidget(self.rename_build_btn)

        self.delete_build_btn = QPushButton("🗑️ Delete")
        self.delete_build_btn.clicked.connect(self.delete_selected_build)
        self.delete_build_btn.setStyleSheet("QPushButton { color: red; }")
        buttons_layout.addWidget(self.delete_build_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # Connect selection changed
        self.builds_table.itemSelectionChanged.connect(self.on_build_selected)

    def refresh_builds_info(self):
        """Refresh builds information"""
        self.builds_table.setRowCount(0)
        self.executables_table.setRowCount(0)
        self.build_details_label.setText("Select a build to view details")

        # Find all build-* directories
        build_dirs = []
        if self.project_root.exists():
            for item in self.project_root.iterdir():
                if item.is_dir() and (item.name.startswith("build") or item.name == "build"):
                    build_dirs.append(item)

        build_dirs.sort()

        for build_dir in build_dirs:
            backend = self._detect_backend(build_dir)
            commit = self._get_git_commit(build_dir)
            size = self._get_directory_size(build_dir)
            size_str = self._format_size(size)
            modified = self._get_last_modified(build_dir)

            row = self.builds_table.rowCount()
            self.builds_table.insertRow(row)

            self.builds_table.setItem(row, 0, QTableWidgetItem(build_dir.name))
            self.builds_table.setItem(row, 1, QTableWidgetItem(backend))
            self.builds_table.setItem(row, 2, QTableWidgetItem(commit))
            self.builds_table.setItem(row, 3, QTableWidgetItem(size_str))
            self.builds_table.setItem(row, 4, QTableWidgetItem(modified))

    def on_build_selected(self):
        """Handle build selection"""
        selected = self.builds_table.selectedItems()
        if not selected:
            self.executables_table.setRowCount(0)
            self.build_details_label.setText("Select a build to view details")
            return

        row = self.builds_table.row(selected[0])
        build_name = self.builds_table.item(row, 0).text()

        build_path = self.project_root / build_name
        self.display_build_executables(build_path)
        self.display_build_details(build_path)

    def display_build_executables(self, build_path: Path):
        """Display executables in the build"""
        self.executables_table.setRowCount(0)

        bin_dir = build_path / "bin"
        if bin_dir.exists():
            executables = []
            for item in bin_dir.iterdir():
                if item.is_file() and (item.suffix == ".exe" or item.stat().st_mode & 0o111):
                    executables.append(item)

            executables.sort()

            for exe in executables:
                size = exe.stat().st_size
                size_str = self._format_size(size)
                modified = self._get_file_modified(exe)

                row = self.executables_table.rowCount()
                self.executables_table.insertRow(row)

                self.executables_table.setItem(row, 0, QTableWidgetItem(exe.name))
                self.executables_table.setItem(row, 1, QTableWidgetItem(size_str))
                self.executables_table.setItem(row, 2, QTableWidgetItem(modified))

    def display_build_details(self, build_path: Path):
        """Display build details"""
        details = []

        backend = self._detect_backend(build_path)
        details.append(f"Backend: {backend}")

        cmake_cache = build_path / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                content = cmake_cache.read_text()
                for line in content.split('\n'):
                    if "CMAKE_BUILD_TYPE" in line and not line.startswith("//"):
                        build_type = line.split("=")[-1].strip()
                        details.append(f"Build Type: {build_type}")
                        break
            except:
                pass

        size = self._get_directory_size(build_path)
        details.append(f"Total Size: {self._format_size(size)}")

        bin_dir = build_path / "bin"
        if bin_dir.exists():
            exes = [f for f in bin_dir.iterdir() if f.is_file()]
            details.append(f"Executables: {len(exes)}")

        self.build_details_label.setText("\n".join(details))

    def open_build_folder(self):
        """Open build folder in file explorer"""
        selected = self.builds_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "Select a build first")
            return

        row = self.builds_table.row(selected[0])
        build_name = self.builds_table.item(row, 0).text()
        build_path = self.project_root / build_name

        if build_path.exists():
            import subprocess
            import sys
            if sys.platform == "win32":
                subprocess.Popen(f"explorer /select,{build_path}")
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(build_path)])
            else:
                subprocess.Popen(["xdg-open", str(build_path)])

    def rename_build_folder(self):
        """Rename build folder"""
        selected = self.builds_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "Select a build to rename")
            return

        row = self.builds_table.row(selected[0])
        old_name = self.builds_table.item(row, 0).text()

        new_name, ok = self._ask_for_new_name(old_name)
        if not ok or not new_name:
            return

        old_path = self.project_root / old_name
        new_path = self.project_root / new_name

        if new_path.exists():
            QMessageBox.warning(self, "Error", f"Folder '{new_name}' already exists")
            return

        try:
            old_path.rename(new_path)
            QMessageBox.information(self, "Success", f"Renamed '{old_name}' → '{new_name}'")
            self.refresh_builds_info()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename folder:\n{str(e)}")

    def delete_selected_build(self):
        """Delete selected build"""
        selected = self.builds_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "Error", "Select a build to delete")
            return

        row = self.builds_table.row(selected[0])
        build_name = self.builds_table.item(row, 0).text()

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete build folder '{build_name}'?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            build_path = self.project_root / build_name

            try:
                import shutil
                shutil.rmtree(build_path)
                QMessageBox.information(self, "Success", f"Deleted '{build_name}'")
                self.refresh_builds_info()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete build:\n{str(e)}")

    @staticmethod
    def _detect_backend(build_path: Path) -> str:
        """Detect build backend from CMakeCache"""
        cmake_cache = build_path / "CMakeCache.txt"
        if not cmake_cache.exists():
            return "Unknown"

        try:
            content = cmake_cache.read_text(errors="ignore")

            if "GGML_HIP=ON" in content:
                return "ROCm/HIP"
            elif "GGML_CUDA=ON" in content:
                return "CUDA"
            elif "GGML_METAL=ON" in content:
                return "Metal"
            elif "GGML_VULKAN=ON" in content:
                return "Vulkan"
            elif "GGML_SYCL=ON" in content:
                return "SYCL"
            else:
                return "CPU"
        except:
            return "Unknown"

    @staticmethod
    def _get_directory_size(path: Path) -> int:
        """Calculate directory size in bytes"""
        total = 0
        if path.exists():
            for item in path.rglob("*"):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except:
                        pass
        return total

    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human readable size"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"

    @staticmethod
    def _get_last_modified(path: Path) -> str:
        """Get last modified date as string"""
        if not path.exists():
            return "—"

        try:
            import datetime
            mtime = path.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return "—"

    @staticmethod
    def _get_file_modified(path: Path) -> str:
        """Get file modified date as string"""
        try:
            import datetime
            mtime = path.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return "—"

    @staticmethod
    def _get_git_commit(build_path: Path) -> str:
        """Get git commit hash from build"""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=build_path.parent,
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except:
            pass

        return "—"

    @staticmethod
    def _ask_for_new_name(current_name: str) -> tuple:
        """Ask user for new folder name"""
        from PyQt6.QtWidgets import QInputDialog, QWidget

        new_name, ok = QInputDialog.getText(
            None,
            "Rename Build",
            f"Enter new name for '{current_name}':",
            text=current_name
        )

        return new_name, ok
