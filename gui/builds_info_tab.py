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
        self._row_by_build_id: dict[int, str] = {}
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
        self.builds_table.setColumnCount(11)
        self.builds_table.setHorizontalHeaderLabels([
            "Name", "Backend", "Source", "Status", "Build ID", "Best Non-MTP", "Best MTP", "Last Bench", "Git Commit", "Size", "Modified"
        ])
        self.builds_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.builds_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.builds_table.horizontalHeader().setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
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
        self._row_by_build_id = {}

        registry = getattr(self.parent, "build_registry", None)
        if registry is not None and hasattr(self.parent, "refresh_build_registry"):
            self.parent.refresh_build_registry()
            registry.update_benchmark_stats_from_history()
            records = sorted(
                registry.list_builds(),
                key=lambda r: (str(r.get("status", "")) != "ready", str(r.get("name", "")).lower()),
            )

            for record in records:
                build_path = Path(record.get("build_dir", "")) if record.get("build_dir") else self.project_root
                backend = str(record.get("backend", "unknown")).upper()
                source = str(record.get("source_type", "fork"))
                status = str(record.get("status", "unknown"))
                build_id = str(record.get("id", ""))
                best_non = str(record.get("bench_best_non_mtp_tps", "") or "-")
                best_mtp = str(record.get("bench_best_mtp_tps", "") or "-")
                last_bench = str(record.get("bench_last_run_at", "") or "-")
                commit = str(record.get("source_ref", "")) or self._get_git_commit(self.project_root)
                size = self._get_directory_size(build_path) if build_path.exists() else 0
                size_str = self._format_size(size)
                modified = self._get_last_modified(build_path)

                row = self.builds_table.rowCount()
                self.builds_table.insertRow(row)

                self.builds_table.setItem(row, 0, QTableWidgetItem(str(record.get("name", build_path.name))))
                self.builds_table.setItem(row, 1, QTableWidgetItem(backend))
                self.builds_table.setItem(row, 2, QTableWidgetItem(source))
                self.builds_table.setItem(row, 3, QTableWidgetItem(status))
                self.builds_table.setItem(row, 4, QTableWidgetItem(build_id))
                self.builds_table.setItem(row, 5, QTableWidgetItem(best_non))
                self.builds_table.setItem(row, 6, QTableWidgetItem(best_mtp))
                self.builds_table.setItem(row, 7, QTableWidgetItem(last_bench))
                self.builds_table.setItem(row, 8, QTableWidgetItem(commit))
                self.builds_table.setItem(row, 9, QTableWidgetItem(size_str))
                self.builds_table.setItem(row, 10, QTableWidgetItem(modified))

                self._row_by_build_id[row] = build_id
            return

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
            self.builds_table.setItem(row, 2, QTableWidgetItem("fork"))
            self.builds_table.setItem(row, 3, QTableWidgetItem("ready"))
            self.builds_table.setItem(row, 4, QTableWidgetItem("—"))
            self.builds_table.setItem(row, 5, QTableWidgetItem("-"))
            self.builds_table.setItem(row, 6, QTableWidgetItem("-"))
            self.builds_table.setItem(row, 7, QTableWidgetItem("-"))
            self.builds_table.setItem(row, 8, QTableWidgetItem(commit))
            self.builds_table.setItem(row, 9, QTableWidgetItem(size_str))
            self.builds_table.setItem(row, 10, QTableWidgetItem(modified))

    def on_build_selected(self):
        """Handle build selection"""
        selected = self.builds_table.selectedItems()
        if not selected:
            self.executables_table.setRowCount(0)
            self.build_details_label.setText("Select a build to view details")
            return

        row = self.builds_table.row(selected[0])
        build_path = self._selected_build_path_from_row(row)
        if build_path is None:
            self.executables_table.setRowCount(0)
            self.build_details_label.setText("Selected build path is not available")
            return

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

        registry = getattr(self.parent, "build_registry", None)
        reg_record = registry.get_by_dir(build_path) if registry is not None else None

        backend = self._detect_backend(build_path)
        details.append(f"Backend: {backend}")
        if reg_record:
            details.append(f"Build ID: {reg_record.get('id', '-')}")
            details.append(f"Source: {reg_record.get('source_type', '-')}")
            details.append(f"Best Non-MTP TPS: {reg_record.get('bench_best_non_mtp_tps', '-') or '-'}")
            details.append(f"Best MTP TPS: {reg_record.get('bench_best_mtp_tps', '-') or '-'}")
            details.append(f"Last benchmark: {reg_record.get('bench_last_run_at', '-') or '-'}")

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
        build_path = self._selected_build_path_from_row(row)
        if build_path is None:
            QMessageBox.warning(self, "Error", "Build path not found in registry")
            return

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
        old_path = self._selected_build_path_from_row(row)
        if old_path is None:
            QMessageBox.warning(self, "Error", "Build path not found in registry")
            return

        new_name, ok = self._ask_for_new_name(old_name)
        if not ok or not new_name:
            return

        new_path = self.project_root / new_name

        if new_path.exists():
            QMessageBox.warning(self, "Error", f"Folder '{new_name}' already exists")
            return

        try:
            old_path.rename(new_path)
            build_id = self._row_by_build_id.get(row, "")
            registry = getattr(self.parent, "build_registry", None)
            if registry is not None and build_id:
                registry.rename_build(build_id, new_name, new_path)
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
        build_path = self._selected_build_path_from_row(row)
        if build_path is None:
            QMessageBox.warning(self, "Error", "Build path not found in registry")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete build folder '{build_name}'?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                import shutil
                shutil.rmtree(build_path)
                build_id = self._row_by_build_id.get(row, "")
                registry = getattr(self.parent, "build_registry", None)
                if registry is not None and build_id:
                    registry.remove_by_id(build_id)
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

    def _selected_build_path_from_row(self, row: int) -> Path | None:
        build_id = self._row_by_build_id.get(row, "")
        registry = getattr(self.parent, "build_registry", None)
        if registry is not None and build_id:
            record = registry.get_by_id(build_id)
            if record and record.get("build_dir"):
                return Path(record["build_dir"])

        # Legacy fallback when registry is not available.
        item = self.builds_table.item(row, 0)
        if item is None:
            return None
        return self.project_root / item.text()
