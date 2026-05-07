"""Download tab - full code will be inserted here - too large to display in preview"""
# Due to character limits, I will create this file using direct extraction from agent output
# File size: ~700 lines - importing from model_downloader threads

import time
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt

from model_downloader import ModelDownloader, DownloadThread, ListFilesThread


class DownloadTabWidget(QWidget):
    """Tab for downloading models from HuggingFace"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.models_dir = parent.models_dir
        self.model_downloader = parent.model_downloader
        self.download_thread = None
        self.list_files_thread = None
        self._current_model_id = None
        self._download_start_time = None
        self.create_ui()

    def create_ui(self):
        """Создание вкладки для загрузки models"""
        layout = QVBoxLayout(self)

        info_label = QLabel("📥 Search and Download Models from HuggingFace")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)

        # Search models
        search_group = QGroupBox("Search Models on HuggingFace")
        search_layout = QVBoxLayout()

        # Search field and buttons
        search_controls_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter query (e.g.: 'llama', 'mistral', 'codellama')...")
        self.search_input.returnPressed.connect(self.search_hf_models)
        search_controls_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.clicked.connect(self.search_hf_models)
        search_controls_layout.addWidget(self.search_btn)

        self.load_popular_btn = QPushButton("⭐ Popular")
        self.load_popular_btn.clicked.connect(self.load_popular_models)
        search_controls_layout.addWidget(self.load_popular_btn)

        search_layout.addLayout(search_controls_layout)

        # Date filter
        date_filter_layout = QHBoxLayout()
        date_filter_layout.addWidget(QLabel("Min Date:"))

        self.filter_year_combo = QComboBox()
        self.filter_year_combo.addItems(["All", "2025", "2024", "2023", "2022"])
        date_filter_layout.addWidget(self.filter_year_combo)

        self.filter_month_combo = QComboBox()
        self.filter_month_combo.addItems([
            "All months",
            "January (01)", "February (02)", "March (03)", "April (04)",
            "May (05)", "June (06)", "July (07)", "August (08)",
            "September (09)", "October (10)", "November (11)", "December (12)"
        ])
        date_filter_layout.addWidget(self.filter_month_combo)

        self.apply_date_filter_btn = QPushButton("🔄 Apply")
        self.apply_date_filter_btn.clicked.connect(self.apply_date_filter)
        date_filter_layout.addWidget(self.apply_date_filter_btn)
        date_filter_layout.addStretch()

        search_layout.addLayout(date_filter_layout)

        # Sorting
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort:"))

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["By Downloads ⬇️", "By Likes ❤️", "By Date Updated 📅", "By Name 🔤"])
        self.sort_combo.currentIndexChanged.connect(self.on_sort_changed)
        sort_layout.addWidget(self.sort_combo)
        sort_layout.addStretch()

        search_layout.addLayout(sort_layout)

        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Results table
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout()

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels([
            "Model Name", "Author", "Downloads", "Likes", "Updated"
        ])
        self.models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.models_table.itemDoubleClicked.connect(self.on_model_double_clicked)
        results_layout.addWidget(self.models_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Selected Model
        selected_group = QGroupBox("Selected Model")
        selected_layout = QVBoxLayout()

        self.selected_model_label = QLabel("Not Selected")
        self.selected_model_label.setStyleSheet("font-weight: bold;")
        selected_layout.addWidget(self.selected_model_label)

        # File selection from repository
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.model_file_combo = QComboBox()
        self.model_file_combo.setEnabled(False)
        file_layout.addWidget(self.model_file_combo)
        selected_layout.addLayout(file_layout)

        selected_group.setLayout(selected_layout)
        layout.addWidget(selected_group)

        # Download Progress
        progress_group = QGroupBox("Download Progress")
        progress_layout = QVBoxLayout()

        self.download_progress = QProgressBar()
        progress_layout.addWidget(self.download_progress)

        self.download_status_label = QLabel("Ready to Download")
        progress_layout.addWidget(self.download_status_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Buttons
        buttons_layout = QHBoxLayout()

        self.download_btn = QPushButton("📥 Download Model")
        self.download_btn.clicked.connect(self.download_model)
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet("QPushButton { font-size: 14px; padding: 8px; }")
        buttons_layout.addWidget(self.download_btn)

        self.cancel_download_btn = QPushButton("❌ Cancel")
        self.cancel_download_btn.setEnabled(False)
        self.cancel_download_btn.clicked.connect(self.cancel_download)
        buttons_layout.addWidget(self.cancel_download_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        # Initialization: загружаем популярные модели
        from PyQt6.QtCore import QThread
        QThread.msleep(100)
        self.load_popular_models()

    def search_hf_models(self):
        """Search Models on HuggingFace"""
        query = self.search_input.text().strip()
        if not query:
            query = "gguf"

        if self.parent.statusBar():
            self.parent.statusBar().showMessage(f"Search models: {query}...")
        self.models_table.setRowCount(0)

        try:
            sort_methods = ["downloads", "likes", "updated", "id"]
            sort_method = sort_methods[self.sort_combo.currentIndex()]

            min_date = None
            year = self.filter_year_combo.currentText()
            month_text = self.filter_month_combo.currentText()

            if year != "All" and month_text != "All months":
                month_num = month_text.split("(")[1].rstrip(")")
                min_date = f"{year}-{month_num}"
            elif year != "All":
                min_date = f"{year}-01"

            results = self.model_downloader.search_models(
                query=query,
                sort=sort_method,
                limit=50,
                min_date=min_date
            )

            self.display_search_results(results)
            date_info = f" (after {min_date})" if min_date else ""
            if self.parent.statusBar():
                self.parent.statusBar().showMessage(f"Found {len(results)} models{date_info}")

        except Exception as e:
            QMessageBox.warning(self, "Error поиска", f"Failed to search:\n{str(e)}")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Error поиска")

    def apply_date_filter(self):
        """Apply date filter"""
        self.search_hf_models()

    def load_popular_models(self):
        """Downloading популярных models"""
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Downloading популярных models...")
        self.models_table.setRowCount(0)

        try:
            results = self.model_downloader.get_popular_gguf_models(limit=30)
            self.display_search_results(results)
            if self.parent.statusBar():
                self.parent.statusBar().showMessage(f"Loaded {len(results)} популярных models")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load popular models:\n{str(e)}")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Error загрузки")

    def display_search_results(self, results: list):
        """Display search results in table"""
        self.models_table.setRowCount(len(results))

        for row, model in enumerate(results):
            name_item = QTableWidgetItem(model["model_name"])
            self.models_table.setItem(row, 0, name_item)

            author_item = QTableWidgetItem(model["author"])
            self.models_table.setItem(row, 1, author_item)

            downloads = model.get("downloads", 0)
            downloads_text = f"{downloads:,}" if downloads else "—"
            self.models_table.setItem(row, 2, QTableWidgetItem(downloads_text))

            likes = model.get("likes", 0)
            likes_text = f"{likes:,}" if likes else "—"
            self.models_table.setItem(row, 3, QTableWidgetItem(likes_text))

            updated = model.get("updated", "")
            updated_text = updated.split("T")[0] if updated else "—"
            self.models_table.setItem(row, 4, QTableWidgetItem(updated_text))

            name_item.setData(Qt.ItemDataRole.UserRole, model["id"])

    def on_model_double_clicked(self, item):
        """Handle model selection"""
        row = item.row()
        name_item = self.models_table.item(row, 0)
        model_id = name_item.data(Qt.ItemDataRole.UserRole)

        self.selected_model_label.setText(f"Model: {model_id}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(f"Loading file list for {model_id}...")

        self.model_file_combo.clear()
        self.model_file_combo.setEnabled(False)
        self.download_btn.setEnabled(False)

        self._current_model_id = model_id

        self.list_files_thread = ListFilesThread(model_id)
        self.list_files_thread.status_signal.connect(self._on_list_files_status)
        self.list_files_thread.finished_signal.connect(self._on_list_files_finished)
        self.list_files_thread.error_signal.connect(self._on_list_files_error)
        self.list_files_thread.start()

    def _on_list_files_status(self, status: str):
        """Handle list files status"""
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(status)

    def _on_list_files_finished(self, files_dict: dict):
        """Handle list files finished"""
        if files_dict:
            for filename, size in files_dict.items():
                size_str = self._format_size(size)
                display_name = f"{filename} ({size_str})"
                self.model_file_combo.addItem(display_name, filename)

            self.model_file_combo.setEnabled(True)
            self.download_btn.setEnabled(True)
            if self.parent.statusBar():
                self.parent.statusBar().showMessage(f"Found {len(files_dict)} files")
        else:
            model_id = getattr(self, "_current_model_id", "unknown")
            QMessageBox.warning(self, "Warning", f"No .gguf files found in repository {model_id}")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("No files found")

    def _on_list_files_error(self, error: str):
        """Handle list files error"""
        QMessageBox.warning(self, "Error", f"Failed to load file list:\n{error}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Error loading files")

    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human readable size"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"

    def on_sort_changed(self):
        """Handle sort changed"""
        if self.search_input.text().strip():
            self.search_hf_models()
        else:
            self.load_popular_models()

    def download_model(self):
        """Downloading модели"""
        if not self.download_btn.isEnabled():
            return

        model_id = self.selected_model_label.text().replace("Model: ", "").strip()
        if not model_id or model_id == "Not Selected":
            QMessageBox.warning(self, "Error", "First select a model from the list")
            return

        filename = self.model_file_combo.currentData()
        if not filename:
            filename = self.model_file_combo.currentText()
            if "(" in filename:
                filename = filename.split("(")[0].strip()

        if not filename:
            QMessageBox.warning(self, "Error", "Select file to download")
            return

        target_path = self.models_dir / filename
        if target_path.exists():
            reply = QMessageBox.question(
                self,
                "File exists",
                f"File {filename} already exists. Download again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.download_status_label.setText(f"Downloading {filename}...")
        self.download_progress.setValue(0)
        self.download_btn.setEnabled(False)
        self.cancel_download_btn.setEnabled(True)
        self.search_btn.setEnabled(False)
        self.load_popular_btn.setEnabled(False)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(f"Downloading {filename}...")

        self._download_start_time = time.time()

        self.download_thread = DownloadThread(model_id, filename, self.models_dir)
        self.download_thread.status.connect(self.on_download_status)
        self.download_thread.finished_signal.connect(self.on_download_finished)
        self.download_thread.error_signal.connect(self.on_download_error)
        self.download_thread.progress.connect(self.on_download_progress)
        self.download_thread.start()

    def cancel_download(self):
        """Cancel current download"""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.stop()
            self.download_status_label.setText("Cancelling download...")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Cancelling download...")
            self.download_thread.wait(2000)
            self._reset_download_ui()
            self.download_status_label.setText("Download cancelled")
            if self.parent.statusBar():
                self.parent.statusBar().showMessage("Download cancelled")

    def _reset_download_ui(self):
        """Reset download UI"""
        self.download_btn.setEnabled(True)
        self.cancel_download_btn.setEnabled(False)
        self.search_btn.setEnabled(True)
        self.load_popular_btn.setEnabled(True)
        self.download_progress.setValue(0)

    def on_download_status(self, status: str):
        """Handle download status"""
        self.download_status_label.setText(status)
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(status)

    def on_download_progress(self, downloaded, total, speed_mbps: float, eta_seconds: float):
        """Handle download progress"""
        downloaded = int(downloaded) if downloaded else 0
        total = int(total) if total else 0

        if total > 0:
            progress = int((downloaded / total) * 100)
            self.download_progress.setValue(progress)

            downloaded_mb = downloaded / 1024 / 1024
            total_mb = total / 1024 / 1024

            if eta_seconds > 3600:
                eta_str = f"{eta_seconds / 3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds / 60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"

            if total_mb >= 1024:
                downloaded_gb = downloaded_mb / 1024
                total_gb = total_mb / 1024
                self.download_status_label.setText(
                    f"Downloading: {downloaded_gb:.2f} / {total_gb:.2f} GB | "
                    f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_str}"
                )
            else:
                self.download_status_label.setText(
                    f"Downloading: {downloaded_mb:.1f} / {total_mb:.1f} MB | "
                    f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_str}"
                )

    def on_download_finished(self, file_path: str):
        """Handle download finished"""
        elapsed = time.time() - getattr(self, "_download_start_time", time.time())
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"

        self.download_progress.setValue(100)
        self.download_status_label.setText(f"[OK] Downloaded: {Path(file_path).name} in {elapsed_str}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage("Download completed")

        self.refresh_models_list()

        QMessageBox.information(
            self,
            "Success",
            f"Model downloaded successfully:\n{file_path}\n\nTime: {elapsed_str}"
        )

        self._reset_download_ui()

    def on_download_error(self, error: str):
        """Handle download error"""
        self.download_status_label.setText(f"[ERROR] {error}")
        if self.parent.statusBar():
            self.parent.statusBar().showMessage(f"Download error: {error}")

        QMessageBox.critical(self, "Download Error", f"Failed to download model:\n{error}")
        self._reset_download_ui()

    def browse_model(self):
        """Left for API compatibility"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели",
            str(self.models_dir),
            "GGUF Files (*.gguf);;All Files (*.*)"
        )
        if file_path and hasattr(self.parent, "model_path_edit"):
            self.parent.model_path_edit.setText(file_path)

    def refresh_models_list(self):
        """Update models lists in parent if needed"""
        if hasattr(self.parent, "refresh_models_list"):
            self.parent.refresh_models_list()
