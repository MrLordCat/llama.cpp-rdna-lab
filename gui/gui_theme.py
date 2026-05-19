"""Application-wide visual polish for the PyQt GUI."""

from __future__ import annotations

import platform

from PyQt6.QtGui import QFont


def apply_modern_theme(app) -> None:
    """Apply a restrained desktop-tool theme to the QApplication."""
    family = "Segoe UI" if platform.system() == "Windows" else "Noto Sans"
    app.setFont(QFont(family, 10))
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background: #15171b;
            color: #e8ebef;
            font-size: 10pt;
        }
        QTabWidget::pane {
            border: 1px solid #343a43;
            background: #15171b;
        }
        QTabBar::tab {
            background: #20242a;
            color: #c9ced6;
            padding: 8px 14px;
            border: 1px solid #343a43;
            border-bottom: none;
            min-width: 120px;
        }
        QTabBar::tab:selected {
            background: #2a3038;
            color: #ffffff;
            font-weight: 600;
        }
        QGroupBox {
            background: #1b1f25;
            border: 1px solid #343b45;
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            background: #1b1f25;
            color: #7bd8c5;
            font-weight: 600;
        }
        QLabel, QCheckBox, QRadioButton {
            background: transparent;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
            background: #101317;
            color: #edf0f4;
            border: 1px solid #3c4652;
            border-radius: 4px;
            padding: 4px;
            selection-background-color: #2f9e8f;
            selection-color: #ffffff;
        }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #4fd1bd;
        }
        QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            background: #1f242b;
            color: #87909c;
            border-color: #303741;
        }
        QPushButton {
            background: #222832;
            color: #edf0f4;
            border: 1px solid #46515e;
            border-radius: 5px;
            padding: 7px 12px;
        }
        QPushButton:hover {
            background: #2b3641;
            border-color: #5bc8b6;
        }
        QPushButton:pressed {
            background: #1d4f4a;
        }
        QPushButton:disabled {
            color: #77808c;
            background: #1d2229;
            border-color: #303741;
        }
        QCheckBox {
            spacing: 7px;
            padding: 2px 4px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            border: 1px solid #596575;
            background: #101317;
        }
        QCheckBox::indicator:hover {
            border-color: #67d4c2;
        }
        QCheckBox::indicator:checked {
            background: #2f9e8f;
            border-color: #67d4c2;
        }
        QCheckBox::indicator:checked:disabled {
            background: #42545b;
            border-color: #4d5965;
        }
        QHeaderView::section {
            background: #242a32;
            color: #edf0f4;
            padding: 6px;
            border: none;
            border-right: 1px solid #343b45;
            border-bottom: 1px solid #343b45;
            font-weight: 600;
        }
        QTableWidget {
            gridline-color: #303741;
            alternate-background-color: #171b20;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #1a1e24;
            border: none;
            width: 12px;
            height: 12px;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #4b5665;
            border-radius: 6px;
            min-height: 24px;
            min-width: 24px;
        }
        QStatusBar {
            background: #1b2027;
            color: #b5bdc8;
            border-top: 1px solid #343b45;
        }
        """
    )