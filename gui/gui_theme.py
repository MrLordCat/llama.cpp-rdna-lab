"""Application-wide visual polish for the PyQt GUI."""

from __future__ import annotations

import os
import platform
import sys

from PyQt6.QtGui import QFont


def apply_modern_theme(app) -> None:
    """Apply a restrained desktop-tool theme to the QApplication."""
    family = "Segoe UI" if platform.system() == "Windows" else "Noto Sans"
    app.setFont(QFont(family, 10))
    # In a PyInstaller bundle assets sit under _MEIPASS/assets; in source they
    # live next to this module.
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(base, "assets").replace("\\", "/")
    app.setStyleSheet(
        (
        """
        QMainWindow, QWidget {
            background: #1c2333;
            color: #e6ebf3;
            font-size: 10pt;
        }
        QTabWidget::pane {
            border: none;
            border-top: 1px solid #313c54;
            background: #1c2333;
            top: -1px;
        }
        QTabBar {
            qproperty-drawBase: 0;
            background: transparent;
        }
        QTabBar::tab {
            background: transparent;
            color: #95a1b6;
            padding: 9px 16px;
            margin-right: 2px;
            border: none;
            border-bottom: 2px solid transparent;
            min-width: 96px;
        }
        QTabBar::tab:hover {
            color: #d3d9e4;
            background: #232c3e;
        }
        QTabBar::tab:selected {
            color: #ffffff;
            background: #232c3e;
            border-bottom: 2px solid #4f7fd0;
            font-weight: 600;
        }
        QGroupBox {
            background: #232c3e;
            border: 1px solid #38455e;
            border-radius: 8px;
            margin-top: 15px;
            padding: 11px 12px 12px 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 8px;
            padding: 1px 7px;
            background: #232c3e;
            color: #7fa8e6;
            font-weight: 600;
            font-size: 10.5pt;
        }
        QLabel[heading="true"] {
            font-size: 15px;
            font-weight: 700;
            color: #8fb3ea;
            padding: 2px 0 9px 0;
            border-bottom: 1px solid #313c54;
            margin-bottom: 2px;
        }
        QLabel[subheading="true"] {
            color: #8fa0bd;
            font-size: 9.5pt;
            padding-bottom: 2px;
        }
        QLabel[muted="true"] {
            color: #7f8a9e;
        }
        QGroupBox[scenarioCard="true"] {
            background: #243150;
            border: 1px solid #3f5f9c;
            border-radius: 8px;
            padding: 12px;
        }
        QGroupBox[scenarioCard="true"]::title {
            background: #243150;
            color: #93b6ef;
        }
        QGroupBox[advancedPanel="true"] {
            background: #1f2839;
            border-color: #33415a;
        }
        QLabel[scenarioDetail="true"] {
            color: #9aa6ba;
            padding: 2px 1px;
        }
        QLabel[preflightState="neutral"] {
            color: #aeb7c2;
            background: #20262d;
            border: 1px solid #3a444f;
            border-radius: 5px;
            padding: 6px 8px;
        }
        QLabel[preflightState="ok"] {
            color: #91e6af;
            background: #193326;
            border: 1px solid #346d4b;
            border-radius: 5px;
            padding: 6px 8px;
        }
        QLabel[preflightState="busy"] {
            color: #f0c674;
            background: #332b18;
            border: 1px solid #756128;
            border-radius: 5px;
            padding: 6px 8px;
        }
        QLabel[preflightState="warning"] {
            color: #ffcc80;
            background: #38291b;
            border: 1px solid #80572f;
            border-radius: 5px;
            padding: 6px 8px;
        }
        QLabel[gridState="ok"] {
            color: #aeb7c2;
            background: #1b2026;
            border: 1px solid #343e48;
            border-radius: 5px;
            padding: 7px 9px;
        }
        QLabel[gridState="busy"] {
            color: #f0c674;
            background: #2d281b;
            border: 1px solid #645527;
            border-radius: 5px;
            padding: 7px 9px;
        }
        QLabel[gridState="error"] {
            color: #ff8a80;
            background: #351f21;
            border: 1px solid #7a3639;
            border-radius: 5px;
            padding: 7px 9px;
        }
        QLabel, QCheckBox, QRadioButton {
            background: transparent;
        }
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {
            background: #151c2b;
            color: #e9eef6;
            border: 1px solid #3a4863;
            border-radius: 5px;
            padding: 5px 7px;
            selection-background-color: #3d6fd6;
            selection-color: #ffffff;
        }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #4f7fd0;
        }
        QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            background: #232b3b;
            color: #828da1;
            border-color: #313c52;
        }
        QPushButton {
            background: #2b344a;
            color: #e9eef6;
            border: 1px solid #445069;
            border-radius: 5px;
            padding: 7px 12px;
        }
        QPushButton:hover {
            background: #33405a;
            border-color: #5a8ce8;
        }
        QPushButton:pressed {
            background: #263a63;
        }
        QPushButton:disabled {
            color: #77808c;
            background: #242c3b;
            border-color: #313c52;
        }
        QPushButton[accent="true"] {
            background: #3d6fd6;
            color: #ffffff;
            border: 1px solid #5a8ce8;
            font-weight: 600;
        }
        QPushButton[accent="true"]:hover {
            background: #4d7fdd;
            border-color: #7ba6ee;
        }
        QPushButton[accent="true"]:pressed {
            background: #2f5cb8;
        }
        QComboBox::drop-down {
            border: none;
            width: 22px;
        }
        QComboBox::down-arrow {
            image: url(@ASSETS@/chevron-down.svg);
            width: 12px;
            height: 12px;
        }
        QComboBox::down-arrow:disabled {
            image: none;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 18px;
            border-left: 1px solid #33415a;
            border-top-right-radius: 5px;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 18px;
            border-left: 1px solid #33415a;
            border-bottom-right-radius: 5px;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background: #2b3750;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image: url(@ASSETS@/chevron-up.svg);
            width: 11px;
            height: 11px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image: url(@ASSETS@/chevron-down.svg);
            width: 11px;
            height: 11px;
        }
        QComboBox QAbstractItemView {
            background: #1b2334;
            color: #e9eef6;
            border: 1px solid #3a4863;
            selection-background-color: #3d6fd6;
            selection-color: #ffffff;
            outline: none;
            padding: 3px;
        }
        QComboBox QAbstractItemView::item {
            min-height: 22px;
            padding: 2px 6px;
            border-radius: 4px;
        }
        QMenu {
            background: #1b2334;
            color: #e9eef6;
            border: 1px solid #3a4863;
            padding: 4px;
        }
        QMenu::item {
            padding: 5px 18px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background: #3d6fd6;
            color: #ffffff;
        }
        QMenu::separator {
            height: 1px;
            background: #33415a;
            margin: 4px 6px;
        }
        QPushButton[chip="true"] {
            background: #212a3b;
            color: #a6b1c4;
            border: 1px solid #3a4863;
            border-radius: 12px;
            padding: 3px 12px;
            font-size: 9pt;
        }
        QPushButton[chip="true"]:hover {
            border-color: #5a8ce8;
            color: #e6ebf3;
            background: #28324a;
        }
        QPushButton[chip="true"]:checked {
            background: #263a63;
            border-color: #4f7fd0;
            color: #dbe8ff;
            font-weight: 600;
        }
        QPushButton[chip="true"]:disabled {
            color: #6a7280;
            background: #1e2636;
            border-color: #2d374a;
        }
        QPushButton[chip="true"]:checked:disabled {
            background: #2a3a5a;
            border-color: #3f5273;
            color: #8fa0bd;
        }
        QToolButton[sectionHeader="true"] {
            background: transparent;
            border: none;
            color: #8fb3ea;
            font-weight: 600;
            font-size: 10.5pt;
            padding: 6px 2px;
        }
        QToolButton[sectionHeader="true"]:hover {
            color: #a9c6f2;
        }
        QCheckBox {
            spacing: 7px;
            padding: 2px 4px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            border: 1px solid #4d5a73;
            background: #151c2b;
        }
        QCheckBox::indicator:hover {
            border-color: #5a8ce8;
        }
        QCheckBox::indicator:checked {
            background: #3d6fd6;
            border-color: #5a8ce8;
        }
        QCheckBox::indicator:checked:disabled {
            background: #3a4661;
            border-color: #46536e;
        }
        QHeaderView::section {
            background: #2a3548;
            color: #e9eef6;
            padding: 6px;
            border: none;
            border-right: 1px solid #38455e;
            border-bottom: 1px solid #38455e;
            font-weight: 600;
        }
        QTableWidget {
            gridline-color: #313c52;
            alternate-background-color: #1b2334;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: transparent;
            border: none;
            width: 11px;
            height: 11px;
            margin: 0;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #3a4661;
            border-radius: 5px;
            min-height: 28px;
            min-width: 28px;
        }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background: #4f628a;
        }
        QScrollBar::add-line, QScrollBar::sub-line {
            width: 0;
            height: 0;
            background: transparent;
        }
        QScrollBar::add-page, QScrollBar::sub-page {
            background: transparent;
        }
        QStatusBar {
            background: #212a3b;
            color: #a6b1c4;
            border-top: 1px solid #38455e;
        }
        QToolTip {
            background: #232c3e;
            color: #e6ebf3;
            border: 1px solid #3f5f9c;
            border-radius: 4px;
            padding: 5px 7px;
        }
        """
        ).replace("@ASSETS@", assets)
    )
