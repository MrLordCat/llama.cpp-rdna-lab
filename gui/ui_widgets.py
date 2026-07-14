"""Shared small UI building blocks: chips, flow layout, status pill, log view."""

from __future__ import annotations

import html

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class FlowLayout(QLayout):
    """Left-to-right layout that wraps items onto new rows (Qt example port)."""

    def __init__(self, parent=None, margin: int = 0, h_spacing: int = 6, v_spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), apply_geometry=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, apply_geometry=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QRect, apply_geometry: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y = effective.x(), effective.y()
        line_height = 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > effective.right() + 1 and line_height > 0:
                x = effective.x()
                y += line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if apply_geometry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def make_chip(text: str, tooltip: str = "", checked: bool = False) -> QPushButton:
    """Checkable pill-shaped toggle; drop-in for QCheckBox usage patterns."""
    chip = QPushButton(text)
    chip.setCheckable(True)
    chip.setChecked(checked)
    if tooltip:
        chip.setToolTip(tooltip)
    chip.setProperty("chip", True)
    chip.setCursor(Qt.CursorShape.PointingHandCursor)
    chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return chip


class StatusPill(QLabel):
    """Rounded status badge; setText keeps state, set_state changes color too."""

    _STATES = {
        "neutral": ("#2a3038", "#aeb6c0", "#3c4652"),
        "ok": ("#1d4331", "#7fe0a7", "#2f7d4f"),
        "busy": ("#4a3a15", "#f0c674", "#8a6d1f"),
        "error": ("#4a1f1f", "#ff8a80", "#8a3535"),
    }

    def __init__(self, text: str = "", state: str = "neutral", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._apply(state)

    def _apply(self, state: str) -> None:
        background, color, border = self._STATES.get(state, self._STATES["neutral"])
        self._state = state
        self.setStyleSheet(
            f"QLabel {{ background: {background}; color: {color}; border: 1px solid {border};"
            " border-radius: 11px; padding: 3px 12px; font-weight: 600; }}"
        )

    def set_state(self, state: str, text: str | None = None) -> None:
        if state != getattr(self, "_state", None):
            self._apply(state)
        if text is not None:
            self.setText("● " + text if not text.startswith("●") else text)


class CollapsibleSection(QWidget):
    """Arrow-header section that shows/hides its content widget.

    Remembers its expanded state in QSettings when settings+key are given.
    """

    def __init__(self, title: str, content: QWidget, settings=None, settings_key: str = "", expanded: bool = True, parent=None):
        super().__init__(parent)
        self._content = content
        self._settings = settings
        self._settings_key = settings_key

        self._header = QToolButton()
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setProperty("sectionHeader", True)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._header)
        layout.addWidget(self._content)

        if self._settings is not None and self._settings_key:
            expanded = self._settings.value(self._settings_key, expanded, type=bool)
        self._header.setChecked(expanded)
        self._on_toggled(expanded)
        self._header.toggled.connect(self._on_toggled)

    def _on_toggled(self, expanded: bool) -> None:
        self._header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)
        if self._settings is not None and self._settings_key:
            self._settings.setValue(self._settings_key, expanded)


# line-classification rules for LogView, first match wins
_LOG_STYLES = [
    ("[ERROR]", "color:#ff8a80;"),
    ("ERROR", "color:#ff8a80;"),
    ("[WARN]", "color:#f0c674;"),
    ("BEST", "color:#4fd1bd; font-weight:600;"),
]


class LogView(QTextEdit):
    """Read-only monospace log with lightweight per-line highlighting.

    Drop-in for QTextEdit used via append()/clear().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)

    @staticmethod
    def _render_line(line: str) -> str:
        escaped = html.escape(line).replace("  ", "&nbsp;&nbsp;")
        for token, style in _LOG_STYLES:
            if token in line:
                return f'<span style="{style}">{escaped}</span>'
        if line.startswith("[INFO]"):
            rest = html.escape(line[len("[INFO]"):])
            return f'<span style="color:#7f8a97;">[INFO]</span><span>{rest}</span>'
        return f"<span>{escaped}</span>"

    def append(self, text: str) -> None:  # noqa: A003 - mirrors QTextEdit API
        lines = str(text).splitlines() or [""]
        for line in lines:
            super().append(self._render_line(line))
