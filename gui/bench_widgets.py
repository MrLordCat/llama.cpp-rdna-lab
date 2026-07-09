"""Shared Qt widget helpers for the benchmark/autotune UI.

Extracted from benchmark_tab.py so panel builders can be split into modules.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QComboBox,
    QHeaderView,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem with numeric sort semantics via UserRole."""

    def __init__(self, text: str, numeric_value: float):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, numeric_value)

    def __lt__(self, other):
        if isinstance(other, QTableWidgetItem):
            left_value = self.data(Qt.ItemDataRole.UserRole)
            right_value = other.data(Qt.ItemDataRole.UserRole)
            if left_value is not None and right_value is not None:
                try:
                    return float(left_value) < float(right_value)
                except (TypeError, ValueError):
                    pass
        return super().__lt__(other)


def create_scroll_panel(widget: QWidget) -> QScrollArea:
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll_area.setMinimumSize(0, 0)
    scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    scroll_area.setWidget(widget)
    return scroll_area


def configure_combo(combo: QComboBox, minimum_contents_length: int = 12) -> None:
    combo.setMinimumContentsLength(minimum_contents_length)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumWidth(80)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def configure_spinbox(spin_box: QSpinBox) -> None:
    spin_box.setMinimumWidth(76)
    spin_box.setMaximumWidth(118)
    spin_box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


def configure_compact_table(table: QTableWidget, column_widths: list[int]) -> None:
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideRight)
    table.setMinimumWidth(0)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    header = table.horizontalHeader()
    header.setMinimumSectionSize(48)
    header.setStretchLastSection(False)
    for column, width in enumerate(column_widths):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(column, width)
