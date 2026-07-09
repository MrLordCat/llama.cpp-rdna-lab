"""Backend-specific parameter panels for the server launch tab.

Each supported build backend (ROCm / Vulkan / CPU) exposes its own sub-tab with
only the parameters that make sense for that backend. The active sub-tab follows
the selected Build Backend and contributes CLI args + environment overrides to
the final llama-server command.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# canonical backend keys used by the build registry
BACKEND_ROCM   = "rocm"
BACKEND_VULKAN = "vulkan"
BACKEND_CPU    = "cpu"

_TAB_ORDER = [BACKEND_ROCM, BACKEND_VULKAN, BACKEND_CPU]
_TAB_TITLES = {
    BACKEND_ROCM:   "🟥 ROCm",
    BACKEND_VULKAN: "🌋 Vulkan",
    BACKEND_CPU:    "🧠 CPU",
}


class _RocmPanel(QWidget):
    """ROCm-specific launch parameters (2x RX 9070 XT rig defaults)."""

    # (display, -dev value or None for all, -sm value or None, -ts value or None)
    DEVICE_CHOICES = [
        ("All GPUs — layer split (default order)", None,          "layer", None),
        ("ROCm1 only — diagnostics",               "ROCm1",      "none",  None),
        ("ROCm0 only — diagnostics",               "ROCm0",      "none",  None),
        ("ROCm1,ROCm0 — layer split (MTP)",         "ROCm1,ROCm0", "layer", "1,1"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Devices:"))
        self.device_combo = QComboBox()
        for display, _dev, _sm in self.DEVICE_CHOICES:
            self.device_combo.addItem(display)
        self.device_combo.setToolTip(
            "Which GPUs the server uses.\n"
            "For large MTP runs, use ROCm1,ROCm0 layer split so weights/KV stay\n"
            "on the two cards instead of spilling one card into RAM. Single-GPU\n"
            "choices are mainly for clean diagnostics."
        )
        dev_row.addWidget(self.device_combo, 1)
        layout.addLayout(dev_row)

        self.peer_copy_check = QCheckBox("Enable GPU peer-to-peer copies (GGML_ROCM_ENABLE_PEER_COPY=1)")
        self.peer_copy_check.setChecked(False)
        self.peer_copy_check.setToolTip(
            "OFF by default: HIP peer copies on Windows/RDNA4 can silently corrupt\n"
            "cross-device tensors (host-staged transfers are used instead).\n"
            "Enable only for testing."
        )
        layout.addWidget(self.peer_copy_check)

        self.hsa_override_check = QCheckBox("Legacy RDNA4 workaround (HSA_OVERRIDE_GFX_VERSION=11.0.0)")
        self.hsa_override_check.setChecked(False)
        self.hsa_override_check.setToolTip(
            "Only needed for ROCm older than 6.4. HIP SDK 7.1 supports gfx1201\n"
            "natively — leave OFF on this rig."
        )
        layout.addWidget(self.hsa_override_check)

        spec_row = QHBoxLayout()
        spec_row.addWidget(QLabel("Spec prefill window (tokens):"))
        self.spec_window_spin = QSpinBox()
        self.spec_window_spin.setRange(0, 131072)
        self.spec_window_spin.setSingleStep(1024)
        self.spec_window_spin.setValue(8192)
        self.spec_window_spin.setToolTip(
            "LLAMA_SPEC_PREFILL_WINDOW: with MTP/DFlash on long prompts, only the\n"
            "last N prompt tokens feed the draft context (skips the expensive\n"
            "full-prompt pass; small acceptance cost). 0 = always feed everything."
        )
        spec_row.addWidget(self.spec_window_spin)
        spec_row.addStretch()
        layout.addLayout(spec_row)

        layout.addStretch(1)

    def args(self) -> list[str]:
        out: list[str] = []
        _display, dev, sm, ts = self.DEVICE_CHOICES[self.device_combo.currentIndex()]
        if dev:
            out.extend(["-dev", dev])
        if sm:
            out.extend(["-sm", sm])
        if ts:
            out.extend(["-ts", ts])
        return out

    def env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.peer_copy_check.isChecked():
            out["GGML_ROCM_ENABLE_PEER_COPY"] = "1"
        if self.hsa_override_check.isChecked():
            out["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        if self.spec_window_spin.value() != 8192:  # 8192 is the built-in default
            out["LLAMA_SPEC_PREFILL_WINDOW"] = str(self.spec_window_spin.value())
        return out

    def to_settings(self) -> dict:
        return {
            "device_index": self.device_combo.currentIndex(),
            "peer_copy": self.peer_copy_check.isChecked(),
            "hsa_override": self.hsa_override_check.isChecked(),
            "spec_window": self.spec_window_spin.value(),
        }

    def from_settings(self, data: dict) -> None:
        idx = int(data.get("device_index", 0))
        if 0 <= idx < self.device_combo.count():
            self.device_combo.setCurrentIndex(idx)
        self.peer_copy_check.setChecked(bool(data.get("peer_copy", False)))
        self.hsa_override_check.setChecked(bool(data.get("hsa_override", False)))
        self.spec_window_spin.setValue(int(data.get("spec_window", 8192)))


class _VulkanPanel(QWidget):
    """Vulkan-specific launch parameters."""

    DEVICE_CHOICES = [
        ("All GPUs — layer split", None,      None),
        ("Vulkan0 only",           "Vulkan0", "none"),
        ("Vulkan1 only",           "Vulkan1", "none"),
    ]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Devices:"))
        self.device_combo = QComboBox()
        for display, _dev, _sm in self.DEVICE_CHOICES:
            self.device_combo.addItem(display)
        dev_row.addWidget(self.device_combo, 1)
        layout.addLayout(dev_row)

        self.large_matmul_check = QCheckBox("Force AMD large matmul path (GGML_VK_FORCE_AMD_LARGE_MATMUL=1)")
        self.large_matmul_check.setChecked(True)
        self.large_matmul_check.setToolTip("Measured faster on RX 9070 XT; disable to compare.")
        layout.addWidget(self.large_matmul_check)

        layout.addStretch(1)

    def args(self) -> list[str]:
        out: list[str] = []
        _display, dev, sm = self.DEVICE_CHOICES[self.device_combo.currentIndex()]
        if dev:
            out.extend(["-dev", dev])
        if sm:
            out.extend(["-sm", sm])
        return out

    def env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.large_matmul_check.isChecked():
            out["GGML_VK_FORCE_AMD_LARGE_MATMUL"] = "1"
        return out

    def to_settings(self) -> dict:
        return {
            "device_index": self.device_combo.currentIndex(),
            "large_matmul": self.large_matmul_check.isChecked(),
        }

    def from_settings(self, data: dict) -> None:
        idx = int(data.get("device_index", 0))
        if 0 <= idx < self.device_combo.count():
            self.device_combo.setCurrentIndex(idx)
        self.large_matmul_check.setChecked(bool(data.get("large_matmul", True)))


class _CpuPanel(QWidget):
    """CPU-only launch parameters."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        info = QLabel(
            "CPU backend: GPU layers are forced to 0.\n"
            "Tip: enable “Load model into RAM (--no-mmap)” in Performance Options —\n"
            "mmap paging usually limits CPU throughput."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.force_ngl0_check = QCheckBox("Force -ngl 0 (recommended for CPU builds)")
        self.force_ngl0_check.setChecked(True)
        layout.addWidget(self.force_ngl0_check)

        layout.addStretch(1)

    def args(self) -> list[str]:
        return ["-ngl", "0"] if self.force_ngl0_check.isChecked() else []

    def env(self) -> dict[str, str]:
        return {}

    def to_settings(self) -> dict:
        return {"force_ngl0": self.force_ngl0_check.isChecked()}

    def from_settings(self, data: dict) -> None:
        self.force_ngl0_check.setChecked(bool(data.get("force_ngl0", True)))


class BackendPanels(QTabWidget):
    """Sub-tabs with backend-specific server parameters.

    The active tab follows the selected Build Backend; args()/env() return the
    contribution of the ACTIVE backend only.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.panels = {
            BACKEND_ROCM:   _RocmPanel(self),
            BACKEND_VULKAN: _VulkanPanel(self),
            BACKEND_CPU:    _CpuPanel(self),
        }
        for key in _TAB_ORDER:
            self.addTab(self.panels[key], _TAB_TITLES[key])

    # -- backend selection ---------------------------------------------------
    def set_backend(self, backend_key: str) -> None:
        key = self._normalize(backend_key)
        if key in _TAB_ORDER:
            self.setCurrentIndex(_TAB_ORDER.index(key))

    def current_backend(self) -> str:
        return _TAB_ORDER[self.currentIndex()]

    @staticmethod
    def _normalize(backend_key: str) -> str:
        key = (backend_key or "").strip().lower()
        if "rocm" in key or "hip" in key:
            return BACKEND_ROCM
        if "vulkan" in key:
            return BACKEND_VULKAN
        if "cpu" in key:
            return BACKEND_CPU
        return key

    # -- command contribution -------------------------------------------------
    def args(self) -> list[str]:
        return self.panels[self.current_backend()].args()

    def env(self) -> dict[str, str]:
        return self.panels[self.current_backend()].env()

    # -- persistence -----------------------------------------------------------
    def to_settings(self) -> dict:
        return {key: panel.to_settings() for key, panel in self.panels.items()}

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for key, panel in self.panels.items():
            if isinstance(data.get(key), dict):
                panel.from_settings(data[key])
