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

# (display, -dev value or None for all, -sm value or None, -ts value or None)
# shared with the bench tab so bench/autotune runs get the same device options
ROCM_BALANCED_DUAL_CHOICE = (
    "ROCm1,ROCm0 - layer split (recommended)", "ROCm1,ROCm0", "layer", "1,1"
)
ROCM_Q4KM_LONG_CONTEXT_CHOICE = (
    "ROCm1,ROCm0 - Q4_K_M 131K+ split (27:37)", "ROCm1,ROCm0", "layer", "27,37"
)
ROCM_Q4KM_LONG_CONTEXT_MIN = 131072

ROCM_DEVICE_CHOICES = [
    ("All GPUs — layer split (default order)", None,          "layer", None),
    ("ROCm1 only — diagnostics",               "ROCm1",      "none",  None),
    ("ROCm0 only — diagnostics",               "ROCm0",      "none",  None),
    ROCM_BALANCED_DUAL_CHOICE,
    ("ROCm0,ROCm1 — layer split (reverse order)", "ROCm0,ROCm1", "layer", "1,1"),
    ROCM_Q4KM_LONG_CONTEXT_CHOICE,
]

VULKAN_DEVICE_CHOICES = [
    ("All GPUs — layer split (default order)",        None,              "layer", None),
    ("Vulkan0 only — diagnostics",                    "Vulkan0",         "none",  None),
    ("Vulkan1 only — diagnostics",                    "Vulkan1",         "none",  None),
    ("Vulkan1,Vulkan0 — layer split (reverse order)", "Vulkan1,Vulkan0", "layer", "1,1"),
    ("Vulkan0,Vulkan1 — layer split (MTP recommended)", "Vulkan0,Vulkan1", "layer", "1,1"),
]


def device_choice_args(choice: tuple) -> list[str]:
    """CLI args (-dev/-sm/-ts) for one device choice tuple."""
    _display, dev, sm, ts = choice
    out: list[str] = []
    if dev:
        out.extend(["-dev", dev])
    if sm:
        out.extend(["-sm", sm])
    if ts:
        out.extend(["-ts", ts])
    return out


def is_qwen36_q4km_model(model_name: str) -> bool:
    """Return whether the measured Q4_K_M placement profile applies."""
    normalized = (model_name or "").lower().replace("-", "_").replace(".", "_")
    return "qwen3_6" in normalized and "27b" in normalized and "q4_k_m" in normalized


def recommended_rocm_device_choice(model_name: str, ctx_size: int) -> tuple:
    if is_qwen36_q4km_model(model_name) and ctx_size >= ROCM_Q4KM_LONG_CONTEXT_MIN:
        return ROCM_Q4KM_LONG_CONTEXT_CHOICE
    return ROCM_BALANCED_DUAL_CHOICE


class _RocmPanel(QWidget):
    """ROCm-specific launch parameters (2x RX 9070 XT rig defaults)."""

    DEVICE_CHOICES = ROCM_DEVICE_CHOICES

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Devices:"))
        self.device_combo = QComboBox()
        for choice in self.DEVICE_CHOICES:
            self.device_combo.addItem(choice[0])
        self.device_combo.setCurrentIndex(3)
        self.device_combo.setToolTip(
            "Which GPUs the server uses.\n"
            "For large MTP runs, use ROCm1,ROCm0 layer split so weights/KV stay\n"
            "on the two cards instead of spilling one card into RAM. Single-GPU\n"
            "choices are mainly for clean diagnostics. Qwen3.6 Q4_K_M at 131K+\n"
            "uses the measured 27:37 split to respect each GPU's WDDM budget."
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
        self.spec_window_spin.setSingleStep(256)
        self.spec_window_spin.setValue(256)
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
        return device_choice_args(self.DEVICE_CHOICES[self.device_combo.currentIndex()])

    def env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.peer_copy_check.isChecked():
            out["GGML_ROCM_ENABLE_PEER_COPY"] = "1"
        if self.hsa_override_check.isChecked():
            out["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        if self.spec_window_spin.value() != 256:  # 256 is the server default
            out["LLAMA_SPEC_PREFILL_WINDOW"] = str(self.spec_window_spin.value())
        return out

    def to_settings(self) -> dict:
        return {
            "device_index": self.device_combo.currentIndex(),
            "peer_copy": self.peer_copy_check.isChecked(),
            "hsa_override": self.hsa_override_check.isChecked(),
            "spec_window": self.spec_window_spin.value(),
            "spec_window_default_256": True,
        }

    def from_settings(self, data: dict) -> None:
        idx = int(data.get("device_index", 3))
        if 0 <= idx < self.device_combo.count():
            self.device_combo.setCurrentIndex(idx)
        self.peer_copy_check.setChecked(bool(data.get("peer_copy", False)))
        self.hsa_override_check.setChecked(bool(data.get("hsa_override", False)))
        spec_window = int(data.get("spec_window", 256))
        if spec_window == 8192 and not data.get("spec_window_default_256", False):
            spec_window = 256
        self.spec_window_spin.setValue(spec_window)

    def apply_model_recommendation(self, model_name: str, ctx_size: int) -> None:
        recommended = recommended_rocm_device_choice(model_name, ctx_size)
        current = self.DEVICE_CHOICES[self.device_combo.currentIndex()]

        # Default/all-GPU and these two profiles are managed automatically.
        # Explicit single-GPU or reverse-order choices remain manual.
        managed = {ROCM_BALANCED_DUAL_CHOICE, ROCM_Q4KM_LONG_CONTEXT_CHOICE}
        if current in managed or (
            current[1] is None and recommended == ROCM_Q4KM_LONG_CONTEXT_CHOICE
        ):
            self.device_combo.setCurrentIndex(self.DEVICE_CHOICES.index(recommended))


class _VulkanPanel(QWidget):
    """Vulkan-specific launch parameters. MTP is supported on Vulkan too."""

    DEVICE_CHOICES = VULKAN_DEVICE_CHOICES

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Devices:"))
        self.device_combo = QComboBox()
        for choice in self.DEVICE_CHOICES:
            self.device_combo.addItem(choice[0])
        self.device_combo.setCurrentIndex(4)
        self.device_combo.setToolTip(
            "Which GPUs the server uses.\n"
            "For MTP runs, Vulkan0,Vulkan1 currently preserves prompt throughput\n"
            "while output tensors remain on Vulkan1. The reverse order is kept\n"
            "for explicit A/B tests. KV stays with its split model layers."
        )
        dev_row.addWidget(self.device_combo, 1)
        layout.addLayout(dev_row)

        self.output_gpu1_check = QCheckBox("Place output tensors on Vulkan1")
        self.output_gpu1_check.setChecked(True)
        self.output_gpu1_check.setToolTip(
            "Moves the large vocabulary/output tensor away from GPU0 while keeping\n"
            "layer-local KV and fast attention. Recommended on a display-attached GPU0."
        )
        layout.addWidget(self.output_gpu1_check)

        self.kv_gpu1_check = QCheckBox("Place all KV cache on Vulkan1 (experimental)")
        self.kv_gpu1_check.setChecked(False)
        self.kv_gpu1_check.setToolTip(
            "Stores every attention K/V cache layer on physical GPU1 while model\n"
            "weights remain layer-split. This prevents GPU0 KV growth, but adds\n"
            "cross-GPU attention transfers and can substantially reduce speed."
        )
        layout.addWidget(self.kv_gpu1_check)

        self.large_matmul_check = QCheckBox("Force AMD large matmul path (GGML_VK_FORCE_AMD_LARGE_MATMUL=1)")
        self.large_matmul_check.setChecked(True)
        self.large_matmul_check.setToolTip("Measured faster on RX 9070 XT; disable to compare.")
        layout.addWidget(self.large_matmul_check)

        spec_row = QHBoxLayout()
        spec_row.addWidget(QLabel("Spec prefill window (tokens):"))
        self.spec_window_spin = QSpinBox()
        self.spec_window_spin.setRange(0, 131072)
        self.spec_window_spin.setSingleStep(256)
        self.spec_window_spin.setValue(256)
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
        if self.large_matmul_check.isChecked():
            out["GGML_VK_FORCE_AMD_LARGE_MATMUL"] = "1"
        if self.output_gpu1_check.isChecked():
            out["LLAMA_OUTPUT_DEVICE"] = "Vulkan1"
        if self.kv_gpu1_check.isChecked():
            out["LLAMA_KV_DEVICE"] = "Vulkan1"
        if self.spec_window_spin.value() != 256:  # 256 is the server default
            out["LLAMA_SPEC_PREFILL_WINDOW"] = str(self.spec_window_spin.value())
        return out

    def to_settings(self) -> dict:
        return {
            "device_index": self.device_combo.currentIndex(),
            "output_gpu1": self.output_gpu1_check.isChecked(),
            "kv_gpu1": self.kv_gpu1_check.isChecked(),
            "large_matmul": self.large_matmul_check.isChecked(),
            "spec_window": self.spec_window_spin.value(),
            "spec_window_default_256": True,
            "device_order_v2": True,
        }

    def from_settings(self, data: dict) -> None:
        idx = int(data.get("device_index", 4))
        if idx == 3 and not data.get("device_order_v2", False):
            idx = 4
        if 0 <= idx < self.device_combo.count():
            self.device_combo.setCurrentIndex(idx)
        self.output_gpu1_check.setChecked(bool(data.get("output_gpu1", True)))
        self.kv_gpu1_check.setChecked(bool(data.get("kv_gpu1", False)))
        self.large_matmul_check.setChecked(bool(data.get("large_matmul", True)))
        spec_window = int(data.get("spec_window", 256))
        if spec_window == 8192 and not data.get("spec_window_default_256", False):
            spec_window = 256
        self.spec_window_spin.setValue(spec_window)


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

    def apply_model_recommendation(self, model_name: str, ctx_size: int) -> None:
        if self.current_backend() == BACKEND_ROCM:
            self.panels[BACKEND_ROCM].apply_model_recommendation(model_name, ctx_size)

    # -- persistence -----------------------------------------------------------
    def to_settings(self) -> dict:
        return {key: panel.to_settings() for key, panel in self.panels.items()}

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for key, panel in self.panels.items():
            if isinstance(data.get(key), dict):
                panel.from_settings(data[key])
