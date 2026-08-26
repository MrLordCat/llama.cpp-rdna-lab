"""Backend-specific parameter panels for the server launch tab.

Each supported build backend (ROCm / Vulkan / CPU) exposes its own sub-tab with
only the parameters that make sense for that backend. The active sub-tab follows
the selected Build Backend and contributes CLI args + environment overrides to
the final llama-server command.
"""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    BACKEND_ROCM:   "ROCm",
    BACKEND_VULKAN: "Vulkan",
    BACKEND_CPU:    "CPU",
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
VULKAN_BALANCED_DUAL_CHOICE = (
    "Vulkan1,Vulkan0 — layer split (recommended)", "Vulkan1,Vulkan0", "layer", "1,1"
)
VULKAN_DISPLAY_FIRST_CHOICE = (
    "Vulkan0,Vulkan1 — layer split (display-first diagnostics)", "Vulkan0,Vulkan1", "layer", "1,1"
)

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
    VULKAN_BALANCED_DUAL_CHOICE,
    VULKAN_DISPLAY_FIRST_CHOICE,
]

# Sentinel appended to every device combo; its args come from the inline editor.
CUSTOM_DEVICE_CHOICE = ("Custom — edit devices / split / ratio below", None, None, None)

SPLIT_MODE_CHOICES = ["auto", "layer", "none", "row"]

# local dual-GPU order each backend prefixes to remote RPC workers
_LOCAL_DEVICE_PREFIX = {BACKEND_ROCM: "ROCm1,ROCm0", BACKEND_VULKAN: "Vulkan1,Vulkan0"}

_RPC_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9_.\-]+:\d{1,5}$")


def parse_rpc_endpoints(text: str) -> list[str]:
    """Valid, de-duplicated host:port endpoints from free-form user text."""
    endpoints: list[str] = []
    for chunk in re.split(r"[,;\s]+", (text or "").strip()):
        if not chunk or not _RPC_ENDPOINT_RE.match(chunk):
            continue
        if int(chunk.rsplit(":", 1)[1]) > 65535:
            continue
        if chunk not in endpoints:
            endpoints.append(chunk)
    return endpoints


def rpc_device_names(count: int) -> list[str]:
    """RPC device ids in --rpc order (one device per endpoint is assumed)."""
    return [f"RPC{i}" for i in range(max(0, int(count)))]


def rpc_server_args(endpoints: list[str]) -> list[str]:
    """`--rpc` args. Must precede -dev so RPC device names already resolve."""
    return ["--rpc", ",".join(endpoints)] if endpoints else []


def rpc_device_choices(backend_key: str, endpoint_count: int) -> list[tuple]:
    """Extra -dev profiles covering the remote RPC workers."""
    remote = ",".join(rpc_device_names(endpoint_count))
    if not remote:
        return []
    local = _LOCAL_DEVICE_PREFIX.get(backend_key, "")
    choices: list[tuple] = []
    if local:
        # no -ts: llama.cpp splits layers by free VRAM, which suits a smaller remote GPU
        choices.append((
            f"{local} + {remote} — layer split (auto ratio)",
            f"{local},{remote}", "layer", None,
        ))
    choices.append((
        f"{remote} only — remote diagnostics",
        remote, "layer" if endpoint_count > 1 else "none", None,
    ))
    return choices


def build_supports_rpc(build_dir: Path | str | None) -> bool | None:
    """RPC support read from CMakeCache; None when it cannot be determined.

    File-based on purpose: probing the binary would start backend discovery.
    """
    if not build_dir:
        return None
    cache = Path(build_dir) / "CMakeCache.txt"
    if not cache.exists():
        return None
    try:
        content = cache.read_text(errors="ignore")
    except OSError:
        return None
    if "GGML_RPC:BOOL=ON" in content:
        return True
    if "GGML_RPC:BOOL=OFF" in content:
        return False
    return None


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


class DevicePlacementFields(QWidget):
    """Inline -dev / -sm / -ts editor for the Custom device profile."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self.devices_input = QLineEdit()
        self.devices_input.setPlaceholderText("Vulkan1,Vulkan0,RPC0")
        self.devices_input.setToolTip(
            "-dev: comma separated device order, e.g. Vulkan1,Vulkan0,RPC0.\n"
            "Remote workers are named RPC0, RPC1, ... in --rpc order."
        )
        row.addWidget(QLabel("-dev"))
        row.addWidget(self.devices_input, 3)

        self.split_combo = QComboBox()
        self.split_combo.addItems(SPLIT_MODE_CHOICES)
        self.split_combo.setCurrentText("layer")
        self.split_combo.setToolTip("-sm: auto leaves the flag out; none keeps everything on the first device.")
        row.addWidget(QLabel("-sm"))
        row.addWidget(self.split_combo)

        self.ratio_input = QLineEdit()
        self.ratio_input.setPlaceholderText("auto")
        self.ratio_input.setToolTip(
            "-ts: layer weights per device, e.g. 27,37 or 16,16,10.\n"
            "Empty = llama.cpp splits proportionally to free VRAM."
        )
        row.addWidget(QLabel("-ts"))
        row.addWidget(self.ratio_input, 2)

        self.devices_input.textChanged.connect(lambda _text: self.changed.emit())
        self.ratio_input.textChanged.connect(lambda _text: self.changed.emit())
        self.split_combo.currentIndexChanged.connect(lambda _index: self.changed.emit())

    def args(self) -> list[str]:
        out: list[str] = []
        devices = self.devices_input.text().strip().replace(" ", "")
        if devices:
            out.extend(["-dev", devices])
        split_mode = self.split_combo.currentText()
        if split_mode != "auto":
            out.extend(["-sm", split_mode])
        ratio = self.ratio_input.text().strip().replace(" ", "")
        if ratio:
            out.extend(["-ts", ratio])
        return out

    def summary(self) -> str:
        return " ".join(self.args()) or "backend default"

    def prefill_from_choice(self, choice: tuple | None) -> None:
        """Seed the fields from a preset so Custom starts from the last profile."""
        if not choice or self.devices_input.text().strip():
            return
        _display, dev, split_mode, ratio = choice
        self.devices_input.setText(dev or "")
        self.split_combo.setCurrentText(split_mode or "auto")
        self.ratio_input.setText(ratio or "")

    def to_settings(self) -> dict:
        return {
            "devices": self.devices_input.text().strip(),
            "split_mode": self.split_combo.currentText(),
            "tensor_split": self.ratio_input.text().strip(),
        }

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        self.devices_input.setText(str(data.get("devices", "")))
        split_mode = str(data.get("split_mode", "layer"))
        if split_mode in SPLIT_MODE_CHOICES:
            self.split_combo.setCurrentText(split_mode)
        self.ratio_input.setText(str(data.get("tensor_split", "")))


class DeviceProfileSelector(QWidget):
    """Devices combo (presets + RPC profiles + Custom) with an inline editor."""

    changed = pyqtSignal()

    def __init__(
        self,
        base_choices: list[tuple],
        backend_key: str,
        default_choice: tuple,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._base_choices = list(base_choices)
        self._backend_key = backend_key
        self._default_choice = default_choice
        self._rpc_endpoint_count = 0
        self._choices: list[tuple] = []
        # Once the user picks an order by hand, stop auto-switching it to the
        # context-based recommendation.
        self.user_overridden = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("Devices:"))
        self.combo = QComboBox()
        combo_row.addWidget(self.combo, 1)
        layout.addLayout(combo_row)

        self.custom_fields = DevicePlacementFields()
        self.custom_fields.setVisible(False)
        layout.addWidget(self.custom_fields)

        self._rebuild_choices()
        self.select_choice(default_choice)
        self.combo.activated.connect(self._on_user_pick)
        self.combo.currentIndexChanged.connect(self._on_index_changed)
        self.custom_fields.changed.connect(self.changed)

    def set_combo_tooltip(self, text: str) -> None:
        self.combo.setToolTip(text)

    # -- choices ---------------------------------------------------------------
    def _rebuild_choices(self) -> None:
        previous = self.combo.currentText()
        self._choices = (
            self._base_choices
            + rpc_device_choices(self._backend_key, self._rpc_endpoint_count)
            + [CUSTOM_DEVICE_CHOICE]
        )
        self.combo.blockSignals(True)
        self.combo.clear()
        for choice in self._choices:
            self.combo.addItem(choice[0])
        idx = self.combo.findText(previous) if previous else -1
        self.combo.setCurrentIndex(idx if idx >= 0 else self._index_of(self._default_choice))
        self.combo.blockSignals(False)
        self._sync_custom_visibility()

    def set_rpc_endpoint_count(self, count: int) -> None:
        count = max(0, int(count))
        if count == self._rpc_endpoint_count:
            return
        self._rpc_endpoint_count = count
        self._rebuild_choices()
        self.changed.emit()

    def _index_of(self, choice: tuple) -> int:
        return self._choices.index(choice) if choice in self._choices else 0

    def is_custom(self) -> bool:
        return self.combo.currentIndex() == len(self._choices) - 1

    def current_choice(self) -> tuple:
        idx = self.combo.currentIndex()
        return self._choices[idx] if 0 <= idx < len(self._choices) else self._choices[0]

    def select_choice(self, choice: tuple) -> None:
        if choice in self._choices:
            self.combo.setCurrentIndex(self._choices.index(choice))

    def base_index(self) -> int:
        """Index into the backend preset list, or -1 for RPC/Custom profiles."""
        idx = self.combo.currentIndex()
        return idx if 0 <= idx < len(self._base_choices) else -1

    # -- command contribution ---------------------------------------------------
    def args(self) -> list[str]:
        if self.is_custom():
            return self.custom_fields.args()
        return device_choice_args(self.current_choice())

    def summary(self) -> str:
        return self.custom_fields.summary() if self.is_custom() else self.combo.currentText()

    # -- events -----------------------------------------------------------------
    def _on_user_pick(self, *_args) -> None:
        self.user_overridden = True

    def _on_index_changed(self, *_args) -> None:
        self._sync_custom_visibility()
        self.changed.emit()

    def _sync_custom_visibility(self) -> None:
        custom = self.is_custom()
        if custom:
            self.custom_fields.prefill_from_choice(self._default_choice)
        self.custom_fields.setVisible(custom)

    # -- persistence -------------------------------------------------------------
    def to_settings(self) -> dict:
        return {
            "display": self.combo.currentText(),
            "custom": self.custom_fields.to_settings(),
        }

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        self.custom_fields.from_settings(data.get("custom", {}))
        display = str(data.get("display", ""))
        idx = self.combo.findText(display) if display else -1
        if idx < 0 and display:
            # Labels evolve with measured recommendations; match the stable prefix.
            prefix = display.split(" — ", 1)[0].split(" - ", 1)[0]
            for candidate in range(self.combo.count()):
                item = self.combo.itemText(candidate)
                if item.split(" — ", 1)[0].split(" - ", 1)[0] == prefix:
                    idx = candidate
                    break
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self._sync_custom_visibility()


class RpcServersPanel(QWidget):
    """Remote llama.cpp RPC workers (--rpc host:port), shared by server/bench."""

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        row = QHBoxLayout()
        self.enable_check = QCheckBox("Use remote workers")
        self.enable_check.setToolTip(
            "Adds --rpc to the launch command. Start llama.cpp rpc-server on each\n"
            "remote host first; the GUI never starts or probes them."
        )
        row.addWidget(self.enable_check)

        self.endpoints_input = QLineEdit()
        self.endpoints_input.setPlaceholderText("192.168.1.60:53333, host:port")
        self.endpoints_input.setToolTip(
            "Comma separated host:port list, in device order.\n"
            "The first endpoint becomes RPC0, the next RPC1, and so on."
        )
        row.addWidget(self.endpoints_input, 1)
        layout.addLayout(row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("scenarioDetail", True)
        layout.addWidget(self.status_label)

        self._build_support: bool | None = None
        self.enable_check.toggled.connect(self._on_changed)
        self.endpoints_input.textChanged.connect(self._on_changed)
        self._refresh_status()

    # -- state ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        return self.enable_check.isChecked() and bool(self.endpoints())

    def endpoints(self) -> list[str]:
        return parse_rpc_endpoints(self.endpoints_input.text())

    def device_count(self) -> int:
        return len(self.endpoints()) if self.enable_check.isChecked() else 0

    def args(self) -> list[str]:
        return rpc_server_args(self.endpoints()) if self.is_enabled() else []

    def set_build_support(self, supported: bool | None) -> None:
        if supported != self._build_support:
            self._build_support = supported
            self._refresh_status()

    def summary(self) -> str:
        return ",".join(self.endpoints()) if self.is_enabled() else ""

    # -- events -----------------------------------------------------------------
    def _on_changed(self, *_args) -> None:
        self.endpoints_input.setEnabled(self.enable_check.isChecked())
        self._refresh_status()
        self.changed.emit()

    def _refresh_status(self) -> None:
        self.endpoints_input.setEnabled(self.enable_check.isChecked())
        if not self.enable_check.isChecked():
            self.status_label.setText("Off — only local GPUs are used.")
            return
        raw = self.endpoints_input.text().strip()
        endpoints = self.endpoints()
        if not endpoints:
            self.status_label.setText(
                "Enter at least one host:port endpoint." if raw
                else "Enter host:port endpoints, e.g. 192.168.1.60:53333."
            )
            return
        names = ", ".join(rpc_device_names(len(endpoints)))
        text = f"{len(endpoints)} endpoint(s) → {names} (usable in -dev / -ts)."
        if self._build_support is False:
            text += " Selected build has GGML_RPC=OFF — --rpc will be rejected."
        self.status_label.setText(text)

    # -- persistence -------------------------------------------------------------
    def to_settings(self) -> dict:
        return {
            "enabled": self.enable_check.isChecked(),
            "endpoints": self.endpoints_input.text().strip(),
        }

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        self.endpoints_input.setText(str(data.get("endpoints", "")))
        self.enable_check.setChecked(bool(data.get("enabled", False)))
        self._refresh_status()


class _RocmPanel(QWidget):
    """ROCm-specific launch parameters (2x RX 9070 XT rig defaults)."""

    DEVICE_CHOICES = ROCM_DEVICE_CHOICES

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.device_selector = DeviceProfileSelector(
            self.DEVICE_CHOICES, BACKEND_ROCM, ROCM_BALANCED_DUAL_CHOICE
        )
        self.device_selector.set_combo_tooltip(
            "Which GPUs the server uses.\n"
            "For large MTP runs, use ROCm1,ROCm0 layer split so weights/KV stay\n"
            "on the two cards instead of spilling one card into RAM. Single-GPU\n"
            "choices are mainly for clean diagnostics. Qwen3.8 Q4_K_M at 131K+\n"
            "uses the measured 27:37 split to respect each GPU's WDDM budget.\n"
            "Custom exposes raw -dev/-sm/-ts for RPC or hand-tuned placements."
        )
        layout.addWidget(self.device_selector)

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
        return self.device_selector.args()

    def env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.peer_copy_check.isChecked():
            out["GGML_ROCM_ENABLE_PEER_COPY"] = "1"
        if self.hsa_override_check.isChecked():
            out["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        if self.spec_window_spin.value() != 256:  # 256 is the server default
            out["LLAMA_SPEC_PREFILL_WINDOW"] = str(self.spec_window_spin.value())
        return out

    def set_rpc_endpoint_count(self, count: int) -> None:
        self.device_selector.set_rpc_endpoint_count(count)

    def to_settings(self) -> dict:
        return {
            "device_index": self.device_selector.base_index(),
            "device_profile": self.device_selector.to_settings(),
            "device_user_overridden": self.device_selector.user_overridden,
            "peer_copy": self.peer_copy_check.isChecked(),
            "hsa_override": self.hsa_override_check.isChecked(),
            "spec_window": self.spec_window_spin.value(),
            "spec_window_default_256": True,
        }

    def from_settings(self, data: dict) -> None:
        if isinstance(data.get("device_profile"), dict):
            self.device_selector.from_settings(data["device_profile"])
        else:
            idx = int(data.get("device_index", 3))
            if 0 <= idx < len(self.DEVICE_CHOICES):
                self.device_selector.select_choice(self.DEVICE_CHOICES[idx])
        self.device_selector.user_overridden = bool(data.get("device_user_overridden", False))
        self.peer_copy_check.setChecked(bool(data.get("peer_copy", False)))
        self.hsa_override_check.setChecked(bool(data.get("hsa_override", False)))
        spec_window = int(data.get("spec_window", 256))
        if spec_window == 8192 and not data.get("spec_window_default_256", False):
            spec_window = 256
        self.spec_window_spin.setValue(spec_window)

    def apply_model_recommendation(self, model_name: str, ctx_size: int) -> None:
        if self.device_selector.user_overridden or self.device_selector.is_custom():
            return
        recommended = recommended_rocm_device_choice(model_name, ctx_size)
        current = self.device_selector.current_choice()

        # Default/all-GPU and these two profiles are managed automatically.
        # Explicit single-GPU, reverse-order or RPC choices remain manual.
        managed = {ROCM_BALANCED_DUAL_CHOICE, ROCM_Q4KM_LONG_CONTEXT_CHOICE}
        if current in managed or (
            current[1] is None and recommended == ROCM_Q4KM_LONG_CONTEXT_CHOICE
        ):
            self.device_selector.select_choice(recommended)


class _VulkanPanel(QWidget):
    """Vulkan-specific launch parameters. MTP is supported on Vulkan too."""

    DEVICE_CHOICES = VULKAN_DEVICE_CHOICES

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.device_selector = DeviceProfileSelector(
            self.DEVICE_CHOICES, BACKEND_VULKAN, VULKAN_BALANCED_DUAL_CHOICE
        )
        self.device_selector.set_combo_tooltip(
            "Which GPUs the server uses.\n"
            "Vulkan1,Vulkan0 is the measured Q4/FP8 order: GPU1 handles the\n"
            "first stage while display-loaded GPU0 is second. Vulkan0,Vulkan1\n"
            "is retained for explicit diagnostics. KV follows its split layers.\n"
            "Custom exposes raw -dev/-sm/-ts for RPC or hand-tuned placements."
        )
        layout.addWidget(self.device_selector)

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

        self.large_matmul_check = QCheckBox("Use tuned AMD large matmul path (wn32)")
        self.large_matmul_check.setChecked(True)
        self.large_matmul_check.setToolTip(
            "Uses the wn32 Q4_K large-matmul variant recovered for recent AMD drivers.\n"
            "Disable to force the generic Vulkan matmul path for comparison."
        )
        layout.addWidget(self.large_matmul_check)

        # D096-M: native fp8 attention (P5) is always enabled on Vulkan; the
        # kernel ignores it for non-f8 KV types, so no separate checkbox needed.
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
        return self.device_selector.args()

    def set_rpc_endpoint_count(self, count: int) -> None:
        self.device_selector.set_rpc_endpoint_count(count)

    def env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.large_matmul_check.isChecked():
            out["GGML_VK_FORCE_AMD_LARGE_MATMUL"] = "1"
            out["GGML_VK_AMD_LARGE_MATMUL_VARIANT"] = "wn32"
        else:
            out["GGML_VK_DISABLE_AMD_LARGE_MATMUL"] = "1"
        if self.output_gpu1_check.isChecked():
            out["LLAMA_OUTPUT_DEVICE"] = "Vulkan1"
        if self.kv_gpu1_check.isChecked():
            out["LLAMA_KV_DEVICE"] = "Vulkan1"
        if self.spec_window_spin.value() != 256:  # 256 is the server default
            out["LLAMA_SPEC_PREFILL_WINDOW"] = str(self.spec_window_spin.value())
        return out

    def to_settings(self) -> dict:
        return {
            "device_index": self.device_selector.base_index(),
            "device_profile": self.device_selector.to_settings(),
            "device_user_overridden": self.device_selector.user_overridden,
            "output_gpu1": self.output_gpu1_check.isChecked(),
            "kv_gpu1": self.kv_gpu1_check.isChecked(),
            "large_matmul": self.large_matmul_check.isChecked(),
            "spec_window": self.spec_window_spin.value(),
            "spec_window_default_256": True,
            "device_order_v3": True,
        }

    def from_settings(self, data: dict) -> None:
        if isinstance(data.get("device_profile"), dict):
            self.device_selector.from_settings(data["device_profile"])
        else:
            idx = int(data.get("device_index", self.DEVICE_CHOICES.index(VULKAN_BALANCED_DUAL_CHOICE)))
            # v2 labeled the display-first dual order as recommended. Move managed legacy dual
            # selections to the measured GPU1-first default once; users can still
            # select the display-first diagnostic profile afterwards.
            if not data.get("device_order_v3", False) and idx in (3, 4):
                idx = self.DEVICE_CHOICES.index(VULKAN_BALANCED_DUAL_CHOICE)
            if 0 <= idx < len(self.DEVICE_CHOICES):
                self.device_selector.select_choice(self.DEVICE_CHOICES[idx])
        self.device_selector.user_overridden = bool(data.get("device_user_overridden", False))
        self.output_gpu1_check.setChecked(bool(data.get("output_gpu1", True)))
        self.kv_gpu1_check.setChecked(bool(data.get("kv_gpu1", False)))
        self.large_matmul_check.setChecked(bool(data.get("large_matmul", True)))
        # fp8_p5 legacy setting is ignored: P5 is always on for Vulkan now
        spec_window = int(data.get("spec_window", 256))
        if spec_window == 8192 and not data.get("spec_window_default_256", False):
            spec_window = 256
        self.spec_window_spin.setValue(spec_window)

    def apply_model_recommendation(self, _model_name: str, _ctx_size: int) -> None:
        if not self.device_selector.user_overridden and not self.device_selector.is_custom():
            self.device_selector.select_choice(VULKAN_BALANCED_DUAL_CHOICE)


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

        # showing/hiding the custom placement row changes the active page height
        for key in (BACKEND_ROCM, BACKEND_VULKAN):
            self.panels[key].device_selector.changed.connect(self.updateGeometry)

        # a QTabWidget is normally as tall as its TALLEST page, which leaves a
        # dead gap under shorter backends. Follow the active page instead by
        # reporting its height from sizeHint and re-querying on tab switch.
        self.currentChanged.connect(lambda _idx: self.updateGeometry())

    def sizeHint(self) -> QSize:
        base = super().sizeHint()
        page = self.currentWidget()
        if page is None:
            return base
        tab_h = self.tabBar().sizeHint().height()
        return QSize(base.width(), page.sizeHint().height() + tab_h + 16)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

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
        elif self.current_backend() == BACKEND_VULKAN:
            self.panels[BACKEND_VULKAN].apply_model_recommendation(model_name, ctx_size)

    def set_rpc_endpoint_count(self, count: int) -> None:
        """Publish the number of --rpc workers so RPC device profiles appear."""
        for key in (BACKEND_ROCM, BACKEND_VULKAN):
            self.panels[key].set_rpc_endpoint_count(count)
        self.updateGeometry()

    # -- persistence -----------------------------------------------------------
    def to_settings(self) -> dict:
        return {key: panel.to_settings() for key, panel in self.panels.items()}

    def from_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        for key, panel in self.panels.items():
            if isinstance(data.get(key), dict):
                panel.from_settings(data[key])
