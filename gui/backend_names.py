"""Backend key <-> display-name mapping shared by the server and bench tabs."""

from __future__ import annotations

_KEY_TO_DISPLAY = {
    "rocm": "ROCm/HIP",
    "cpu": "CPU",
    "vulkan": "Vulkan",
}

_DISPLAY_TO_KEY = {display: key for key, display in _KEY_TO_DISPLAY.items()}


def display_backend_from_key(key: str) -> str:
    return _KEY_TO_DISPLAY.get(key.lower(), key)


def backend_key_from_display(display: str) -> str:
    return _DISPLAY_TO_KEY.get(display, display.lower())
