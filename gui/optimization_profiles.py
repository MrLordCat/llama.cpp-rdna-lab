"""Shared GUI optimization defaults and server presets."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_NGRAM_MATCH = 24
DEFAULT_NGRAM_MIN = 48
DEFAULT_NGRAM_MAX = 64

KV_INDEX_TO_NAME = {
    0: "f16",
    1: "bf16",
    2: "f32",
    3: "q8_0",
    7: "q4_0",
}

KV_NAME_TO_INDEX = {value: key for key, value in KV_INDEX_TO_NAME.items()}

ACTIVE_QWEN36_27B_SERVER_PRESET = {
    "gpu_layers": 999,
    "context": 12288,
    "batch": 4096,
    "ubatch": 1024,
    "threads": 8,
    "parallel": 1,
    "kv": "f16",
    "flash_attn": True,
    "disable_thinking": False,
    "spec_type": "None",
    "extra_args": "--spec-type none\n-fit off",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "notes": "Current cold-first prompt-heavy lane: ctx=12288, b=4096, ub=1024, f16 KV, no speculative decoding. Use q4_0 KV if VRAM is tight.",
}

SERVER_PRESETS = {
    "Qwen3.6 27B Active": ACTIVE_QWEN36_27B_SERVER_PRESET,
    "Default": {
        "gpu_layers": 99,
        "context": 32768,
        "batch": 2048,
        "ubatch": 512,
        "threads": 8,
        "parallel": 1,
        "kv": "f16",
        "flash_attn": True,
        "spec_type": "None",
        "extra_args": "",
    },
    "Fast": {
        "gpu_layers": 99,
        "context": 4096,
        "batch": 4096,
        "ubatch": 1024,
        "threads": 4,
        "parallel": 1,
        "kv": "q8_0",
        "flash_attn": True,
        "spec_type": "None",
        "extra_args": "",
    },
    "Quality": {
        "gpu_layers": 99,
        "context": 16384,
        "batch": 2048,
        "ubatch": 1024,
        "threads": 16,
        "parallel": 1,
        "kv": "f16",
        "flash_attn": True,
        "spec_type": "None",
        "extra_args": "",
    },
    "Balanced": {
        "gpu_layers": 99,
        "context": 8192,
        "batch": 2048,
        "ubatch": 512,
        "threads": 8,
        "parallel": 1,
        "kv": "q8_0",
        "flash_attn": True,
        "spec_type": "None",
        "extra_args": "",
    },
    "VRAM Limited": {
        "gpu_layers": 20,
        "context": 4096,
        "batch": 512,
        "ubatch": 256,
        "threads": 4,
        "parallel": 1,
        "kv": "q4_0",
        "flash_attn": True,
        "spec_type": "None",
        "extra_args": "",
    },
}


def server_preset_names() -> list[str]:
    return list(SERVER_PRESETS.keys())


def get_server_preset(name: str) -> dict[str, object] | None:
    preset = SERVER_PRESETS.get(name)
    return deepcopy(preset) if preset is not None else None