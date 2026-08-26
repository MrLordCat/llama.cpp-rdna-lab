"""Hand-built GGUF headers.

A few hundred bytes are enough to prove the parser walks the key-value block
correctly and to pin the memory formulas to numbers real runs printed, and the
suite stays runnable without a models directory.
"""

from __future__ import annotations

import struct
from pathlib import Path

_STRING, _UINT32, _ARRAY = 8, 4, 9

#: the shape Qwen3.5/3.8 declares, and the one the memory tests are pinned to
QWEN35_27B: dict[str, int] = {
    "attention.head_count": 24,
    "attention.head_count_kv": 4,
    "attention.key_length": 256,
    "attention.value_length": 256,
    "full_attention_interval": 4,
    "nextn_predict_layers": 1,
    "ssm.conv_kernel": 4,
    "ssm.state_size": 128,
    "ssm.inner_size": 6144,
    "ssm.group_count": 16,
}


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", _STRING) + _string(value)
    return _string(key) + struct.pack("<I", _UINT32) + struct.pack("<I", value)


def _token_array(key: str, count: int) -> bytes:
    """The tokenizer list a real header carries; the parser must step over it."""
    body = struct.pack("<I", _STRING) + struct.pack("<Q", count)
    body += b"".join(_string(f"tok{index}") for index in range(count))
    return _string(key) + struct.pack("<I", _ARRAY) + body


def write_gguf(path: Path, *, architecture: str = "qwen3", layers: int = 48,
               context: int | None = 40960, embedding: int = 0, vocab: int = 64,
               hparams: dict[str, int] | None = None, pad_to: int = 0) -> Path:
    """A header with the keys GUI 2.0 reads, and nothing it does not."""
    pairs = [_kv("general.architecture", architecture), _kv("general.name", "Test Model")]
    pairs.append(_kv(f"{architecture}.block_count", layers))
    if context is not None:
        pairs.append(_kv(f"{architecture}.context_length", context))
    if embedding:
        pairs.append(_kv(f"{architecture}.embedding_length", embedding))
    for key, value in (hparams or {}).items():
        pairs.append(_kv(f"{architecture}.{key}", value))
    # last, as llama.cpp writes it: everything above must be read before it
    pairs.append(_token_array("tokenizer.ggml.tokens", vocab))

    header = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
              + b"".join(pairs))
    # file size stands in for the weights, so some tests need a plausible one
    path.write_bytes(header + b"\0" * max(0, pad_to - len(header)))
    return path
