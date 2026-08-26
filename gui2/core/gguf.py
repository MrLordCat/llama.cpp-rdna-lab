"""What a GGUF says about itself, read without loading it.

Only the header is parsed -- a few kilobytes off the front of the file, no GPU
and no llama.cpp binary involved. The layer count and the trained context
length bound the sliders on the server page, so a model trained for 32k cannot
be asked for 128k by accident. The attention and SSM shapes feed the VRAM
estimate in `gui2.core.memory`.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"

#: how llama.cpp names the parts of a split model (llama_split_path)
SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)

# GGUF value types
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4), _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}

#: everything the sliders and the VRAM estimate need, by key suffix
_WANTED_SUFFIXES = (
    ".block_count", ".context_length", ".embedding_length",
    ".attention.head_count", ".attention.head_count_kv",
    ".attention.key_length", ".attention.value_length",
    ".attention.sliding_window",
    ".full_attention_interval", ".nextn_predict_layers",
    ".ssm.conv_kernel", ".ssm.state_size", ".ssm.inner_size", ".ssm.group_count",
)
_WANTED_KEYS = ("general.architecture", "general.name", "general.size_label")

#: the vocabulary is the length of this array, and the array itself is the
#: megabytes of header we exist to avoid reading
TOKENS_KEY = "tokenizer.ggml.tokens"


@dataclass(frozen=True, slots=True)
class ModelFacts:
    """Bounds and labels for one model file."""

    path: Path
    architecture: str = ""
    name: str = ""
    size_label: str = ""
    n_layers: int | None = None
    n_ctx_train: int | None = None
    n_embd: int | None = None
    #: bytes of the whole model, which for a split model is all of its parts
    file_bytes: int = 0
    #: parts found on disk, and parts the name says there should be
    parts: int = 1
    declared_parts: int = 1
    n_vocab: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    sliding_window: int = 0
    #: >1 means only every n-th layer keeps a KV cache; the rest are recurrent
    full_attention_interval: int = 0
    #: NextN/MTP blocks appended past the main stack, not run in the main pass
    nextn_layers: int = 0
    ssm_conv: int = 0
    ssm_state: int = 0
    ssm_inner: int = 0
    ssm_group: int = 0
    error: str = ""

    @property
    def known(self) -> bool:
        return self.n_layers is not None or self.n_ctx_train is not None

    @property
    def missing_parts(self) -> int:
        """Parts the name promises that are not next to this file."""
        return max(0, self.declared_parts - self.parts)

    @property
    def head_dim_k(self) -> int:
        """Key width per head; older headers leave it to be derived."""
        if self.key_length:
            return self.key_length
        if self.n_embd and self.n_head:
            return self.n_embd // self.n_head
        return 0

    @property
    def head_dim_v(self) -> int:
        return self.value_length or self.head_dim_k

    @property
    def n_embd_k_gqa(self) -> int:
        """K row width of one token in one layer, llama.cpp's own name for it."""
        return (self.n_head_kv or self.n_head) * self.head_dim_k

    @property
    def n_embd_v_gqa(self) -> int:
        return (self.n_head_kv or self.n_head) * self.head_dim_v

    @property
    def summary(self) -> str:
        parts = [part for part in (self.architecture, self.size_label) if part]
        if self.n_layers is not None:
            parts.append(f"{self.n_layers} layers")
        if self.n_ctx_train is not None:
            parts.append(f"trained for {context_text(self.n_ctx_train)}")
        return " · ".join(parts)


def split_group(path: Path) -> tuple[list[Path], int]:
    """Every part of a split model that exists, and how many there should be.

    llama.cpp writes the parts as `<stem>-00001-of-00003.gguf`, refuses to load
    any but the first, and finds the rest from that name itself
    (`llama_get_list_splits`). So the name is the whole of the convention --
    just as well, because the parts past the first carry no architecture to
    read and nothing else could identify them.
    """
    match = SPLIT_RE.match(path.name)
    if not match:
        return [path], 1
    total = int(match["total"])
    candidates = [path.parent / f"{match['stem']}-{index:05d}-of-{total:05d}.gguf"
                  for index in range(1, total + 1)]
    return [part for part in candidates if part.is_file()], total


def is_first_part(path: Path) -> bool:
    """False only for the parts of a split model that cannot be loaded alone."""
    match = SPLIT_RE.match(path.name)
    return match is None or int(match["index"]) <= 1


def context_text(tokens: int) -> str:
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return str(tokens)


def _read(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return data


def _scalar(stream: BinaryIO, kind: int):
    if kind in _FIXED:
        fmt, size = _FIXED[kind]
        return struct.unpack(fmt, _read(stream, size))[0]
    if kind == _STRING:
        length = struct.unpack("<Q", _read(stream, 8))[0]
        return _read(stream, length).decode("utf-8", "replace")
    raise ValueError(f"unsupported GGUF type {kind}")


def _array_header(stream: BinaryIO) -> tuple[int, int]:
    kind = struct.unpack("<I", _read(stream, 4))[0]
    count = struct.unpack("<Q", _read(stream, 8))[0]
    return kind, count


def _skip_array_body(stream: BinaryIO, kind: int, count: int) -> None:
    if kind in _FIXED:
        stream.seek(_FIXED[kind][1] * count, 1)
        return
    if kind == _STRING:
        for _ in range(count):
            length = struct.unpack("<Q", _read(stream, 8))[0]
            stream.seek(length, 1)
        return
    if kind == _ARRAY:
        for _ in range(count):
            _skip_array_body(stream, *_array_header(stream))
        return
    raise ValueError(f"unsupported GGUF array type {kind}")


def _parse(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    with path.open("rb") as stream:
        if _read(stream, 4) != MAGIC:
            raise ValueError("not a GGUF file")
        struct.unpack("<I", _read(stream, 4))[0]  # version
        _read(stream, 8)  # tensor count
        kv_count = struct.unpack("<Q", _read(stream, 8))[0]

        for _ in range(kv_count):
            key_length = struct.unpack("<Q", _read(stream, 8))[0]
            key = _read(stream, key_length).decode("utf-8", "replace")
            kind = struct.unpack("<I", _read(stream, 4))[0]
            if kind == _ARRAY:
                array_kind, count = _array_header(stream)
                if key == TOKENS_KEY:
                    values[key] = count
                    # the token list is the last thing worth reading and the
                    # first thing worth not walking, so leave once it appears
                    if _complete(values):
                        break
                _skip_array_body(stream, array_kind, count)
                continue
            value = _scalar(stream, kind)
            if key in _WANTED_KEYS or key.endswith(_WANTED_SUFFIXES):
                values[key] = value
    return values


def _complete(values: dict[str, object]) -> bool:
    """True once the architecture block has been seen in full."""
    return "general.architecture" in values and any(
        key.endswith(".block_count") for key in values)


_cache: dict[tuple[str, int, int], ModelFacts] = {}


def read_facts(path: Path | str) -> ModelFacts:
    """Facts for a model file; cached until the file changes."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as exc:
        return ModelFacts(path=path, error=str(exc))

    parts, declared = split_group(path)
    total_bytes = 0
    for part in parts:
        try:
            total_bytes += part.stat().st_size
        except OSError:
            pass

    # a part arriving later changes the size, and so must change the key
    key = (str(path), stat.st_mtime_ns, total_bytes, len(parts))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        values = _parse(path)
    except (OSError, ValueError, struct.error) as exc:
        facts = ModelFacts(path=path, error=str(exc))
    else:
        architecture = str(values.get("general.architecture", ""))

        def by_suffix(suffix: str) -> int | None:
            for key_name, value in values.items():
                if key_name.endswith(suffix) and isinstance(value, int):
                    return int(value)
            return None

        def count(suffix: str) -> int:
            return by_suffix(suffix) or 0

        facts = ModelFacts(
            path=path,
            architecture=architecture,
            name=str(values.get("general.name", "")),
            size_label=str(values.get("general.size_label", "")),
            n_layers=by_suffix(".block_count"),
            n_ctx_train=by_suffix(".context_length"),
            n_embd=by_suffix(".embedding_length"),
            file_bytes=total_bytes,
            parts=len(parts),
            declared_parts=declared,
            n_vocab=int(values.get(TOKENS_KEY, 0) or 0),
            n_head=count(".attention.head_count"),
            n_head_kv=count(".attention.head_count_kv"),
            key_length=count(".attention.key_length"),
            value_length=count(".attention.value_length"),
            sliding_window=count(".attention.sliding_window"),
            full_attention_interval=count(".full_attention_interval"),
            nextn_layers=count(".nextn_predict_layers"),
            ssm_conv=count(".ssm.conv_kernel"),
            ssm_state=count(".ssm.state_size"),
            ssm_inner=count(".ssm.inner_size"),
            ssm_group=count(".ssm.group_count"),
        )

    _cache[key] = facts
    return facts
