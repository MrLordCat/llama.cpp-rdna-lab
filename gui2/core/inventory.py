"""Builds and models discovered from the filesystem.

Build capabilities come from CMakeCache.txt on purpose: probing llama-server
with --help/--version starts backend discovery and risks a driver drop.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gui2.core.gguf import is_first_part, split_group

BACKEND_FLAGS = (("GGML_HIP", "rocm"), ("GGML_VULKAN", "vulkan"))
SERVER_NAMES = ("llama-server.exe", "llama-server")

_cache_lock = threading.Lock()
_cache: dict[Path, tuple[float, dict[str, bool]]] = {}


@dataclass(frozen=True, slots=True)
class Build:
    path: Path
    name: str
    backend: str
    server_bin: Path | None
    supports_rpc: bool | None
    #: when llama-server was last linked; 0 when there is none to ask
    built_at: float = 0.0

    @property
    def usable(self) -> bool:
        return self.server_bin is not None

    @property
    def built_text(self) -> str:
        """How old this binary is, which is what "is it current" really asks."""
        if not self.built_at:
            return "never built"
        hours = (time.time() - self.built_at) / 3600
        if hours < 1:
            return f"{max(1, int(hours * 60))} min ago"
        if hours < 48:
            return f"{hours:.0f} h ago"
        return f"{hours / 24:.0f} days ago"

    @property
    def built_on(self) -> str:
        if not self.built_at:
            return "-"
        return datetime.fromtimestamp(self.built_at).strftime("%Y-%m-%d %H:%M")


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: Path
    name: str
    #: all the parts together, for a model that comes in parts
    size_bytes: int
    parts: int = 1
    declared_parts: int = 1

    @property
    def is_split(self) -> bool:
        return self.declared_parts > 1

    @property
    def missing_parts(self) -> int:
        return max(0, self.declared_parts - self.parts)

    @property
    def is_mtp(self) -> bool:
        """Whether this conversion kept its NextN block.

        The name is the only evidence: the GGUFs in this lab do not carry
        `nextn_predict_layers`, and the layer count only says "one more than
        the base" to someone who already knows the base. Same rule as
        `agent_workload_bench.is_mtp_model_name`, so the two agree on what a
        run gets filed as.
        """
        name = self.name.lower()
        return "-mtp" in name or "_mtp" in name or name.endswith("mtp.gguf") or "nextn" in name

    @property
    def is_mmproj(self) -> bool:
        # both conventions in the wild: mmproj-model-f16.gguf and Model.mmproj-F16.gguf
        return "mmproj" in self.name.lower()

    @property
    def size_text(self) -> str:
        gib = self.size_bytes / (1024 ** 3)
        return f"{gib:.1f} GiB" if gib >= 1 else f"{self.size_bytes / (1024 ** 2):.0f} MiB"


def _cmake_flags(build_dir: Path) -> dict[str, bool]:
    """`NAME:BOOL=` values from CMakeCache.txt, cached until the file changes."""
    cache_file = build_dir / "CMakeCache.txt"
    try:
        stamp = cache_file.stat().st_mtime
    except OSError:
        return {}

    with _cache_lock:
        cached = _cache.get(cache_file)
        if cached is not None and cached[0] == stamp:
            return cached[1]

    flags: dict[str, bool] = {}
    try:
        for line in cache_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, separator, value = line.partition(":BOOL=")
            if separator:
                flags[key.strip()] = value.strip().upper() in {"ON", "1", "TRUE", "YES"}
    except OSError:
        return {}

    with _cache_lock:
        _cache[cache_file] = (stamp, flags)
    return flags


def _server_binary(build_dir: Path) -> Path | None:
    for name in SERVER_NAMES:
        candidate = build_dir / "bin" / name
        if candidate.is_file():
            return candidate
    return None


def _backend_of(build_dir: Path, flags: dict[str, bool]) -> str:
    for key, backend in BACKEND_FLAGS:
        if flags.get(key):
            return backend
    lowered = build_dir.name.lower()
    for _key, backend in BACKEND_FLAGS:
        if backend in lowered:
            return backend
    return "cpu"


def _linked_at(server_bin: Path | None) -> float:
    """When the binary was written, not when the tree was configured.

    CMakeCache.txt is older than every rebuild that reused it, so it answers a
    different question than the one being asked.
    """
    try:
        return server_bin.stat().st_mtime if server_bin else 0.0
    except OSError:
        return 0.0


def read_build(build_dir: Path) -> Build:
    flags = _cmake_flags(build_dir)
    server_bin = _server_binary(build_dir)
    return Build(
        path=build_dir,
        name=build_dir.name,
        backend=_backend_of(build_dir, flags),
        server_bin=server_bin,
        supports_rpc=flags.get("GGML_RPC"),
        built_at=_linked_at(server_bin),
    )


def discover_builds(root: Path) -> list[Build]:
    """`build-*` directories under `root`, freshest usable one first.

    Newest first rather than alphabetical: with eight build directories the
    question is almost always "which one did I just build", and a name sort
    answers it only by accident.
    """
    try:
        candidates = [entry for entry in root.iterdir()
                      if entry.is_dir() and entry.name.startswith("build-")]
    except OSError:
        return []
    builds = [read_build(entry) for entry in candidates]
    return sorted(builds, key=lambda build: (not build.usable, -build.built_at, build.name))


def discover_models(models_dir: Path) -> list[ModelFile]:
    """One entry per model, not per file.

    A split model is listed once, under the part llama.cpp will actually
    accept; its size is all of its parts together. Offering part two of three
    as something to launch would only produce an error message.
    """
    try:
        entries = sorted(models_dir.glob("*.gguf"), key=lambda path: path.name.lower())
    except OSError:
        return []
    models: list[ModelFile] = []
    for path in entries:
        if not is_first_part(path):
            continue
        parts, declared = split_group(path)
        size = 0
        for part in parts:
            try:
                size += part.stat().st_size
            except OSError:
                pass
        models.append(ModelFile(path=path, name=path.name, size_bytes=size,
                                parts=len(parts), declared_parts=declared))
    return models


def find_build(builds: list[Build], name_or_path: str) -> Build | None:
    if not name_or_path:
        return None
    target = os.path.normcase(name_or_path)
    for build in builds:
        if os.path.normcase(build.name) == target or os.path.normcase(str(build.path)) == target:
            return build
    return None
