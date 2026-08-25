"""Builds and models discovered from the filesystem.

Build capabilities come from CMakeCache.txt on purpose: probing llama-server
with --help/--version starts backend discovery and risks a driver drop.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

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

    @property
    def usable(self) -> bool:
        return self.server_bin is not None


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: Path
    name: str
    size_bytes: int

    @property
    def is_mtp(self) -> bool:
        return "_mtp" in self.name.lower()

    @property
    def is_mmproj(self) -> bool:
        return self.name.lower().startswith("mmproj")

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


def read_build(build_dir: Path) -> Build:
    flags = _cmake_flags(build_dir)
    return Build(
        path=build_dir,
        name=build_dir.name,
        backend=_backend_of(build_dir, flags),
        server_bin=_server_binary(build_dir),
        supports_rpc=flags.get("GGML_RPC"),
    )


def discover_builds(root: Path) -> list[Build]:
    """`build-*` directories under `root`, usable ones first."""
    try:
        candidates = sorted(entry for entry in root.iterdir()
                            if entry.is_dir() and entry.name.startswith("build-"))
    except OSError:
        return []
    builds = [read_build(entry) for entry in candidates]
    return sorted(builds, key=lambda build: (not build.usable, build.name))


def discover_models(models_dir: Path) -> list[ModelFile]:
    try:
        entries = sorted(models_dir.glob("*.gguf"), key=lambda path: path.name.lower())
    except OSError:
        return []
    models: list[ModelFile] = []
    for path in entries:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        models.append(ModelFile(path=path, name=path.name, size_bytes=size))
    return models


def find_build(builds: list[Build], name_or_path: str) -> Build | None:
    if not name_or_path:
        return None
    target = os.path.normcase(name_or_path)
    for build in builds:
        if os.path.normcase(build.name) == target or os.path.normcase(str(build.path)) == target:
            return build
    return None
