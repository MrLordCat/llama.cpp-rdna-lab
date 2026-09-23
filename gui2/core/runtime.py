"""What a launched build needs on PATH before its binary can load at all.

On Windows the loader resolves shared libraries through PATH, so a ROCm build
finds ``amdhip64_7.dll`` only when the SDK's ``bin`` directory is on it, and a
MinGW-linked binary needs its runtime DLLs from the toolchain. Without this the
process dies before it prints anything: ``exited with 3221225781``
(``0xC0000135``, DLL not found).

On Linux the same need is expressed as ``LD_LIBRARY_PATH``, which
:attr:`gui2.config.AppConfig.library_paths` fills from ``gui2.config.json``.

The directory list mirrors ``scripts/bench2.py:runtime_path_prepend()`` on
purpose: the GUI and the benchmark script must launch one build the same way,
or a server started from the GUI runs against a different ROCm runtime than the
one bench2 measured with.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: where the ROCm SDKs live when a build's CMakeCache cannot say
_ROCM_HOME = Path(r"C:\Program Files\AMD\ROCm")

#: MinGW runtime shared libraries (libgcc/libstdc++), which the shader
#: generator and any MinGW-linked binary need on PATH
_MINGW_BIN = Path(r"C:\Strawberry\c\bin")

def _cmake_compiler_dir(build_dir: Path | None) -> Path | None:
    """The directory of the compiler that produced a build.

    Read from CMakeCache.txt so adjacent SDKs stay apart: ``build-rocm72``
    compiled by ROCm 7.2 must load 7.2's runtime even when 7.1 is installed
    too, and pairing a build with the wrong runtime silently changes results.
    """
    if build_dir is None:
        return None
    try:
        text = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:") and "=" in line:
            return Path(line.split("=", 1)[1].strip()).parent
    return None

def _rocm_bin_fallback() -> Path | None:
    """The ROCm ``bin`` to use when CMakeCache has no compiler.

    ``HIP_PATH``/``ROCM_PATH`` win because they are what the user built with;
    otherwise the newest installed SDK, since a hard-coded version would aim a
    build at an older runtime than the one that produced it.
    """
    for variable in ("HIP_PATH", "ROCM_PATH"):
        value = os.environ.get(variable, "").strip()
        if value and (Path(value) / "bin").is_dir():
            return Path(value) / "bin"
    try:
        versions = sorted(
            (entry for entry in _ROCM_HOME.iterdir() if (entry / "bin").is_dir()),
            key=lambda entry: [int(part) if part.isdigit() else 0
                               for part in entry.name.replace("-", ".").split(".")],
        )
    except OSError:
        return None
    return (versions[-1] / "bin") if versions else None

def path_prepend(build_dir: Path | None, backend: str) -> list[str]:
    """Directories to put in front of PATH for this build, in order.

    Empty without a build directory: the answer depends on which build is being
    launched, and guessing one would be worse than leaving PATH alone.
    """
    if sys.platform != "win32" or build_dir is None:
        return []
    backend = (backend or "").lower()
    directories: list[Path] = []
    if backend in {"rocm", "hip"}:
        compiler = _cmake_compiler_dir(build_dir)
        if compiler is not None and compiler.is_dir():
            directories.append(compiler)
        else:
            fallback = _rocm_bin_fallback()
            if fallback is not None:
                directories.append(fallback)
    elif backend == "vulkan" and _MINGW_BIN.is_dir():
        directories.append(_MINGW_BIN)

    seen: set[str] = set()
    out: list[str] = []
    for directory in directories:
        text = str(directory)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out

def merged_path(directories: list[str], current: str) -> str:
    """``directories`` first, then the existing PATH without duplicates."""
    parts = [part for part in current.split(os.pathsep) if part]
    return os.pathsep.join([*directories, *(part for part in parts if part not in directories)])
