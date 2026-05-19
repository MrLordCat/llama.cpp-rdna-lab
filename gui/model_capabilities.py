"""Lightweight model capability probes used by GUI autotune flows."""

from __future__ import annotations

from pathlib import Path


_MTP_SCAN_CACHE: dict[str, tuple[int, int, bool]] = {}


def _scan_file_for_nextn_marker(path: Path) -> bool:
    marker = b"nextn"
    chunk_size = 4 * 1024 * 1024
    overlap = len(marker) - 1
    tail = b""

    try:
        with path.open("rb") as file_obj:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                data = (tail + chunk).lower()
                if marker in data:
                    return True
                tail = data[-overlap:] if len(data) > overlap else data
    except Exception:
        return False

    return False


def model_supports_mtp(model_path: Path | str) -> bool:
    path = Path(model_path)
    name = path.name.lower()

    if "mtp" in name or "nextn" in name:
        return True

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    if not resolved.exists() or not resolved.is_file():
        return False

    try:
        stat = resolved.stat()
        cache_key = str(resolved)
        cached = _MTP_SCAN_CACHE.get(cache_key)
        if cached and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
            return cached[2]

        has_mtp = _scan_file_for_nextn_marker(resolved)
        _MTP_SCAN_CACHE[cache_key] = (stat.st_size, stat.st_mtime_ns, has_mtp)
        return has_mtp
    except Exception:
        return False
