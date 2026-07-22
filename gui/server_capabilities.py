"""Non-executing llama-server capability discovery for GUI workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LOCAL_FORK_SPEC_MODES = (
    "none",
    "ngram-mod",
    "draft",
    "eagle3",
    "mtp",
    "ngram-mtp",
)
UNKNOWN_SPEC_MODES = ("none",)
MTP_SPEC_MODES = {"mtp", "ngram-mtp"}

_SPEC_ALIASES = {
    "draft-mtp": "mtp",
    "ngram_mod": "ngram-mod",
    "ngram_mtp": "ngram-mtp",
}


@dataclass(frozen=True, slots=True)
class ServerCapabilities:
    """Capabilities resolved without loading or executing the server binary."""

    spec_modes: tuple[str, ...]
    known: bool
    source: str
    note: str = ""

    @property
    def supports_mtp(self) -> bool:
        return bool(MTP_SPEC_MODES.intersection(self.spec_modes))


class ServerCapabilityResolver:
    """Resolve server features from sidecar/build metadata only.

    Launching ``llama-server --help`` is deliberately forbidden here: loading a
    GPU backend just to inspect CLI flags can trigger driver discovery and is
    unsafe while another server or benchmark is active.
    """

    def __init__(self, project_root: Path, build_registry: Any = None):
        self.project_root = Path(project_root)
        self.build_registry = build_registry

    @staticmethod
    def _normalize_spec_modes(values: Iterable[object]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_value in values:
            mode = str(raw_value).strip().lower()
            mode = _SPEC_ALIASES.get(mode, mode)
            if not mode or mode in seen:
                continue
            seen.add(mode)
            normalized.append(mode)
        if "none" not in seen:
            normalized.insert(0, "none")
        return tuple(normalized)

    @staticmethod
    def _sidecar_modes(payload: object) -> tuple[str, ...]:
        if not isinstance(payload, dict):
            return ()
        capabilities = payload.get("capabilities", payload)
        if not isinstance(capabilities, dict):
            return ()
        modes = capabilities.get("spec_modes")
        if not isinstance(modes, list):
            return ()
        return ServerCapabilityResolver._normalize_spec_modes(modes)

    @staticmethod
    def _read_sidecar(path: Path) -> tuple[str, ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ()
        return ServerCapabilityResolver._sidecar_modes(payload)

    def _registry_record(self, server_bin: Path) -> dict[str, Any] | None:
        registry = self.build_registry
        if registry is None:
            return None
        try:
            build_id = registry.detect_build_id_from_server_bin(str(server_bin))
            record = registry.get_by_id(build_id) if build_id else None
        except Exception:
            return None
        return record if isinstance(record, dict) else None

    @staticmethod
    def _sidecar_candidates(server_bin: Path, record: dict[str, Any] | None) -> list[Path]:
        candidates = [
            server_bin.with_name(f"{server_bin.name}.capabilities.json"),
            server_bin.with_name("llama-server.capabilities.json"),
            server_bin.parent.parent / "llama-server.capabilities.json",
        ]
        if record:
            build_dir = str(record.get("build_dir", "")).strip()
            if build_dir:
                candidates.append(Path(build_dir) / "llama-server.capabilities.json")
        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def resolve(self, server_bin: Path) -> ServerCapabilities:
        server_bin = Path(server_bin)
        record = self._registry_record(server_bin)

        for sidecar in self._sidecar_candidates(server_bin, record):
            if not sidecar.is_file():
                continue
            modes = self._read_sidecar(sidecar)
            if modes:
                return ServerCapabilities(
                    spec_modes=modes,
                    known=True,
                    source=f"sidecar:{sidecar.name}",
                )

        if record and str(record.get("source_type", "")).strip().lower() == "fork":
            return ServerCapabilities(
                spec_modes=LOCAL_FORK_SPEC_MODES,
                known=True,
                source="build-registry:fork",
            )

        return ServerCapabilities(
            spec_modes=UNKNOWN_SPEC_MODES,
            known=False,
            source="conservative-default",
            note=(
                "Capability metadata is unavailable; only spec=none is enabled. "
                "Add llama-server.capabilities.json next to the build to enable more modes."
            ),
        )

    def select_spec_modes(
        self,
        server_bin: Path,
        requested: str | Iterable[str],
        *,
        mtp_compatible: bool,
    ) -> list[str]:
        """Filter a requested spec set through server and model capabilities."""

        capabilities = self.resolve(server_bin)
        allowed = set(capabilities.spec_modes)

        if isinstance(requested, str):
            raw_values = [value.strip() for value in requested.split(",") if value.strip()]
        else:
            raw_values = [str(value).strip() for value in requested if str(value).strip()]

        normalized_request = [
            _SPEC_ALIASES.get(value.lower(), value.lower()) for value in raw_values
        ]
        is_auto = not normalized_request or normalized_request == ["auto"] or normalized_request == ["all"]
        candidates = list(capabilities.spec_modes) if is_auto else normalized_request

        resolved: list[str] = []
        seen: set[str] = set()
        for mode in candidates:
            if mode not in allowed or mode in seen:
                continue
            if not mtp_compatible and mode in MTP_SPEC_MODES:
                continue
            seen.add(mode)
            resolved.append(mode)

        requested_mtp_only = bool(normalized_request) and all(
            mode in MTP_SPEC_MODES for mode in normalized_request
        )
        if not resolved and requested_mtp_only and not mtp_compatible and "none" in allowed:
            return ["none"]
        return resolved