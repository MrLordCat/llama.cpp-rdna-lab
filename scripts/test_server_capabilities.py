#!/usr/bin/env python3
"""Offline safety tests for gui/server_capabilities.ServerCapabilityResolver.

These tests guard the driver-safety invariant that server capability discovery
never launches ``llama-server`` (loading a GPU backend just to read CLI flags
can trigger driver discovery and is unsafe while a server or benchmark runs).
They run without a GPU, server or model.

Usage:
    python scripts/test_server_capabilities.py
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "gui" / "server_capabilities.py"

# GUI modules that resolve spec capabilities must never reintroduce a
# `llama-server --help` probe (loading a GPU backend to read CLI flags is
# forbidden by AGENTS.md driver-safety rules).
CAPABILITY_CALLER_FILES = (
    ROOT / "gui" / "benchmark_tab.py",
    ROOT / "gui" / "build_tab.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("server_capabilities", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass(slots=True) can resolve annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRegistry:
    """Minimal stand-in for BuildVersionRegistry used by the resolver."""

    def __init__(self, record: dict | None):
        self._record = record

    def detect_build_id_from_server_bin(self, server_bin: str) -> str:
        return "build-1" if self._record is not None else ""

    def get_by_id(self, build_id: str) -> dict | None:
        return self._record if build_id == "build-1" else None


def test_source_never_launches_a_process(m) -> None:
    """The resolver must not shell out: no process-launching primitives."""
    source = inspect.getsource(m)
    for forbidden in ("subprocess", "Popen", "run_hidden", "os.system", "os.popen"):
        assert forbidden not in source, f"capability resolver references {forbidden!r}"
    # The module itself must not import subprocess.
    assert getattr(m, "subprocess", None) is None, "subprocess leaked into resolver module"
    print("  OK  capability source launches no process")


def test_gui_callers_have_no_server_help_probe(_m) -> None:
    """Autotune callers must not spawn a llama-server --help capability probe.

    A bench-script ``--help`` probe (agent_workload_bench.py) stays allowed; the
    forbidden pattern is a server-binary help probe and its old helper names.
    """
    forbidden_markers = (
        "_server_help",
        "ProbeThread",
        "_start_server_help_probe",
        'server_bin), "--help"',
        "server_bin, '--help'",
    )
    for path in CAPABILITY_CALLER_FILES:
        source = path.read_text(encoding="utf-8")
        for forbidden in forbidden_markers:
            assert forbidden not in source, f"{path.name} reintroduced {forbidden!r}"
    print("  OK  GUI callers carry no server --help capability probe")


def test_unknown_build_is_conservative(m) -> None:
    """With no metadata the resolver enables only spec=none and flags unknown."""
    with tempfile.TemporaryDirectory() as tmp:
        resolver = m.ServerCapabilityResolver(Path(tmp), None)
        server_bin = Path(tmp) / "bin" / "llama-server.exe"
        caps = resolver.resolve(server_bin)
        assert caps.spec_modes == ("none",), caps.spec_modes
        assert caps.known is False, caps
        assert caps.supports_mtp is False, caps
        assert caps.note, "conservative default should carry an explanatory note"
    print("  OK  unknown build resolves to conservative spec=none")


def test_fork_build_enables_full_modes(m) -> None:
    """A registered fork build exposes the local speculative mode set."""
    with tempfile.TemporaryDirectory() as tmp:
        record = {"source_type": "fork", "build_dir": str(Path(tmp) / "build")}
        resolver = m.ServerCapabilityResolver(Path(tmp), _FakeRegistry(record))
        caps = resolver.resolve(Path(tmp) / "build" / "bin" / "llama-server.exe")
        assert caps.spec_modes == m.LOCAL_FORK_SPEC_MODES, caps.spec_modes
        assert caps.known is True, caps
        assert caps.supports_mtp is True, caps
    print("  OK  fork build enables full spec set")


def test_sidecar_metadata_wins(m) -> None:
    """A capabilities sidecar next to the binary overrides registry defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir(parents=True)
        server_bin = bin_dir / "llama-server.exe"
        server_bin.write_bytes(b"stub")
        sidecar = bin_dir / "llama-server.exe.capabilities.json"
        sidecar.write_text(
            json.dumps({"capabilities": {"spec_modes": ["none", "draft-mtp", "ngram_mod"]}}),
            encoding="utf-8",
        )
        resolver = m.ServerCapabilityResolver(Path(tmp), None)
        caps = resolver.resolve(server_bin)
        # draft-mtp and ngram_mod aliases normalise to mtp and ngram-mod.
        assert caps.spec_modes == ("none", "mtp", "ngram-mod"), caps.spec_modes
        assert caps.known is True, caps
        assert caps.supports_mtp is True, caps
        assert caps.source.startswith("sidecar:"), caps.source
    print("  OK  sidecar metadata parsed and aliases normalised")


def test_select_filters_mtp_for_incompatible_model(m) -> None:
    """auto selection drops MTP modes when the model cannot support them."""
    with tempfile.TemporaryDirectory() as tmp:
        record = {"source_type": "fork", "build_dir": str(Path(tmp))}
        resolver = m.ServerCapabilityResolver(Path(tmp), _FakeRegistry(record))
        server_bin = Path(tmp) / "llama-server.exe"
        modes = resolver.select_spec_modes(server_bin, "auto", mtp_compatible=False)
        assert modes, "auto should still resolve non-MTP modes"
        assert not (set(modes) & m.MTP_SPEC_MODES), modes
        assert "none" in modes, modes
    print("  OK  MTP modes filtered for incompatible model")


def test_select_mtp_only_degrades_to_none(m) -> None:
    """An explicit MTP-only request on a non-MTP model degrades to none."""
    with tempfile.TemporaryDirectory() as tmp:
        record = {"source_type": "fork", "build_dir": str(Path(tmp))}
        resolver = m.ServerCapabilityResolver(Path(tmp), _FakeRegistry(record))
        server_bin = Path(tmp) / "llama-server.exe"
        modes = resolver.select_spec_modes(server_bin, "mtp,ngram-mtp", mtp_compatible=False)
        assert modes == ["none"], modes
    print("  OK  MTP-only request degrades to none")


def test_select_explicit_subset_is_preserved(m) -> None:
    """Explicit supported modes pass through unchanged and in request order."""
    with tempfile.TemporaryDirectory() as tmp:
        record = {"source_type": "fork", "build_dir": str(Path(tmp))}
        resolver = m.ServerCapabilityResolver(Path(tmp), _FakeRegistry(record))
        server_bin = Path(tmp) / "llama-server.exe"
        modes = resolver.select_spec_modes(server_bin, "ngram-mod,none", mtp_compatible=True)
        assert modes == ["ngram-mod", "none"], modes
    print("  OK  explicit subset preserved")


def test_select_drops_unknown_modes_on_unknown_build(m) -> None:
    """On a conservative build only none survives, unsupported modes are dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        resolver = m.ServerCapabilityResolver(Path(tmp), None)
        server_bin = Path(tmp) / "llama-server.exe"
        modes = resolver.select_spec_modes(server_bin, "mtp,ngram-mod,none", mtp_compatible=True)
        assert modes == ["none"], modes
    print("  OK  unsupported modes dropped on conservative build")


def main() -> int:
    m = _load_module()
    tests = [
        test_source_never_launches_a_process,
        test_gui_callers_have_no_server_help_probe,
        test_unknown_build_is_conservative,
        test_fork_build_enables_full_modes,
        test_sidecar_metadata_wins,
        test_select_filters_mtp_for_incompatible_model,
        test_select_mtp_only_degrades_to_none,
        test_select_explicit_subset_is_preserved,
        test_select_drops_unknown_modes_on_unknown_build,
    ]
    failures = 0
    for t in tests:
        try:
            t(m)
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
