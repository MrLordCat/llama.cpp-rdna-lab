#!/usr/bin/env python3
"""Compatibility wrapper around the legacy v1 benchmark (bench2 era).

The v1 agent-workload benchmark is preserved verbatim at
``scripts/legacy/agent_workload_bench.py``. This wrapper keeps the old entry
point and any external callers (GUI Benchmark tab, other bench scripts)
working unchanged: it re-runs the legacy file with the same ``__name__``, so
both ``python scripts/agent_workload_bench.py ...`` and
``import agent_workload_bench`` behave exactly as before.

New measurements should use ``scripts/bench2.py``; legacy output data was
archived to ``build_logs/archive/agent-workload-legacy-2026-08/``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parent / "legacy" / "agent_workload_bench.py"

if not LEGACY.exists():
    sys.exit(f"agent_workload_bench: legacy module missing: {LEGACY}")

runpy.run_path(str(LEGACY), run_name=__name__)
