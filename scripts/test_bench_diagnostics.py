#!/usr/bin/env python3
"""Offline regression tests for agent_workload_bench diagnostics parsing.

Runs without a GPU/server: it parses captured *.server.log files and asserts the
MTP/decode diagnostics behave correctly. Intended to guard the
baseline-vs-speculative comparison tooling.

Usage:
    python scripts/test_bench_diagnostics.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "build_logs" / "agent-workload"


def _load_module():
    spec = importlib.util.spec_from_file_location("awb", ROOT / "scripts" / "agent_workload_bench.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _find_log(*names: str) -> Path | None:
    for n in names:
        p = LOGS / f"{n}.server.log"
        if p.exists():
            return p
    return None


def test_decode_sentinel_excluded(m) -> None:
    """A run that decoded a single token must not report ~1e6 tok/s."""
    log = _find_log("vscode-rocm-mtp-fix-d3", "diag-2gpu-mtp-postreboot-r1")
    if log is None:
        print("  SKIP test_decode_sentinel_excluded (no log)")
        return
    d = m.parse_server_log_diagnostics(log)
    mx = float(d["decode_eval_tps"]["max"])
    assert mx < 100000.0, f"decode sentinel leaked into stats: max={mx}"
    print(f"  OK  decode sentinel excluded (max={mx}, degenerate={d['decode_degenerate_count']})")


def test_mtp_acceptance_parsed(m) -> None:
    """MTP runs expose an acceptance ratio derived from #acc/#gen draft tokens."""
    log = _find_log(
        "rocm-dual-layer-mtp-polish-mt256-n8-r1",
        "rocm1-mtp-polish-short-n8-r1",
        "diag-2gpu-mtp-postreboot-r1",
        "vscode-rocm-mtp-fix-d3",
    )
    if log is None:
        print("  SKIP test_mtp_acceptance_parsed (no log)")
        return
    d = m.parse_server_log_diagnostics(log)
    assert d.get("mtp_present"), "mtp_present should be True for an MTP log"
    acc = d["mtp_acceptance"]
    assert 0.0 <= acc <= 1.0, f"acceptance out of range: {acc}"
    # Sanity: parsed acceptance == acc_tokens / gen_tokens (when gen > 0).
    if d["mtp_gen_tokens"] > 0:
        expect = d["mtp_acc_tokens"] / d["mtp_gen_tokens"]
        assert abs(acc - expect) < 1e-4, f"acceptance mismatch {acc} != {expect}"
    print(f"  OK  mtp acceptance parsed ({acc:.2%}, "
          f"{d['mtp_acc_tokens']}/{d['mtp_gen_tokens']})")


def test_low_acceptance_hint(m) -> None:
    """A low-acceptance MTP run should surface a bottleneck hint."""
    log = _find_log(
        "rocm-dual-layer-mtp-polish-mt256-n8-r1",
        "rocm1-mtp-polish-short-n8-r1",
        "diag-2gpu-mtp-postreboot-r1",
        "vscode-rocm-mtp-fix-d3",
    )
    if log is None:
        print("  SKIP test_low_acceptance_hint (no log)")
        return
    d = m.parse_server_log_diagnostics(log)
    hints = m.build_bottleneck_hints([], d)
    if d.get("mtp_present") and d["mtp_gen_tokens"] > 0 and d["mtp_acceptance"] < 0.30:
        assert any("acceptance" in h for h in hints), f"missing low-acceptance hint: {hints}"
        print("  OK  low-acceptance hint emitted")
    else:
        print("  OK  acceptance healthy or no drafts; no low-acceptance hint required")


def test_baseline_decode_preserved(m) -> None:
    """A healthy baseline run keeps its real decode tok/s and reports no MTP."""
    log = _find_log("diag-2gpu-none-postreboot-r1", "diag-1gpu-none-r1")
    if log is None:
        print("  SKIP test_baseline_decode_preserved (no log)")
        return
    d = m.parse_server_log_diagnostics(log)
    assert d["decode_eval_tps"]["mean"] > 1.0, "baseline decode tok/s should be > 1"
    assert not d.get("mtp_present"), "baseline run should not be flagged as MTP"
    print(f"  OK  baseline decode preserved (mean={d['decode_eval_tps']['mean']})")


def test_prefixed_upstream_timings(m) -> None:
    """Timestamp-prefixed upstream server timings retain decode and acceptance."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "upstream.server.log"
        log.write_text(
            "0.38 I slot print_timing: prompt eval time = 26804.37 ms / 29563 tokens "
            "( 0.91 ms per token, 1102.92 tokens per second)\n"
            "0.39 I slot print_timing:        eval time = 3079.32 ms / 128 tokens "
            "(24.06 ms per token, 41.57 tokens per second)\n"
            "0.39 I slot print_timing: draft acceptance = 0.78070 "
            "(89 accepted / 114 generated), mean len = 3.34\n",
            encoding="utf-8",
        )

        d = m.parse_server_log_diagnostics(log)
        assert d["prompt_eval_tps"]["mean"] == 1102.92, d
        assert d["decode_eval_tps"]["mean"] == 41.57, d
        assert d["mtp_acc_tokens"] == 89, d
        assert d["mtp_gen_tokens"] == 114, d
        print("  OK  timestamp-prefixed upstream timings parsed")


def test_canonical_history_retention(m) -> None:
    """Canonical refresh must not re-import benchmark rows older than July 2026."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        history = out_dir / "BENCH_HISTORY.csv"
        with history.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=m.HISTORY_FIELDS)
            writer.writeheader()
            writer.writerow({"timestamp": "2026-06-30 23:59:59", "run_id": "old"})
            writer.writerow({"timestamp": "2026-07-01 00:00:00", "run_id": "kept"})

        count = m.refresh_canonical_history(out_dir)
        rows = m._read_history_csv(out_dir / m.CANONICAL_BENCH_RUNS_CSV)
        assert count == 1, f"expected one retained row, got {count}"
        assert [row["run_id"] for row in rows] == ["kept"], rows
        print("  OK  canonical history retention boundary enforced")


def main() -> int:
    m = _load_module()
    tests = [
        test_decode_sentinel_excluded,
        test_mtp_acceptance_parsed,
        test_low_acceptance_hint,
        test_baseline_decode_preserved,
        test_prefixed_upstream_timings,
        test_canonical_history_retention,
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
