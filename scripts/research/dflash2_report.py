#!/usr/bin/env python3
"""Classify DFlash2 parity/stability artifacts and summarize server logs.

The report deliberately separates bit-exact parity from runtime stability.
Upstream issue ggml-org/llama.cpp#27407 documents deterministic greedy
divergence caused by batched speculative verification even without DFlash2.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


RATE_RE = re.compile(
    r"draft acceptance rate\s*=\s*(?P<rate>[0-9.]+)\s*"
    r"\(\s*(?P<accepted>\d+) accepted\s*/\s*(?P<generated>\d+) generated\)"
)
PROMPT_TPS_RE = re.compile(
    r"prompt eval time\s*=.*?([0-9.]+) tokens per second\)", re.MULTILINE
)
DECODE_TIMING_RE = re.compile(
    r"^\s*eval time\s*=\s*([0-9.]+) ms\s*/\s*(\d+) tokens .*?"
    r"([0-9.]+) tokens per second\)",
    re.MULTILINE,
)


def _waves(payload: dict[str, Any], phase: str) -> list[list[dict[str, Any]]]:
    value = payload.get("phases", {}).get(phase, {})
    waves = value.get("waves", []) if isinstance(value, dict) else []
    return waves if isinstance(waves, list) else []


def _hashes_by_column(waves: list[list[dict[str, Any]]], width: int) -> list[list[str]]:
    columns: list[list[str]] = [[] for _ in range(width)]
    for wave in waves:
        for index, sample in enumerate(wave[:width]):
            value = str(sample.get("text_sha256", ""))
            if value:
                columns[index].append(value)
    return columns


def _distinct(columns: list[list[str]]) -> list[int]:
    return [len(set(values)) for values in columns]


def _all_stable(distinct: list[int]) -> bool:
    return bool(distinct) and all(value == 1 for value in distinct)


def parse_server_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rates = [
        {
            "rate": float(match.group("rate")),
            "accepted": int(match.group("accepted")),
            "generated": int(match.group("generated")),
        }
        for match in RATE_RE.finditer(text)
    ]
    prompt_tps = [float(value) for value in PROMPT_TPS_RE.findall(text)]
    decode_matches = DECODE_TIMING_RE.findall(text)
    decode_tps = [
        float(tps)
        for elapsed_ms, tokens, tps in decode_matches
        if int(tokens) > 1 and float(elapsed_ms) > 0.0 and float(tps) < 100000.0
    ]
    return {
        "path": str(path),
        "request_count": len(decode_matches),
        "decode_sample_count": len(decode_tps),
        "decode_degenerate_count": len(decode_matches) - len(decode_tps),
        "acceptance": rates,
        "acceptance_mean": statistics.mean(item["rate"] for item in rates) if rates else None,
        "prompt_tps_mean": statistics.mean(prompt_tps) if prompt_tps else None,
        "decode_tps_mean": statistics.mean(decode_tps) if decode_tps else None,
    }


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prompts = payload.get("prompts", [])
    width = len(prompts)
    if width <= 0:
        raise ValueError("artifact contains no prompts")

    serial_target = _hashes_by_column(_waves(payload, "serial_target"), width)
    serial_spec = _hashes_by_column(_waves(payload, "serial_spec"), width)
    hetero_target = _hashes_by_column(_waves(payload, "heterogeneous_target"), width)
    hetero_spec = _hashes_by_column(_waves(payload, "heterogeneous_spec"), width)

    target_reference = [values[0] if values else "" for values in serial_target]
    serial_spec_parity = [
        bool(target_reference[index]) and bool(values) and all(value == target_reference[index] for value in values)
        for index, values in enumerate(serial_spec)
    ]

    identical_target_hashes = [
        str(sample.get("text_sha256", ""))
        for wave in _waves(payload, "identical_target")
        for sample in wave
        if sample.get("text_sha256")
    ]
    identical_spec_hashes = [
        str(sample.get("text_sha256", ""))
        for wave in _waves(payload, "identical_spec")
        for sample in wave
        if sample.get("text_sha256")
    ]

    boundary_phase = payload.get("phases", {}).get("max_token_boundaries", {})
    boundary_cases = boundary_phase.get("cases", []) if isinstance(boundary_phase, dict) else []
    boundary_results = []
    for case in boundary_cases:
        target = case.get("target", {})
        spec = case.get("spec", {})
        max_tokens = int(case.get("max_tokens", 0) or 0)
        target_tokens = int(target.get("usage", {}).get("completion_tokens", 0) or 0)
        spec_tokens = int(spec.get("usage", {}).get("completion_tokens", 0) or 0)
        boundary_results.append({
            "max_tokens": max_tokens,
            "target_tokens": target_tokens,
            "spec_tokens": spec_tokens,
            "passed": (
                max_tokens > 0
                and target_tokens == max_tokens
                and spec_tokens == max_tokens
                and target.get("text_sha256") == spec.get("text_sha256")
            ),
        })

    summary = {
        "serial_target_distinct": _distinct(serial_target),
        "serial_spec_distinct": _distinct(serial_spec),
        "serial_spec_matches_target": serial_spec_parity,
        "heterogeneous_target_distinct": _distinct(hetero_target),
        "heterogeneous_spec_distinct": _distinct(hetero_spec),
        "identical_target_distinct": len(set(identical_target_hashes)),
        "identical_spec_distinct": len(set(identical_spec_hashes)),
        "max_token_boundaries": boundary_results,
    }

    findings: list[dict[str, str]] = []
    target_stable = _all_stable(summary["serial_target_distinct"])
    spec_stable = _all_stable(summary["serial_spec_distinct"])
    identical_target_stable = summary["identical_target_distinct"] <= 1
    identical_spec_stable = summary["identical_spec_distinct"] <= 1
    hetero_target_stable = _all_stable(summary["heterogeneous_target_distinct"])
    hetero_spec_stable = _all_stable(summary["heterogeneous_spec_distinct"])

    if not target_stable:
        findings.append({
            "code": "TARGET_STATE_INSTABILITY",
            "severity": "error",
            "detail": "Serial non-speculative target output changed across repeats.",
        })
    if not spec_stable:
        findings.append({
            "code": "SPEC_SERIAL_STATE_INSTABILITY",
            "severity": "error",
            "detail": "Serial DFlash2 output changed across repeats; inspect recurrent/KV cleanup.",
        })
    if spec_stable and not all(serial_spec_parity):
        findings.append({
            "code": "BATCHED_VERIFY_NUMERICAL_DIVERGENCE",
            "severity": "info",
            "detail": "Stable speculative output differs from sequential target; compare with upstream #27407.",
        })
    if not identical_target_stable:
        findings.append({
            "code": "IDENTICAL_TARGET_MULTISLOT_INSTABILITY",
            "severity": "error",
            "detail": "Identical prompts diverged across target-only slots.",
        })
    if not identical_spec_stable:
        findings.append({
            "code": "IDENTICAL_SPEC_MULTISLOT_INSTABILITY",
            "severity": "error",
            "detail": (
                "Identical prompts diverged across DFlash2 slots; investigate shared state, "
                "races, or batched-verification ordering."
            ),
        })
    if not hetero_target_stable:
        findings.append({
            "code": "TARGET_BATCH_SHAPE_SENSITIVITY",
            "severity": "warning",
            "detail": "Target-only output changes across heterogeneous concurrent wave shapes.",
        })
    if not hetero_spec_stable and identical_spec_stable:
        findings.append({
            "code": "HETEROGENEOUS_BATCH_SHAPE_SENSITIVITY",
            "severity": "warning",
            "detail": "DFlash2 varies only for heterogeneous concurrent prompts; track as batched numerical sensitivity.",
        })
    failed_boundaries = [item["max_tokens"] for item in boundary_results if not item["passed"]]
    if failed_boundaries:
        findings.append({
            "code": "MAX_TOKEN_BOUNDARY_MISMATCH",
            "severity": "error",
            "detail": f"Target/spec full-output parity failed at max_tokens={failed_boundaries}.",
        })
    if not findings:
        findings.append({
            "code": "STABLE_AND_BIT_EXACT",
            "severity": "ok",
            "detail": "All recorded controls are stable and speculative serial output matches target.",
        })

    return {"summary": summary, "findings": findings}


def render_markdown(payload: dict[str, Any], analysis: dict[str, Any], log_stats: dict[str, Any] | None = None) -> str:
    cfg = payload.get("configuration", {})
    summary = analysis["summary"]
    lines = [
        "# DFlash2 Lab Report",
        "",
        f"- Timestamp: `{payload.get('timestamp', 'unknown')}`",
        f"- Endpoint: `{cfg.get('url', 'unknown')}`",
        f"- Parallel slots: `{cfg.get('parallel', 'unknown')}`",
        f"- Speculative n_max: `{cfg.get('spec_n_max', 'unknown')}`",
        f"- Max output tokens: `{cfg.get('max_tokens', 'unknown')}`",
        f"- Server environment overrides: `{cfg.get('env_overrides') or {}}`",
        "",
        "## Stability matrix",
        "",
        "| Signal | Distinct hashes |",
        "| --- | --- |",
        f"| Serial target | `{summary['serial_target_distinct']}` |",
        f"| Serial DFlash2 | `{summary['serial_spec_distinct']}` |",
        f"| Heterogeneous target | `{summary['heterogeneous_target_distinct']}` |",
        f"| Heterogeneous DFlash2 | `{summary['heterogeneous_spec_distinct']}` |",
        f"| Identical target slots | `{summary['identical_target_distinct']}` |",
        f"| Identical DFlash2 slots | `{summary['identical_spec_distinct']}` |",
        (
            f"| Max-token boundaries | "
            f"`{sum(1 for item in summary['max_token_boundaries'] if item['passed'])}/"
            f"{len(summary['max_token_boundaries'])}` |"
        ),
        f"| Serial DFlash2 matches target | `{summary['serial_spec_matches_target']}` |",
        "",
        "## Findings",
        "",
    ]
    for finding in analysis["findings"]:
        lines.append(f"- **{finding['code']}** ({finding['severity']}): {finding['detail']}")

    if log_stats is not None:
        lines.extend([
            "",
            "## Server log",
            "",
            f"- Requests with timing blocks: `{log_stats['request_count']}`",
            f"- Mean prompt tok/s: `{log_stats['prompt_tps_mean']}`",
            f"- Mean decode tok/s: `{log_stats['decode_tps_mean']}`",
            f"- Excluded degenerate decode blocks: `{log_stats['decode_degenerate_count']}`",
            f"- Mean draft acceptance rate: `{log_stats['acceptance_mean']}`",
        ])

    lines.extend([
        "",
        "## Interpretation contract",
        "",
        "Bit-exact mismatch is recorded independently from stability. Upstream issue",
        "`ggml-org/llama.cpp#27407` shows that batched speculative verification can",
        "change greedy near-tie decisions even in `draft-simple`; DFlash2 can amplify",
        "that numerical effect without corrupting acceptance or coherent output.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a DFlash2 lab JSON artifact")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    analysis = analyze_payload(payload)
    log_stats = parse_server_log(args.server_log) if args.server_log else None

    if args.json:
        rendered = json.dumps({"analysis": analysis, "server_log": log_stats}, indent=2, ensure_ascii=False)
    else:
        rendered = render_markdown(payload, analysis, log_stats)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())