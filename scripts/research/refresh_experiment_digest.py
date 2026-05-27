#!/usr/bin/env python3
"""Refresh the compact research experiment digest from RESULTS_LOG.md."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_LOG = ROOT / "docs" / "research" / "RESULTS_LOG.md"
DIGEST = ROOT / "docs" / "research" / "EXPERIMENTS_DIGEST.md"


FAMILY_SUMMARIES = [
    (
        "Speculative/formula foundations",
        "E001-E006, E028-E030, E060, E107, E111-E113, E250-E251",
        "ngram/reuse is a real repeated-session path; cold-first MTP/ngram claims need strict no-reuse/no-prime separation.",
        "Local Q3/Q4 MTP cold-target escapes failed or timed out; do not promote MTP without a compatible fast GGUF and prompt-overhead proof.",
        "Keep speculative work as a separate repeated/decode program, not a dense 27B cold-prefill claim.",
    ),
    (
        "ROCm allocator/residency",
        "E007-E008, E044, E123",
        "ROCm compute-vbuffer chunking fixed the RDNA4 ubatch cliff while preserving native large ubatch.",
        "Chunk-size/env sweeps after the fix are mostly negative controls, not a new acceleration route.",
        "Use `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` only as a causality/rollback control.",
    ),
    (
        "ROCm large Q3_K prefill",
        "E045-E059, E103-E106, E228-E256",
        "The practical bottleneck is large Q3_K prefill through staged fp16 + rocBLAS/hipBLAS; simple dequant/MMQ/compute16/library switches are exhausted.",
        "Persistent fp16 cache, existing MMQ override, hipBLASLt grouped GEMM, fast16 f32-output compute, Q3FlashMatmul scalar/WMMA, src1 reuse stack, and ubatch fine retunes failed.",
        "A future ROCm win needs a new compressed-GEMM/layout topology with point proof before server integration.",
    ),
    (
        "130k major topology",
        "D002-D035",
        "Vulkan reached the old 2 TPS target through the D012 q3quad + GLU stack; D035 hardens those route pieces plus a narrow host-KV guard as defaults and recovers the fresh slow pocket to `1.8736 TPS`, still below D012. ROCm is recentered at `1.5200 TPS` and paused after D013-D027.",
        "ROCm ub256, raw-storage escape, forced cublas/dequant, no-mmap, src1 quant reuse, y32/w2, GLU-only, dense staging, b4 loads, Q3Flash active-shape promotion, wider-N scalar Q3Flash, dual-Y MMQ, vbuffer single-chunk, multi-row WMMA Q3Flash, upstream-stock rollback, streaming dequant+rocBLAS chunking, pair-only FFN SwiGLU WMMA, naive whole-FFN streaming, expanded persistent Q3_K layout, compact signed-nibble unpack-only layout, Vulkan activation-only/naive-streaming whole-FFN, old all-Q3 storage/helper/Q8/tile families, compact Q3S layout-body work, FA-only pivot, q3-octa/LOAD_VEC_A=8 repeat, and broad host-KV-as-speed-route sweeps all failed.",
        "Continue Vulkan speed work only with a true Q3_K compute body or compressed-dot route: D032 says FA can be a stack component only after Q3 has roughly `1.18-1.20x` local point/static evidence, D033 rejects wider per-invocation Q3 dequant, and D035 is stability hardening rather than a 2.4 TPS route; reopen ROCm only with a stronger topology proof.",
    ),
    (
        "ROCm decode parity",
        "E149-E201",
        "E151 kept RDNA4 Q3_K ncols=1 `nwarps=2` as a real decode win; Vulkan remains faster on decode-heavy lanes.",
        "Generic fusion removal/addition, wider wave64 transfer, preload/fused-pair microforms, occupancy-only changes, and padded-storage variants did not close the gap.",
        "Treat decode parity as a separate route-body/layout problem; do not mix with 12k prompt-heavy wall claims.",
    ),
    (
        "Vulkan Q3_K coopmat prefill",
        "E061-E102, E257-E265",
        "E082/E086/E102 are kept: corrected Q3_K large matmul path is default on the local AMD proprietary coopmat device. E257 is the archived dense 12k Vulkan baseline.",
        "Nearby stride/helper/tile/Q8/int-dot/f16/queue/mmap/batch/transpose-A/F16-src1 routes are rejected or invalid for the archived 12k lane.",
        "E265 establishes the active 130k quick baseline; post-E265 work must start as a major topology or residency design, not another helper/no-code probe.",
    ),
    (
        "Vulkan long-context / CPU fallback",
        "E125-E147, E265",
        "Vulkan 64k has useful decode but loses full wall to ROCm because Q3_K + FA prefill dominate; `-ngl 0` CPU fallback keeps `--no-mmap` as a practical local profile.",
        "FA tile flips, split-k forcing, q8/f16 KV pivots, large Q3_K tiles, BK shrink, existing predequant, and broad persistent Q3_K layout are rejected.",
        "Keep 64k as a diagnostic lane; use E265 for quick 130k speed claims and reserve full-fill for explicit residency stress.",
    ),
    (
        "Practical model/profile routes",
        "E241-E255",
        "A3B practical profile clears the user-visible cold >10 TPS target with E255 r3 `22.1407 TPS`.",
        "This is not an apples-to-apples dense 27B-Q3 speedup.",
        "Keep A3B as a GUI practical profile and keep dense 27B kernel work separate.",
    ),
]


def split_table_row(line: str) -> list[str]:
    text = line.strip()
    if not text.startswith("|") or not text.endswith("|"):
        return []
    return [cell.strip() for cell in text.strip("|").split("|")]


def parse_results_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in RESULTS_LOG.read_text(encoding="utf-8").splitlines():
        cells = split_table_row(line)
        if len(cells) != 8:
            continue
        if cells[0] in {"Date", "---"} or cells[1] in {"ID", "---"}:
            continue
        if not re.fullmatch(r"[DE]\d+[A-Za-z0-9-]*(?:/[A-Za-z]\d+[A-Za-z0-9-]*)*", cells[1]):
            continue
        rows.append(
            {
                "date": cells[0],
                "id": cells[1],
                "name": cells[2],
                "baseline": cells[3],
                "candidate": cells[4],
                "delta": cells[5],
                "decision": cells[6],
                "artifacts": cells[7],
            }
        )
    return rows


def short_decision(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def main() -> int:
    rows = parse_results_rows()
    now = dt.datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = [
        "# Experiment Digest",
        "",
        f"Updated: {now}.",
        "",
        "This is the compact historical base for performance work. `RESULTS_LOG.md` remains the detailed ledger; this file groups the evidence so agents can pick the next route without rereading every E/D note.",
        "",
        "## Current Truth",
        "",
        "| Area | Current conclusion |",
        "| --- | --- |",
        "| Active dense 130k target | `Qwen3.6-27B-Q3_K_S`, `ctx=131072,b=512,q4_0/q4_0,spec=none`, cold/no-reuse/no-prime, `real-context-chars=24576`, `max_tokens=16`, thinking on. Vulkan D012 `ub=256` remains the active speed baseline at `2.0013 TPS`; D035 recovers the fresh default slow pocket to `1.8736 TPS` but is stability hardening, not a new speed baseline. The new speed target is `2.4 TPS`. ROCm `ub=128` baseline is `1.5200 TPS` and paused after D013-D027. |",
        "| 130k residency constraint | RX 9070 XT 16 GB is not expected to keep dense 27B + 130k KV/context/working set fully VRAM-resident. RAM-spill/residency/PCIe/startup diagnostics are part of the metric, not noise. |",
        "| Archived dense Vulkan 12k | E257 remains the short-context reference: `ctx=12288,b=7168,ub=1024,q4_0/q4_0,spec=none`, cold/no-reuse/no-prime, thinking on, `7.0319 TPS` r3. |",
        "| Vulkan post-E265 | D005/D012 show the 130k Vulkan path: split-K plus q3quad/GLU stack clears 2 TPS with documented opt-in env and `--no-mmap`; D034 identifies the current 130k slow pocket as residency-driven; D035 promotes guarded route defaults plus a narrow host-KV guard to recover default stability. D028 retargets Vulkan to `2.4 TPS`; D029 rejects activation-only/naive-streaming whole-FFN, D030 rejects nearby old all-Q3 storage/helper/Q8/tile families, D031 rejects compact Q3S layout-body work, D032 shows FA-only cannot carry, and D033 rejects q3-octa/LOAD_VEC_A=8. Next speed work needs a true Q3_K compute body/compressed-dot route; FA can stack only after Q3 reaches roughly `1.18-1.20x` local evidence. |",
        "| ROCm dense cold target | D002/D013-D027 recenter ROCm at `1.5200 TPS` and reject nearby ubatch, storage escape, cublas/dequant, no-mmap, src1 quant, GLU-only, current-MMQ staging/load/barrier, Q3Flash active-shape, wider-N scalar Q3Flash, dual-Y, vbuffer single-chunk, multi-row WMMA Q3Flash, upstream-stock rollback, streaming dequant+rocBLAS chunking, pair-only FFN SwiGLU WMMA, naive whole-FFN streaming, expanded persistent Q3_K layout, and compact signed-nibble unpack-only layout routes. Future work needs larger FFN/Q3_K dataflow, not another selector sweep. |",
        "| Practical >10 TPS | E255 A3B profile reaches `22.1407 TPS` r3, but it is a model/profile result, not dense 27B-Q3 acceleration. |",
        "| Benchmark history | New runs should update `BENCH_RUNS.csv`, `BENCH_RECENT.md`, and `BENCH_LANES.md` through `scripts/agent_workload_bench.py`. |",
        "",
        "## Route-Family Conclusions",
        "",
        "| Family | Covered IDs | Key keep | Key rejection / lesson | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family, ids, keep, reject, next_action in FAMILY_SUMMARIES:
        lines.append(f"| {family} | {ids} | {keep} | {reject} | {next_action} |")

    lines += [
        "",
        "## Compact Experiment Ledger",
        "",
        f"Rows parsed from `docs/research/RESULTS_LOG.md`: {len(rows)}.",
        "",
        "| Date | ID | Short name | Delta | Decision | Artifacts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['date']} | {row['id']} | {row['name']} | {short_decision(row['delta'], 90)} | "
            f"{short_decision(row['decision'])} | {short_decision(row['artifacts'], 160)} |"
        )

    DIGEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {DIGEST} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())