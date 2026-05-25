# E245 ROCm Q3FlashMatmul P0/P2

## Metadata

- Experiment ID: E245
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 / Q3FlashMatmul
- Target lane: ROCm cold-first Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`,
  `ubatch=2048`, q4/q4 KV, FlashAttention on, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a direct tiled Q3_K compressed matmul can eventually beat the
  current `Q3_K -> fp16 staging -> rocBLAS` route on hot prompt-heavy FFN shapes.
- Mechanism: dequantize Q3_K blocks inside the matmul tile and reuse the
  dequantized A tile across an N tile, avoiding large fp16 staging traffic.
- Why now: E228 recentered the bottleneck on Q3_K GEMM-side work, and E244 shows
  graph-level chunk coalescing alone is not enough for the `+20%` cold target.

## Math / Theory

- Assumptions:
  - current same-snapshot E245 cold baseline r1 is `7.58 TPS`;
  - E228 robust Q3_K split is `3783.195 ms`, with `508.206 ms` src0 conversion
    and `2906.403 ms` GEMM;
  - P0 must first prove correctness and point-level viability before runtime use.
- Expected speedup corridor:
  - P0 scalar-tile implementation may regress; that is acceptable only as a
    diagnostic if it identifies the next kernel body requirement;
  - runtime promotion needs at least `10%` point win on a hot shape.
- Failure conditions:
  - correctness mismatch versus dequant+rocBLAS;
  - point timing slower than baseline;
  - memory allocation or residency errors on hot-shape scout.

## Implementation Plan

1. Minimal code surface to change:
   - add standalone `scripts/research/rocm_q3flashmatmul_scout.cpp`;
   - no runtime route changes in P0.
2. Guard rails:
   - compare against in-process dequant+rocBLAS baseline;
   - small correctness shape before hot point;
   - no wall claim from standalone scout alone.
3. Rollback path:
   - remove standalone scout if it proves useless; no runtime rollback needed.

## Implemented Variants

- Baseline: Q3_K padded blocks dequantized to f16 A, then rocBLAS `A^T * B`
  with f32 output.
- P0 scalar tile: direct `16x16x256` tiled Q3_K x f16 matmul with f32
  accumulation.
- P1 WMMA tile: direct RDNA4 wave32
  `wmma_f32_16x16x16_f16` body with Q3_K dequantized directly into the A
  fragment.
- P2 WMMA reuse tile: shared-A reuse across `8` N-waves, so one `16x16` Q3
  dequant tile feeds `N=128` worth of WMMA fragments.

Correctness note: first P1 attempt had a wrong B-fragment mapping; the small
gate caught it (`wmma_max_abs ~= 3.0`). Transposing B fragment load fixed the
layout, and P1/P2 then matched the rocBLAS baseline on small shapes.

## Benchmark Plan

- Baseline command:
  - current cold wall baseline: `e245-rocm12k-q3flash-baseline-r1`.
  - standalone scout baseline: dequant Q3_K padded to f16, then rocBLAS GEMM.
- Candidate command:
  - standalone direct Q3FlashMatmul P0 kernel in the same scout.
- Number of runs:
  - wall baseline: `runs=1`;
  - scout: warmup/iters per shape.
- Artifacts path:
  - `build_logs/agent-workload/e245-*`.

## Metrics

- aggregate completion TPS (wall baseline)
- baseline point ms
- q3flash point ms
- speedup ratio
- max abs / relative output difference
- memory allocation success/failure

## Result

- Outcome: standalone scout built and validated; runtime promotion rejected.
- Wall baseline: `e245-rocm12k-q3flash-baseline-r1` measured `7.58 TPS` on the
  active cold-first lane.
- Correctness:
  - scalar P0 small gate passed (`max_abs ~= 0`);
  - corrected P1/P2 WMMA small gate passed (`wmma_max_abs ~= 0`,
    `wmma_reuse_max_abs ~= 0`).
- Point timing summary:

| Shape | Baseline ms | P0 scalar ms | P0 speedup | P1 WMMA ms | P1 speedup | P2 reuse ms | P2 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `128x512@n128` | `0.0025` | `0.0467` | `0.0535x` | `0.1391` | `0.0180x` | `0.1603` | `0.0156x` |
| `17408x5120@n128` | `2.3884` | `10.9891` | `0.2173x` | `26.9049` | `0.0888x` | `7.7691` | `0.3074x` |
| `5120x17408@n128` | `2.1506` | `12.3964` | `0.1735x` | `24.0208` | `0.0895x` | `9.7115` | `0.2214x` |
| `17408x5120@n2048` | `5.1259` | `141.7664` | `0.0362x` | `217.1527` | `0.0236x` | `80.7312` | `0.0635x` |

- Delta: all direct Q3FlashMatmul variants are slower than the in-process
  `dequant -> rocBLAS` baseline, far below the `>=1.10x` point gate.
- Confidence: high for rejection of the current body; correctness passed and the
  hot point is decisively negative.
- Recommendation: do not wire Q3FlashMatmul into runtime. Keep the scout as a
  layout/correctness harness. Continue H42 only with a fundamentally different
  compressed-GEMM topology that avoids repeated A-dequant per N tile and does
  not surrender rocBLAS-level matrix-core occupancy.

## Notes

- P0 is expected to be a truth-finding kernel, not necessarily the final route.
- If P0 regresses, the likely next step is a matrix-core or larger-body design,
  not runtime integration of the scalar P0 kernel.
- P2 proved A-tile reuse is directionally useful (`P1 -> P2` improved hot
  `17408x5120@n128` from `26.90 ms` to `7.77 ms`), but this is still only
  `0.31x` of rocBLAS at `n128` and collapses to `0.064x` at `n2048` because the
  Q3 tile is still repeated per `N=128` group.
- The current negative result is not a correctness failure after the B-fragment
  fix; it is a route-body/topology failure.
- Follow-up E246 pivot: keep rocBLAS and stream Q3_K dequantized row chunks into
  it instead of replacing GEMM. That P4 scout is positive on multiple hot shapes.

## Artifacts

- `scripts/research/rocm_q3flashmatmul_scout.cpp`
- `build_logs/agent-workload/e245-rocm12k-q3flash-baseline-r1.diagnostics.md`
- `build_logs/agent-workload/e245-q3flash-small-wmma-reuse-r1.csv`
- `build_logs/agent-workload/e245-q3flash-hot17408-n128-wmma-reuse-r1.csv`
- `build_logs/agent-workload/e245-q3flash-hot5120-n128-wmma-reuse-r1.csv`
- `build_logs/agent-workload/e245-q3flash-hot17408-n2048-wmma-reuse-r1.csv`
- `docs/research/experiments/E246_rocm_q3k_streaming_cublas_pipeline.md`
