# E249 ROCm hipBLASLt Grouped GEMM Gate

## Metadata

- Experiment ID: E249
- Date: 2026-05-25
- Owner: Copilot
- Target lane: Qwen3.6-27B-Q3_K_S cold-first ROCm lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, `--spec-type none`, no reuse, no prime
- Goal: find a high-ceiling route toward the current `10 TPS` cold target from the post-E248 `~7.9 TPS` baseline.

## Hypothesis

- Statement: ROCm 7.1 `hipBLASLt` grouped GEMM may execute sibling FFN `gate/up` GEMMs with better scheduling than the current two-call rocBLAS/hipBLAS path, without using the already-rejected `rocblas_gemm_batched_ex` route.
- Mechanism: grouped GEMM can expose the two same-shape GEMMs to one library scheduler and choose a grouped kernel/solution that amortizes launch and scheduling overhead while preserving separate output buffers.
- Why this is distinct from rejected routes:
  - E232 tested `rocblas_gemm_batched_ex` and found it catastrophically slow.
  - E238 tested a single tall rocBLAS GEMM; standalone looked positive, but the in-runtime tall path regressed badly.
  - This gate tests `hipBLASLt` grouped GEMM, a different ROCm 7.1 API and solution family.

## Theory / Gate

- Current target requires roughly `10 / 7.96 = 1.26x` aggregate speedup after E248.
- A gate/up-only grouped route is only worth runtime work if the local hot pair speedup is large, because graph integration would add library linkage, algorithm caching, workspace ownership, and graph-safety constraints.
- Strong pass: grouped pair is at least `1.15x` faster than two separate rocBLAS GEMMs on the main `m=17408,n=2048,k=5120` shape and does not regress the tail.
- Weak pass: `1.05x-1.15x`; keep only as a stack clue unless runtime integration looks very cheap.
- Reject: grouped ties/regresses or fails to find a supported algorithm.

## Implementation Plan

1. Add standalone scout `scripts/research/rocm_hipblaslt_grouped_gemm_scout.cpp`.
2. Compile it manually with ROCm 7.1 `hipcc` and link `hipblaslt` + `rocblas`.
3. Compare:
   - current-like `rocblas_gemm_ex` separate pair;
   - `hipBLASLt` grouped pair for the same A/B/D layout.
4. Only if point timing passes, consider an env-gated runtime prototype.

## Benchmark Plan

- Main shape: `m=17408,n=2048,k=5120`.
- Tail shape: `m=17408,n=1382,k=5120` or the nearest current trace tail.
- Optional reverse/down shape: `m=5120,n=2048,k=17408`.
- Iteration default: `warmup=8`, `iters=20`.
- Artifact path: `build_logs/agent-workload/e249-hipblaslt-grouped-*.csv`.

## Result

- Outcome: rejected at API/algorithm gate.
- Build notes:
  - `hipcc` on Windows loses normal quoted `Program Files` include/lib paths; short path `C:/PROGRA~1/AMD/ROCm/7.1` plus `-x none` before import libraries was required.
  - The scout compiles and links with `libhipblaslt.dll.a` and `rocblas.lib`.
- Runtime gate:
  - exact current contract `f16 x f16 -> f32`, compute `f32`: `hipBLASLt grouped returned no algorithms`;
  - `f16 x f16 -> f32`, compute `fast16`: no algorithms;
  - `f16 x f16 -> f16`, compute `fast16`: no algorithms.
- The fallback from `algoGetHeuristic()` to `getAllAlgos(... HIPBLASLT_GROUPED_GEMM ...)` also returned no usable algorithms, so this is not only a heuristic-list issue.

## Decision

- Reject `hipBLASLt` grouped GEMM as a current ROCm 7.1 Windows route for the active Q3_K FFN pair. Do not build runtime linkage or graph integration on top of this API unless a future ROCm release exposes grouped algorithms for these shapes/types.
- Keep the standalone scout as a diagnostic/linkage probe.
