# E238 ROCm FFN Tall GEMM Gate

## Metadata

- Experiment ID: E238
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `0fef64355`
- Target lane: H42 ROCm large-Q3_K prefill route, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse

## Hypothesis

- Statement: a graph-level FFN `gate/up` route might replace two same-shape `17408x5120@2048` rocBLAS GEMMs with one taller `34816x5120@2048` GEMM sharing the same activation matrix.
- Mechanism: a single taller GEMM could choose a better rocBLAS topology or reduce dispatch overhead while producing vertically concatenated `gate` and `up` outputs for the later SWIGLU stage.
- Why now: E218 rejected the cheap shared-`src1` staging subcase, and E232 rejected pointer-batched rocBLAS pair GEMM. The remaining library-level fusion question is whether a single tall GEMM improves the GEMM body itself.

## Math / Theory

- Assumptions:
  - E217 top bucket `(17408,5120,2048)` totals `1415.111 ms`, with `1147.612 ms` in GEMM and `82.721 ms` in `src1` staging.
  - A tall route only matters if the GEMM portion itself improves; launch/staging savings alone are below the +20% cold target.
  - Runtime integration would require graph fusion and possibly output layout handling, so the standalone GEMM point signal must be clearly positive before code work.
- Expected speedup corridor:
  - Strong candidate: one tall GEMM beats two separate GEMMs by at least `5%` on the main `n=2048` bucket and does not regress the `n=1345` tail.
  - Weak/no candidate: tall GEMM ties or regresses, which means graph fusion cannot justify its complexity.
- Failure conditions:
  - rocBLAS already saturates the individual GEMMs;
  - taller `m` selects a worse solution or hits workspace/cache limits;
  - any local win is too small to pay for graph/runtime changes.

## Implementation Plan

1. Minimal code surface to change:
   - add standalone diagnostic utility `scripts/research/rocm_rocblas_tall_pair_scout.cpp`.
2. Guard rails:
   - compare only rocBLAS point timing: two separate GEMMs vs one `m*2` GEMM with contiguous synthetic A/D buffers;
   - if point timing is not positive, do not build graph fusion;
   - if point timing is positive, try only an env-gated runtime point prototype before any wall A/B.
3. Rollback path:
   - revert the runtime prototype if its point trace regresses; keep the standalone scout only as diagnostic evidence.

## Benchmark Plan

- Compile:
  - `hipcc -std=c++17 scripts/research/rocm_rocblas_tall_pair_scout.cpp ... -lrocblas`
- Candidate shapes:
  - main: `(m=17408,n=2048,k=5120)`
  - tail: `(m=17408,n=1345,k=5120)`
- Number of runs:
  - one sequential scout per shape, `warmup=8`, `iters=20`.
- Artifacts path:
  - `build_logs/agent-workload/e238-rocblas-tall-pair-*.csv`.

## Metrics

- separate pair average ms
- tall GEMM average ms
- relative to separate pair
- projected runtime decision; no wall speed claim unless point gate passes

## Result

- Outcome: rejected and runtime prototype reverted.
- Delta:
  - Standalone scout looked promising after warmup:
    - main `m=17408,n=2048,k=5120`: separate pair `5.6638 ms`, tall `4.5001 ms`, relative `0.7945x` in r3.
    - tail `m=17408,n=1345,k=5120`: separate pair `4.1002 ms`, tall `3.4625 ms`, relative `0.8445x` in r2.
  - Runtime point gate failed badly:
    - route activated `230` times before the hard timeout.
    - robust `ncols=2048` tall route: `187` rows, `sum_ms=4304.43`, `gemm_ms=2158.94`, `glu_ms=1939.47`.
    - robust `ncols=1382` tail: `41` rows, `sum_ms=657.67`, `gemm_ms=321.65`, `glu_ms=292.71`.
    - E217 current route comparator for `(17408,5120,2048)` is `189` gate/up pairs with bucket total `1415.111 ms` and GEMM `1147.612 ms`.
    - Therefore the in-runtime tall GEMM alone is already slower than the current two-GEMM route, before counting the fused output/GLU work.
- Confidence: high for rejection. The point regression is large enough that no cold wall A/B is justified.
- Recommendation: do not implement graph-level FFN tall GEMM on the current hipBLAS/cublas runtime path. Standalone rocBLAS body wins are not sufficient unless the same solution/topology is available through the actual llama.cpp runtime API and the route point timing moves in-runtime.

## Notes

- This is intentionally distinct from E232: it does not use `rocblas_gemm_batched_ex`; it tests whether one normal GEMM with doubled `m` is a better body for the gate/up pair.
- Why the hypothesis looked plausible: direct rocBLAS synthetic timing can choose a better tall shape after warmup, and solution-index scouting showed several fast tall variants for `m=34816,n=2048,k=5120`.
- Why it did not transfer: the runtime `cublasGemmEx` path used by llama.cpp did not reproduce the synthetic tall speed; synchronized route timing showed about `11.5 ms` GEMM per main FFN pair, while the existing split route is about `6.1 ms` GEMM per pair. The prototype also had to materialize a `2*out` float buffer and run SWIGLU over it.
- Workflow correction: library-shape scouts can only open a candidate. Every such candidate still needs an in-runtime point gate before wall A/B, and solution-index evidence is not actionable until the runtime path can select the same solution contract.
