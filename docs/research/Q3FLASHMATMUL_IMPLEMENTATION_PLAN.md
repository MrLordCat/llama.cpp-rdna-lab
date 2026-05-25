# Q3FlashMatmul Implementation Plan

## Goal

Build a ROCm/RDNA4 Q3_K route that works like a FlashAttention-style compressed
matmul: keep Q3_K weights compressed, dequantize inside the matmul tile, and
avoid large fp16 staging buffers when this beats the current `Q3_K -> fp16 ->
rocBLAS` route.

This is an H42 route-body experiment, not a selector tweak. The implementation
must be promoted only if point timing and wall A/B are positive.

## Current Baseline

- Active lane: Qwen3.6-27B-Q3_K_S, ROCm, `ctx=12288`, `batch=6144`,
  `ubatch=2048`, q4/q4 KV, FlashAttention on, `spec=none`, no reuse, thinking on.
- E245 current cold baseline r1: `7.58 TPS`.
- E228 robust Q3_K split: `3783.195 ms`, with `508.206 ms` src0 conversion and
  `2906.403 ms` GEMM.
- Main target shapes:
  - `17408 x 5120 @ n=2048`
  - `5120 x 17408 @ n=2048`
  - `10240 x 5120 @ n=2048`
  - `6144 x 5120 @ n=2048`

## Promotion Gates

1. Standalone correctness gate:
   - compare direct Q3_K output against `dequant_q3_K_to_fp16 + rocBLAS`;
   - start with small shapes, then one hot-shape point.
2. Standalone point gate:
   - direct Q3FlashMatmul must beat baseline point timing by at least `10%` on a
     hot shape before runtime integration.
3. Runtime opt-in gate:
   - route must be guarded by an env knob and limited to ROCm/RDNA4/Q3_K hot
     shapes at first.
4. Wall A/B gate:
   - same-lane control/candidate, no reuse, no prime, thinking on;
   - promote only if no-trace wall improves, not just synchronized point timing.

## P0 Scope

- Add standalone `scripts/research/rocm_q3flashmatmul_scout.cpp`.
- Implement a simple tiled direct Q3_K padded-storage matmul:
  - Q3_K padded blocks as A (`m x k`);
  - f16 B (`k x n`);
  - f32 D (`m x n`);
  - tile M/N by `16 x 16`, K by `256`.
- Baseline inside the same scout:
  - dequantize Q3_K padded blocks to f16;
  - call rocBLAS GEMM with the same contract used by llama.cpp.

P0 is intentionally conservative. If it regresses, do not wire it into runtime;
use the measured failure mode to design P1.

## E245 Status

Standalone P0/P1/P2 was implemented in
`scripts/research/rocm_q3flashmatmul_scout.cpp` and rejected for runtime
promotion:

- P0 scalar tile was correct but slower than `dequant -> rocBLAS`.
- P1 RDNA4 WMMA tile initially exposed a B-fragment layout bug; after fixing the
  transposed B fragment, correctness matched the baseline.
- P2 shared-A reuse improved the WMMA route substantially, but still missed the
  point gate by a wide margin: `17408x5120@n128` was `0.307x` of baseline and
  `17408x5120@n2048` was `0.063x` of baseline.

Decision: keep the standalone scout as a correctness and topology harness, but
do not integrate Q3FlashMatmul into llama.cpp runtime from these bodies. A future
P3 must solve broad-N A-dequant reuse and matrix-core occupancy before another
runtime branch is justified.

## E246 Streaming Pipeline Pivot

The direct Q3FlashMatmul body is not the winning route, but the same scout now
has a positive P4 path: chunked Q3_K dequantization feeding rocBLAS GEMM through
bounded fp16 staging. This keeps the library GEMM body instead of replacing it.

Measured point gates:

- `17408x5120@n2048`, chunk `6144`: `5.1528 -> 4.3862 ms` (`1.1748x`).
- `5120x17408@n2048`, chunk `4096`: `5.3959 -> 4.6166 ms` (`1.1688x`).
- `10240x5120@n2048`, chunk `6144`: `3.2504 -> 2.5051 ms` (`1.2975x`).
- `6144x5120@n2048`, chunk `4096`: `1.9787 -> 1.6113 ms` (`1.2281x`).

Decision: continue with P4 as the runtime candidate, not direct P0/P1/P2/P3.
Runtime integration must use persistent auxiliary rocBLAS handles/streams and an
env gate; per-call handle creation or unsafe graph-capture stream work would
invalidate the scout result.

Runtime follow-up result: the env-gated server prototype activated, but did not
convert into a wall win. The safe synchronous route regressed the active lane
after correct stream dependencies were added, and the async route hit real-server
timeouts even after private stream/event-lifetime fixes. Keep P4 as a standalone
topology clue only until a new graph-safe async design exists.

## E248 Adjacent src1 F16 Reuse

The productive runtime side-route from this branch is not Q3FlashMatmul itself,
but removing duplicate activation staging around adjacent cuBLAS calls. E248 adds
an opt-in adjacent-only fp16 cache for `src1`:

- env: `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE=1`;
- threshold: `GGML_EXPERIMENTAL_CUBLAS_SRC1_F16_REUSE_MIN_NCOLS`;
- trace smoke confirmed `429` adjacent hits on the prompt (`ffn_up`, `z`,
  `Vcur`, `Kcur`);
- final 3-run active-lane A/B: `7.8981 -> 7.9608` aggregate TPS (`+0.79%`),
  median `8.02 -> 8.17`, prompt mean `5988.23 -> 5948.21 ms`.

Decision: keep E248 as opt-in positive micro-route, not default. It is small but
runtime-positive, and it avoids broad fp16 weight residency.

## Risks

- rocBLAS may remain faster because the custom kernel is scalar FMA rather than
  a matrix-core route.
- Q3 unpacking can cause register pressure cliffs.
- Larger tiles can trigger LDS cliffs.
- Full-prompt outputs can consume hundreds of MiB; do not keep full FFN
  intermediates resident as a shortcut.

## Rollback

- The P0 scout is standalone and not part of normal builds.
- Runtime files must remain unchanged until P0 passes point gates.
