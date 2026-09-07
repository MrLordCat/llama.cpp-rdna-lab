# W16: MMVQ K-stream prefetch (MLP candidate)

Date: 2026-09-07

Status: **REJECTED** (measured 2026-09-07). The 3-iteration K loop is not
MLP-limited at 100% occupancy: `__builtin_prefetch` is expressible (LLVM IR
`llvm.prefetch.p1` survives; smoke output bit-identical), but the L2 49K
A-B-A shows a small regression.

Scope: next candidate after W13 C1b. The C1b resource audit showed the
fused/non-fused MMVQ paths are at 100% occupancy and the shared/reduce
machinery is exhausted, yet the decode stream only reaches `~250-280 GB/s`
of `644 GB/s` peak = `~40-45%` (W13 traffic model). The remaining gap was
hypothesized to be **memory-level parallelism (latency), not instruction
width**: the weight loads are already vectorized (`dwordx4` x2 = 32 B per
K-iteration per thread), but there is **no software prefetch** in the
K-iteration loop.

## Evidence

- `vexdot/vecdotq.cuh: vec_dot_q4_K_q8_1`: per K-block a thread reads
  `v[0], v[1]` (2 x `dwordx4`, 32 B of q4_K weights) + 8 x q8_1 ints; the
  loop is unrolled; no explicit prefetch in either the dot or the caller.
- `mmvq.cu: mul_mat_vec_q` K-loop:
  `for (kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter)`
  - `blocks_per_iter = gridDim.x` portion; production lane: ~3 iterations
    per thread. Each iteration issues its weight loads, then computes
    dp4a chains - the loads for iteration `i+1` are not issued until the
    current iteration has finished consuming them.
- Full codebase search: no `prefetch`/`__builtin_prefetch` in the CUDA/ROCm
  sources (only a bias "prefetch" comment).
- At 100% occupancy (8 CTAs x 256 threads = 2048 threads/SM) the SM has
  plenty of warps; a per-iteration latency of ~200-400 ns (L2) would still
  leave VGPR-limited stalls if the loop body is short and dependent. This is
  the classic "3 iterations, no prefetch" pattern that a single software
  prefetch per iteration can hide.

## Prototype (W16)

- `mul_mat_vec_q` gained an env-gated `GGML_MMVQ_RDNA4_QWEN_PREFETCH=1`
  (host static gate passed as a kernel arg; default off). In the K-loop, for
  `kbx_next = kbx + blocks_per_iter < blocks_per_row_x`, it prefetched:
  - qwen-hot x blocks (q3_K/q4_K/q6_K) via `sizeof(block_qN_K)` byte offset
    for each `rows_per_cuda_block` row;
  - the q8_1 y blocks for each `ncols_dst` column.
- Build: clean with ROCm 10 clang for gfx1201. `__builtin_prefetch` emits
  `llvm.prefetch.p1` in IR, so it is not silently dropped (unlike H80's NT
  inline-asm case).
- Correctness: deterministic smoke at ctx=4096 produced byte-identical
  output with and without the gate (same seed/prompt; f8 KV).

## Measurement (2026-09-07, bench2 L2 A-B-A)

Same contract as the C1b re-check: ctx 49152, b/ub 512/512, f8_e4m3 KV,
flash on, spec none, no-warmup, seed 42, ROCm1,ROCm0 -sm layer -ts 1,1,
30609 prompt / 256 out; library pinned per variant, env set only for the
candidate.

| round | variant | decode tps | prefill tps | aggregate |
| --- | --- | --- | --- | --- |
| A | C1b (control) | 22.7642 | 1763.5 | 8.9504 |
| B | C1b + prefetch | 22.6221 | 1748.5 | 8.8820 |
| A' | C1b (control) | 22.7709 | 1751.6 | 8.9144 |

- Decode control interpolation = (22.7642 + 22.7709) / 2 = 22.7676;
  prefetch = 22.6221 -> **-0.64%** (candidate below both controls).
- Prefill also slightly lower (1748.5 vs 1763.5/1751.6); aggregate -0.5%.
- The 3-iteration loop at 2048 threads/SM already keeps enough loads in
  flight; the extra prefetch instructions are dead work. The 40-45% of
  "peak BW" model is not a per-thread MLP ceiling (W13 traffic model counts
  the theoretical DRAM ceiling, which this kernel does not approach at
  ncols_dst=1 for other reasons - weight read is the dominant but not the
  only factor).

## Verdict

- REJECTED. Prototype reverted (mmvq.cu byte-identical to HEAD; clean
  rebuild verified). Gate removed - there is no `GGML_MMVQ_RDNA4_QWEN_PREFETCH`
  in the tree.
- Lesson: like H80, a "the toolchain can express it" result is not a
  "performance win"; only the A-B-A decides. Next MMVQ candidate should
  change the *memory system* (e.g. L2 residency / weight preload channel) or
  the *geometry*, not add instructions to an already-latency-covered loop.
- Re-open only if a future trace (`GGML_TRACE_MMVQ_TIMING_SYNC=1`) shows
  the K-loop is really memory-stalled; the current evidence says it is not.

## Artifacts

- `build_logs/bench/q38-w16-prefetch/` (base-a, cand-b, base-a2, full logs);
  smoke logs `/tmp/smoke_base.log`, `/tmp/smoke_pf.log`.
- W13 "C1b fresh Linux re-check" remains valid; this W16 did not touch it.

