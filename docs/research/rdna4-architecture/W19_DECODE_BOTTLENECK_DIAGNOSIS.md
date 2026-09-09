# W19: Decode bottleneck diagnosis (SPM profile on gfx1201)

Date: 2026-09-08

Scope: first hardware-counter (SPM) diagnosis of the decode hot path on
Linux ROCm 10 / gfx1201, separate from kernel-trace only. Answers where the
decode is actually limited: compute/issue vs memory bandwidth vs launch gap.
This run confirms the direction guidance from W16/W17: "next MMVQ candidate
must change memory system or geometry, not add instructions" - and adds the
missing part: inside the MMVQ kernel the SQ is busy (compute-bound), so the
correct lever is fewer instructions, i.e. the W18 exact INT4 dot8 plan.

## Method

- Tool: `rocprofv3` (`/home/chris/rocm/bin/rocprofv3`) with
  `--pmc` + `--kernel-trace --stats --summary -D -u msec`.
- Workload: Qwen3.8-27B-Q4_K_M, short run: eval 10 prompt tokens,
  200 decode tokens, ctx 4096, b/ub 256/128, f8_e4m3 K/V, flash on,
  spec none, ROCm1,ROCm0 -sm layer -ts 1,1, no warmup, seed 42.
  Same binary/contract as the W13/W16 L2 validation except context/batch,
  which is fine for decode-only attribution; L2 numbers from earlier runs.
- Counter group 1 (worked): `GPUBusy GPU_UTIL L2CacheHit MemUnitBusy
  OccupancyPercent SQ_BUSY_CYCLES` (6 counters).
- Counter group 2 (did NOT work): `WAVE_ISSUE_WAIT WAVE_DEP_WAIT VALUInsts
  ValuPipeIssueUtil SQ_INSTS_VALU SQ_INSTS_LDS` -> rocprofv3 collected
  0 PMC rows (group not available on this agent/config; no crash, just empty).
  20+counter requests cause error 38 "Request exceeds capabilities".
- Output: `/tmp/vkprof4/d4_results.db` (931 MB) and
  `/tmp/vkprof5/d5_results.db` (1.4 GB, empty PMC).
  Summary: `/tmp/vkprof4/d4_/tmp/vkprof4/summary.txt.txt`, parsed by
  `scripts/research/rocprof_summary_top.py`.

## Results (kernel time = 7144.86 ms over 200 decode tokens)

| kernel | % of kernel time | ms/token | calls/token |
| --- | --- | --- | --- |
| mul_mat_vec_q<(ggml_type)12, 1, true, true...> (fused q4_K) | 45.89% | 16.39 | 110 |
| mul_mat_vec_q<(ggml_type)12, 1, false, true...> (q4_K) | 14.61% | 5.22 | 112 |
| mul_mat_vec_q<(ggml_type)14, 1, false, true...> | 10.30% | 3.68 | 33 |
| mul_mat_vec_q<(ggml_type)14, 1, true, false...> | 10.85% | 3.87 | 32 |
| mul_mat_vec_q<(ggml_type)13, 1, false...> | 5.09% | 1.82 | 48 |
| **MMVQ family total** | **86.74** | **30.98** | **~335** |
| k_get_rows_float | 1.73% | 0.62 | 98 |
| quantize_q8_1 | 1.52% | 0.54 | 337 |
| rms_norm / mul_mat_vec_f | 2.65% | 0.84 | 226 |
| gated_delta_net_cuda | 0.88% | 0.31 | 48 |
| flash_attn_ext_f16 | 0.57% | 0.20 | 16 |
| remaining (concat/ssm/rope/unary) | ~5% | ~1.8 | ~600 |

- Fused q4_K (type 12, ncols=1, `small_k=true`) = 22 091 calls / 110 per token,
  avg 0.1484 ms, max 1.037 ms. Non-fused q4_K = 22 487 calls.
- Kernel time per token = 35.72 ms; observed decode is ~22.8 tps
  (43.9 ms/token) -> ~8.2 ms/token (≈19%) is outside kernel time
  (launch/sync/graph).

## Counter attribution (fused q4_K)

Per dispatched fused q4_K kernel:

| counter | value | interpretation |
| --- | --- | --- |
| GPUBusy | 100.0 | kernel occupies GPU the whole time (expected) |
| SQ_BUSY_CYCLES | ~470 000 per SIMD-set | see below |
| L2CacheHit / MemUnitBusy / OccupancyPercent | 0 (not collected) | inst. not valid in this group |

- Duration 0.1484 ms at ~2.4 GHz = ~356k cycles; at boost ~3.2-3.4 GHz
  = ~475-505k cycles.
- `SQ_BUSY_CYCLES` ≈ 470k ≈ 93-100% of total cycles at boost: the shader
  queue is busy issuing/executing almost every cycle.
- Conclusion: the Q4_K MMVQ kernel is **compute/issue-bound**, not memory
  latency-bandwidth bound. This explains:
  - 59% of peak BW (the kernel does not leave enough memory pipeline slack),
  - why software prefetch (W16, -0.64%) and occupancy 50->100% (C1b, +1.1%)
    were exhausted - they address memory/occupancy, but the limit is issue
    throughput,
  - the remaining gap is instruction count (see W18).

## Why the instruction count is the lever (code-level)

`vec_dot_q4_K_q8_1_impl_vmmq` (vecdotq.cuh:635) per 32-elem group:
- `dot1` (weights x activations): 2x2 DP4A (8-byte vectors) = 4 instructions;
- `dot2` (sum of activations, for the exact dmin correction): 4 more DP4A;
- plus FP scale math (d8 * (dot1*sc) - ...).
So ~50% of dot instructions are helper `dot2`, and `dot1` uses 4-MAC DP4A when
8-MAC `v_dot8_u32_u4` exists. See W18 for the exact nibble decomposition that
removes this load bit-exactly.

## Launch gap (structural)

- 110+ MMVQ launches/token (each ~0.15 ms avg) + ~337 quantize_q8_1 +
  ~226 rms_norm/mul_mat_vec_f + 48 gated_delta_net: ~700 device launches/token.
- Any per-launch overhead (host submit, graph node, sync) multiplies by this
  count; ~19% of wall time is outside kernels.
- Next structural step after W18: reduce launches (CUDA graph reuse across
  steps, merge/reorder per-layer nodes). Not a compute change; separate W-note.

## Interpretation (updates W19) - SQ busy is NOT proof of compute/issue-bound

- 2026-09-08 W18 follow-up: `dot2` (dmin activation sum) hoisted out of the
  Q4_K row loop removes ~50% of DP4A in the fused q4_K dot-loop
  (bit-identical, verified). L1 A-B-A: control 24.1908 vs candidate 24.2095
  decode = **+0.08% (noise)**; prefill neutral. So DP4A/ALU work is NOT the
  limiter.
- Revised interpretation: `SQ_BUSY_CYCLES ≈ 100%` also includes the cycles
  spent issuing **memory** requests (v_dot* loads and vector loads), so it is
  consistent with a memory-system/issue-limited kernel, not an ALU-bound one.
- The actual limiter candidates: weight-stream arrival (K-block loads,
  40-45% of peak BW per W13), per-thread memory parallelism/coalescing, and
  L2 behavior. W16's lesson stands: "must change memory system or geometry,
  not add instructions".
- Next experiment: collect GL2C_EA_RDREQ_{32,64,128,256}B + GL2C_HIT/MISS
  per kernel to see request size / cache-line efficiency (previous 6-counter
  group had L2CacheHit=0; try a minimal 3-counter group without extra pmc).

## Limitations / follow-ups

- Only one burst of 200 tokens, one GPU pair; no cold/steady split. Absolute
  percentages stable across kernels but per-token numbers are one sample.
- `WAVE_DEP_WAIT`/`VALUInsts` etc. unavailable in this rocprofv3 version on
  gfx1201; if needed, use `RADV_DEBUG=shaderstats` path (Vulkan) or
  `rocprofv3-avail info <counter>` to find a supported group.
- No GPU code changed; all analysis from counters + summary.
