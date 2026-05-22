# E167 ROCm Q3_K Small-K Row-Warp Probe

## Metadata

- Experiment ID: E167
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E166 rejection
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: RDNA4 Q3_K small-k MMVQ may benefit from assigning one warp to one output row instead of splitting K across both warps for every row.
- Mechanism: The current `nwarps=2`, `rows_per_block=2` route has both warps compute partial sums for both rows, then performs a shared-memory cross-warp reduction. A row-warp route keeps the same two rows per block but lets `threadIdx.y` own one row, avoiding cross-warp shared reduction and per-row shared traffic.
- Risk: Each warp then walks the full K range instead of half, so the K loop has fewer lanes per row and may lose latency hiding. E164 already showed that improving occupancy/grid geometry can still regress if the shape loses enough parallelism.

## Analytical Gate

E163 resource trace says Q3_K small-k MMVQ dominates parsed MMVQ time:

- fused `ncols_x=5120`, `grid.x=8704`: `341.640 ms`;
- fused `ncols_x=17408`, `grid.x=2560`: `211.049 ms`;
- direct Q3_K small-k buckets together are also material.

This route touches the same high-share buckets without changing quant math. A `2%` wall target requires only a few percent local improvement across the Q3_K small-k buckets, so a structural row/warp split is worth a temporary probe.

## Implementation Plan

1. Add a temporary branch inside `mul_mat_vec_q` for `type == GGML_TYPE_Q3_K && small_k && ncols_dst == 1 && rows_per_cuda_block == nwarps && nwarps > 1`.
2. Use `threadIdx.y` as the local row index, `threadIdx.x` as the warp lane, and reduce only within the warp.
3. Keep generic MMVQ untouched for all non-Q3_K or non-small-k routes.
4. Revert unless r1/r3 and resource trace both beat the E151/E163 baseline.

## Benchmark Plan

- Candidate r1/r3: active H39 speed lane with `--max-tokens 128`.
- Resource trace: active H39 trace lane with `--max-tokens 16`, `GGML_TRACE_MMVQ_RESOURCES=1`.
- Artifacts path: `build_logs/agent-workload/e167-*`.

## Result

| Stack | Aggregate TPS | Decode eval | Decision |
| --- | ---: | ---: | --- |
| E151 promoted best | `30.3145` | `32.2467 tok/s` | baseline |
| E167 row-warp r1 | `27.2886` | `29.37 tok/s` | regression |

Resource trace:

| Q3_K small-k bucket | E163 clean | E167 row-warp | Change |
| --- | ---: | ---: | --- |
| fused `ncols_x=5120`, `grid.x=8704` | `0.355 ms`, `84 regs`, `512 B shared`, `87.5% occ` | `0.383 ms`, `90 regs`, `0 B shared`, `100% occ` | slower |
| fused `ncols_x=17408`, `grid.x=2560` | `0.219 ms`, `84 regs`, `512 B shared`, `87.5% occ` | `0.242 ms`, `90 regs`, `0 B shared`, `100% occ` | slower |
| direct `ncols_x=5120`, `grid.x=5120` | `0.156 ms`, `88 regs`, `256 B shared`, `87.5% occ` | `0.165 ms`, `75 regs`, `0 B shared`, `100% occ` | slower |
| direct `ncols_x=5120`, `grid.x=3072` | `0.124 ms`, `88 regs`, `256 B shared`, `87.5% occ` | `0.128 ms`, `75 regs`, `0 B shared`, `100% occ` | slower |
| Total parsed MMVQ trace | `1075.567 ms` | `1133.910 ms` | slower |

## Decision

- Reject and revert.
- The row-warp route removed shared-memory partial reductions, but it doubled each warp's K traversal and lost the K-split latency hiding that matters on the high-share fused shapes.
- Workflow correction: do not treat shared-memory removal or 100% reported occupancy as sufficient for small-k Q3_K. For this lane, preserving K-split parallelism is more valuable than removing the cross-warp reduction.

## Artifacts

- `build_logs/agent-workload/e167-rocm-decode-q4-q3-rowwarp-r1.diagnostics.md`
- `build_logs/agent-workload/e167-rocm-decode-q4-q3-rowwarp-resources-r1.server.log`
