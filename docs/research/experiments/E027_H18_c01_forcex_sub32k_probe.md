# E027 H18 C01 Force-X Sub-32KiB Probe

## Metadata

- Date: 2026-05-16
- Target lane: `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `batch=6144`, `ubatch=192`, KV `q4_0/q4_0`, `review_bug+patch_sim`, no-reuse, thinking on
- Baseline: `c01-e015-rdna4-y64w4-r3-retest-20260516 = 9.4111 TPS`
- Fresh trace: `build_logs/agent-workload/c01-return-20260516-r1-resources.server.log`
- Hypothesis: H18

## Fresh C01 Gate

- Trace-overhead TPS: `6.4253`
- Shape gate: `type=11`, `ncols_max=192`, count `26524`, PASS
- Steady `MUL_MAT forward` split:
  - `mul_mat_q_direct|q3_K`: `12171.789 ms`, `78.31%`
  - `cublas_backend|f32`: `2101.367 ms`, `13.52%`
  - `mul_mat_q_direct|q4_K`: `964.349 ms`, `6.20%`
- Q3 component proxy:
  - steady `compute_core_q3`: `12171.789 ms`, `84.13%`
  - steady `fallback_cublas`: `2101.367 ms`, `14.52%`
  - steady `dequant_load_vec_q3`: `195.523 ms`, `1.35%`
- Active Q3 geometry:
  - `mmq_x=96`, `mmq_y=64`, `block_threads=128`
  - shared `35712` bytes, regs `160`, `max_blocks_per_sm=1`, waves `4.00`

## Analytic Gate

The active `x96/y64` bucket uses two x-tiles for `ncols=192`.

- `x72` looked attractive on paper because projected shared is about `32256` bytes, below `32 KiB`.
- Trace showed it is not a valid runtime override: RDNA4 WMMA granularity is `16`, so `72` is rejected and the route falls back to `mmq_x=96`.
- This is good: `mmq_x=72` is not a safe WMMA tile width because the final `j0` block would not align with the `16`-column tile geometry.
- `x64` is valid and also below `32 KiB`, but needs three x-tiles instead of two.

## Measurements

| Candidate | Result | Decision |
| --- | ---: | --- |
| `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=72` | `9.53 TPS` non-trace r1 | invalid/no-op; trace still selected `mmq_x=96` |
| `x72` trace | `MMQ type=11 ncols_max=192`: `9526.997 -> 9474.024 ms` | no causal claim; override did not apply |
| `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=64` | `8.90 TPS` non-trace r1 | reject |

## Decision

- Reject `x64`: the extra x-tile cost is larger than the shared/occupancy benefit.
- Reject `x72`: invalid override/no-op due RDNA4 WMMA granularity.
- Do not continue force-x sweeps unless the kernel tile geometry changes. The valid below-32KiB point (`x64`) is decisively slower, and the best two-tile points stay above `32 KiB` without a layout change.

## Next

- C01 remains dominated by Q3_K MMQ, but simple force-x is now closed for this lane.
- Future Q3 work needs a real layout or scheduling change, not another x-selector value.
- If switching away from Q3 internals, the only C01 subcenter with a meaningful ceiling is the F32 batched/cuBLAS group (`~13-15%` of steady `MUL_MAT forward`), but E023 already rejected the simple `GemmEx` route.
