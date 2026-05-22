# E168 ROCm Q3_K Streaming Vec-Dot Probe

## Metadata

- Experiment ID: E168
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E167 rejection
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: Q3_K MMVQ may improve if `vec_dot_q3_K_q8_1` streams `q8_1` values inside the dot loop instead of preloading `u[QR3_K]` and `d8[QR3_K]` arrays before calling the implementation helper.
- Mechanism: E165/E166 showed Q3_K fused work is sensitive to live values and register pressure. Removing the temporary q8 arrays may reduce register pressure or scheduling pressure across both fused and direct Q3_K small-k buckets without changing quant math or launch geometry.
- Risk: The compiler may already scalarize the arrays optimally, or streaming loads may reduce scheduling freedom and regress.

## Analytical Gate

The target covers the full Q3_K MMVQ family used by the H39 decode lane:

- fused Q3_K buckets: `5120->8704`, `17408->2560`, `6144->2560`;
- direct Q3_K buckets: attention, gate, Q/K/V, FFN down.

E163 says these buckets dominate parsed MMVQ time. This probe is allowed only as a temporary source change with resource trace confirmation.

## Implementation Plan

1. Replace only the Q3_K MMVQ wrapper body with a streaming loop.
2. Keep `vec_dot_q3_K_q8_1_impl_mmvq` intact for comparison and MMQ code untouched.
3. Build, run active H39 r1, then resource trace.
4. Revert unless the highest-share fused bucket improves and r3 beats E151.

## Result

| Stack | Aggregate TPS | Decode eval | Decision |
| --- | ---: | ---: | --- |
| E151 promoted best | `30.3145` | `32.2467 tok/s` | baseline |
| E168 streaming-dot r1 | `28.4937` | `30.80 tok/s` | regression |

Resource trace:

| Q3_K bucket | E163 clean | E168 streaming-dot | Change |
| --- | ---: | ---: | --- |
| fused `ncols_x=5120`, `grid.x=8704` | `0.355 ms`, `84 regs` | `0.362 ms`, `84 regs` | slower |
| fused `ncols_x=17408`, `grid.x=2560` | `0.219 ms`, `84 regs` | `0.222 ms`, `84 regs` | slower |
| direct `ncols_x=5120`, `grid.x=5120` | `0.156 ms`, `88 regs` | `0.161 ms`, `84 regs` | slower |
| direct `ncols_x=5120`, `grid.x=3072` | `0.124 ms`, `88 regs` | `0.123 ms`, `84 regs` | tie/noise |
| Total parsed MMVQ trace | `1075.567 ms` | `1087.373 ms` | slower |

## Decision

- Reject and revert.
- The compiler already handles the original `u[QR3_K]`/`d8[QR3_K]` arrays well enough for the dominant fused path. Streaming did not reduce fused-register count and made the highest-share buckets slower.
- Workflow correction: treat local-array removal in inlined vec-dot helpers as low-confidence unless a compile-resource report shows an actual register reduction in the target fused instantiation.

## Artifacts

- `build_logs/agent-workload/e168-rocm-decode-q4-q3-streaming-dot-r1.diagnostics.md`
- `build_logs/agent-workload/e168-rocm-decode-q4-q3-streaming-dot-resources-r1.server.log`
