# E036 H18 C01 E020 Pre-Sync Companion

## Metadata

- Experiment ID: E036
- Date: 2026-05-17
- Owner: Copilot
- Type: trace-only companion analysis (no runtime code changes)
- Lane: C01 (`ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, no-reuse)

## Goal

Explain why E020 improved the target MMQ bucket but stayed runtime-neutral.

## Inputs

- Baseline trace: `build_logs/agent-workload/c01-e015-rdna4-y64w4-trace-r1.server.log`
- Candidate trace: `build_logs/agent-workload/c01-e020-q3-halfscale-compact-trace-r1b.server.log`
- Filter: `mul_mat_q_case: timing type=11 ... ncols_max=192`

## Aggregated MMQ q3 bucket comparison

- E015:
  - count: `26524`
  - pre_sync_ms: `1282.705`
  - enqueue_ms: `135.698`
  - sync_ms: `9446.242`
  - total_ms: `9579.561`
- E020:
  - count: `26524`
  - pre_sync_ms: `1476.664`
  - enqueue_ms: `153.422`
  - sync_ms: `9326.433`
  - total_ms: `9479.791`

Delta (`E020 - E015`):

- pre_sync_ms: `+193.959`
- enqueue_ms: `+17.724`
- sync_ms: `-119.809`
- total_ms: `-99.770`

Interpretation:

- E020 really reduced MMQ q3 kernel-side sync time.
- A large pre-sync increase consumed most of that gain inside the same bucket.
- This matches the earlier outcome: target bucket improved, but end-to-end TPS stayed neutral.

## H06 gate snapshot (trace-based)

From `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`:

- `GGML_TRACE_CUDA_NODE_TIMING` total: `3303.800 ms`
- attention/QKV/RoPE-related node-name subset (regex: `rope|rot|attn|q_|k_|v_|wq|wk|wv|query|key|value`):
  - sum: `575.093 ms`
  - share: `17.41%`

Practical ceiling estimate from this slice:

- 10% improvement inside slice: about `+1.74%` of CUDA_NODE total
- 20% improvement inside slice: about `+3.48%` of CUDA_NODE total

## Decision

- E036 verdict: `keep as diagnostic evidence`
- No runtime code changes in this experiment.

## Next action

1. Keep E020 reverted as default.
2. If revisiting compact layout, pair it with explicit pre-sync mitigation probe and compare serialized traces.
3. Start H06 implementation gate because it still has a realistic multi-percent ceiling on this lane.
