# C03 - MUL_MAT fused

## Current cost snapshot

- Center: `CUDA_NODE op=MUL_MAT kind=fused`
- sum_ms: `326.936`
- count: `2298`
- avg_ms: `0.142`
- Priority: `P3`

Source trace:
- `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`

## Planned trace steps

1. Build top `name` and `ne` breakdown for fused nodes.
2. Compare baseline vs best C01 candidate to detect coupled wins/losses.
3. Identify whether fused path is compute-bound or sync-bound.
4. Queue focused A/B only after C01 closure.
