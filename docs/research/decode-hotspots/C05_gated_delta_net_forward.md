# C05 - GATED_DELTA_NET forward

## Current cost snapshot

- Center: `CUDA_NODE op=GATED_DELTA_NET kind=forward`
- sum_ms: `149.095`
- count: `1008`
- avg_ms: `0.148`
- Priority: `P5`

Source trace:
- `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`

## Planned trace steps

1. Separate decode-relevant chunks from prefill-dominant chunks.
2. Collect chunk-size and route traces for current decode lane.
3. Re-evaluate only after C01/C02 results are stabilized.
