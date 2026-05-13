# C04 - RMS_NORM fused

## Current cost snapshot

- Center: `CUDA_NODE op=RMS_NORM kind=fused`
- sum_ms: `209.981`
- count: `4389`
- avg_ms: `0.048`
- Priority: `P4`

Source trace:
- `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`

## Planned trace steps

1. Baseline route and shape histogram for RMS_NORM fused nodes.
2. Check coupling with MUL_MAT/MMVQ candidates.
3. Only pursue if C01/C02 no longer provide higher ROI.
