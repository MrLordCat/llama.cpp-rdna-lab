# E302: ROCm True One-Token Decode Node Trace

Date: 2026-07-14

## Purpose

E299 initially captured the first prompt chunk rather than token generation.
This follow-up advanced the graph skip window until graph 7/8 showed
one-column tensors (`ne=(...,1,1,1)`) on both devices.

The diagnostic used dual ROCm GPUs, a 161-token prompt, two output tokens, and
per-node synchronization. Absolute times are heavily inflated by the sync and
are not wall-speed measurements; route ranking is the useful result.

## Ranked Trace

The two device graphs contained 1,494 timed groups:

| Operation | Groups | Distorted sync time |
| --- | ---: | ---: |
| MUL_MAT | 497 | 66.529 ms |
| UNARY | 160 | 22.360 ms |
| RMS_NORM | 209 | 15.667 ms |
| GET_ROWS | 98 | 7.763 ms |
| CPY | 97 | 7.611 ms |
| L2_NORM | 96 | 7.111 ms |
| ADD | 97 | 7.067 ms |
| GATED_DELTA_NET | 48 | 4.641 ms |
| CONCAT | 48 | 3.664 ms |
| SSM_CONV | 48 | 3.537 ms |
| FLASH_ATTN_EXT | 16 | 1.980 ms |

Q3_K matvec remains the largest actionable family:

- fused Q3_K on ROCm1: 74 groups / 13.402 ms;
- fused Q3_K on ROCm0: 69 groups / 12.715 ms;
- direct Q3_K on ROCm1: 74 groups / 8.653 ms;
- direct Q3_K on ROCm0: 71 groups / 8.566 ms.

Together these Q3_K groups account for about 43.3 ms of the 154.4 ms
sync-distorted trace. Short-context FlashAttention is too small to lead the
next optimization branch.

## Decision

Keep Q3_K fused/direct MMVQ as the primary ROCm decode target. Treat UNARY,
RMS/L2 norm, and graph-node count as the secondary route if the next structural
Q3_K candidates are exhausted. Do not use this trace's absolute milliseconds
as a throughput prediction.

Primary artifact: `e302-lol-rocm-decode-node-trace-skip6`.

