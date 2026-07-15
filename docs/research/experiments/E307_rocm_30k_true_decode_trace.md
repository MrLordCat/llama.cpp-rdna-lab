# E307: ROCm True Decode Trace After a 30K Prompt

Date: 2026-07-14

## Goal

Recheck the one-token dual-GPU decode critical path after a genuinely long
prompt. Earlier short-context traces could understate the contribution of
flash attention as the KV cache grows.

## Method

The run used Qwen3.6-27B-Q3_K_S, dual ROCm in `ROCm1,ROCm0` order, layer
split, q8 KV, flash attention, context 49,152, batch/ubatch 8192/1024, and no
speculative decoding. The injected prompt contained 30,073 tokens.

A graph census found 64 backend graphs. The first synchronized attempt at
skip 60 captured the final 377-token prompt chunk. Moving the trace to skip 62
captured the real one-token decode graph; its flash-attention query shape had
`q_rows=1`. Synchronization distorts absolute wall time, so only the operation
shares below are used for diagnosis.

## True Decode Profile

| Operation | ROCm1 | ROCm0 |
| --- | ---: | ---: |
| MUL_MAT | 40.81% | 44.56% |
| RMS_NORM | 10.21% | 9.58% |
| Flash attention | 8.91% | 10.35% |
| UNARY | 8.09% | 6.66% |
| GET_ROWS | 5.23% | - |
| CPY | 4.92% | - |
| ADD | 4.84% | - |
| L2_NORM | 4.84% | - |
| GDN | 3.20% | - |

ROCm1 recorded 771 groups over 73.825 ms and ROCm0 recorded 725 groups over
64.099 ms. The first flash-attention launch on each device included roughly
2.9 ms of cold enqueue overhead; subsequent launches were about 0.42-0.49 ms.

## Conclusion

Long-context flash attention grows into a meaningful secondary cost, but it
is not the primary limiter at 30K tokens. Q3/matrix multiplication still owns
41-45% of the synchronized decode profile. Continue prioritizing the Q3_K
MMVQ path, while treating flash-attention work as a later long-context lever.

Artifacts use the prefixes `e307-clean-rocm-32k-graph-census-r1`,
`e307-clean-rocm-32k-decode-node-trace-r1`, and
`e307-clean-rocm-32k-true-decode-node-trace-r1`.
