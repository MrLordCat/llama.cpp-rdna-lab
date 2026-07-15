# E306: ROCm Paired L2-Norm Fusion

Date: 2026-07-14

## Hypothesis

The true one-token trace in E302 contains 96 `L2_NORM` nodes: a Q/K pair in
each of 48 recurrent layers immediately before GDN. A paired kernel could
preserve both outputs while reducing those 96 launches to 48.

## Prototype

An opt-in `GGML_CUDA_L2_PAIR_FUSION=1` route detected the Qwen graph pattern,
skipped the intervening metadata view, and launched one grid over both source
tensors. A trace smoke confirmed 48 fused pairs and zero separate L2 launches
in the captured dual-GPU graph. The server completed the request without an
error.

## Clean A/B/B/A

The matched lane used an 8,604-token prompt, 256 generated tokens, dual ROCm,
16K context, batch/ubatch 8192/1024, q8 KV, and no speculative decoding. No
game or other intentional GPU workload was active.

| Route | Prompt TPS | Decode TPS | Aggregate TPS |
| --- | ---: | ---: | ---: |
| Control A1 | 1720.40 | 29.19 | 18.55 |
| Paired B1 | 1748.04 | 29.08 | 18.61 |
| Paired B2 | 1717.23 | 27.75 | 17.94 |
| Control A2 | 1731.23 | 29.08 | 18.55 |
| Control mean | 1725.82 | 29.14 | 18.55 |
| Paired mean | 1732.64 | 28.42 | 18.28 |

The paired route changed prompt throughput by only `+0.40%`, reduced decode by
`2.47%`, and reduced aggregate throughput by `1.48%`.

## Decision

Reject and remove the prototype. HIP graph replay already amortizes host
dispatch, and halving this small kernel family does not reduce the critical
path. The combined grid instead makes decode measurably slower. Do not revisit
paired L2 launch fusion without evidence that graph replay is unavailable or
that the normalization body itself has changed.

Artifacts use the prefix `e306-clean-rocm-l2pair-*`.

