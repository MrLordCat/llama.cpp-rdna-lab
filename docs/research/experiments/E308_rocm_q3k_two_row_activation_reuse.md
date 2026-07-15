# E308: ROCm Q3_K Two-Row Activation Reuse

Date: 2026-07-14

## Hypothesis

The direct Q3_K N=1 MMVQ kernel computes two output rows per block. Both rows
multiply the same q8_1 activation, but the production implementation invokes
the dot helper twice and loads that activation twice. Loading it once and
accumulating both weight rows could reduce decode memory traffic.

## Prototype

An opt-in `GGML_MMVQ_Q3K_ROWPAIR=1` route was limited to direct, non-fused,
small-K Q3_K with N=1 and two rows per block. Prompt kernels, fused gate/up,
MTP N>1, and all other quantization formats were unchanged. Backend operation
tests passed all 11 Q3_K shapes, including N=1 through N=9.

The shared activation increased compiler resource use substantially:

| Route | Registers | Modeled occupancy |
| --- | ---: | ---: |
| Production | 54 | 100% |
| Two-row reuse | 85 | 87.5% |

## Clean A/B/B/A

The matched lane used an 8,604-token prompt, 256 generated tokens, dual ROCm,
16K context, batch/ubatch 8192/1024, q8 KV, and no speculative decoding. No
game or other intentional GPU workload was active.

| Route | Prompt TPS | Decode TPS | Aggregate TPS |
| --- | ---: | ---: | ---: |
| Control A1 | 1742.73 | 27.92 | 18.11 |
| Reuse B1 | 1605.02 | 27.06 | 17.24 |
| Reuse B2 | 1747.73 | 28.10 | 18.20 |
| Control A2 | 1737.18 | 27.57 | 17.94 |
| Control mean | 1739.96 | 27.75 | 18.03 |
| Reuse mean | 1676.38 | 27.58 | 17.72 |

The candidate changed prompt throughput by `-3.65%`, decode by `-0.59%`, and
aggregate throughput by `-1.69%`.

## Decision

Reject and remove the prototype. Reusing q8_1 values does reduce explicit
loads, but keeping both Q3_K rows live raises register pressure enough to erase
the benefit. Do not revisit this exact two-row inline form unless the helper
can be reorganized to remain near the production 54-register footprint.

Artifacts use the prefixes `e308-clean-rocm-q3-rowpair-*` and
`e308-clean-rocm-q3-control-resource-r1`.
