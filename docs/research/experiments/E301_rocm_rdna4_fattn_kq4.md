# E301: ROCm RDNA4 Vector FlashAttention KQ4 Gate

Date: 2026-07-14

## Hypothesis

The RDNA4 vector FlashAttention path used two threads per quantized KQ dot.
Increasing that specialization to four threads could shorten the quantized KQ
section after E293 restored the rocWMMA prompt route.

## Correctness

- The ROCm backend built successfully.
- Focused `FLASH_ATTN_EXT` coverage passed, including the local Qwen forms with
  head size 256, q8_0 K/V, grouped-query ratios, and both `nb=1` and `nb=16`.
- A broad unrelated FlashAttention test hit the existing
  `fattn-wmma-f16.cu:631` fatal gate; the candidate-specific focused tests did
  not fail.

## Game-Loaded A/B

Single free GPU, 8,606 prompt tokens and 128 output tokens:

| Variant | Prompt TPS | Decode TPS |
| --- | ---: | ---: |
| KQ4 candidate r1 | 1170.59 | 35.33 |
| KQ2 control | 1146.39 | 34.69 |
| KQ4 candidate r2 | 1159.06 | 35.33 |

Dual GPU, 21,636 prompt tokens and 128 output tokens:

| Variant | Prompt TPS | Decode TPS |
| --- | ---: | ---: |
| KQ4 candidate r1 | 1791.04 | 26.81 |
| KQ2 control | 1799.23 | 27.17 |
| KQ4 candidate r2 | 1787.57 | 27.02 |

The candidate improved the single-GPU screen by about 1.8%, but its dual-GPU
decode mean was 26.915 versus 27.17 for control, a 0.94% regression.

## Decision

Reject for the dual-GPU target and restore `nthreads_KQ_q=2`. The production
`ggml-hip.dll` was rebuilt on the control path. This also confirms that
short-context vector FlashAttention is not the main remaining dual decode
bottleneck.

