# E314: ROCm RMS_NORM + MUL + IMROPE fusion

Date: 2026-07-15

## Goal

Test whether eliminating the intermediate Q/K normalization tensor and the
separate RoPE launch can improve ROCm decode on Qwen3.6.

## Finding

Qwen3.6 uses interleaved MRoPE (`mode=40`) with a 256-element head and 64
rotated dimensions. The existing Vulkan `RMS_NORM + MUL + ROPE` fusion only
accepts normal and NeoX modes, so it is not active for this model.

An opt-in HIP kernel was implemented with exact IMROPE section selection. A
route probe confirmed that the three graph nodes were fused, and the
deterministic 128-token response matched the control preview, length, and token
count.

## Result

Dual `ROCm1,ROCm0`, equal layer split, output on ROCm0, q8 KV, context 49,152,
batch/ubatch 8192/1024, 161-token prompt, 128-token deterministic decode:

| Route | Decode tok/s |
| --- | ---: |
| Control r1 | 32.77 |
| Fused | 29.81 |
| Control r2 | 31.88 |
| Control mean | 32.33 |

The fused route was 7.8% slower than the control mean. Prompt performance was
neutral within run variance.

## Decision

Reject and remove the fusion. Combining the reduction, broadcast multiply,
IMROPE position selection, and trigonometric work increased the cost of the
hot kernel more than the removed launch and intermediate traffic saved.

Artifacts use the prefix `e314-rocm-dual-rms-rope-*`.
