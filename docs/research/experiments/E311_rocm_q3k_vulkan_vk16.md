# E311: ROCm Q3_K Vulkan-style VK16 decode kernel

Date: 2026-07-14

## Goal

Test whether Vulkan's 16-values-per-lane Q3_K decode layout can reduce the
remaining RDNA4 MMVQ cost on Windows ROCm without changing quantization or
model output.

## Implementation

An opt-in `GGML_MMVQ_Q3K_RDNA4_VK16=1` route was added for padded Q3_K,
`N=1`, no IDs, and K divisible by 512. It uses one wave per output row and is
available for direct and fused gate/up dispatch.

The compiled gfx1201 kernels use 36 VGPR for direct MMVQ and 60 VGPR for the
fused route, with no LDS. Machine-code inspection showed that clang already
emits `global_load_b128` for Q3 payload, hmask, and Q8 activation loads. Manual
vector-load rewrites therefore have no remaining load-width opportunity.

## Results

On a matched dual-GPU short lane, the production control reached 29.82 decode
tok/s. Two VK16 runs reached 30.76 and 30.92 tok/s, a 3.15-3.69% decode gain,
while prompt throughput changed by about -0.5%.

The gain did not survive the correct 30K topology. With output on the last
device, VK16 reached 26.34 tok/s versus 26.72 tok/s for the adjacent control.
Earlier long A/B/A runs that forced output back to ROCm1 were neutral on
average, but E312 shows that topology is not a valid production baseline.

## Rejected variants

- Vulkan-style Q8_1 x4 activation layout: -1.5% decode.
- Two-row activation reuse: 60/93 VGPR and slower; direct-only was also slower.
- Cooperative scale loads through shuffles: -5.4% decode.
- K-loop unroll 2: -2.1%; unroll 4 increased pressure to 83/112 VGPR.

All rejected code was removed.

## Decision

Keep VK16 as an off-by-default short-decode research gate. Do not enable it for
the long-prompt production profile. The remaining long dual-GPU cost is
topology and cross-device scheduling, not a missing scalar load-width tweak.

Primary artifacts use `e311-rocm-dual-*` prefixes.
