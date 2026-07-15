# E289: ROCm Q3_K Packed Subtract

Date: 2026-07-14

## Root Cause

The hot Q3_K MMVQ and fused FFN helpers reconstructed four signed 3-bit values
with `__vsubss4(vl, vh)`. Here each low lane is `0..3` and each high lane is
exactly `0` or `4`, so saturation can never occur. On gfx1201/clang 21 the
saturating intrinsic nevertheless produced expensive code and excessive live
state.

The replacement flips the high bit, performs a biased packed subtraction, and
restores the byte sign bias:

```cpp
const uint32_t vals = uint32_t(vl) | (uint32_t(vh) ^ 0x04040404u);
return int(((vals ^ 0x80808080u) - 0x04040404u) ^ 0x80808080u);
```

The bias prevents borrow propagation between bytes. Exhaustive host validation
covered all `4^4 * 2^4 = 4096` packed input combinations with zero mismatches.

## Kernel Resources

| Q3_K N=1 route | Before | After |
| --- | ---: | ---: |
| direct MMVQ registers | 88 | 54 |
| direct modeled occupancy | 87.5% | 100% |
| fused pair registers | 94 | 62 |
| fused modeled occupancy | 100% | 100% |

This is a compiler/code-generation fix, not a scheduling or acceptance change.

## Performance

The clean single-GPU 207-prompt/128-output r3 lane measured:

| Build | Aggregate TPS | Prompt TPS | Decode TPS |
| --- | ---: | ---: | ---: |
| saturating subtract | 29.43 | 753.81 | 31.58 |
| packed biased subtract | 34.14 | 753.49 | 37.10 |
| delta | +16.0% | -0.04% | +17.5% |

On the dual-GPU 7,923-prompt/32-output lane, decode improved from `26.72` to
`28.43 tok/s` (`+6.4%`). The first prompt remained cold, while the two warm
prompts averaged `1,824.44 tok/s`, `+0.76%` over the E286 warm baseline. Prompt
eval is therefore neutral.

The exact current-build MTP A/B used 7,729 prompt and 256 output tokens:

| Speculation | Aggregate TPS | Prompt TPS | Decode TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| none | 19.01 | 1,725.85 | 28.66 | - |
| MTP n4 | 24.09 | 1,685.56 | 42.78 | 63.76% |
| delta | +26.7% | -2.3% | +49.3% | - |

This restores a useful ROCm MTP gain while keeping the prompt tax within the
accepted few-percent budget.

The E291 31,997-token prompt follow-up confirms that the normal decode gain
survives long context (`22.30 -> 26.44 tok/s` versus E284), while MTP n3 remains
`32.34 tok/s`. At that length FA/KV and multi-column verify dominate more of
the speculative path, so the relative MTP gain narrows to `22.3%` even though
it stays positive with only a `0.9%` prompt tax.

## Correctness and Decision

- Full HIP build completed successfully.
- Focused `test-backend-ops` Q3_K `MUL_MAT` coverage passed `11/11` shapes,
  including `N=1..9`.
- All baseline/candidate response previews were identical and all server runs
  completed without errors.
- Keep the packed helper for direct, raw fused-pair, and padded fused-pair Q3_K
  paths.

Primary artifacts:

- `e289-rocm1-mmvq-resources-short-none-r1.*`;
- `e289-rocm1-q3-sub4twiddle-resources-short-none-r1.*`;
- `e289-rocm1-pairdot-on-short128-none-r1r3.*`;
- `e289-rocm1-q3-sub4twiddle-short128-none-r1r3.*`;
- `e289-rocm-dual-q3-sub4twiddle-12k-none-r1r3.*`;
- `e289-rocm-dual-q3-sub4twiddle-12k-none256-r1r3.*`;
- `e289-rocm-dual-q3-sub4twiddle-12k-mtp-n4-r1r3.*`.
