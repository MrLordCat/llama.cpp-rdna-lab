# E286: ROCm Windows Fast Math

Date: 2026-07-14

## Scope

Upstream enables HIP fast math with infinity-preserving semantics. Its normal
`COMPILE_LANGUAGE:HIP` expression does not affect the Windows build because
CMake marks HIP sources as `CXX` and invokes clang with `-x hip`. This port
adds `GGML_HIP_FAST_MATH` and applies the upstream flags directly to the HIP
source list when `CXX_IS_HIPCC` is active:

- `-funsafe-math-optimizations`;
- `-ffast-math`;
- `-fno-finite-math-only`.

The last flag is required because attention masks rely on infinities.

## A/B Results

The matched dual-GPU lane used Qwen3.6-27B Q3_K_S, 7,923 prompt tokens, 32
generated tokens, `b8192/ub1024`, q8 K/V, layer split `1,1`, and no MTP.
Both variants ran three requests in one server process with prompt reuse and
the explicit prime pass disabled.

| Build | Cold prompt | Warm prompt mean | Decode mean |
| --- | ---: | ---: | ---: |
| fast math off | 1,574.31 | 1,796.66 | 26.09 |
| fast math on | 1,589.01 | 1,810.74 | 26.72 |
| delta | +0.93% | +0.78% | +2.42% |

The 207-prompt-token single-GPU gate was neutral: `749.74 / 32.84` with fast
math versus `750.40 / 32.78` without it. Response previews remained coherent
and identical across the long A/B; no NaN or server errors were observed.

The large cold-to-warm prompt jump exists in both variants and is therefore a
HIP/rocBLAS process warmup effect, not prompt-cache reuse or a fast-math gain.

## Result

Keep `GGML_HIP_FAST_MATH=ON` by default. The improvement is small but
repeatable on the long dual-GPU route, and the option permits a direct rollback
with `-DGGML_HIP_FAST_MATH=OFF`. This does not close the ROCm/Vulkan gap: the
dominant broad Q3_K prompt route still stages to F16 and executes in rocBLAS,
whose kernels are outside this target's compile flags.

Primary artifacts:

- `e286-rocm1-fastmath-short-none-r1`;
- `e286-rocm1-fastmath-short-none-r2r4`;
- `e286-rocm-dual-nofast-12k-none-r1r3`;
- `e286-rocm-dual-fastmath-12k-none-r1`;
- `e286-rocm-dual-fastmath-12k-none-r2r4`.
