# D093: Vulkan Q4_K_M prompt recovery on current AMD drivers

Date: 2026-08-04
Status: accepted local workaround
Hardware: dual Radeon RX 9070 XT 16 GB
Driver: AMD display `32.0.31035.1003` (`2026-07-24`)
Primary build: `build-vulkan`

## Problem

Qwen3.6-27B Q4_K_M Vulkan prompt evaluation fell from the July
`~1052-1229 tok/s` range to about `381-401 tok/s`. The same slowdown was
visible through GUI autotune. Updating from the first slow driver
`32.0.22042.14002` to `32.0.31035.1003` did not recover performance, and a
driver rollback is not part of the accepted operating plan.

The regression reproduced with:

- the current Vulkan source and binary;
- a rebuilt pre-D4 source snapshot;
- the preserved July Vulkan binary;
- both compute and graphics queue selection.

The Qwen3.5-9B control remained healthy (dual Vulkan `1104 tok/s`), so this was
not a general dual-GPU or PCIe failure. Kernel timing localized the regression
to the large Q4_K matmul route selected by the AMD proprietary-driver profile.

## Route gate

All rows below use the same Q4_K_M `ctx=12288,b=8192,ub=1024,q8_0/q8_0`
diagnostic lane. One-token route scouts are used only to rank shader variants.

| Route | Prompt tok/s | Decision |
| --- | ---: | --- |
| previous automatic large route / base | 316.75 | reject |
| `bn64` | 269.99 | reject |
| `wmiter1` | 316.88 | reject |
| `block64` | 327.36 | reject |
| `block128` | 330.45 | reject |
| generic path (large matmul disabled) | 915.39 | viable fallback |
| `wn32` | 1094.53 | accept |

A normal 64-token adjacent run with explicit `wn32` produced
`1470.51 prompt / 26.51 decode tok/s`, versus
`381.51 / 27.05` on the broken default. The route changes prompt throughput
without a material decode regression.

## Accepted change

For AMD proprietary-driver, discrete RDNA3-class devices with cooperative
matrix support, the automatic large-matmul variant is changed from `bn256` to
`wn32`.

Explicit controls remain available:

- `GGML_VK_AMD_LARGE_MATMUL_VARIANT` overrides the automatic variant;
- `GGML_VK_DISABLE_AMD_WN32_DEFAULT=1` disables the tuned default;
- legacy `GGML_VK_DISABLE_AMD_BN256_DEFAULT=1` remains an alias for disabling
  the current tuned default so existing A/B commands keep their meaning;
- `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1` forces the generic route.

The GUI server and benchmark/autotune paths explicitly select `wn32`. Turning
off the Vulkan large-matmul checkbox now explicitly selects the generic route,
rather than silently falling back to the same automatic AMD optimization.

## Post-build validation

The accepted code was rebuilt in the canonical `build-vulkan` directory. No
parallel or replacement production build was created.

| Lane | Prompt tok/s | Decode tok/s | Artifact label |
| --- | ---: | ---: | --- |
| 12K, 6,729 prompt tokens | 1563.57 | 26.35 | `validate-20260804-q4km-vulkan-short12k-default-wn32-r1` |
| 131K, 59,213 prompt tokens | 1171.94 | 21.31 | `validate-20260804-q4km-vulkan-long131k-p60k-default-wn32-r1` |

Both validation runs explicitly removed the force/variant environment variables,
therefore they validate the new backend default rather than the diagnostic
override. The long result is `2.92x` the adjacent broken-default
`401.19 tok/s` result and is slightly above the explicit-`wn32` confirmation
(`1157.32 tok/s`).

## Operational decision

Keep `build-vulkan` as the only production Vulkan build and rebuild it in
place after Vulkan changes. Temporary source/binary A/B builds are diagnostic
only and must not be promoted or wired into the GUI.
