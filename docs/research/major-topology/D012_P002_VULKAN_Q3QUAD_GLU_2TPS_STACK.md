# D012 P002 Vulkan Q3_K Quad + GLU 2 TPS Stack

Date: 2026-05-26

Status: kept as a measured Vulkan P002 opt-in stack; GLU fast path is kept as a general contiguous split fast path.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`, cold-first, no reuse, no v2 prime, thinking on.
- Required env for the confirmed stack: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256`, `GGML_VK_QK_LOW_TILE_SPLIT_K=3`, `GGML_VK_Q3K_QUAD_DEQUANT=1`.
- Comparison anchor: D005 `d005-vulkan-default-splitk-confirm3`, `1.7898 TPS`, prompt `934.81 tok/s`, decode `43.59 tok/s`.
- Confirmed candidate: D012 `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3`, `2.0013 TPS`, prompt `1053.1067 tok/s`, decode `42.7233 tok/s`.

## Mechanism

D008-D012 stacked three source/topology changes on top of D005:

1. `bn256` AMD large-matmul variant for the active large Q3_K matmul family.
2. Separate compile-time `matmul_q3_k_f32_quad` pipeline selected only for high-share Q3_K prefill shapes.
3. A GLU contiguous split fast path that avoids row/stride division arithmetic for the common FFN split-GLU case.

The Q3_K quad pipeline is intentionally separate from the default Q3_K shader so
the helper does not raise register pressure for every shape. It is gated by
`GGML_VK_Q3K_QUAD_DEQUANT=1` and only applies to aligned `Q3_K x F32` matmul
when `n >= 128` and the shape is one of:

- `m=17408,k=5120`
- `m=5120,k=17408`
- `m=5120,k=6144`
- `m=6144,k=5120`
- `m=12288,k=5120`

The `m=10240,k=5120` bucket is deliberately excluded. A point trace looked
slightly positive, but the full wall run collapsed to `0.3783 TPS` with prompt
eval `190.63 tok/s`, so that shape is a residency/resource cliff in the separate
quad pipeline.

## Point Evidence

The separate q3quad pipeline on top of `bn256 + lowtile3` improved the point
route without regressing the excluded `m=10240` bucket:

| Trace | Parsed total ms | Gate/up Q3_K ms | Down Q3_K ms | All Q3_K ms | `m=10240` ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| D008 `bn256 + lowtile3` | `7591.73` | `2857.34` | `1480.10` | `5876.48` | `626.92` |
| D009 separate q3quad pipeline | `7330.07` | `2759.96` | `1417.34` | `5691.67` | `627.79` |

Artifacts:

- `build_logs/agent-workload/d008-vulkan-130k-bn256-lowtile3-route-ceiling.md`.
- `build_logs/agent-workload/d009-vulkan-130k-q3quad-pipeline-bn256-lowtile3-route-ceiling.md`.
- `build_logs/agent-workload/d010-vulkan-130k-q3quad-bn256-lowtile3-fulltrace-route-ceiling.md`.

The route trace confirmed `matmul_q3_k_f32_quad_f16acc_aligned_l` on the selected
shapes and not on `m=10240,k=5120`.

## Wall Validation

| Variant | Runs | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| D005 default split-K | 3 | `1.7898` | `934.81` | `43.59` | baseline |
| D008 `bn256 + lowtile3` | 1 | `1.9344` | `1021.00` | `38.79` | positive but below target |
| D009 q3quad pipeline stack | 1 | `2.0032` | `1054.74` | `42.30` | first r1 target hit |
| D009 q3quad pipeline stack | 3 | `1.9935` | `1048.98` | `42.5933` | positive, but target not confirmed |
| D012 q3quad + GLU fast path stack | 1 | `2.0042` | `1055.19` | `42.41` | best r1 |
| D012 q3quad + GLU fast path stack | 3 | `2.0013` | `1053.1067` | `42.7233` | keep; target confirmed |

D012 improves the D005 r3 anchor by `+11.82%` wall TPS and prompt eval by
`+12.65%`. Decode is slightly below D005 but remains above the D008 `bn256`
decode regression and no longer prevents the `2 TPS` target from clearing on r3.

Final confirmation artifact:

- `build_logs/agent-workload/d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3.diagnostics.md`.

## Rejected Neighbors

| Probe | Result | Decision |
| --- | --- | --- |
| `ub320` after `bn256 + lowtile3` | point prompt `763.64 tok/s` | reject; still below `ub256` |
| `ub320 + GGML_VK_ENABLE_MEMORY_PRIORITY=1` | point prompt `186.75 tok/s` | reject; catastrophic on this stack |
| `GGML_VK_Q3K_FFN_DOWN_SPLIT_K=6` with q3quad | full r1 `1.9943 TPS` | reject; does not convert |
| q3quad as a single shader body or runtime branch | point traces regressed due shader footprint; `m=10240` worsened badly | reject |
| include `m=10240,k=5120` in separate q3quad pipeline | point `m=10240` improved to `607.76 ms`, full r1 collapsed to `0.3783 TPS` | reject and exclude |
| `lowtile4` | full r1 `1.9877 TPS` | reject |
| `lowtile2` | r1 `2.0009`, r3 `1.9926`; point total `7334.15 ms` vs `7330.07` for lowtile3 | reject as final setting |
| no `GGML_VK_ALLOW_GRAPHICS_QUEUE` | full r1 `0.3679 TPS` class cliff | reject; graphics queue remains part of this lane |
| `GGML_VK_DISABLE_MMVQ=1` | decode `42.7 -> 34.7 tok/s`, full r1 `1.9805` | reject |
| vector-return form for q3quad dequant | prompt `1024.43 tok/s`, full r1 `1.9476` | reject and revert |

## Code Policy

Kept source changes:

- `ggml-vulkan.cpp`: env-gated `bn256` large-matmul variant, low-tile split-K diagnostic, q3quad pipeline creation/selection, and D005 FFN-down split-K default.
- `mul_mm_funcs.glsl`: compile-time `Q3_K_QUAD_DEQUANT` helper path for the separate q3quad pipeline.
- `vulkan-shaders-gen.cpp`: q3quad shader variants for non-coopmat2 matmul.
- `glu_main.glsl`: default contiguous split fast path for GLU.

Default/promotion status:

- GLU fast path is default because it is fail-closed by stride/shape predicates and preserves the generic fallback.
- Q3_K quad, `bn256`, and low-tile split-K remain opt-in gates for the measured P002 stack until broader correctness/resource coverage decides whether they should become defaults.
- The exact command/env stack above is the canonical D012 2 TPS evidence; do not claim this result for default Vulkan launches without those gates.

## Next Work

The confirmed P002 Vulkan stack has reached the old `2 TPS` target. As of
2026-05-27 the user retargeted Vulkan to `2.4 TPS`, so D012 is now the active
baseline rather than the endpoint. Promotion hardening remains useful but is
lower priority while the speed target is open:

- add a compact correctness/smoke gate for the q3quad pipeline shapes;
- decide whether `bn256`, lowtile3, and q3quad should become RDNA4/Q3_K defaults
  or remain GUI/benchmark opt-ins;
- use D028 before new speed work: `2.4 TPS` requires `1.1992x` wall over D012,
  about `1.387x` local on dense FFN or `1.260x` local on all-Q3;
- keep m10240 excluded unless a separate resource fix removes the full-run cliff;
- update GUI/autotune launch presets only after promotion policy is chosen.