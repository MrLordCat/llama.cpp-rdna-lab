# D001 P002 Vulkan Q3_K Prompt Route

Status: superseded by `ubatch=256` recenter and S001 runtime rejection.

Parent program: P002 130k dense Qwen3.6 Vulkan/ROCm residency route.

## Decision

Original D001 decision started P002 Vulkan source work from T2, not T3:

- Primary candidate: shader-native/backend-private Q3_K prompt layout scout for
  the active `matmul_q3_k_f32_f16acc_aligned_l` route.
- Hold candidate: broad dense FFN gate/up fusion, unless a separate design proves
  enough B/activation reuse to matter on `ubatch=128`.

Original reason: the P002 quick lane used `n=128`, matching the current large matmul
`BN=128` tile. There is no N-tile A-side reuse to recover in the dominant
`17408 x 128 x 5120` and `5120 x 128 x 17408` shapes. A fused FFN route can
still reduce launches or reuse B/activation tiles, but it no longer has the
old `n=1024` N-reuse argument as its main ceiling.

## Inputs

- Evidence: [P002_VULKAN_130K_EVIDENCE.md](P002_VULKAN_130K_EVIDENCE.md).
- Scout artifact: `build_logs/agent-workload/p002-vulkan-q3k-layout-route-gate.txt`.
- Active source route:
  - `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: `ggml_vk_mul_mat_q_f16`, `ggml_vk_guess_matmul_pipeline`, `GGML_VK_MATMUL_ROUTE_TRACE`.
  - `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`: large matmul tile constants and shared-memory staging.
  - `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl`: Q3_K pair dequant and A-tile load.
  - `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`: `block_q3_K`, packed16, packed32 layouts.

## P002 Route Facts

Parsed Vulkan perf time from the P002 routepack:

| Bucket | Parsed share | Total ms |
| --- | ---: | ---: |
| `MUL_MAT q3_K` | 79.41% | 7941.70 |
| `FLASH_ATTN_EXT` | 7.24% | 724.12 |
| `MUL_MAT f32` | 6.41% | 640.94 |
| `MUL_MAT q4_K` | 4.91% | 491.27 |

Top Q3_K shapes:

| Shape | Parsed share | Total ms |
| --- | ---: | ---: |
| `MUL_MAT q3_K m=17408 n=128 k=5120` | 39.33% | 3933.65 |
| `MUL_MAT q3_K m=5120 n=128 k=17408` | 18.35% | 1835.36 |
| `MUL_MAT q3_K m=10240 n=128 k=5120` | 9.27% | 926.85 |

P002 layout gate with `BM=128,BN=128`:

| Shape | Base workgroups | Base A pair-dequants | N-reuse2 A pairs | N-reuse4 A pairs |
| --- | ---: | ---: | ---: | ---: |
| `ffn_gate_up 17408x128x5120` | 272 | 89,128,960 | 89,128,960 | 89,128,960 |
| `ffn_down 5120x128x17408` | 40 | 44,564,480 | 44,564,480 | 44,564,480 |
| total hot FFN proxy | 312 | 133,693,440 | 133,693,440 | 133,693,440 |

This makes simple N-reuse fusion analytically closed for P002 quick. The lane is
still Q3_K dominated, but the useful mechanism must reduce Q3_K unpack/dequant,
memory layout cost, or another high-share route body cost without adding a
multi-dispatch temp/sync path.

## Rejection Fence

Do not reopen these as the first P002 prototype:

- Existing per-node fp16 predequant fallback: rejected because it adds a large
  temp, sync boundary, and extra global write/read traffic.
- Existing split-K route for the reverse Q3_K shape: near-neutral/negative and
  adds partial-output traffic plus reduce dispatch.
- Larger `BN`/`BM` or smaller `BK` within the current shared shader body: prior
  probes show VGPR/LDS/live-state or K-loop/barrier cost dominates.
- Full persistent fp16 or int8 FFN layouts: too much extra memory for a 16 GiB
  long-context lane.
- Broad Q3_K transpose-A as a single layout for prompt and decode: prompt-only
  improvement was offset by decode regression on the archived lane.

## Candidate T2: Compact Q3_K Prompt Layout Scout

Memory gate from the P002 layout tool:

| Tensor group | Current Q3 copy | Signed-nibble alternate | Delta |
| --- | ---: | ---: | ---: |
| FFN gate+up | 4.57 GiB | 5.31 GiB | +765.00 MiB |
| FFN down | 2.28 GiB | 2.66 GiB | +382.50 MiB |
| FFN all | 6.85 GiB | 7.97 GiB | +1.12 GiB |

Full fp16 and int8 alternates are rejected. A compact signed-nibble or equivalent
backend-private layout is memory-plausible only as a narrow opt-in scout. It may
remove hmask/qs bit extraction from the shader body, but it does not remove scale
decode or FMA work, so it needs a static SPIR-V/resource gate before any server
A/B.

Required S001 scout before runtime selection:

1. Add or generate a compile-only Q3_K shader-layout variant, not selected by
   default and not wired into normal inference.
2. Compare SPIR-V fingerprints against `matmul_q3_k_f32_aligned_f16acc_cm1.spv`:
   `OpShift*`, `OpBitwise*`, `OpLoad`, `OpFConvert`, `OpFMul`, and total function
   body size.
3. Capture `GGML_VK_PIPELINE_STATS=matmul_q3_k` for the variant if the driver
   exposes resource lines; require no scratch and no obvious resource blow-up.
4. Only after static/resource proof, wire a default-off runtime gate for a narrow
   tensor subset and run paired cold P002 A/B.

Acceptance to proceed from scout to runtime prototype:

- SPIR-V shows a material reduction in Q3_K unpack/bit-manipulation work.
- No new scratch and no large LDS/VGPR increase relative to the current route.
- Extra memory is scoped to FFN gate/up or another justified subset, not all Q3_K
  tensors by default.
- Decode correctness is protected: unset gate must be baseline behavior; enabled
  gate either keeps decode on the original layout or proves decode-safe reads.

## Held Candidate T3: Dense FFN Fusion

For P002 quick, FFN fusion is not first because `n=128` gives no A-side N-reuse.
The only plausible savings are B/activation reuse, launch count, and possibly GLU
traffic. That can be worthwhile later, but it needs its own B-traffic ceiling
model before source edits. A launch-only fusion remains rejected.

If revived, T3 must show:

- B/activation traffic saved per layer/chunk is large enough to affect a route
  whose parsed time is dominated by Q3_K matmul.
- The shader does not double live accumulators enough to recreate the old VGPR
  regression.
- The down projection is addressed or the design explains why gate/up-only is
  still above the P002 build threshold.

## Next Step

Do not continue the signed-nibble runtime route as-is. S001 static scout reduced
SPIR-V size/op counts, but the measured runtime path failed the P002 constraints:
all-Q3 signed storage failed the 130k fit check, and a narrow `hot5` opt-in
completed at `1.5186 TPS` versus `1.5798 TPS` same-session control. The runtime
prototype was reverted.

Next design should refresh P002 around the confirmed Vulkan `ubatch=256` lane.
That reopens the old `n=128` no-reuse assumption: any T3/FFN or Q3_K route must
model the current two-`BN=128` tile behavior before source edits.