# Vulkan Route Map - llama.cpp-rdna-lab

This document maps the current Vulkan backend routes in this fork. It is meant
to sit next to `docs/research/ROCM_ROUTE_MAP.md`: ROCm is the preferred RX 9070
XT path, while Vulkan is the fallback/backend-comparison path. The top-level
Qwen TPS atlas is `docs/research/QWEN_TPS_ROUTE_ATLAS.md`, and the measured
route companion is `docs/research/ROUTE_METRICS_AND_GAPS.md`. Treat this as a
route atlas and cleanup guide, not as a deletion plan.

## Scope

Local target:

- OS/GPU: Windows 11, AMD Radeon RX 9070 XT, RDNA4.
- Preferred backend: ROCm/HIP SDK 7.1.
- Fallback backend: Vulkan.
- Main performance lane remains Qwen3.6-27B prompt-heavy under ROCm, but Vulkan
  is important for fallback correctness, feature comparison, and possible route
  ideas that can be ported back to ROCm.

Current Vulkan implementation is mostly contained in:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/*.comp`
- `ggml/src/ggml-vulkan/vulkan-shaders/*.glsl`
- generated build outputs from `vulkan-shaders-gen`

## Executive Summary

| Route family | Current local role | Keep/prune verdict |
| --- | --- | --- |
| Matmul matrix route | Main Vulkan `MUL_MAT` path for wide batches, with F32/F16/BF16 and quantized weights through dequant/matmul shaders | Keep. It is the core fallback compute path |
| Matmul vector route | Decode/small-column route, including quantized weight x F32/F16 inputs and optional Q8_1 staging when integer dot is useful | Keep. This is the Vulkan analogue of a decode-heavy fast path |
| `MUL_MAT_ID` routes | MoE/expert matrix and vector routes with `count_experts` helper | Keep if model coverage matters; possible profile gate only if MoE is explicitly dropped |
| FlashAttention | Vulkan FA path with scalar/coopmat/coopmat2 capability gates and quantized K/V coverage | Keep. It is one of the most backend-specific areas and useful for fallback comparisons |
| Graph fusions | Multi-add, matmul+bias, RMS_NORM+MUL(+ROPE), ROPE+SET_ROWS, TopK MoE patterns | Keep. These change the real route graph and can hide plain op costs |
| Training/optimizer/image routes | Optimizers, losses, conv/image utilities | Possible local-minimal profile candidates, not safe source deletions today |

Important cleanup rule: Vulkan shader source and generator logic are tightly
coupled. Removing an apparently unused `.comp` file can break generated pipeline
fields, `supports_op(...)`, or feature-test builds. Prefer build-profile gates
over source deletion.

## Build-Time Route Map

| Layer | Files/options | Route role |
| --- | --- | --- |
| User option | `GGML_VULKAN` in `ggml/CMakeLists.txt` | Enables the Vulkan backend |
| Package/tooling | `find_package(Vulkan COMPONENTS glslc REQUIRED)` in `ggml/src/ggml-vulkan/CMakeLists.txt` | Requires Vulkan SDK and `glslc` |
| Backend library | `ggml-vulkan.cpp`, `../../include/ggml-vulkan.h` | Host backend, scheduler hooks, device setup, graph dispatch |
| Shader generator | `vulkan-shaders-gen` external project | Generates `ggml-vulkan-shaders.hpp` and per-shader `.cpp` files into the build tree |
| Shader sources | `vulkan-shaders/*.comp`, `*.glsl` | Compute kernels and common shader helpers |
| Feature tests | `feature-tests/coopmat.comp`, `coopmat2.comp`, `integer_dot.comp`, `bfloat16.comp` | Decide compile definitions for shader variants |

Compile-time feature macros:

| Macro | Meaning |
| --- | --- |
| `GGML_VULKAN_COOPMAT_GLSLC_SUPPORT` | `glslc` can compile `GL_KHR_cooperative_matrix` shaders |
| `GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT` | `glslc` can compile `GL_NV_cooperative_matrix2` shaders |
| `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` | integer-dot shader variants can be built |
| `GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT` | BF16 shader variants can be built |
| `GGML_VULKAN_CHECK_RESULTS` | Inserts CPU comparison/check route around each op |
| `GGML_VULKAN_DEBUG`, `GGML_VULKAN_MEMORY_DEBUG`, `GGML_VULKAN_VALIDATE`, `GGML_VULKAN_SHADER_DEBUG_INFO` | Debug, memory, validation, shader info routes |

## Backend Life-Cycle Map

| Phase | Entry points | Notes |
| --- | --- | --- |
| Registration | `ggml_backend_vk_reg()`, `ggml_backend_vk_init()` | Creates backend devices from detected Vulkan physical devices |
| Device discovery | `ggml_vk_init()`, device feature probing around extension/property blocks | Sets vendor/architecture, fp16, BF16, integer dot, subgroup, coopmat, memory limits, queue behavior |
| Buffers | `ggml_backend_vk_buffer_type()`, `ggml_backend_vk_host_buffer_type()` | Device buffers and host-visible buffer route; no ROCm-style row split buffer type |
| Upload/download | `ggml_backend_vk_set_tensor_*_async()`, `ggml_backend_vk_get_tensor_*_async()` | Uses direct host-visible copies where possible and command-buffer transfer otherwise |
| Backend copy | `ggml_backend_vk_cpy_tensor_async()` | Same-backend copies, with transfer queue and synchronization state |
| Graph optimization | `ggml_vk_graph_optimize()` | Reorders nearby independent nodes but preserves known fusion patterns |
| Graph execution | `ggml_backend_vk_graph_compute()` -> `ggml_vk_build_graph()` -> `ggml_vk_compute_forward()` | Batches nodes into Vulkan command buffers, submits periodically, and can log timings |
| Validation route | `GGML_VULKAN_CHECK_RESULTS` with `ggml_vk_check_results_0/1()` | Forces per-node checks; diagnostic only |

## Runtime Device/Env Map

Clear diagnostic/force env vars before speed claims unless they are the explicit
candidate.

| Env var | Route area | Use |
| --- | --- | --- |
| `GGML_VK_VISIBLE_DEVICES` | device selection | Limits visible Vulkan devices |
| `GGML_OP_OFFLOAD_MIN_BATCH` | scheduler/offload | Minimum op batch size for offload decisions |
| `GGML_VK_MATMUL_ROUTE_TRACE` | matmul | Logs unique Vulkan matmul pipeline/type/shape decisions |
| `GGML_VK_FA_ROUTE_TRACE` | FlashAttention | Logs unique Vulkan FA path/type/tile/split/mask decisions |
| `GGML_VK_PERF_LOGGER` | timing | Timestamp timing logger |
| `GGML_VK_PERF_LOGGER_CONCURRENT` | timing | Groups timings by concurrent command batches |
| `GGML_VK_PERF_LOGGER_FREQUENCY` | timing | Controls logger frequency |
| `GGML_VK_PIPELINE_STATS` | pipeline diagnostics | Filters pipeline stats logging |
| `GGML_VK_SYNC_LOGGER` | synchronization | Logs sync points |
| `GGML_VK_MEMORY_LOGGER` | memory | Logs Vulkan memory events |
| `GGML_VK_DEBUG_MARKERS` | debug markers | Enables debug utils labels when extension exists |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE` | graph optimizer | Disables graph reordering |
| `GGML_VK_DISABLE_FUSION` | graph fusions | Disables fusion layer |
| `GGML_VK_DISABLE_MULTI_ADD` | fusion | Disables multi-add fusion only |
| `GGML_VK_DISABLE_MMVQ` / `GGML_VK_FORCE_MMVQ` | matvec selector | Overrides MMVQ-like Vulkan Y-quantization decision |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` | matmul/FA | Disables integer dot route even if device supports it |
| `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_COOPMAT2` | matmul/FA | Disables cooperative matrix routes |
| `GGML_VK_DISABLE_BFLOAT16` | BF16 | Disables BF16 route |
| `GGML_VK_DISABLE_F16` | FP16 | Disables FP16 route |
| `GGML_VK_FORCE_AMD_LARGE_MATMUL`, `GGML_VK_DISABLE_AMD_LARGE_MATMUL`, `GGML_VK_AMD_LARGE_MATMUL_VARIANT` | AMD matmul shader selection | AMD large-matmul experiments |
| `GGML_VK_DISABLE_ASYNC`, `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`, `GGML_VK_ALLOW_GRAPHICS_QUEUE` | queue/async | Queue and async behavior diagnostics |
| `GGML_VK_PREFER_HOST_MEMORY`, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM`, `GGML_VK_ALLOW_SYSMEM_FALLBACK` | memory placement | Host/VRAM placement probes |
| `GGML_VK_FORCE_MAX_ALLOCATION_SIZE`, `GGML_VK_FORCE_MAX_BUFFER_SIZE`, `GGML_VK_SUBALLOCATION_BLOCK_SIZE` | memory limits | Memory stress/compat diagnostics |
| `GGML_VK_ENABLE_MEMORY_PRIORITY` | memory priority | Enables memory priority path |
| `GGML_VULKAN_SKIP_CHECKS`, `GGML_VULKAN_OUTPUT_TENSOR` | check-results | Controls validation output when checks are compiled in |

## Matmul Route Map

Entry point: `ggml_vk_mul_mat(...)`.

Dispatch order:

1. Huge `src0` split-by-M route if dst has no batch and `src0` exceeds
   `maxStorageBufferRange`.
2. Permuted F16/F32 vector route:
   `ggml_vk_mul_mat_vec_p021_f16_f32(...)`.
3. Non-contiguous F16/F32 vector route:
   `ggml_vk_mul_mat_vec_nc_f16_f32(...)`.
4. General mat-vec route when `dst->ne[1] == 1`, or `dst->ne[1] <= 8` and
   `src1` has no higher batch:
   `ggml_vk_mul_mat_vec_q_f16(...)`.
5. General matrix route:
   `ggml_vk_mul_mat_q_f16(...)`.

Supported `src0` type set in `supports_op(...)`:

- Dense: `F32`, `F16`, `BF16`.
- Classic quants: `Q1_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`.
- K-quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`.
- IQ: `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`,
  `IQ3_S`, `IQ4_XS`, `IQ4_NL`.
- FP4: `MXFP4`, `NVFP4`.

The support list is intentionally limited to the maintained dense, classic,
K-quant, IQ and FP4 formats listed above.

### Matrix Route

Code path:

- `ggml_vk_mul_mat_q_f16(...)`
- `ggml_vk_get_mul_mat_mat_pipeline(...)`
- `ggml_vk_guess_matmul_pipeline(...)`
- `ggml_vk_guess_split_k(...)`
- `ggml_vk_matmul(...)`

Pipeline families:

| Family | Pipeline fields | Shader area |
| --- | --- | --- |
| Dense F32/F16/BF16 | `pipeline_matmul_f32`, `pipeline_matmul_f16`, `pipeline_matmul_f16_f32`, `pipeline_matmul_bf16` | `mul_mm*.comp`, `mul_mm_funcs.glsl` |
| Quant x F16/F32 | `pipeline_dequant_mul_mat_mat[...]`, `pipeline_dequant_mul_mat_mat_f16[...]` | `mul_mmq.comp`, `mul_mmq_funcs.glsl`, `dequant_funcs*.glsl` |
| Quant x Q8_1 staged Y | `pipeline_dequant_mul_mat_mat_q8_1[...]`, `pipeline_quantize_q8_1_x4` | `quantize_q8_1.comp`, `mul_mmq*.glsl` |
| Split-K reduce | `pipeline_matmul_split_k_reduce` | `mul_mat_split_k_reduce.comp` |

Mechanics:

- `x_non_contig` triggers a copy/reformat into `prealloc_x`.
- `y_non_contig` or BF16/F16 mismatch can trigger reformat into `prealloc_y`.
- If integer dot is available and `src1` is contiguous F32 with element count
  divisible by 4, Vulkan first tries to quantize Y to Q8_1 and use an integer
  dot matmul route.
- If no direct quant matmul pipeline exists, `src0` is dequantized to F16/BF16
  and routed through dense matmul.
- Large `k >= 2048` and under-filled shader cores can enable Split-K, followed
  by `pipeline_matmul_split_k_reduce`.
- `ggml_vk_guess_matmul_pipeline(...)` chooses small/medium/large and aligned
  variants. Coopmat2 uses a different crossover heuristic than scalar/coopmat1.

Measured 64k Q3_K route note:

- Active hot Q3_K shapes route directly through
  `matmul_q3_k_f32_f16acc_aligned_l`, not through the fallback predequant path.
- E139 forced the existing fallback for large Q3_K shapes as a route gate:
  `Q3_K -> fp16 prealloc_x -> matmul_f16_f32_f16acc_aligned_l`.
  It routed correctly but regressed pp7488 `969.61 -> 743.65`. Narrow gates
  also lost: only `m>=17000` measured `832.27`, only `k>=17000` measured
  `929.40`.
- The f16 fallback pipeline resource stats were good
  (`77 VGPR`, `44 SGPR`, `22528 B LDS`, `0 scratch`), so the failure is the
  multi-dispatch route topology: a large fp16 temp, sync boundary, and extra
  global write/read traffic. Do not use existing per-node predequant as the
  future Q3_K repack strategy.
- E140 forced existing matmul split-K for the hot reverse Q3_K shape
  `m=5120,n=1024,k=17408`. Split-K2 measured `966.21` and split-K4 measured
  `964.46` vs the direct `968.74` baseline. The route is near-neutral but not
  positive; the shape already has enough workgroups, and partial-output
  traffic/reduce overhead cancels the theoretical shorter K-loop benefit.
- E143 tested the larger-N warptile route family for the active large Q3_K
  shader. Static scout showed plain `BN192` is unsafe under the current A-load
  map, while `BN192/WN96` and `BN256` variants are layout-valid. The real pp
  gate rejected all valid variants: default `974.19`, `bn192-wn96` `760.78`
  (`139 VGPR / 25088 B LDS`), `bn192-wm128-wn96` `137.71`
  (`171 VGPR` plus scratch), and `bn256-*` about `660` (`165 VGPR /
  29696 B LDS`). Larger N tiles reduce A-dequant/workgroup proxies, but current
  `mul_mm.comp` pays too much in live fragments, LDS, and occupancy.
- E144 tested the opposite resource direction with `BK16`. It lowered resources
  from `113 VGPR / 20480 B LDS` to `70 VGPR / 12288 B LDS`, but pp7488 fell
  `972.77 -> 587.52` because K-loop and barrier cadence doubled. `BK64` was
  rejected statically because Q3 shader LDS would be `36864 B`, above the local
  32 KiB shared-memory budget.
- E146 tested `BM256` as a B/workgroup-reduction route. Static workgroups and
  B reload halve while A-pair dequant stays flat. Runtime resources changed to
  `94 VGPR / 45 SGPR / 31744 B LDS / 0 scratch`, but pp7488 fell
  `972.84 -> 916.62`. Larger M tiles in current `mul_mm.comp` are closed unless
  a new topology also fixes near-limit LDS/occupancy.
- E147 tested the larger Q3_K layout/repack branch analytically. Persistent
  FFN fp16/int8 alternates would add `25.03 GiB` / `9.09 GiB`, so they are not
  viable for the 16 GiB 64k lane. Signed-nibble layout is memory-plausible
  (`+1.12 GiB` for FFN) but low-confidence because current Q3_K SPIR-V is only
  modestly heavier than f16/Q4_K and prior unpack/scale simplifications were
  negative.

### Mat-Vec Route

Code path:

- `ggml_vk_mul_mat_vec_q_f16(...)`
- `ggml_vk_should_use_mmvq(...)`
- `ggml_vk_get_dequantize_mul_mat_vec(...)`

Pipeline families:

- `pipeline_dequant_mul_mat_vec_f32_f32`
- `pipeline_dequant_mul_mat_vec_f16_f32`
- `pipeline_dequant_mul_mat_vec_q8_1_f32`
- `pipeline_mul_mat_vec_p021_f16_f32`
- `pipeline_mul_mat_vec_nc_f16_f32`

Selector notes:

- The route can handle `ne11 == 1` decode-style vectors.
- It can also treat `ne11 <= 8` as a small batch when `ne12*ne13 == 1`.
- For AMD, `ggml_vk_should_use_mmvq(...)` rejects Q8_1 Y-quantization when
  `k < 2048`; for `Q8_0`, it only prefers this mode on older AMD GCN.
- With integer dot and suitable F32 `src1`, the route may quantize Y to Q8_1.
- It supports fusing one or two bias/scale follow-up ops when the graph fusion
  layer marks them as `MUL_MAT_ADD`, `MUL_MAT_ADD_ADD`, etc.

### `MUL_MAT_ID` Routes

Entry point: `ggml_vk_mul_mat_id(...)`.

Route order:

1. If `ids->ne[1] <= 8` and `src0` is F32/F16/quantized, use
   `ggml_vk_mul_mat_vec_id_q_f16(...)`.
2. Otherwise use `ggml_vk_mul_mat_id_q_f16(...)`.

Matrix-ID route details:

- Counts expert usage through `pipeline_count_experts`.
- Uses `ggml_vk_get_mul_mat_mat_id_pipeline(...)` and the same dense/quant/Q8_1
  staging ideas as normal matmul.
- Dispatches `ggml_vk_matmul_id(...)` with `ids` and `expert_count_buf`.

Vector-ID route details:

- Uses `ggml_vk_get_dequantize_mul_mat_vec_id(...)`.
- Can quantize F32 `src1` to Q8_1 under the same integer-dot constraints.
- Supports fusing `ADD_ID` and `MUL` follow-up nodes.

Cleanup note: these routes are separate from dense Qwen3.6, but they matter for
MoE models. Do not delete them for Qwen-only speed work; gate them only through
a local build-profile experiment if MoE support is intentionally dropped.

## FlashAttention Route Map

Entry point: `ggml_vk_flash_attn(...)`, support gate:
`ggml_backend_vk_device_supports_op(... GGML_OP_FLASH_ATTN_EXT ...)`.

Support gate highlights:

- Q must be `F32`; output must be `F32`.
- Mask, when present, must be `F16`.
- Sinks, when present, must be `F32`.
- K/V head sizes must be multiples of 8.
- K/V types: `F32`, `F16`, `Q8_0`, `Q5_1`, `Q5_0`, `Q4_1`, `Q4_0`;
  `Q1_0` is accepted only with coopmat2.
- Mismatched K/V types are accepted only with coopmat2.
- Without coopmat2, scalar/coopmat1 FA needs subgroup shuffle and vote support.

Pipeline families:

| Family | Pipeline fields/shaders | Notes |
| --- | --- | --- |
| Main FA | `pipeline_flash_attn_f32_f16[...]` | `flash_attn*.comp`, `flash_attn_base.glsl`, `flash_attn_mmq_funcs.glsl` |
| Mask optimization | `pipeline_fa_mask_opt` | `flash_attn_mask_opt.comp` |
| Split-K reduce | `pipeline_flash_attn_split_k_reduce` | `flash_attn_split_k_reduce.comp` |

Route knobs:

- `GGML_VK_FA_ROUTE_TRACE` for default-off route diagnostics.
- `GGML_VK_DISABLE_COOPMAT`
- `GGML_VK_DISABLE_COOPMAT2`
- `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`
- `GGML_VK_DISABLE_BFLOAT16`
- `GGML_VK_DISABLE_F16`

Measured Qwen3.6 64k Vulkan route:

- E131 active route: `flash_attn_f32_f16_aligned_f32accq4_0`.
- Path and types: `coopmat1`, `q4_0/q4_0`, `q=f32`.
- Main geometry: `HSK=256`, `HSV=256`, `Br=16`, `Bc=64`,
  `D_split=8`, `row_split=4`, `workgroup_size=256`,
  `subgroup_size=64`.
- Main chunks: `N=1024`, `KV=1024..57344`, `split_k=1`,
  `use_mask_opt=1`; tail chunk: `N=178`, `KV=57600`.
- Negative gates: disabling mask-opt regressed; forced FA f16acc did not beat
  the full 64k best; forced SHMEM staging fell back to scalar FA.
- Driver resource stats for the main route: `98 VGPR`, `76 SGPR`,
  `26112 B LDS`, `0 scratch`.
- E133 shape timing: `FLASH_ATTN_EXT` totals `33965.16 ms`; the largest
  individual tail chunks are `N=1024,KV=57344` (`1168.85 ms`),
  `KV=56320` (`1136.25 ms`), and `KV=55296` (`1122.66 ms`).
  Future FA claims should show the long-KV tail moved and stayed on coopmat1.
- E134 route ceiling says FA alone would need about `1.494x` local speedup to
  match the ROCm 64k wall. Treat FA as a co-primary branch, but do not spend it
  on another simple `Bc`/mask/f16acc toggle.
- E138 forced the existing split-k/reduce path from `KV>=8192`; it routed to
  `split_k=2` but regressed prompt eval from `666.87` to `96.29 tok/s` because
  the route adds temp writes, a sync boundary, and a reduce dispatch per FA
  node. Do not repeat split-k forcing without redesigning the reduce topology.
- E141 used KV dtype as an upper-bound route gate. f16/f16 improved pp7488 only
  `970.03 -> 996.00 tok/s` (`+2.68%`) and failed real 64k server fit
  (`16183 MiB` projected Vulkan device use vs `15221 MiB` free). q8_0/q8_0
  regressed to `940.03 tok/s`. Keep q4/q4 for 64k and optimize the current
  single-dispatch q4 coopmat1 shader directly; do not build an f16 KV
  staging/cache route unless a future design also solves residency.
- E142 tested a larger query-row route, `Br32/Bc32`, to reuse each long-KV pass
  across twice as many rows while staying under LDS limits. The route stayed on
  coopmat1 q4/q4 but regressed pp7488 `971.09 -> 896.97` and raised resources
  to `133 VGPR / 83 SGPR / 27136 B LDS`; a f16acc companion still measured only
  `922.22` with `134 VGPR`. Do not pursue larger-`Br` cm1 without reducing
  per-row live state.
- E145 tested the remaining simple cm1 split knob, `D_split=4/16`, while
  keeping `Br16/Bc64,row_split=4` and q4/q4. Both stayed on the same pipeline
  and reported the same `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch`, but
  regressed pp7488 `978.88 -> 953.24/951.54`. Do not repeat `D_split` retunes;
  the next FA route needs a shader-body or long-KV-tail redesign.

Cleanup note: FlashAttention is one of the highest-value comparison surfaces
between ROCm and Vulkan. If compile pressure becomes a problem, prefer a
reduced-shader profile rather than deleting shaders.

## Graph Fusion Map

Fusion is selected in `ggml_backend_vk_graph_compute(...)` before
`ggml_vk_build_graph(...)`.

| Fusion name in code | Pattern | Launch route |
| --- | --- | --- |
| `MULTI_ADD` | chains of `ADD` | `ggml_vk_multi_add(...)` |
| `MUL_MAT_ADD` | `MUL_MAT` + `ADD` | matvec/matmul route with bias descriptor when possible |
| `MUL_MAT_ADD_ADD` | `MUL_MAT` + `ADD` + `ADD` | matvec route can pass two bias descriptors |
| `MUL_MAT_ID_ADD_ID` | `MUL_MAT_ID` + `ADD_ID` | ID matvec/matmul fused route |
| `MUL_MAT_ID_MUL` | `MUL_MAT_ID` + `MUL` | ID matvec scale fusion |
| `MUL_MAT_ID_ADD_ID_MUL` | expert matmul + bias + scale | ID fused route |
| `RMS_NORM_MUL` | `RMS_NORM` + `MUL` | `ggml_vk_rms_norm(...)` fused variant |
| `RMS_NORM_MUL_ROPE` | norm + scale + rope | RMS/rope fused pipeline when shape gates pass |
| `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` | norm + scale + rope + view + KV write | largest KV-update fusion route |
| `ROPE_VIEW_SET_ROWS` | rope + view + set rows | rope/set-rows fused route |
| `TOPK_MOE_*` | several softmax/sigmoid/argsort/getrows MoE patterns | `ggml_vk_topk_moe(...)` |

Known gap for H38: Vulkan does not currently have a dense FFN
`MUL_MAT + MUL_MAT + GLU` prefill fusion route. CUDA/ROCm has a related
graph matcher/executor for `mul_mat_vec` decode, but it is limited to
`ncols_dst=1`. E134 says a launch/post-op-only port is not enough; a useful
Vulkan branch must reuse the B/activation tile or change the Q3_K layout for
the `m=17408,n=1024,k=5120` gate/up route. E135 adds default-off
`GGML_VK_FFN_ROUTE_TRACE=1` and proves the target graph exposes this branch on
the real 64k server lane: `63 x q3_K SWIGLU` prefill candidates per graph.

Fusion can be disabled globally with `GGML_VK_DISABLE_FUSION` or partly with
`GGML_VK_DISABLE_MULTI_ADD`. Any route timing must state whether fusion was
enabled.

## All-Op Route Atlas

| Route family | GGML ops | Main shader/code area | Support notes | Cleanup note |
| --- | --- | --- | --- | --- |
| Metadata/no-op | `NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE` | graph switch only | Accepted as graph-owned nodes, no compute shader | Keep for scheduler/graph consistency |
| Copy/layout | `CPY`, `CONT`, `DUP` | `ggml_vk_cpy(...)`, `copy*.comp`, `contig_copy.comp`, `copy_transpose.comp`, `copy_to_quant.comp`, `copy_from_quant.comp` | Supports dense copies, F32<->selected quant copies, same-type contiguous quant copies | Keep; required by prealloc/staging routes |
| Row gather/scatter | `GET_ROWS`, `SET_ROWS` | `get_rows*.comp`, `set_rows.comp` | Supports dense, many quant types, and I32 ids; Vulkan supports more quant get-rows types than HIP in places | Keep for embeddings/KV/model coverage |
| Binary/broadcast | `ADD`, `SUB`, `MUL`, `DIV`, `ADD1`, `ADD_ID`, `REPEAT`, `REPEAT_BACK`, `ACC`, `SET`, `CONCAT` | `add*.comp`, `sub.comp`, `mul.comp`, `div.comp`, `add_id.comp`, `repeat*.comp`, `acc.comp`, `concat.comp`, `multi_add.comp` | Mostly F32/F16 gates; `ADD_ID` is F32/F32/I32 | Keep; fusion inputs and common graph surface |
| Unary/activation | `UNARY`, `GLU`, `SILU_BACK`, `LEAKY_RELU`, `SQR`, `SQRT`, `SIN`, `COS`, `LOG`, `CLAMP`, `SCALE`, `FILL`, `ARANGE` | `generic_unary_head.glsl`, `gelu*.comp`, `silu.comp`, `swiglu*.comp`, `glu*.glsl`, `scale.comp`, etc. | F32/F16 for most unary/GLU, F32-only for several scalar math ops | Keep for modern model coverage |
| Norms | `NORM`, `RMS_NORM`, `RMS_NORM_BACK`, `GROUP_NORM`, `L2_NORM` | `norm.comp`, `rms_norm*.comp`, `group_norm.comp`, `l2_norm.comp` | Contiguous gates for norm/group/l2; RMS has fusion paths | Keep; decode traces often include RMS_NORM hotspots |
| Attention/position | `ROPE`, `ROPE_BACK`, `FLASH_ATTN_EXT`, `DIAG_MASK_INF`, `SOFT_MAX`, `SOFT_MAX_BACK` | `rope*.comp`, `flash_attn*.comp`, `diag_mask_inf.comp`, `soft_max*.comp` | FA has strict type/device gates; softmax is F32 with optional F16/F32 mask | Keep; core transformer surface |
| Matmul/expert | `MUL_MAT`, `MUL_MAT_ID` | `mul_mm*.comp`, `mul_mmq*.glsl`, `mul_mat_vec*.comp`, `count_experts.comp` | Detailed above | Keep |
| Reduction/sort | `SUM`, `SUM_ROWS`, `MEAN`, `CUMSUM`, `ARGMAX`, `COUNT_EQUAL`, `ARGSORT`, `TOP_K` | `sum_rows.comp`, `cumsum*.comp`, `argmax.comp`, `count_equal.comp`, `argsort*.comp`, `topk*.comp` | Argsort large requires Vulkan memory model; top-k has power-of-two pipeline availability gates | Keep for sampling/MoE; some training-only reductions may be profile-gated |
| Conv/image | `IM2COL`, `IM2COL_3D`, `CONV_2D`, `CONV_2D_DW`, `CONV_TRANSPOSE_1D`, `CONV_TRANSPOSE_2D`, `POOL_2D`, `PAD`, `ROLL`, `UPSCALE`, `TRI`, `DIAG` | `im2col*.comp`, `conv*.comp`, `pool2d.comp`, `pad.comp`, `roll.comp`, `upscale.comp`, `tri.comp`, `diag.comp` | Useful for multimodal/image/audio models, not Qwen text-only hot path | Candidate for an inference-text-only profile, not deletion by default |
| State-space/RWKV | `RWKV_WKV6`, `RWKV_WKV7`, `GATED_DELTA_NET`, `SSM_SCAN`, `SSM_CONV` | `wkv*.comp`, `gated_delta_net.comp`, `ssm_scan.comp`, `ssm_conv.comp` | GDN requires F32 and `S_v` in 32/64/128; SSM_SCAN supports Mamba2-like shapes with subgroup support | Keep for model coverage; profile-gate only if explicitly out of scope |
| Optimizer/math utilities | `OPT_STEP_ADAMW`, `OPT_STEP_SGD`, `SOLVE_TRI`, `TIMESTEP_EMBEDDING`, `ARGMAX`, `COUNT_EQUAL` | `opt_step*.comp`, `solve_tri.comp`, `timestep_embedding.comp` | Mostly F32 utility/training surface | Good first candidates for local inference-only build-profile study |

## Shader Source Families

| Shader family | Files | Route purpose |
| --- | --- | --- |
| Matmul matrix | `mul_mm*.comp`, `mul_mmq.comp`, `mul_mm_funcs.glsl`, `mul_mmq_funcs.glsl`, `mul_mmq_shmem_types.glsl` | Dense/quant matrix multiply |
| Matmul vector | `mul_mat_vec*.comp`, `mul_mat_vecq.comp`, `mul_mat_vecq_funcs.glsl`, `mul_mat_vec_base.glsl`, `mul_mat_vec_iface.glsl` | Decode/small-column matvec |
| Quant/dequant | `dequant_*.comp`, `dequant_funcs*.glsl`, `dequant_head.glsl`, `quantize_q8_1.comp` | Explicit type conversion and staging |
| FlashAttention | `flash_attn*.comp`, `flash_attn_base.glsl`, `flash_attn_mmq_funcs.glsl` | FA scalar/coopmat/quant K/V routes |
| Common ops | `generic*.glsl`, unary/binary `.comp` files | Elementwise/broadcast/math surface |
| MoE/top-k | `topk*.comp`, `topk_moe.comp`, `count_experts.comp`, `argsort*.comp` | Sampling and expert gating |
| State models | `ssm_*.comp`, `wkv*.comp`, `gated_delta_net.comp` | Mamba/RWKV/GDN support |
| Conv/image | `conv*.comp`, `im2col*.comp`, `pool2d.comp` | Non-text and multimodal support |

## Pruning Candidates

Do not delete now:

- Core `ggml-vulkan.cpp` graph/device/buffer code.
- Any matmul, matvec, quant/dequant, copy, RoPE, softmax, norm, or FA shader.
- Shader generator files. Generated pipeline names and host code assume the
  source/generator contract remains intact.

Possible future build-profile gates:

- Optimizer/loss/training utilities for an inference-only binary.
- Conv/image/timestep routes for a text-only Qwen profile.
- RWKV/Mamba/GDN routes if those model families are explicitly out of scope.
- MoE routes only if MoE models are explicitly dropped.
- Debug/check/perf logger compile routes for release builds.

Recommended cleanup method:

1. Add a named CMake option/profile.
2. Exclude a route family at build time.
3. Configure and build Vulkan.
4. Run representative GUI launch and fallback model smoke tests.
5. Only then consider source-level cleanup, and only if upstream sync cost is
   acceptable.

## Quick Probe Recipes

Matmul route topology:

```bash
GGML_VK_MATMUL_ROUTE_TRACE=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

FlashAttention route topology:

```bash
GGML_VK_FA_ROUTE_TRACE=1 \
python scripts/repo_snapshot_context_bench.py --label-prefix <label> ...
```

Dense FFN gate/up route topology:

```bash
GGML_VK_FFN_ROUTE_TRACE=1 \
python scripts/repo_snapshot_context_bench.py --label-prefix <label> ...
```

Vulkan timing:

```bash
GGML_VK_PERF_LOGGER=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

Fusion-disabled control:

```bash
GGML_VK_DISABLE_FUSION=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

MMVQ-style selector controls:

```bash
GGML_VK_DISABLE_MMVQ=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

```bash
GGML_VK_FORCE_MMVQ=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

These probes can change synchronization, memory placement, or selector behavior.
Use them as diagnostics, not as defaults, unless a same-lane A/B proves the
candidate.
