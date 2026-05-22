# ROCm Route Map - RX 9070 XT / Qwen3.6

This document maps the current ROCm/HIP backend routes in this fork, with a
deep focus on `MUL_MAT`, the measured speed/latency evidence behind each hot
route, and conservative cleanup candidates. It is a research map, not a
deletion plan. Source deletion should be done only after a separate build-option
or pruning experiment proves that the route is unused for the intended local
profile.

## Scope

Local target:

- OS/GPU: Windows 11, AMD Radeon RX 9070 XT, RDNA4, `gfx1201`.
- Backend: ROCm/HIP SDK 7.1 through the shared `ggml/src/ggml-cuda/` backend.
- Main model lane: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Current cold-first prompt-heavy lane: `ctx=12288`, `batch=6144`,
  `ubatch=2048`, KV `q4_0/q4_0`, `spec=none`, no reuse, thinking on.
- Important adjacent lane: Q4_K_S prompt-heavy, where RDNA4 Q4_K/Q5_K MMQ was
  promoted by E070.

Terminology note: upstream names this backend `cuda`; under ROCm it is compiled
with HIP and routes through rocBLAS/hipBLAS where the code says cuBLAS.

Companion backend map: `docs/research/VULKAN_ROUTE_MAP.md`.
Top-level Qwen TPS atlas: `docs/research/QWEN_TPS_ROUTE_ATLAS.md`.
Metrics and bottleneck companion: `docs/research/ROUTE_METRICS_AND_GAPS.md`.

## Executive Summary

Current route facts for the local Qwen/RDNA4 environment:

| Route family | Current local role | Keep/prune verdict |
| --- | --- | --- |
| `cublas_backend` / hipBLAS | Required for large Q3_K prefill when RDNA4 MMQ selector rejects `ne11 > 192`; stages Q3_K weights to fp16, converts `src1`, runs `cublasGemmEx`/rocBLAS with 32f compute on RDNA4 | Keep. It is the active large Q3_K prefill route and still beats existing MMQ for large Q3_K shapes |
| `mul_mat_q` / MMQ | Required for Q3_K decode/medium batches, Q4_K/Q5_K prompt path, MoE/expert routes, and quantized fallback avoidance | Keep. E015 and E070 are real wins; only large-Q3_K selector overrides are rejected |
| `mul_mat_vec_q` / MMVQ | Required for quantized decode/small batch and `MUL_MAT_ID` small expert work; includes local Qwen-hot RDNA4 small-k behavior | Keep. E151 restores and confirms RDNA4 `Q3_K/ncols_dst=1` `nwarps=2` on the current decode lane |
| `mul_mat_f` / MMF | Dense small matrix path for F32/F16/BF16 when alignment and size gates pass | Keep. Needed as generic dense small route; force-wide probes were rejected, not the route itself |
| `mul_mat_vec_f` / MMVF | Dense vector/small batch route for F32/F16/BF16 | Keep. Used for dense skinny shapes and avoids hipBLAS overhead in narrow cases |
| `batched_cublas` | Dense KQ/KQV multi-batch route when FlashAttention is not handling the case | Keep for general backend compatibility |
| Negative Q3_K cache/MMQ prototypes | E104 persistent fp16 cache and E105 existing-MMQ override regressed | Do not keep. They were reverted and should remain out unless a new design changes the mechanism |

The main future route opportunity is not another selector toggle. E049/E054/E103
show that large Q3_K prefill repeatedly pays real Q3_K -> fp16 conversion cost,
but E104 shows persistent fp16 residency hurts more than it saves, and E105 shows
the existing MMQ kernel loses on the large Q3_K prompt shapes. The credible H35
route is a shape-specific fused Q3_K x F16 RDNA4 GEMM that avoids persistent
fp16 residency and competes directly with the current hipBLAS route.

H39 adds a separate decode-parity focus against Vulkan. Do not conflate it with
the large-Q3_K prefill route above: E116 shows the short-decode ROCm q4 route at
about `29.625 tok/s` decode eval while Vulkan q4/f16 reaches about
`40.9-41.2 tok/s`. E149 audits the gap and rejects the simplified explanation
that ROCm lacks fusion; this backend already has RMS, rope/set-rows, unary, SSM,
FFN decode fusions and HIP graph capture. The fresh E149 non-sync ROCm trace is
matmul-dominated (`MUL_MAT forward 77.84%`) and leaves norm/rope/set-rows at
only about `3.26%`, with Q3_K direct route counts as the first target
(`mul_mat_vec_q_direct,q3_K 929`, `mul_mat_q_direct,q3_K 349`). The E149 sync
companion moderates the exact share but confirms the same priority:
`MUL_MAT forward+fused 53.80%` versus `RMS_NORM+ROPE+SET_ROWS 11.83%` after the
initial section is excluded. The first decode-only shape-delta table narrows the
target further: ROCm Q3_K matvec time is split between `mul_mat_vec_q_fused`
(`63.78%`) and `mul_mat_vec_q_direct` (`36.22%`), led by FFN shapes
`m=17408,n=1,k=5120` and `m=5120,n=1,k=17408`. The Vulkan q4 perf comparator is
also Q3-led (`MUL_MAT_VEC q3_K 50.67%` plus `MUL_MAT_ADD_VEC q3_K 19.38%`)
while `ROPE+SET_ROWS` is only `0.60%`. E150 then rejects disabling ROCm fusion
(`30.08 -> 28.61 tok/s` decode), so the route is useful but still the main
optimization target. E151 restores RDNA4 `Q3_K/ncols_dst=1` `nwarps=2` and
moves the same short-decode gate from clean post-rebuild `29.77 tok/s` to
`32.2467 tok/s` (`+8.32%` decode). This is a real first H39 win, but still
leaves about a `1.27x` gap to the E116 Vulkan q4 comparator. Next H39 work
E152 confirms the residual Q3_K split remains fused/direct dominated after the
first win (`64.62%` fused, `35.38%` direct of parsed Q3_K MMVQ time) and fixes
the route-delta parser so MMVQ `grid.x` is normalized back to logical rows when
`rows_per_block=2`. Next H39 work should design a larger Q3_K-specific MMVQ
branch if the residual fused/direct split still supports it, not a standalone
fusion port or fusion removal.

## Build-Time Route Map

ROCm is not a standalone source backend in this tree. It is a HIP build of the
shared `ggml/src/ggml-cuda/` sources:

| Layer | Files/options | Route role |
| --- | --- | --- |
| User option | `GGML_HIP` in `ggml/CMakeLists.txt` | Enables the HIP backend instead of pure CUDA |
| Windows compiler gate | `ggml/src/ggml-hip/CMakeLists.txt` | Requires ROCm `clang++` or `hipcc`; this is why local ROCm config must use Ninja/ROCm clang, not a Visual Studio generator |
| ROCm packages | `hip`, `hipblas`, `rocblas`, optional `rccl` | Provides runtime, BLAS route, and optional collectives |
| Source collection | `ggml/src/ggml-hip/hip-source-bundles.cmake` | Pulls `../ggml-cuda/*.cu` and selected template instances into `ggml-hip` |
| Backend defines | `GGML_USE_HIP`, `GGML_USE_CUDA`, `GGML_HIP_GRAPHS`, `GGML_HIP_NO_VMM`, `GGML_CUDA_NO_FA`, `GGML_HIP_MMVQ_FOCUSED_PROFILE` | Chooses HIP aliases, graph behavior, VMM path, FlashAttention inclusion, and local reduced profiles |

HIP source profiles:

| Profile | What builds | Intended use |
| --- | --- | --- |
| `default` | All `ggml-cuda/*.cu`, MMQ/MMF instances, FlashAttention tile/MMA instances, selected FA vec instances unless `GGML_CUDA_FA_ALL_QUANTS=ON` | Normal ROCm backend |
| `qwen-fa-reduced` | Replaces the normal FA driver with `fattn-qwen-reduced.cpp`, removes the broad `fattn.cu`/`fattn-tile.cu` source path, keeps non-disabled FA support | Local Qwen/RDNA4 build-pressure and FA route A/B |
| `mmvq-focused` / `mmvq-isolated` | Reduced profile plus `GGML_CUDA_NO_FA`; excludes FA vec/tile/WMMA source pressure and defines `GGML_HIP_MMVQ_FOCUSED_PROFILE` | MMVQ/decode-path build and route isolation |

Cleanup implication: the existing profiles are the right place to test "lighter
ROCm-only" builds. Source deletion from `ggml-cuda/` should be last resort
because that directory is shared by CUDA, HIP, MUSA-style aliases, template
instances, and upstream sync.

## Backend Life-Cycle Map

| Phase | Entry points | Notes for route work |
| --- | --- | --- |
| Registration | `ggml_backend_cuda_reg()`, `ggml_backend_cuda_init()` in `ggml-cuda.cu` | HIP still registers through CUDA-named symbols behind `GGML_USE_HIP` |
| Device props | `ggml_cuda_info()`, `ggml_backend_cuda_device_get_props()` | RDNA4 behavior is mostly inferred through `cc`, HIP feature flags, and local selector code |
| Buffers | `ggml_backend_cuda_buffer_type()`, `ggml_backend_cuda_split_buffer_type()`, `ggml_backend_cuda_host_buffer_type()` | Normal GPU buffers, row-split weight buffers, and pinned/host buffers are distinct routes |
| Tensor upload/download | `set_tensor_async`, `get_tensor_async`, `set/get_tensor_2d_async` | Uses direct device copies for GPU buffers; split buffers walk devices/ranges |
| Device-to-device copy | `ggml_backend_cuda_cpy_tensor_async()` | Same-backend copy route, with peer-copy behavior controlled by compile options |
| Graph execution | `ggml_backend_cuda_graph_compute()` -> `ggml_cuda_graph_evaluate_and_capture()` -> `ggml_cuda_compute_forward()` | The op switch is only one layer; graph capture, update checks, and fusions can change what actually launches |
| Synchronization | `ggml_backend_cuda_synchronize()`, events | Required when adding timing traces; sync traces must not be used as speed claims |

Graph/fusion route layer:

| Fusion/optimization | Entry points | Route effect |
| --- | --- | --- |
| CUDA/HIP graphs | `GGML_HIP_GRAPHS`, `ggml_cuda_graph_check_compability()`, `ggml_cuda_graph_update_required()` | Captures compatible graphs and replays them until shapes/pointers require update |
| TopK MoE fusion | `ggml_cuda_topk_moe_fusion()`, `ggml_cuda_op_topk_moe()` | Replaces multi-op gating subgraphs with a fused MoE route |
| Rope/set-rows fusion | `ggml_cuda_should_fuse_rope_set_rows()`, `ggml_cuda_op_rope_fused()` | Avoids a separate set-rows/write path after rope when layout gates pass |
| Elementwise chains | `ggml_cuda_op_fused_add()`, `ggml_cuda_op_fused_mul()` | Fuses repeated add/mul broadcast work |
| RMS norm fusions | `ggml_cuda_op_rms_norm_fused()`, `ggml_cuda_op_rms_norm_fused_add()` | Handles common norm+mul/add layouts without separate kernels |
| SSM/unary companions | `ggml_cuda_op_ssm_conv(..., bias_add_node, silu_dst)`, `ggml_cuda_op_unary_mul()`, `ggml_cuda_op_relu_sqr()`, `ggml_cuda_op_softcap()` | Local graph layer can redirect a supported op into a fused helper instead of the plain switch route |

## Main Dispatch Map

Code entry point: `ggml_cuda_mul_mat(...)` in `ggml/src/ggml-cuda/ggml-cuda.cu`.

Route selection is ordered, not independent:

1. Establish candidate booleans by tensor type, dst type, `src1` type, split
   buffer status, and padding safety:
   - `use_mul_mat_vec_f`: dense F32/F16/BF16 x F32 -> F32.
   - `use_mul_mat_f`: dense matrix route, not quantized.
   - `use_mul_mat_vec_q`: quantized x F32 -> F32, `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`.
   - `use_mul_mat_q`: quantized x F32 -> F32.
2. Refine those booleans with per-device selector functions:
   - `ggml_cuda_should_use_mmq(...)` in `mmq.cu`.
   - `ggml_cuda_should_use_mmf(...)` in `mmf.cu`.
   - `ggml_cuda_should_use_mmvf(...)` in `mmvf.cu`.
3. Dispatch in priority order:
   - direct `mul_mat_vec_f` if non-split and selected;
   - direct `mul_mat_f` if non-split and selected;
   - direct `mul_mat_vec_q` if non-split and selected;
   - direct `mul_mat_q` if non-split and selected;
   - dense `batched_cublas` for multi-batch dense cases;
   - backend/split wrappers for selected vector/MMQ routes;
   - final fallback `cublas_backend`.

Diagnostic route trace:

- `GGML_TRACE_CUDA_MUL_MAT_ROUTE=1` logs the selected route and the selector
  booleans for normal `MUL_MAT`.
- `GGML_TRACE_MUL_MAT_ID_ROUTE=1` or `GGML_TRACE_CUDA_MUL_MAT_ROUTE=1` logs the
  `MUL_MAT_ID` route.

## All-Op Route Atlas

This table maps the non-`MUL_MAT` operations that the ROCm/HIP backend can
claim through `ggml_backend_cuda_device_supports_op(...)` and launch through
`ggml_cuda_compute_forward(...)`. It is intentionally route-oriented: use it to
find where an op enters the backend, where its kernel family lives, and whether
it is likely relevant to Qwen/RDNA4 performance work.

| Route family | GGML ops | Main files | Support/shape notes | Qwen/RDNA4 cleanup note |
| --- | --- | --- | --- | --- |
| Metadata/no-op | `NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE` | `ggml-cuda.cu` switch only | Accepted as backend-owned graph nodes but no kernel launch | Keep; removing would break graph ownership/scheduler behavior |
| Copy/layout | `DUP`, `CPY`, `CONT` | `cpy.cu`, `cpy-utils.cuh`, `convert.cu`, `convert.cuh`, `dequantize.cuh` | Dense F32/F16/BF16 copies, F32<->Q4/Q5/Q8/IQ4_NL/I32 conversions, same-type contiguous copies | Keep; these are staging routes for almost every fallback and pruning experiment |
| Row gather/scatter | `GET_ROWS`, `GET_ROWS_BACK`, `SET_ROWS` | `getrows.cu`, `set-rows.cu` | `GET_ROWS` supports F16/F32/BF16/I32 plus classic Q1/Q4/Q5/Q8; `SET_ROWS` also covers TurboQuant/KV types in this fork | Keep for embeddings, KV cache edits, GUI/TurboQuant paths |
| Binary/broadcast | `ADD`, `ADD1`, `SUB`, `MUL`, `DIV`, `REPEAT`, `REPEAT_BACK` | `binbcast.cu`, `add-id.cu` | Generic broadcast kernels plus fused add/mul variants in graph layer | Keep; common C01 trace nodes and fusion inputs |
| Acc/set/concat/shape | `ACC`, `SET`, `CONCAT`, `PAD`, `PAD_REFLECT_1D`, `ROLL`, `ARANGE`, `FILL`, `UPSCALE` | `acc.cu`, `set.cu`, `concat.cu`, `pad.cu`, `pad_reflect_1d.cu`, `roll.cu`, `arange.cu`, `fill.cu`, `upscale.cu` | Mostly F32/F16/BF16 or simple contiguous gates, with some I32 support | Not a prefill bottleneck; prune only in a strict inference-only build profile |
| Unary activations | `UNARY`, `SILU_BACK`, `LEAKY_RELU`, `SQR`, `SQRT`, `SIN`, `COS`, `CLAMP`, `LOG` | `unary.cu`, `clamp.cu`, `scale.cu`, `softcap.cu` | Contiguous source gates for `UNARY`; includes GELU/SILU/TANH/EXP/ELU/XIELU/etc. | Keep; GLU/activation routes are common in modern architectures |
| GLU/gated activations | `GLU` variants | `unary.cu` | Supports `REGLU`, `GEGLU`, `SWIGLU`, `SWIGLU_OAI`, `GEGLU_ERF`, `GEGLU_QUICK` with contiguous-1 input | Keep; relevant to feed-forward route experiments |
| Norms | `NORM`, `RMS_NORM`, `RMS_NORM_BACK`, `GROUP_NORM`, `L2_NORM` | `norm.cu` | Norms are broadly supported; backprop route requires contiguous input | Keep; C01 traces repeatedly flag RMS_NORM as a small but real node family |
| Reductions | `SUM`, `SUM_ROWS`, `MEAN`, `CUMSUM`, `ARGMAX`, `COUNT_EQUAL` | `sum.cu`, `sumrows.cu`, `mean.cu`, `cumsum.cu`, `argmax.cu`, `count-equal.cu` | `SUM`/`SUM_ROWS`/`MEAN` prefer contiguous rows; `COUNT_EQUAL` is I32 compare | Mostly auxiliary; keep unless a server-inference-only build proves unused |
| Sorting/top-k | `TOP_K`, `ARGSORT` | `top-k.cu`, `argsort.cu`, `topk-moe.cu` | Without CUB, `TOP_K`/`ARGSORT` gate to small row widths; with CUB broader | Keep for sampling/MoE; topk-moe fusion depends on this area |
| RoPE/KV edits | `ROPE`, `ROPE_BACK`, fused rope+set-rows | `rope.cu`, `set-rows.cu` | Requires contiguous-2/row-compatible layout; fused path handled before plain switch | Keep; decode route and KV cache integration depend on it |
| Attention | `FLASH_ATTN_EXT` | `fattn.cu`, `fattn-*.cuh`, `fattn-wmma-f16.cu`, `fattn-qwen-reduced.cpp`, template instances | Gate is `ggml_cuda_flash_attn_ext_supported(...)`; HIP profile can reduce or disable this source area | Candidate for build-profile trimming, not source deletion, because Qwen and non-Qwen lanes may diverge |
| Matmul | `MUL_MAT`, `MUL_MAT_ID`, `OUT_PROD` | `ggml-cuda.cu`, `mmq.cu/.cuh`, `mmvq*.cu`, `mmf.cu`, `mmvf.cu`, `mmid.cu`, `out-prod.cu` | Detailed in the following sections; split buffers only legal for `MUL_MAT` | Keep; primary performance search space |
| Convolution/image | `IM2COL`, `IM2COL_3D`, `CONV_2D`, `CONV_2D_DW`, `CONV_TRANSPOSE_1D`, `CONV_TRANSPOSE_2D`, `POOL_2D` | `im2col.cu`, `conv2d.cu`, `conv2d-dw.cu`, `conv-transpose-1d.cu`, `conv2d-transpose.cu`, `pool2d.cu` | Mostly dense F32/F16 gates; useful for multimodal and non-LLM models | Possible future local-minimal prune candidate only if GUI drops these model classes |
| State-space/RWKV | `SSM_CONV`, `SSM_SCAN`, `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET` | `ssm-conv.cu`, `ssm-scan.cu`, `wkv.cu`, `gla.cu`, `gated_delta_net.cu` | SSM gates include strict state/head shape constraints; GDN is disabled for MUSA but active for HIP | Keep if the fork wants broad model coverage; possible non-Qwen build-profile gate |
| Loss/optimizer | `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_BACK`, `OPT_STEP_ADAMW`, `OPT_STEP_SGD`, `SOLVE_TRI`, `TRI`, `DIAG`, `DIAG_MASK_INF` | `cross-entropy-loss.cu`, `opt-step-*.cu`, `solve_tri.cu`, `tri.cu`, `diag.cu`, `diagmask.cu` | Training/fine-tuning/math utility surface | Least relevant to local inference speed; safest candidates for an inference-only profile experiment |
| TurboQuant/local | `TURBO_WHT`, `TQ3_0`, `TKV2_0`, `TKV3_0`, `TKV4_0` type paths | `turbo-wht.cu`, `set-rows.cu`, `mmq.cu/.cuh`, `mmvq*.cu`, FA vec instances | `TURBO_WHT` requires F32 contiguous; TQ/KV types appear in set rows and selected quant routes | Do not prune in this fork without a separate TurboQuant compatibility decision |

Support and launch are separate layers. A route can be listed in
`supports_op(...)` but still dispatch through a fused graph path, a BLAS
fallback, or a selector-specific kernel variant at runtime.

## `cublas_backend` / hipBLAS Route

Code path:

- Dispatch wrapper: `ggml_cuda_op_mul_mat(...)` in `ggml-cuda.cu`.
- Per-split implementation: `ggml_cuda_op_mul_mat_cublas(...)`.

Mechanics:

- BF16 path: if hardware supports BF16 and `src0` is BF16, convert `src1` to
  BF16 when needed, run `cublasGemmEx(... CUDA_R_16BF ..., CUBLAS_COMPUTE_32F)`,
  convert dst to F32.
- FP16 path: if fast fp16 hardware is available and `src0` is F16 or quantized,
  stage quantized `src0` to fp16, convert `src1` to fp16 when needed, then run
  `cublasGemmEx`. On RDNA4 the default compute path is 32f unless explicitly
  forced to 16f.
- F32 path: convert inputs to F32 when needed and run `cublasSgemm`.

Large Q3_K prefill route:

- On RDNA4, `ggml_cuda_should_use_mmq(GGML_TYPE_Q3_K, ..., ne11, ...)` returns
  true only up to `ne11 <= 192` for non-expert work.
- Therefore Q3_K prompt shapes like `ncols=2048` fall to `cublas_backend`.
- This route dequantizes/stages Q3_K `src0` to fp16 before hipBLAS.

Key evidence:

| Experiment | Finding | Decision |
| --- | --- | --- |
| E046 | Forcing 16f cublas compute regressed `11.7908 -> 11.4146 TPS` (`-3.19%`) | Keep RDNA4 32f compute default |
| E048 | `ROCBLAS_USE_HIPBLASLT=1` was noise: `11.5443 -> 11.5557 TPS` (`+0.10%`) | Do not promote hipBLASLt for this lane |
| E049 | Q3_K split timing: `src0 32.29%`, `src1 6.74%`, `GEMM 60.97%`; target `6144x5120@ncols2048` had `src0` at `78.23%` of local split time | Hotspot is Q3_K staging for the target shape, not generic GEMM toggles |
| E054 | Q3_K `src0_alloc_ms=6.12 ms`, `src0_convert_ms=3370.32 ms`; target convert `1430.88 ms` | Allocation/pool reuse is not the bottleneck; conversion/store is |
| E103 | `2792` Q3_K route rows, `349` unique tensor/range keys, all repeated `8` times; unlimited fp16 cache would save `2852.549 ms` convert but need `42.002 GiB` | Reuse is real, but full fp16 cache is not practical |
| E104 | Persistent fp16 cache regressed: baseline `11.74 TPS`, full `attn_gate` cache `9.56 TPS`, 480 MiB cache `11.59 TPS` | Do not keep persistent fp16 cache |
| E105 | Existing-MMQ route override regressed/tied: baseline `11.74 TPS`, candidates `11.54`, `11.68`, `11.44 TPS` | Do not route large Q3_K prefill through current MMQ |

Diagnostics:

- `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`: stage timing rows.
- `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`: adds alloc vs convert timing for fp16 path.
- `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=<n>`: filters small/decode rows.
- `GGML_TRACE_CUBLAS_Q3K_ROUTE=1`: Q3_K staging reuse/key trace.
- `GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=<n>`: prompt-shape filter.
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` and
  `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`: route experiments only; clear before
  speed claims.

Cleanup notes:

- Keep split/detail/Q3K trace if H35 remains active; they are default-off and
  are the only cheap way to prove whether a new fused route is acting on the
  intended shape.
- Do not reintroduce E104 cache code without a new mechanism that avoids the
  residency penalty.
- Do not reintroduce E105 selector override to existing MMQ for large Q3_K.

## MMQ Route (`mul_mat_q`)

Code path:

- Selector and op wrappers: `ggml/src/ggml-cuda/mmq.cu`.
- Kernel templates and resource/timing diagnostics: `ggml/src/ggml-cuda/mmq.cuh`.
- `src1` staging: `quantize_mmq_q8_1_cuda(...)` or FP4 staging for Blackwell
  FP4 types.

Mechanics:

- Quantized `src0` remains in its quantized format.
- F32 `src1` is quantized into the MMQ Q8_1 staging layout.
- `mmq_args` carries tensor strides, channel/sample dimensions, optional expert
  ids, and `use_stream_k`.
- `ggml_cuda_mul_mat_q_switch_type(...)` instantiates the matching type kernel
  for Q1/Q4/Q5/Q8/K-quants/IQ types and FP4 types.
- On RDNA4, current MMQ geometry is `mmq_y=64`, `nwarps=4` from E015.

Current RDNA4 selector policy in `ggml_cuda_should_use_mmq(...)`:

| Type group | RDNA4 non-expert threshold |
| --- | ---: |
| `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` | `ne11 <= 256` |
| `Q4_K`, `Q5_K` | `ne11 <= GGML_MMQ_RDNA4_Q4K_MAX_NE11`, default `1024` |
| `Q2_K`, `Q3_K`, `Q6_K` | `ne11 <= 192` |
| Other supported quant types | `ne11 <= 128` |
| `n_experts >= 64` | always true for supported quant types |

Important knobs:

- `GGML_MMQ_RDNA4_Q4K_MAX_NE11=<n>`: rollback/override for the E070 Q4_K/Q5_K
  threshold; `192` restores the old gate.
- `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=<n>`: backend/split MMQ Stream-K threshold
  experiment knob. It is not a default speed claim.
- `GGML_CUDA_FORCE_MMQ_RUNTIME=1`: broad force-MMQ escape hatch. It was rejected
  for the Q3_K lanes; use only as a diagnostic.
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=<8..128 multiple of 8>`: diagnostic force-x
  knob in `mmq.cuh`; E016 rejected force-x points below the selected default.
- `GGML_RDNA4_MOE_MMQ_STAGING=1`: opt-in MoE staging experiment path; not a
  dense Q3_K default.

Key evidence:

| Experiment | Finding | Decision |
| --- | --- | --- |
| E013 | RDNA4 Q3_K decode fast path improved `9.1629 -> 9.3847 TPS` (`+2.42%`) | Keep Q3_K decode policy |
| E015 | RDNA4 `mmq_y=64,nwarps=4` improved `9.3974 -> 9.6080 TPS` (`+2.24%`), target Q3 bucket `9949.928 -> 9551.391 ms` | Keep RDNA4 MMQ geometry |
| E016 | Force-x points after E015 all below reference (`x64=9.02`, `x80=8.20`, `x112=9.06`, `x128=8.77` vs `9.6080`) | Keep default selector, do not force-x |
| E050 | Large Q3_K forced-MMQ target shape was `37.52%` slower than cublas split; broad forced route `10.05 TPS` | Do not send large Q3_K prefill to current MMQ |
| E070 | Q4_K/Q5_K threshold `ne11<=1024` improved Q4 pp512 `57.30 -> 246.60 tok/s`; full Q4 lane recovered from `122.23s` timeout-scale to `28.44s`, prompt `330.42 tok/s` | Keep Q4_K/Q5_K RDNA4 selector extension |
| E105 | Narrow Q3_K existing-MMQ override did not beat baseline (`11.74 -> 11.54/11.68/11.44 TPS`) | Do not add Q3 large-prefill selector override |

Diagnostics:

- `GGML_TRACE_MMQ_TIMING=1`: logs `mul_mat_q_case` timing.
- `GGML_TRACE_MMQ_TIMING_SYNC=1`: synchronizes stream for measured kernel time;
  diagnostic only.
- `GGML_TRACE_MMQ_TIMING_PRE_SYNC=1`: pre-sync companion for queueing effects.
- `GGML_TRACE_MMQ_RESOURCES=1`: adds registers, dynamic shared memory,
  max blocks/SM, occupancy, and waves/SM.

Cleanup notes:

- MMQ is not pruneable for this fork. It is a kept path for Q3 decode/C01 and
  Q4_K/Q5_K prompt work.
- What is pruneable is negative probe surface: do not keep new force-MMQ or
  large-Q3 override code unless it is default-off, documented, and has a fresh
  positive same-lane A/B.
- The diagnostic force knobs could be hidden behind a local debug build option
  later, but source removal would reduce our ability to reproduce old route
  studies.

## MMVQ Route (`mul_mat_vec_q`)

Code path: `ggml/src/ggml-cuda/mmvq.cu`.

Mechanics:

- Used for quantized matrix-vector and very small matrix-batch work where
  `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`.
- Maps quant types to `vec_dot_*_q8_1` functions through `get_vec_dot_q_cuda(...)`.
- Has architecture-specific parameter tables, including `MMVQ_PARAMETERS_RDNA4`.
- Supports `MUL_MAT_ID` small expert routing via per-type max-batch gates.
- Supports fusion for `ncols_dst=1` when gate/bias fusion args are present.

RDNA4 Qwen-hot behavior:

- `ggml_cuda_mmvq_is_qwen_hot_type(...)` returns true for `Q3_K`, `Q4_K`, and
  `Q6_K`.
- For RDNA4, `ncols_dst=1`, and Qwen-hot types, `should_use_small_k(...)`
  defaults to `small_k=true` unless disabled.
- E151 confirms the current-tree policy: RDNA4 `Q3_K/ncols_dst=1` uses
  `nwarps=2`. Because Qwen-hot `small_k` is enabled, this lets
  `calc_rows_per_block(...)` use two rows per block instead of staying at one.

Important knobs:

- `GGML_MMVQ_QWEN_FORCE_SMALL_K=1`: force Qwen-hot small-k.
- `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1`: disable Qwen-hot small-k.
- `GGML_TRACE_MMVQ_SMALL_K=1`: logs small-k decision lines.
- `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`,
  `GGML_TRACE_MMVQ_TIMING_PRE_SYNC=1`: timing diagnostics.

Key evidence:

| Experiment/checkpoint | Finding | Decision |
| --- | --- | --- |
| Early C01 small-k A/B | `26.30 -> 26.66 TPS` in decode trace; route moved to `small_k=1` | Kept as Qwen-hot RDNA4 policy |
| C01 two-task validation | Default small-k `28.02/28.06 TPS`, disabled `27.86 TPS`; trace `26.68` vs `26.46 TPS` | Keep default small-k |
| E013 | Historical Q3_K MMVQ `nwarps=2` note improved paired control `9.1629 -> 9.3847 TPS`; Q3_K `nwarps=4` follow-up regressed `9.3847 -> 9.2136` | Historical prior for the current E151 policy |
| E151 | Current-tree RDNA4 Q3_K `nwarps=2` improves clean post-rebuild r3 `28.1123 -> 30.3145 TPS`, decode `29.77 -> 32.2467 tok/s`; live server sanity output is normal | Keep RDNA4 Q3_K `nwarps=2`; collect post-E151 residual trace before larger changes |
| E152 | Post-E151 sync trace confirms `nwarps=2`, `small_k=1`, `block=(32,2,1)` for Q3_K and residual Q3_K split fused `64.62%` / direct `35.38%` | Use as topology evidence only; sync timing is not a speed claim |

Cleanup notes:

- MMVQ is required for decode and small expert routes. Do not remove it.
- Generic NVIDIA/older-AMD parameter tables are not useful for the local RX 9070
  XT profile, but deleting them from the shared backend would make upstream sync
  painful. If codebase reduction becomes a real goal, prefer a build-time
  `GGML_HIP_RDNA4_ONLY` style exclusion experiment instead of source deletion.

## Dense Routes: MMF, MMVF, Batched hipBLAS

### MMVF (`mul_mat_vec_f`)

Code path: `ggml/src/ggml-cuda/mmvf.cu`.

Current selector highlights:

- Requires dense F32/F16/BF16 `src0`, F32 `src1`, F32 dst.
- Requires even/aligned dimensions and strides.
- On RDNA4 F16 with fp16 MMA available, selected only for `ne11 <= 5`.
- F32 dense vector selection is similarly narrow.

Role:

- Avoids heavy library overhead for dense vector/skinny cases.
- Seen in C01 route splits as a smaller but real `mul_mat_vec_f_direct|f32`
  share.

### MMF (`mul_mat_f`)

Code path: `ggml/src/ggml-cuda/mmf.cu`.

Current selector highlights:

- Rejects quantized types.
- Requires stride/alignment and row-block compatibility.
- Non-`MUL_MAT_ID` route generally rejects `src1_ncols > 16`.
- Supports F32/F16/BF16 only when the architecture has a compatible MMA route.

Evidence:

- E023 RDNA4 F32 GemmEx/route probe regressed `9.6080 -> 9.42 TPS` and target
  avg `0.1712 -> 0.1850 ms`.
- E044-R1 forced F32 MMF route hit a runtime hard timeout and was reverted.

Role:

- Keep as a narrow dense route and generic backend capability.
- Do not force it for wide F32 C01/SSM shapes on this machine without a fresh
  activation proof.

### Batched hipBLAS

Code path: `ggml_cuda_mul_mat_batched_cublas(...)` in `ggml-cuda.cu`.

Selector highlights:

- Non-split dense F16/BF16/F32 cases.
- Both inputs must be non-transposed.
- `src1->ne[2] * src1->ne[3] > 1`.
- Mostly relevant for dense KQ/KQV multi-batch when FlashAttention is not the
  active route.

Cleanup notes:

- These dense routes are not the large Q3_K prefill bottleneck, but they are
  generic fallback routes. Prune only through a dedicated local-minimal build
  profile after confirming no active Qwen/RDNA4 workload touches them.

## `MUL_MAT_ID` / Expert Routes

Code path: `ggml_cuda_mul_mat_id(...)` in `ggml-cuda.cu`, with MMQ/MMVQ support
in `mmq.cu`, `mmq.cuh`, and `mmvq.cu`.

Route order:

1. If `src1` and dst are F32 and `ne2 <= MMVQ_MAX_BATCH_SIZE`:
   - quantized `src0`: use MMVQ if `ne2 <= get_mmvq_mmid_max_batch(type, cc)`;
   - dense AMD `src0`: use MMVF.
2. Else if `ggml_cuda_should_use_mmq(type, cc, ne12, n_experts=ne02)`, use MMQ.
3. Else if dense MMF selector passes, use MMF.
4. Else use sorted/fallback path with stream synchronization.

Evidence:

- E034/E035 non-C01 MoE route scans found that routed expert MMQ is already used
  for some expert shapes, while broad force-MMQ/staging attempts were unstable or
  regressed.
- H13 RDNA4 MoE MMQ staging remains an opt-in research idea, not a dense default.

Cleanup notes:

- Do not remove expert route support if this fork may test MoE/Stormrage-like
  models. It is separate from the dense Qwen3.6 Q3_K prefill route.
- Any future MoE pruning should be a model-support policy decision, not a Qwen
  dense-lane optimization.

## Diagnostic Environment Map

Clear these before speed claims unless they are the explicit candidate:

| Env var | Route area | Use |
| --- | --- | --- |
| `GGML_TRACE_CUDA_MUL_MAT_ROUTE` | generic dispatch | Route topology trace |
| `GGML_TRACE_MUL_MAT_ID_ROUTE` | expert dispatch | `MUL_MAT_ID` route trace |
| `GGML_TRACE_CUBLAS_SPLIT_TIMING` | hipBLAS fallback | Stage timing |
| `GGML_TRACE_CUBLAS_SPLIT_DETAIL` | hipBLAS fallback | Alloc vs convert split |
| `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS` | hipBLAS fallback | Prompt-shape filter |
| `GGML_TRACE_CUBLAS_Q3K_ROUTE` | Q3_K hipBLAS staging | Reuse/key trace |
| `GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS` | Q3_K hipBLAS staging | Prompt-shape filter |
| `GGML_TRACE_MMQ_TIMING` | MMQ | MMQ timing rows |
| `GGML_TRACE_MMQ_TIMING_SYNC` | MMQ | Sync timing, diagnostic only |
| `GGML_TRACE_MMQ_TIMING_PRE_SYNC` | MMQ | Queueing/pre-sync companion |
| `GGML_TRACE_MMQ_RESOURCES` | MMQ | Registers/LDS/occupancy resource fields |
| `GGML_TRACE_MMVQ_SMALL_K` | MMVQ | Qwen-hot small-k decisions |
| `GGML_TRACE_MMVQ_TIMING` | MMVQ | MMVQ timing rows |
| `GGML_TRACE_MMVQ_TIMING_SYNC` | MMVQ | Sync timing, diagnostic only |
| `GGML_CUDA_FORCE_MMQ_RUNTIME` | selector | Broad force-MMQ diagnostic; rejected for Q3_K lane |
| `GGML_MMQ_RDNA4_Q4K_MAX_NE11` | MMQ selector | Q4_K/Q5_K threshold override/rollback |
| `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11` | backend/split MMQ | Stream-K threshold probe |
| `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X` | MMQ kernel selection | Diagnostic force-x probe |
| `GGML_MMVQ_QWEN_FORCE_SMALL_K` | MMVQ | Force Qwen-hot small-k |
| `GGML_MMVQ_QWEN_DISABLE_SMALL_K` | MMVQ | Disable Qwen-hot small-k |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F` | hipBLAS fp16 path | Rejected route experiment |
| `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` | hipBLAS fp16 path | Force 32f compute experiment |
| `ROCBLAS_USE_HIPBLASLT` | rocBLAS/hipBLASLt | Rejected/noise for active Q3_K lanes |
| `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK` | allocator/residency | Negative control for E008; do not use for speed claims |

## Speed/Latency Evidence Table

| ID | Area | Baseline | Candidate / diagnostic | Verdict |
| --- | --- | ---: | ---: | --- |
| E008 | ROCm compute vbuffer residency | `302.87 tok/s` at bad `ctx32768,ub904` single chunk | `1038.19 tok/s` default chunk at `ub904`, `1114.58 tok/s` at `ub1024` | Keep allocator fix; use single chunk only as negative control |
| E013 | MMVQ Q3_K decode | `9.1629 TPS` | `9.3847 TPS` | Historical prior for Q3_K `nwarps=2` |
| E151 | ROCm decode parity / MMVQ Q3_K | `28.1123 TPS`, `29.77 tok/s` decode | `30.3145 TPS`, `32.2467 tok/s` decode | Keep RDNA4 Q3_K `nwarps=2`; real server sanity passed |
| E152 | Post-E151 residual trace | diagnostic | Q3_K fused `64.62%`, direct `35.38%`; parser corrected for `rows_per_block=2` | Route branch remains Q3_K-led |
| E015 | MMQ RDNA4 geometry | `9.3974 TPS` | `9.6080 TPS` | Keep `mmq_y=64,nwarps=4` |
| E045 | Prefill ubatch recenter | `11.4240 TPS` (`ub1024`) | `11.6534 TPS` (`ub2048`) | Use `ub2048` as current prompt-heavy search baseline |
| E046 | cublas compute16 | `11.7908 TPS` | `11.4146 TPS` | Reject compute16 default |
| E048 | hipBLASLt | `11.5443 TPS` | `11.5557 TPS` | Reject as noise |
| E049 | cublas split | diagnostic | Q3_K `src0 32.29%`; target `src0 78.23%` | Use for hotspot selection |
| E050 | large Q3_K current MMQ | cublas target split `1839.27 ms` | broad MMQ target `2529.35 ms` | Reject existing-MMQ large-Q3 route |
| E054 | Q3_K staging detail | diagnostic | `src0_convert_ms=3370.32 ms`, alloc `6.12 ms`; target convert `1430.88 ms` | Conversion/store is target, not allocation |
| E055/E056 | Q3_K half2 store | E053/E056 controls | r1 small apparent gain, r3 `11.6726 -> 11.6375 TPS` | Reject and revert |
| E070 | Q4_K/Q5_K MMQ selector | Q4 pp512 `57.30 tok/s`; full lane `122.23s`, prompt `64.39 tok/s` | pp512 `246.60 tok/s`; full lane `28.44s`, prompt `330.42 tok/s` | Keep Q4_K/Q5_K threshold |
| E103 | Q3_K staging reuse | diagnostic | `2792` rows, `349` keys, all repeated `8x`; full cache footprint `42.002 GiB` | Keep trace, avoid unlimited cache |
| E104 | persistent Q3_K fp16 cache | `11.74 TPS` | `9.56 TPS` full cache, `11.59 TPS` 480 MiB cache | Reject/reverted |
| E105 | existing-MMQ Q3_K override | `11.74 TPS` | `11.54`, `11.68`, `11.44 TPS` | Reject/reverted |

## Cleanup / Pruning Candidates

### Do not delete

- `cublas_backend`: active large Q3_K prefill route and generic fallback.
- MMQ: kept for Q3_K decode/medium batches, Q4_K/Q5_K prompt, MoE/expert paths.
- MMVQ: kept for Qwen-hot decode and small `MUL_MAT_ID` routes.
- MMF/MMVF/batched cublas: generic dense support and fallback safety.
- E049/E054/E103 style default-off diagnostics while H35 remains active.

### Already rejected, should stay out of runtime code

- Persistent Q3_K fp16 staging cache from E104.
- Existing-MMQ selector override for large Q3_K prefill from E105.
- Large-Q3 compute16 default from E046.
- Broad `GGML_CUDA_FORCE_MMQ_RUNTIME` as a profile/default.
- Q3_K dequant128, half2 store packing, explicit unroll4 conversion variants.
- hipBLASLt/Stream-K env defaults for current Q3_K lanes.

### Candidates for a future local-minimal build study

These are not safe source deletions today. They are possible build-profile gates
if the fork decides to prioritize a local RX 9070 XT/Qwen-only binary:

- NVIDIA-only and MUSA-only branches inside the shared CUDA backend. Removing
  them from source would make upstream sync harder; a HIP-only compile option is
  safer.
- Rare quant families not used by the local Qwen profiles. This must be checked
  against TurboQuant (`TQ3_0`) and any GUI-supported model presets before pruning.
- MoE/expert staging experiments if the fork explicitly drops MoE route research.
- Diagnostic force knobs such as Q3 force-x if route-map reproducibility is no
  longer valued. Prefer a debug-build option over deletion.

### Current H35 design gate

Before coding a new ROCm Q3_K fused route, require all of the following:

1. Target the measured hot family first: Q3_K `row_diff=6144`, `ne00=5120`,
   `ne10=5120`, `ncols=2048`, plus tail chunks around `1259/1278`.
2. Avoid persistent fp16 residency. E104 proves saved conversion can still lose
   wall time through VRAM/residency pressure.
3. Do not route through current MMQ. E050/E105 prove the existing MMQ path is
   slower for large Q3_K prefill shapes.
4. Provide a local ceiling model: E054 target conversion is `1430.88 ms`; a
   route must plausibly save at least `20-25%` of the relevant local conversion
   or improve conversion+GEMM together enough for about `>=2%` wall.
5. Add default-off activation and trace proving the intended tensors/shapes use
   the new route before a lane benchmark.
6. Compare only cold-first, no-reuse, same-lane controls.

## Quick Route Probe Recipes

Route topology only:

```bash
GGML_TRACE_CUDA_MUL_MAT_ROUTE=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

Large Q3_K hipBLAS split detail:

```bash
GGML_TRACE_CUBLAS_SPLIT_TIMING=1 \
GGML_TRACE_CUBLAS_SPLIT_DETAIL=1 \
GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024 \
python scripts/agent_workload_bench.py --label <label> ...
```

Q3_K staging reuse:

```bash
GGML_TRACE_CUBLAS_Q3K_ROUTE=1 \
GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=1024 \
python scripts/agent_workload_bench.py --label <label> ...
```

MMQ timing/resource probe:

```bash
GGML_TRACE_MMQ_TIMING=1 \
GGML_TRACE_MMQ_TIMING_SYNC=1 \
GGML_TRACE_MMQ_RESOURCES=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

MMVQ small-k trace:

```bash
GGML_TRACE_MMVQ_SMALL_K=1 \
GGML_TRACE_MMVQ_TIMING=1 \
python scripts/agent_workload_bench.py --label <label> ...
```

All sync traces are diagnostic only. Do not turn sync-trace TPS into a speed
claim.
