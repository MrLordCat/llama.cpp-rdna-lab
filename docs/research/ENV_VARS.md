# Environment variable registry (RDNA4 research tree)

State as of commit `f704ad8f2` (2026-08-14, phase-3 debt cleanup). This file is
the single registry of the surviving env-var surface. The debt cleanup removed
the diagnostic scaffolding; see "Removed by the phase-3 cleanup" below.

Conventions: all envs are read once at backend/first-use time (no hot reload).
ROCm attention envs are read per-op at dispatch time; Vulkan tuning envs are
read once at backend init (some cached per pipeline family).

## ROCm/HIP - flash attention (f8/Q8 routes)

| Var | Purpose |
| --- | --- |
| `GGML_ROCM_FATTN_F8_REFERENCE` | force the f16-converted reference FA route for f8 KV |
| `GGML_ROCM_FATTN_F8_NATIVE_KQ` | opt-in native fp8 WMMA KQ route (shape-gated) |
| `GGML_ROCM_FATTN_F8_NATIVE_V` | opt-in native fp8 WMMA PV route (default enabled when KQ native) |
| `GGML_ROCM_FATTN_Q8_V_DIRECT_WMMA` | q8 V direct WMMA route (D=256/cols16/32, nwarps=4) |
| `GGML_ROCM_FATTN_Q8_CHUNKED_WMMA` | q8 chunked WMMA route |
| `GGML_FATTN_WMMA_FORCE_COLS_PER_BLOCK` | force cols-per-block for WMMA FA (debug/experiments) |
| `GGML_QWEN_FA_REDUCED_FORCE` | force the Qwen reduced-FA path |

## ROCm/HIP - MMQ / MMVQ / MoE / Q3K routes

| Var | Purpose |
| --- | --- |
| `GGML_CUDA_FORCE_MMQ_RUNTIME` | force MMQ runtime (no graph) |
| `GGML_MMQ_RDNA4_Q4Q5_FORCE_MMQ_X` | force MMQ variant X for Q4/Q5 |
| `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X` | force MMQ variant X for Q3 |
| `GGML_MMQ_RDNA4_PQ2_FORCE_MMQ_X` | force MMQ variant X for PQ2 |
| `GGML_MMQ_RDNA4_Q4K_MAX_NE11` | Q4_K MMQ threshold (max rows) |
| `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11` | stream-K MMQ threshold |
| `GGML_MMVQ_Q3K_DISABLE_PAIRDOT` | disable pairdot for Q3_K MMVQ |
| `GGML_MMVQ_Q3K_RDNA4_VK16` | Q3_K MMVQ VK16 variant toggle |
| `GGML_MMVQ_QWEN_DISABLE_SMALL_K` | disable small-K MMVQ for Qwen shapes |
| `GGML_MMVQ_QWEN_FORCE_SMALL_K` | force small-K MMVQ for Qwen shapes |
| `GGML_MMVQ_RDNA4_Q3K_MAX_BATCH` | Q3_K MMVQ max batch |
| `GGML_RDNA4_Q3K_SMALLN_DP4A` | Q3_K small-N DP4A route |
| `GGML_RDNA4_MOE_MMQ_STAGING` | MoE MMQ staging mode |
| `GGML_OP_OFFLOAD_MIN_BATCH` | offload minimum batch size |
| `GGML_CUDA_Q3K_PADDED_*` | Q3_K padded dequant storage/probe switches |

## ROCm/HIP - graphs, memory, WDDM

| Var | Purpose |
| --- | --- |
| `GGML_CUDA_GRAPH_OPT` / `GGML_CUDA_DISABLE_FUSION` | graph optimization / fusion toggles |
| `GGML_CUDA_ALLOW_RDNA4_GRAPH_OPT` | enable RDNA4-specific graph optimizations |
| `GGML_HIP_DISABLE_GRAPHS` | disable HIP graphs entirely |
| `GGML_CUDA_WDDM_STRICT_GLOBAL` / `GGML_CUDA_NO_WDDM_BUDGET` | WDDM memory budget controls |
| `GGML_CUDA_NO_PINNED` / `GGML_CUDA_REGISTER_HOST` | pinned/host memory controls |
| `GGML_CUDA_P2P` / `GGML_CUDA_NO_PEER_COPY_RUNTIME` | peer-to-peer copy switches |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | unified memory (VMM) toggle |
| `GGML_HIP_POOL_CACHE_LIMIT_MB` | HIP buffer pool cache limit |
| `GGML_CUDA_FORCE_DP4A` / `FORCE_CUBLAS_COMPUTE_16F/32F` | kernel selection overrides |
| `GGML_GDN_CHUNK_SIZE` | GDN chunk size |

## Vulkan - flash attention

| Var | Purpose |
| --- | --- |
| `GGML_VK_FA_FORCE_SCALAR` | force the scalar FA route |
| `GGML_VK_FA_F8_DIRECT` | f8 KV direct scalar decode route (inline dequant) |
| `GGML_VK_FA_F8_NATIVE` | native fp8 coopmat FA route (prefill) |
| `GGML_VK_FA_NO_PRECONVERT` | disable f8 preconvert for prefill |
| `GGML_VK_FA_F32ACC` | f32 accumulation toggle |
| `GGML_VK_FA_BLOCK_COLS` / `GGML_VK_FA_BLOCK_ROWS` | FA block size overrides |
| `GGML_VK_FA_DISABLE_SPLIT_K` / `GGML_VK_FA_FORCE_SPLIT_K` | split-K controls |
| `GGML_VK_FA_ROW_SPLIT` | row split for the split-K path |
| `GGML_VK_FA_SCALAR_BC/BR/RS/DSPLIT/NOMASK/SUBGROUP/STAGING` | scalar-FA tuning knobs |
| `GGML_VK_FA_ROUTE_TRACE` / `GGML_VK_FA_F8_DUMP` / `GGML_VK_FA_OUTDUMP` / `GGML_VK_FA8_HOSTDUMP` | route tracing and host dumps (debug) |

## Vulkan - MMVQ / matmul / device

| Var | Purpose |
| --- | --- |
| `GGML_VK_FORCE_MMVQ` / `GGML_VK_DISABLE_MMVQ` | MMVQ route switches |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` | disable int-dot kernels |
| `GGML_VK_DISABLE_COOPMAT` / `GGML_VK_DISABLE_COOPMAT2` | disable coopmat |
| `GGML_VK_DISABLE_F16` / `GGML_VK_DISABLE_BFLOAT16` | dtype availability toggles |
| `GGML_VK_ALLOW_AMD_F32_BATCHED_MUL_MAT_VEC_5_PLUS` | allow batched MMV f32 |
| `GGML_VK_AMD_LARGE_MATMUL_VARIANT` / `FORCE/DISABLE_AMD_LARGE_MATMUL` | large-matmul variant |
| `GGML_VK_DISABLE_AMD_BN256_DEFAULT` / `DISABLE_AMD_WN32_DEFAULT` | AMD default-kernel toggles |
| `GGML_VK_Q3K_QUAD_DEQUANT` / `GGML_VK_Q3K_FFN_DOWN_SPLIT_K` | Q3_K dequant/split-K |
| `GGML_VK_QK_LOW_TILE_SPLIT_K` / `DISABLE_QK_LOW_TILE_DEFAULT` | QK low-tile route |
| `GGML_VK_DISABLE_MULTI_ADD` / `GGML_VK_CONCAT_FAST` / `GGML_VK_DISABLE_FUSION` | fusion/multi-add toggles |
| `GGML_VK_VISIBLE_DEVICES` | device selection |
| `GGML_VK_ALLOW_GRAPHICS_QUEUE` / `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` / `GGML_VK_DISABLE_ASYNC` | queue placement |
| `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` / `FORCE_MAX_BUFFER_SIZE` / `SUBALLOCATION_BLOCK_SIZE` | memory sizing |
| `GGML_VK_PREFER_HOST_MEMORY` / `ALLOW_SYSMEM_FALLBACK` / `DISABLE_HOST_VISIBLE_VIDMEM` | host/device memory policy |
| `GGML_VK_ENABLE_MEMORY_PRIORITY` (+`_SMALL`, `_LARGE_THRESHOLD_MB`) | memory priority |
| `GGML_VK_TENSOR_ALLREDUCE` / `_BF16` | tensor allreduce |
| `GGML_VK_DEVICE_GROUP_TRACE` / `GGML_VK_INIT_TRACE[_VERBOSE]` | device group / init traces |

## Vulkan - observability

| Var | Purpose |
| --- | --- |
| `GGML_VK_PERF_LOGGER[_CONCURRENT][_FREQUENCY]` | perf logger |
| `GGML_VK_MEMORY_LOGGER` / `GGML_VK_SYNC_LOGGER` / `GGML_VK_COMM_TIMING` | memory/sync/comm timing |
| `GGML_VK_PIPELINE_CREATE_TRACE` / `GGML_VK_PIPELINE_STATS` | pipeline observability |
| `GGML_VK_MATMUL_ROUTE_TRACE` / `GGML_VK_FFN_ROUTE_TRACE` | route tracing |
| `GGML_VK_MM_DISABLE_SPLIT_K` / `GGML_VK_MM_TRACE_SPLIT` | MM split-K |
| `GGML_VK_DEBUG_MARKERS` | debug markers |
| `GGML_VK_MAX_NODES_PER_SUBMIT` | submission batching |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE` | graph optimization toggle |

## ROCm/HIP - observability

`GGML_TRACE_*`: CUDA graph state/timing/alloc/diff, node timing, pool,
WDDM budget, host stage, mul-mat route, MMQ/MMVQ path+timing+resources,
FATTN path/launch config/selected/alloc size, GDN path+timing, GLU fill/timing,
mul-mat-id route, CUBLAS Q3K route/split/src1-reuse, async cross-device stage.
These are diagnostic-only and have no effect on results.

## Server-side (llama-server / common)

| Var | Purpose |
| --- | --- |
| `LLAMA_CACHE` | server cache directory |
| `LLAMA_MTP_DEVICE_HANDOFF` | MTP draft/target device handoff |
| `LLAMA_MTP_RS_SEQ_MAX` | MTP reject-sample sequence cap |
| `LLAMA_VK_MTP_KV_LAST_F16` | keep the MTP KV (last) layer in f16 |
| `LLAMA_SPEC_PREFILL_SPARSE_CHUNK/STRIDE/WINDOW` | sparse spec-prefill tuning |
| `LLAMA_SPEC_TOKEN_TRACE` / `LLAMA_SPEC_VERIFY_TIMING` | speculative decode traces |
| `LLAMA_SPEC_RS_SEQ_MAX` | reject-sample sequence cap |
| `LLAMA_DFLASH_*` | DFlash chunk/ubatch controls |
| `LLAMA_DELTA_NET_*` | Delta-Net chunk policy |
| `LLAMA_CHECKPOINT_TIMING` | checkpoint timing trace |

## Bench harness (`scripts/agent_workload_bench.py`)

| Var | Purpose |
| --- | --- |
| `LLAMA_BENCH_ALLOW_HARD_KILL` | allow hard-killing a hung server at teardown |
| `LLAMA_BENCH_ALLOW_AUTO_FIT` | allow the hipMemGetInfo-based auto-fit probe (disabled by `-fit off` to avoid the ROCm startup path) |

## Removed by the phase-3 cleanup (`f704ad8f2`)

- `GGML_ROCM_FATTN_PHASE_CENSUS` (+ `_BLOCKS`/`_PHASES` constants, `GGML_TRACE_FATTN_PHASE_CENSUS`) - D102 per-phase clock64 census
- `GGML_VK_FA_F8_NATIVE_DECODE` - D096 diagnostic route (48% regression, closed)
- `GGML_VK_FA_F8_P2` / `_P3` / `_P4` / `_P5` - D096 fp8 direction variants (closed on Vulkan, D4.1)
- `GGML_VK_FA_HALF_CMP` - in-process A/B splitter
- GUI defaults referencing `GGML_VK_FA_F8_P5` were removed with the same commit.

## Notes

- Full raw name list (180 GGML_* entries) can be regenerated with:
  `grep -rnoE 'getenv\("GGML_[A-Z0-9_]+"\)' ggml/src/ggml-vulkan ggml/src/ggml-cuda ggml/src/ggml-hip`
- The registry intentionally omits upstream `GGML_CUDA_*` tuning switches
  unrelated to the RDNA4 research lanes unless they appear in the lanes' docs.
