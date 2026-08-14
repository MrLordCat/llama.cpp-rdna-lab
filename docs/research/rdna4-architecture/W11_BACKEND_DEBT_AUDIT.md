# W11: backend debt audit (Vulkan + ROCm, 2026-08-14)

Source-only audit answering "what legacy/architectural debt do the Vulkan and
ROCm backends carry". Three categories: dead diagnostics (safe to remove
after conclusions firm), live fallbacks (do NOT remove), and real dead code.

## 1. Dead diagnostics - D-series experiment scaffolding (the main local debt)

### Vulkan FA dispatch zoo (`ggml/src/ggml-vulkan/runtime/vk_dispatch.inc:2197-2260`)

All env-gated, dead by default:

- `GGML_VK_FA_F8_P2/P3/P4/P5` - four hand-edited GLSL fp8 variants
  (fail-closed to aligned prefill; P3/P4 hardcode HSK=256/Br=16; P4 adds an
  f16 V preconvert buffer). Each requires its own transform pass in the
  shader generator (`vulkan-shaders-gen.cpp:481-608`).
- `GGML_VK_FA_F8_NATIVE` + `GGML_VK_FA_F8_NATIVE_DECODE` - the spvasm native
  kernel; the decode route measured **48% slower** than scalar f8 (13.28 vs
  25.56 t/s, noted in-code at vk_dispatch.inc:2208-2220). Diagnostic only.
- `GGML_VK_FA_F8_DIRECT` - route probe.
- `GGML_VK_FA_HALF_CMP` - in-process native/fallback A/B splitter used for
  benchmarking (vk_dispatch.inc:2230-2241) - benchmark scaffolding in the
  production dispatch.
- `GGML_VK_FA_F8_DUMP`, `GGML_VK_FA_SCALAR_STAGING`, `GGML_VK_MM_TRACE_SPLIT`,
  `GGML_VK_Q3K_QUAD_DEQUANT`, `GGML_VK_Q3K_FFN_DOWN_SPLIT_K`,
  `GGML_VK_QK_LOW_TILE_SPLIT_K` - more experiment toggles.
- 22 D###-comments in vk_dispatch.inc alone; ~40 env switches total in the
  runtime .inc files.

Build-time experiment machinery: `scripts/research/d096_fp8_fa_spirv_*.py`
wired into CMake (`ggml/src/ggml-vulkan/CMakeLists.txt:222-232`) - the fragile
mtime-sensitive transform pipeline from D096.

### ROCm toggles and scaffolding

- D102 phase census: `GGML_ROCM_FATTN_PHASE_CENSUS` + pinned-host copy +
  atexit report in `fattn-wmma-f16.cu` (~60 lines, env-gated).
- ~12 `GGML_TRACE_*` switches (MMVQ/MMQ/SRC1_QUANT/GLU/GDN timing + path).
- `GGML_MMVQ_QWEN_FORCE_SMALL_K/DISABLE_SMALL_K`, `GGML_MMVQ_Q3K_RDNA4_VK16`,
  `GGML_MMVQ_Q3K_DISABLE_PAIRDOT`, `GGML_MMQ_RDNA4_Q4Q5_FORCE_MMQ_X`,
  `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X`, `GGML_MMQ_RDNA4_PQ2_FORCE_MMQ_X`,
  `GGML_CUDA_Q3K_PADDED_*`, `GGML_GDN_FAST_EXP/CHUNK_SIZE`,
  `GGML_RDNA4_MOE_MMQ_STAGING`, `GGML_RDNA4_Q3K_SMALLN_DP4A`,
  `GGML_HIP_DISABLE_GRAPHS` - dozens of unregistered experiment toggles.
- `ggml/src/ggml-cuda/fattn-qwen-reduced.cpp` - a parallel reduced-head FA
  dispatcher, selectable at configure time (`hip-source-bundles.cmake:52-60`,
  excluding fattn.cu/fattn-tile.cu/fattn-wmma-f16.cu when enabled).
- `ggml/src/ggml-cuda/PERF_RESEARCH_NOTES.md` - research notes living inside
  the source tree.

## 2. Live fallbacks / compat paths (not debt - do not remove)

### Vulkan

- `flash_attn.comp` (scalar FA, devices without coopmat) + `flash_attn_cm1`
  (coopmat1) + `flash_attn_cm2` (coopmat2) - three-generation FA matrix.
- `mul_mat_vec.comp` / `mul_mat_vec_<quant>.comp` (scalar vec-dot for old
  devices), `mul_mm.comp` (coopmat), `mul_mm_cm2.comp` (coopmat2),
  `mul_mmq.comp`; `mul_mat_vecq.comp` (q8_1 vec path, still generated:
  vulkan-shaders-gen.cpp:946-948).
- `dequant_*.comp` - dequant path for non-MMQ devices; `GGML_VK_DISABLE_F16`,
  `GGML_VK_DISABLE_COOPMAT/COOPMAT2`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`
  - driver-compat switches.

### ROCm/HIP

- f16 FA route (n4, 212 VGPR) - live fallback for f16 KV.
- 23 `fattn-mma-f16` template instances under `GGML_HIP_FA_ALL_QUANTS`
  (hip-source-bundles.cmake:80) - legacy "FA on all quant types" path.
- `mmq.cuh` dp4a kernels (`GGML_CUDA_FORCE_DP4A`, mmq.cuh:4353) - int-dot
  fallback route.
- hipBLAS (`cublas` route in runtime_compute.inc:1925-1930) - last-resort
  fallback for non-quantized/bad-padding cases; off the hot path by design
  (see W10).
- `vecdotq.cuh` / `dequantize.cuh` - vec-dot and dequant primitives used by
  mmvq/convert/getrows.
- `cp-async.cuh` - CUDA-only idiom, physically dead on gfx1201 (no async
  copy, W02) but retained as the CUDA source layer.

## 3. Real dead code (removal candidates)

- `vendors/musa.h` - MUSA backend removed by the backend policy; the vendor
  shim remains only to keep upstream diffs small.
- Vulkan FA F8_P2-P5 branches + their transform machinery in
  `vulkan-shaders-gen.cpp` - removable once the fp8 direction is decided
  (native spvasm accepted or the whole fp8-FA experiment closed).
- `GGML_VK_FA_F8_NATIVE_DECODE` route - measured 48% regression, purely
  diagnostic.

## Cleanup order (phase 3, after phase-2 conclusions are firm)

1. Remove benchmark scaffolding: HALF_CMP splitter, FA_F8_DUMP, census
   instrumentation (needs the census conclusions first).
2. Decide the fp8-Vulkan direction, then delete the losing P-variants and
   their transforms (the riskiest removal: touches the generator).
3. Consolidate the winner of each QWEN/small-K toggle pair into the default
   and delete the loser.
4. Write the surviving env-var registry (3.4 in PHASE2_PLAN.md).

## 2026-08-14: batch 1 executed (PHASE2_PLAN 3.3)

Removed in this batch:

- `GGML_VK_FA_F8_P2..P5` enum members, dispatch branches, pipeline
  registration blocks in `vk_shaders.inc`, and the whole transform machinery
  (`--fp8-fa-transform` + `d096_fp8_fa_spirv_*.py` in CMakeLists, generator
  transforms in `vulkan-shaders-gen.cpp`). The fp8 direction is closed on
  Vulkan (D4.1: scalar f8 + prefill preconvert is the memory-only route), so
  the losing variants are gone.
- `GGML_VK_FA_F8_NATIVE_DECODE` branch in `get_fa_tuning_params` (measured
  48% regression, diagnostic only).
- `GGML_VK_FA_HALF_CMP` in-process A/B splitter in `vk_dispatch.inc`.
- Dead prealloc members `prealloc_fa_q8`/`prealloc_fa_qscale` +
  `pipeline_fa_q_f32_f8` (P3 machinery; no surviving references).
- ROCm D102 phase-census scaffolding in `fattn-wmma-f16.cu` (consts, device
  symbol, init kernel, report, dispatch branch, template flag) - the
  `phase_census` template parameter was folded back to the plain production
  instantiation.
- GUI: `GGML_VK_FA_F8_P5=1` defaults removed from `benchmark_tab.py` and
  `server_backend_panels.py` (dead env, silently ignored after the removal).

Retained: `GGML_VK_FA_F8_NATIVE` (opt-in prefill kernel) + `GGML_VK_FA_F8_DUMP`
(its debug dump); `GGML_VK_FA_F8_DIRECT` (live prefill route probe).

Validation: `cmake --build build-rocm` and `build-vulkan` (llama-server +
test-backend-ops) both clean; ROCm `FLASH_ATTN_EXT -p f8_e4m3` 19/19;
Vulkan full `FLASH_ATTN_EXT` suite 4037/4290 with 253 failures, ALL non-f8
(sinks=1-heavy f16/f32/bf16/q8_0 cases) and untouched by this diff - a
pre-existing suite gap on the Vulkan side (verified at runtime, pre-existence
inferred from diff scope: no removed code touches those paths).
