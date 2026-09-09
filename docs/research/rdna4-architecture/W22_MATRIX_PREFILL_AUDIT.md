# W22: RDNA4 MMQ already uses WMMA-I8 (matrix hardware is busy on prefill)

Date: 2026-09-09

## Finding (source verification, no code change)

W21 proposed an env-gated MMQ-i8-WMMA prototype. Source audit + runtime
trace shows **it is already implemented and active**:

1. `ggml/src/ggml-cuda/mma.cuh:1200`: AMD_WMMA branch for
   `mma(tile<16,16,int> D, tile<16,8,int> A, tile<16,8,int> B)` calls
   `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` (m16n16k16 int8->i32,
   RDNA4 only).
2. `mmq.cuh` `load_tiles_mxfp4` (and other int quant types) write the
   MMA layout (`MMQ_MMA_TILE_X_K_Q8_1` int32 x_qs + float x_df) when
   `AMD_WMMA_AVAILABLE`.
3. `mmq_type_traits<...,GGML_TYPE_MXFP4>` on non-Blackwell maps
   `vec_dot_mma = vec_dot_q8_0_q8_1_mma` (the AMD WMMA-I8 vector dot).
4. `mmq_select_vec_dot<...,use_dp4a=false>` returns `vec_dot_mma` whenever
   `AMD_MFMA || TURING || AMD_WMMA`; `launch_mul_mat_q` always launches
   `mul_mat_q<type, mmq_x, need_check>` (default `use_dp4a=false`), so the
   MMA (WMMA-I8) path is used on RDNA4 for all non-special types.
5. Runtime trace (`GGML_TRACE_MMQ_PATH=1`): `mul_mat_q_case type=12..14
   smalln_dp4a=0` confirms the general path (only Q3_K small-N and
   Q8_0 `GGML_CUDA_FORCE_DP4A` use DP4A).

Therefore the earlier W21 statement "int types (MXFP4/Q4_K) stay DP4A"
is CORRECTED: on RDNA4 prefill MMQ already runs on the INT8 WMMA matrix
hardware.

## Why prefill is still not faster

- W17 profile: MMQ = 7.8% of prefill time (q4 27.67 ms / 353 ms) - the
  WMMA-I8 compute is already ~6x cheaper than DP4A, so MMQ is NOT the
  prefill limiter; the remaining time is src1 Q8_1 quantization, norms,
  attention, and the F32 pipeline.
- W20 measured MXFP4 prefill neutral vs Q4_K: consistent - the weight
  format does not change the non-MMQ parts; MMQ is already fast enough.
- W21 microbench (186 TMAC/s) only confirms the ceiling; the kernel is not
  near it because it is not the bottleneck.

## Consequences for the R9700 question

- "llama.cpp leaves RDNA4 matrix hardware underutilized" is only true for
  **decode** (MMVQ ncols=1, cannot use WMMA by design - W20
  bandwidth-limited). For **prefill** the matrix hardware is already used;
  the R9700-style 280 tok/s gain cannot come from an MMQ WMMA port.
- Decode does not have a free matrix-hardware win; its remaining levers are
  weight-stream geometry (MXFP4 already +15.3%), speculative (MTP +22%),
  or GDN.

## Artifacts

- Source lines: `mma.cuh:1200`, `mmq.cuh:872/1032/3460/3618/4509`,
  `mmq.cu:4329` (launch), `mmq.cu:138` (dispatch).
- Runtime trace: `/tmp/w22_trace_server.log` (GGML_TRACE_MMQ_PATH=1).
