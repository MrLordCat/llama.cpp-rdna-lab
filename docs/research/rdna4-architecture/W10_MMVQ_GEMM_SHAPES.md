# W10: MMVQ/MMQ gemm shape audit (decode/prefill routes)

Date: 2026-08-14

Source-only audit of the MUL_MAT dispatch for the 49K lane (Qwen3.6-27B
Q4_K_M, n_embd = 5120, n_ff = 17408, dense, 64 layers). No GPU runs.

## Dispatch chain (runtime/runtime_compute.inc:1760-1930)

Route order in `ggml_cuda_mul_mat`:

1. `mul_mat_vec_f` - f16/bf16/f32 weights, small M (mmvf.cu);
2. `mul_mat_f` - non-quantized, larger M (mmf.cu);
3. `mul_mat_vec_q` - quantized weights, `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`
   (= 8, mmvq.cuh:5) and `ggml_cuda_should_use_mmvq` (mmvq.cu:279);
4. `mul_mat_q` - quantized weights, MMQ (mmq.cu:425);
5. `batched_cublas` - non-quantized multi-batch without FA;
6. backend/split variants of 1-4;
7. `cublas` - final fallback (hipBLAS), non-quantized or bad-padding cases.

## Decode path (the 49K token stream)

- Activations src1: f32, `ne[1] = 1..4` (tokens) -> route 3, **MMVQ**.
- `ggml_cuda_should_use_mmvq` on RDNA4: Q3_K is capped at batch 1
  (`GGML_MMVQ_RDNA4_Q3K_MAX_BATCH`, default 1, mmvq.cu:280-296); every other
  quant type falls through to `ne11 <= 8`.
- Kernel geometry (RDNA4 table, mmvq.cu calc_nwarps): ncols 1-4 -> nwarps 4;
  ncols 5-8 -> nwarps 2. Comment notes nwarps=8 was tested and rejected for
  complex vec_dot types (IQ2/IQ3).
- Shapes: M = 1..4, K = 5120 (Q4_K_M blocks of 256 -> 20 blocks per row),
  N in {5120 (attn/out), 17408 (FFN gate/up), 5120 (FFN down)}.
- The MMVQ kernel reads the full weight matrix (~16 GB total) per token and
  converts activations to Q8_1 on the fly - the dominant decode traffic
  (consistent with D102's ~17 GB/token weight-stream observation).
- Local tuning toggles live here: `GGML_MMVQ_Q3K_RDNA4_VK16` /
  `GGML_MMVQ_Q3K_DISABLE_PAIRDOT` / `GGML_MMVQ_RDNA4_Q3K_MAX_BATCH`. The
  former `GGML_MMVQ_QWEN_FORCE_SMALL_K` / `GGML_MMVQ_QWEN_DISABLE_SMALL_K`
  toggles were consolidated into the auto small_k policy during phase-3 debt
  cleanup (W13 audit: Qwen-hot RDNA4 ncols==1 -> small_k on for Q3_K/Q4_K,
  off for Q6_K).

## Prefill path

- Quantized weights, large M -> route 4, **MMQ stream-k** (mmq.cu), with
  RDNA4-specific Q4_K/Q5_K wide tiles (template instances
  `mmq-instance-q4_k.cu` / `mmq-instance-q5_k.cu` compiled with
  `GGML_MMQ_RDNA4_Q4Q5_Y128_W8=1`, hip-source-bundles.cmake:70-75).
- Stream-k threshold: `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11` (default 256);
  Q4_K MMQ cap `GGML_MMQ_RDNA4_Q4K_MAX_NE11` (default 1024).

## hipBLAS status

- The `cublas` route (rocBLAS) is the **last-resort fallback**: it is
  reached only for non-quantized src0 (f16/f32/bf16), bad-padding compute
  buffers, or shapes no custom kernel claims. For the quantized production
  lane the weights never take it; f16 compute tensors are handled by
  mmvf/mmf. It is legacy compat, not a hot path (consistent with the W11
  debt audit).
- `GGML_CUDA_FORCE_CUBLAS` exists only as a compile-time escape hatch
  (mmq.cu:469).

## Conclusions for 2.5

1. Decode = MMVQ with M<=4: the gemm is extremely memory-bound (K=5120,
   ~16 GB weights re-read per token). The kernel-side lever is not FLOPs but
   the Q8_1 activation conversion + address stream; the RDNA4 table already
   tunes nwarps per type.
2. Two oddities worth a follow-up (cheap to test in phase 2):
   - Q3_K is capped at batch 1 on RDNA4 while other types allow 8 (mmvq.cu)
     - a missed prefill-latency-free decode opportunity for Q3 lanes;
   - the QWEN small-K toggles are env-gated variants that won or lost during
     D-series work; the winner should be the default, the loser removed
     (feeds 3.1/3.4).
3. No hipBLAS gap to chase: it is off the hot path by design.
