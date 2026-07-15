# E292 ROCm Q3_K Packed Dequant Probe

## Metadata

- Date: 2026-07-14
- Target: Qwen3.6-27B Q3_K_S ROCm prompt-eval path on dual RX 9070 XT
- Rollback: `GGML_CUDA_Q3K_PADDED_DEQUANT_PACKED=0`

## Hypothesis

The padded Q3_K staging kernel currently expands four values per thread through
four scalar low-bit/high-mask expressions. Loading the four bytes as packed
32-bit words and performing lane-safe biased subtraction may reduce load and
branch instructions without changing the 64-thread launch shape or output
layout.

This is distinct from the rejected E051 thread-count, E055/E239 half2-store,
and E057 explicit-unroll probes: it changes the value reconstruction body.

## Gates

1. Compare generated code size, VGPR count, and spills with the baseline kernel.
2. Pass focused Q3_K `MUL_MAT` backend correctness tests.
3. Measure synchronized Q3_K `src0_convert_ms` before a wall benchmark.
4. Keep only if the local conversion improvement survives a clean prompt lane
   without a material decode or memory regression.

## Result

- The packed lane reconstruction passed an exhaustive host-side check across
  every low-byte value, high-mask value, 2-bit shift, high-mask bit, and lane.
- Focused ROCm Q3_K `MUL_MAT` correctness passed `11/11` supported cases.
- gfx1201 code generation improved:
  - kernel size `1216 -> 800` bytes;
  - instructions `210 -> 142`;
  - global loads `14 -> 10`;
  - VGPRs `15 -> 12`, with zero spills in both variants.
- Synchronized 7.9k-prompt Q3_K trace:
  - `src0_convert_ms 820.041 -> 753.251` (`-8.14%`);
  - full traced Q3_K route `4481.660 -> 4462.722 ms` (`-0.42%`), as rocBLAS
    GEMM remains dominant.
- Clean dual-GPU `b8192/ub1024` prompt results:
  - 7.9k prompt, 16 output, r3: `1739.80 -> 1766.33 tok/s` (`+1.52%`);
  - 7.8k prompt, 128 output, r3: `1728.80 -> 1745.01 tok/s` (`+0.94%`);
  - 30.1k prompt, 16 output, r1: `1363.73 -> 1373.58 tok/s` (`+0.72%`).
- A Q3_K route trace with `min_ncols=1` found zero cuBLAS Q3_K staging
  calls in the short-prompt/decode lane at `ubatch=128`. Decode uses MMVQ, so
  the packed staging kernel cannot cause the noisy 16-token decode deltas.

## Decision

Promote the packed padded-Q3_K dequant kernel as the HIP default. Keep the
original kernel compiled as a runtime rollback with
`GGML_CUDA_Q3K_PADDED_DEQUANT_PACKED=0`. CUDA behavior is unchanged.

Primary artifacts:

- `e292-rocm-q3k-packed-dequant-{control,candidate}-trace-r1.*`;
- `e292-rocm-q3k-packed-dequant-{control,candidate}-clean-r3.*`;
- `e292-rocm-q3k-packed-dequant-{control,candidate}-128-r3.*`;
- `e292-rocm-long49ctx-packed-dequant-{control,candidate}-r1.*`;
- `e292-rocm-q3k-route-decode-check-r1.*`.
