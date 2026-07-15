# E287: ROCm Src1 Reuse and Warmup

Date: 2026-07-14

## Scope

This experiment tested two low-risk explanations for the remaining ROCm prompt
gap: repeated source-activation staging and incomplete server warmup. The lane
was the E286 dual-RX-9070-XT profile with 7,923 prompt tokens, 32 output tokens,
`b8192/ub1024`, q8 K/V, and MTP disabled.

## Results

| Variant | Cold prompt | Warm prompt median | Decode | Extra VRAM |
| --- | ---: | ---: | ---: | ---: |
| E286 default | 1,589.01 | 1,807.17 | 26.72 | baseline |
| persistent src1 reuse | 1,601.60 | 1,820.32 | 26.53 | about 34 MiB/GPU |
| normal server warmup | 1,580.35 | 1,802.69 | 26.78 | none |

Persistent src1 reuse improves warm prompt throughput by only `0.73%`, costs
about 34 MiB on each device, and slightly lowers decode. It remains an opt-in
diagnostic instead of the default.

The normal server warmup does not exercise the wide Q3_K-to-F16 rocBLAS shapes.
The first real prompt therefore remains slower than later prompts in the same
process. This is library/solution initialization, not prompt-cache reuse: the
benchmark explicitly disabled both reuse and its prime pass.

## Result

- Keep src1 reuse opt-in only.
- Do not treat the current server warmup as a wide-GEMM warmup.
- Optimize or deliberately prime the actual Q3_K/rocBLAS prefill route instead
  of retaining another full activation buffer by default.

Primary artifacts:

- `e287-rocm-dual-src1reuse-12k-none-r1r3.*`;
- `e287-rocm-dual-serverwarmup-12k-none-r1r3.*`.
