# E072 Vulkan Q3_K MMQ Sub4 Bit-Twiddle

## Metadata

- Experiment ID: E072
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master after Q4 ROCm fix `172fa02c8`, with E071 docs pending
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: Q3_K MMQ prefill can recover the remaining Vulkan raw prompt gap by replacing inner-loop `unpack8 -> subtract 4 -> pack32` with byte-wise 32-bit subtract, matching the already-optimized Q3_K MMVQ path.
- Mechanism: Perf logger shows Q3_K `MUL_MAT` dominates the lane. The hottest shapes are `m=17408,n=1024,k=5120` and `m=5120,n=1024,k=17408`; both run through MMQ with Q8_1 activations. Current MMQ subtracts the Q3_K offset inside every dot product using vector unpack/repack, while MMVQ already uses `((vals ^ 0x80808080) - 0x04040404) ^ 0x80808080` to subtract 4 per byte without lane unpacking.
- Why now: E071 tile retargeting reached near parity but not a significant raw prompt win. The perf breakdown points at Q3_K MMQ rather than FATTN/GDN/f32 matmul.

## Math / Theory

- Assumptions: The dot loop is ALU/register limited enough that removing repeated unpack/repack lowers Q3_K MMQ kernel time; byte-wise subtract is equivalent because packed values occupy only the low nibble of each byte before subtracting 4.
- Expected speedup corridor: +1% to +4% prompt eval if the compiler does not already optimize the unpack/repack away.
- Failure conditions: The compiler already lowers both forms similarly, the bit-twiddle increases register pressure, or MMQ is memory/LDS-bound rather than arithmetic-bound.

## Implementation Plan

1. Minimal code surface to change: Q3_K branch of `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl` only.
2. Guard rails: rebuild only Vulkan server/bench, run active lane with current `wm32-wn32` best profile, compare against same-session E071 Vulkan and ROCm refresh.
3. Rollback path: revert the single shader hunk if the active lane is a tie/regression.

## Benchmark Plan

- Baseline command: E071 `wm32-wn32` refresh r3: prompt `1165.33 tok/s`, wall `7.9505 TPS`.
- Candidate command: same active lane with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` and `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`.
- Number of runs: 1 for first gate; 3 only if prompt eval beats ROCm `1172.4467 tok/s` or clearly improves over Vulkan baseline.
- Artifacts path: `build_logs/agent-workload/e072-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split

## Result

- Outcome: regression / no code kept
- Delta: First active-lane gate reached `1161.4` prompt tok/s and `7.9234` wall TPS, below the E071 Vulkan `wm32-wn32` refresh at `1165.33` prompt tok/s and below ROCm `1172.4467` prompt tok/s.
- Confidence: medium. The result is close enough to normal single-run noise that it is not proof the bit-twiddle is intrinsically slower, but it does not provide the needed direction or magnitude.
- Recommendation: revert the shader change and do not keep this micro-optimization. The compiler or RDNA4 scheduler likely already handles the existing unpack/repack acceptably, or the bit-twiddle adds pressure without reducing the dominant latency.

## Notes

- Perf logger before the patch aggregated approximately `4522.98 ms` in Q3_K `MUL_MAT`, `658.87 ms` in `FLASH_ATTN_EXT`, `241.96 ms` in `GATED_DELTA_NET`, `151.00 ms` in `RMS_NORM_MUL`, `114.82 ms` in f32 `MUL_MAT`, and `45.41 ms` in `SSM_CONV` for the diagnostic run.
- Q3_K shape totals in that diagnostic were led by `m=17408,n=1024,k=5120` at `2053.22 ms` and `m=5120,n=1024,k=17408` at `1151.25 ms`.
- Artifact: `build_logs/agent-workload/e072-vulkan-q3k-mmq-sub4-r1-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`.