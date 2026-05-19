# E073 Vulkan Q3_K MMQ Packed32 Loads

## Metadata

- Experiment ID: E073
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master after Q4 ROCm fix `172fa02c8`, E071/E072 docs pending
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: Q3_K MMQ prefill can improve by loading Q3_K `qs`/`hmask` as packed32 words in `block_a_to_shmem`, avoiding four packed16 loads and unpack/repack work per `iqs`.
- Mechanism: Q3_K MMVQ already uses `block_q3_K_packed32`. MMQ still reconstructs eight values from four 16-bit pairs, then packs them into the same byte/nibble layout. For MMQ `iqs=0..3`, two 32-bit `qs` words and two 32-bit `hmask` words contain exactly the same eight bytes currently read as four 16-bit pairs.
- Why now: E071 perf logger shows Q3_K MMQ dominates prompt time; E072 dot-loop subtract rewrite did not help, so the next narrow target is MMQ load/repack overhead rather than the dot loop.

## Math / Theory

- Assumptions: Q3_K MMQ is at least partly load/repack or instruction-throughput limited, and the packed32 path reduces instructions without increasing register pressure enough to hurt occupancy.
- Expected speedup corridor: +1% to +5% prompt eval; enough to close or slightly beat the current ROCm prompt target if it lands well.
- Failure conditions: Driver coalesces the old packed16 loads well, added 32-bit temporaries increase pressure, or the hot Q3_K MMQ shapes are dominated by memory bandwidth/dot throughput instead of repack cost.

## Implementation Plan

1. Minimal code surface to change: Q3_K `block_a_to_shmem` in `ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq_funcs.glsl`.
2. Guard rails: no numerical layout change; `vals0 | (vals1 << 4)` preserves the current packed nibble format consumed by `mmq_dot_product`.
3. Rollback path: revert the single shader hunk if the active lane is a tie/regression.

## Benchmark Plan

- Baseline command: E071 `wm32-wn32` refresh r3: prompt `1165.33 tok/s`, wall `7.9505 TPS`; ROCm prompt target `1172.4467 tok/s`.
- Candidate command: active lane with `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` and `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`.
- Number of runs: 1 for first gate; 3 only if prompt eval beats ROCm or clearly improves over E071 Vulkan.
- Artifacts path: `build_logs/agent-workload/e073-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split

## Result

- Outcome: regression / no code kept
- Delta: First active-lane gate reached `1162.4` prompt tok/s and `7.9321` wall TPS, below the E071 Vulkan `wm32-wn32` refresh at `1165.33` prompt tok/s and below ROCm `1172.4467` prompt tok/s.
- Confidence: medium. Like E072, the result is close to noise but not promising enough to keep or spend 3-run confirmation on.
- Recommendation: revert the shader hunk. The packed32 load-side rewrite does not solve the Q3_K MMQ bottleneck on the active lane.

## Notes

- Do not confuse this with E067, which rewrote Q3_K packed32 handling in the dequant `mul_mm` path and regressed. E073 targets the active Q8_1 MMQ prefill shader identified by perf logger.
- Artifact: `build_logs/agent-workload/e073-vulkan-q3k-mmq-packed32-r1-reposnapshot-b4096-ub1024-ctx12288-q3ks.diagnostics.md`.