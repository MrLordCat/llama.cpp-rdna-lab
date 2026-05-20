# E096 SPIR-V Opcode Summary Tooling

## Metadata

- Experiment ID: E096
- Date: 2026-05-20
- Owner: Copilot
- Type: workflow/tooling + shader-route fingerprint
- Target lane: H31 Vulkan Q3_K prefill research

## Hypothesis

- Statement: Generated SPIR-V opcode summaries can provide a cheap no-build fingerprint for Vulkan shader route changes before benchmark work.
- Mechanism: Counting KHR cooperative matrix ops, barriers, loads/stores, and integer/bit ops helps distinguish real route changes from source edits that compile to the same broad shader class.
- Why now: E092/E093 closed many blind source/tile probes; E095 showed the active AMD route is KHR coopmat, not NV coopmat2.

## Math / Theory

- Assumptions: SPIR-V opcode counts do not predict runtime speed alone, but they can gate whether a candidate changes the expected mechanism.
- Expected speedup corridor: none; this is a diagnostic filter.
- Failure conditions: Do not use opcode counts as a speed claim; use them to decide whether a build/benchmark is worth considering.

## Implementation Plan

1. Minimal code surface to change: add `scripts/research/spirv_op_summary.py`.
2. Guard rails: read existing generated `.spv` files only; no build or benchmark.
3. Rollback path: remove the tool if replaced by a richer shader-analysis pipeline.

## Benchmark Plan

- Baseline command: `python scripts/research/spirv_op_summary.py build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q3_k_f32_aligned_f16acc_cm1.spv build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q3_k_f32_aligned_f16acc.spv --top 35`
- Candidate command: same tool against candidate-generated `.spv` after a justified build.
- Number of runs: 1.
- Artifacts path: `build_logs/agent-workload/e096-spirv-op-summary-q3k.md`

## Metrics

Active KHR coopmat Q3_K shader (`matmul_q3_k_f32_aligned_f16acc_cm1.spv`) focus counts:

| Op | Count |
| --- | ---: |
| `OpCooperativeMatrixLoadKHR` | `2` |
| `OpCooperativeMatrixMulAddKHR` | `1` |
| `OpCooperativeMatrixStoreKHR` | `3` |
| `OpTypeCooperativeMatrixKHR` | `4` |
| `OpControlBarrier` | `6` |
| `OpLoad` | `256` |
| `OpStore` | `89` |
| `OpIAdd` | `99` |
| `OpIMul` | `81` |

Plain non-cm Q3_K shader (`matmul_q3_k_f32_aligned_f16acc.spv`) has no `OpCooperativeMatrix*KHR` ops and only `2` `OpControlBarrier` ops.

## Result

- Outcome: keep tool and artifact.
- Delta: no TPS claim.
- Confidence: high for opcode fingerprinting of generated SPIR-V; runtime resource usage still requires pipeline stats/driver compile logs.
- Recommendation: before any future shader-route claim, compare generated SPIR-V summaries for baseline vs candidate. If the candidate does not change the expected focus ops or only changes low-ceiling scalar helper patterns, skip benchmark unless another gate justifies it.

## Notes

- Surprises: the cm1 and non-cm binaries are easy to distinguish cheaply, which is useful for guarding against accidental route/fallback confusion.
- Follow-up action: use SPIR-V summaries together with pipeline stats when evaluating any `BK`, load-vector, or route-selection candidate.
