# E087 Vulkan 12k Q3_K Correct LOAD_VEC_A=8 Probe

## Metadata

- Experiment ID: E087
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E086 corrected Q3_K LOAD_VEC_A=4 kept
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: The E086 Q3_K pair-load improvement may continue to `LOAD_VEC_A=8` if reduced load-loop invocation count offsets higher per-invocation dequant pressure.
- Mechanism: For `BK=32`, `LOAD_VEC_A=8` maps each A-load row to four Q3_K pair indices and writes four `f16vec2` shared-memory slots.
- Risk: Four dequant pairs per invocation may raise VGPR/ALU pressure enough to undo the scheduling benefit.

## Benchmark Plan

- Baseline: E086 stride18 + corrected `LOAD_VEC_A=4`, fixed pp7488 r3 `961.82 tok/s`, workload prompt eval `934.8 tok/s`.
- Candidate: corrected Q3_K `LOAD_VEC_A=8`.
- Runs: r1 gate first; r3 only if promising.

## Result

- Outcome: reject and revert to E086 `LOAD_VEC_A=4`.
- Delta: fixed pp7488 r1 was `947.44 tok/s`, below E086 r3 `961.82 tok/s` and slightly below E086 r1 `949.65 tok/s`.
- Recommendation: keep Q3_K corrected `LOAD_VEC_A=4`; `LOAD_VEC_A=8` likely adds too much per-invocation dequant/register pressure for the smaller load-loop count.

Artifact:

- `build_logs/agent-workload/e087-vulkan-q3-correct-loadvec8-pp7488.md`