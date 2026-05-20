# E088 Vulkan 12k Q3_K LOAD_VEC_A=4 Pair-Scale Probe

## Metadata

- Experiment ID: E088
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E086 kept and E087 rejected
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: E086's corrected Q3_K `LOAD_VEC_A=4` can be improved by dequanting the two adjacent Q3_K pairs with a shared scale decode.
- Mechanism: `LOAD_VEC_A=4` maps one invocation to pair indices `2*i` and `2*i+1`; these adjacent pairs share block, scale index, high-mask halfsplit, and dequant multiplier. Computing `us/dl` once may lower ALU/register pressure.
- Risk: Extra helper/out-parameter structure can raise compiler pressure or block inlining.

## Benchmark Plan

- Baseline: E086 fixed pp7488 r3 `961.82 tok/s`.
- Candidate: paired Q3_K dequant helper for `LOAD_VEC_A=4`.
- Runs: r1 gate first; r3 only if promising.

## Result

- Outcome: reject and revert to E086.
- Delta: fixed pp7488 r1 was `959.89 tok/s`, effectively neutral but still below E086 r3 `961.82 tok/s`.
- Recommendation: keep the simpler E086 two-call `dequant_q3_k_pair()` path. The paired helper did not provide a measurable gain and adds code complexity.

Artifact:

- `build_logs/agent-workload/e088-vulkan-q3-loadvec4-pairscale-pp7488.md`