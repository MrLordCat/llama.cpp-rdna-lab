# E090 Vulkan 12k Q3_K LOAD_VEC_A=4 Packed32 Pair Probe

## Metadata

- Experiment ID: E090
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E086 kept and E087-E089 rejected
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: E086's two adjacent Q3_K pair loads may benefit from `packed32` `qs/hmask` loads when used only inside the corrected `LOAD_VEC_A=4` branch.
- Mechanism: Pair indices `2*i` and `2*i+1` map to adjacent uint16 `qs` and `hmask` words, so one uint32 load can provide both pair inputs.
- Risk: Prior wider packed32 Q3_K probes regressed; extra shifts may outweigh fewer memory operations.

## Benchmark Plan

- Baseline: E086 fixed pp7488 r3 `961.82 tok/s`.
- Candidate: Q3_K `LOAD_VEC_A=4` paired packed32 helper.
- Runs: r1 gate first; revert if below E086.

## Result

- Outcome: reject and revert to E086.
- Delta: fixed pp7488 r1 was `951.79 tok/s`, below E086 r3 `961.82 tok/s`.
- Recommendation: keep packed16/simple pair helper path. The packed32 paired load adds shift/register pressure and does not beat E086.

Artifact:

- `build_logs/agent-workload/e090-vulkan-q3-loadvec4-packed32pair-pp7488.md`