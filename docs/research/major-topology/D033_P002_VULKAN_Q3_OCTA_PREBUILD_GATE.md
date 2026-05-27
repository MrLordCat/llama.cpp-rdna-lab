# D033 P002 Vulkan Q3 Octa Prebuild Gate

Date: 2026-05-27.

## Candidate

`Q3_K` octa dequant body: a `LOAD_VEC_A=8` q3quad successor that would compute
eight values per invocation with one scale group.

## Gate

- Command: `python scripts/research/vulkan_q3k_prebuild_gate.py --candidate "D033 Q3_K octa dequant body: LOAD_VEC_A=8 q3quad successor computing eight values per invocation with one scale group" --local-gain-pct 20 --require-target-closing`.
- Artifact: `build_logs/agent-workload/d033-vulkan-q3-octa-prebuild-gate.md`.

## Result

The prebuild gate rejects this candidate before build:

- Required local speedup if only the hotspot changes: `1.2081x` (`+20.81%`).
- Candidate estimate: `+20.00%`, projected total `1.1360x`, below the target
  closure gate.
- Matched prior: E087 corrected Q3_K `LOAD_VEC_A=8`, measured `-1.50%` vs E086
  (`947.44` pp7488 r1 vs `961.82`).

## Decision

Reject q3-octa/`LOAD_VEC_A=8` as the D033 route. It is too close to the measured
negative E087 family and does not change enough of the Q3_K dataflow to be worth
a shader build or server A/B.