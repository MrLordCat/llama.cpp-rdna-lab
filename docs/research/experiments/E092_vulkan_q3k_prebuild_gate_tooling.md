# E092 Vulkan Q3_K Prebuild Gate Tooling

## Metadata

- Experiment ID: E092
- Date: 2026-05-20
- Owner: Copilot
- Type: workflow/tooling
- Target lane: H31 Vulkan Q3_K prefill research

## Problem

The H31 loop was spending too much time on cheap but low-information shader probes. Many nearby ideas were already measured negative, yet they were easy to repeat in slightly different forms. After E093, the accepted baseline is E086 source-only, and the remaining ROCm gap is too large for blind `1-2%` micro-edits to be the default strategy.

## Tool

Added `scripts/research/vulkan_q3k_prebuild_gate.py`.

The tool runs without building. It:

- reads current Q3_K shader state from `mul_mm.comp`, `mul_mm_funcs.glsl`, and `vulkan-shaders-gen.cpp`;
- reports current stride/load-vector/static tile metrics;
- reports a Q3 dequant-reuse sanity model;
- detects leftovers from rejected Q3_K probes;
- computes Amdahl-style required local speedup from current Vulkan pp to ROCm pp;
- matches candidate text or an optional diff against H31 history in `RESULTS_LOG.md`;
- emits a prebuild decision such as `skip-build`, `needs-mechanism-estimate`, or `build-candidate`.

Current default target math after E093 correction:

- baseline pp: accepted E086 `961.82 tok/s`
- target pp: ROCm `1097.66 tok/s`
- hotspot share: E078 Q3_K MUL_MAT `0.7184`
- required local speedup if only that hotspot changes: `+20.81%`

## Smoke Checks

```bash
python scripts/research/vulkan_q3k_prebuild_gate.py
python scripts/research/vulkan_q3k_prebuild_gate.py --candidate "try Q3_K LOAD_VEC_A=8 with packed32 pair helper"
python scripts/research/vulkan_q3k_prebuild_gate.py --candidate "new Q3_K algorithmic shared-memory layout that could reduce dequant+coopmat route cost" --local-gain-pct 21 --require-target-closing
```

Observed behavior:

- current tree: detects stride18, Q3_K `LOAD_VEC_A=4`, corrected loadvec4, no rejected probe leftovers;
- dequant sanity: reports `1024` pair dequants per A tile, `512` A load invocations, `0.875` repeated scale-decode share, and the E088 `-0.20%` calibration for scale/helper reuse;
- bad analogue: `loadvec8 + packed32` is rejected without a build;
- dequant-reuse analogue: `block-level dequant reuse with fused pair decode` is `skip-or-doc-only` unless a separate instruction/load model shows substantial pair-count, LDS traffic, or coopmat-work reduction;
- BK-depth analogue: `BK=64` is `needs-resource-proof` because static scout shows only a K-loop/barrier tradeoff while full-K dequant/B traffic is unchanged and Q3 shared memory rises to `34816 B`;
- high-ceiling candidate: a claimed `+21%` local hotspot idea passes the target-closing gate.
- E093 correction: `wn48` is no longer the default accepted baseline because static scout marks it invalid for `BN=128`.

Smoke artifact:

- `build_logs/agent-workload/e092-vulkan-q3k-prebuild-gate-smoke.md`

## Integration

- Added VS Code task: `research: vulkan q3 prebuild gate`.
- Updated `docs/research/PERF_WORKSPACE.md` and `docs/research/HYPOTHESES.md` to require the gate before new H31 shader/code probes.

## Decision

Keep the tool. Future H31 work should first pass this prebuild gate unless it is purely a documentation update or a fixed benchmark control.