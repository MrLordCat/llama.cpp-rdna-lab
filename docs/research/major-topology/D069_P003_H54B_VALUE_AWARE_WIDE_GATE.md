# D069 P003 H54-B value-aware wide gate

Date: 2026-05-28
Owner: Copilot/perf workspace
Mode: theory-only (wide analytical gate)

## Context

D067 and D068 passed fast representative gates for H54-B. D069 extends evidence to
all detected Q4 tensors with per-tensor element cap for stable runtime.

## Command

- `python scripts/research/q4_c2_value_aware_gate.py --model models/Qwen3.6-27B-Q4_K_S.gguf --sample 0 --sample-strategy spread --max-elements-per-tensor 131072 --label q4c2-value-aware-qwen36-27b-q4ks-wide-r3`

## Scope

- Q4 tensors analyzed: `348` (all detected Q4 tensors)
- Sampled elements: `45,613,056`
- Per-tensor cap: `131,072` elements

## Results

From artifacts:

- `build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-wide-r3.q4_c2_value_aware_gate.json`
- `build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-wide-r3.q4_c2_value_aware_gate.md`

Summary:

- Original entropy: `3.864270 bpw`
- New entropy: `3.277495 bpw`
- Entropy delta: `-0.586775 bpw`
- Weighted NRMSE: `0.101327`
- Feasible tensors under corridor upper bound `3.77`: `348/348`

Gate verdict:

- **PASS** on wide scope.

## Decision

- H54-B remains the active continuation route.
- Evidence is now consistent across fast (24), representative (48), and wide (348)
  analytical scopes.

## Next

1. Define explicit quality contract for pass/fail (bounded weighted NRMSE and/or
   bounded relative error vs baseline policy).
2. Add decode-complexity proxy for value-aware decode path and compare against
   Ck-4 budget envelope.
3. Execute final full-gate package and decide prototype authorization for H54-B.
