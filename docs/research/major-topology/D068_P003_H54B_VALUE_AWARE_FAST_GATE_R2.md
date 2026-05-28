# D068 P003 H54-B value-aware fast gate (representative R2)

Date: 2026-05-28
Owner: Copilot/perf workspace
Mode: theory-only (fast analytical gate, representative spread sample)

## Context

D067 produced a first positive H54-B signal on 24 tensors. D068 extends that to a
broader representative sample and adds structured artifacts (JSON/MD) from the gate
script.

## Scope and command

Script:

- `scripts/research/q4_c2_value_aware_gate.py`

Run:

- `python scripts/research/q4_c2_value_aware_gate.py --model models/Qwen3.6-27B-Q4_K_S.gguf --sample 48 --sample-strategy spread --max-elements-per-tensor 262144 --label q4c2-value-aware-qwen36-27b-q4ks-fast-r2`

## Method details

- Q4 tensors are sampled with spread strategy across tensor list (not only head).
- Each tensor is capped at `262,144` elements for stable fast-gate latency.
- For each tensor:
  - baseline nibble entropy is measured,
  - Lloyd-Max 16-level codebook is fitted,
  - value-aware re-quant entropy is measured,
  - MSE and normalized RMSE are reported.

## Results

Artifacts:

- `build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-fast-r2.q4_c2_value_aware_gate.json`
- `build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-fast-r2.q4_c2_value_aware_gate.md`

Key summary:

- Tensors analyzed: `48`
- Total sampled elements: `12,582,912`
- Original entropy: `3.865866 bpw`
- New entropy: `3.277582 bpw`
- Entropy delta: `-0.588283 bpw`
- Weighted NRMSE: `0.101195`
- Feasible tensors under corridor upper bound `3.77`: `48/48`

Gate verdict:

- **PASS** on representative fast-gate scope.

## Decision

- Keep H54-B as the active continuation path.
- Move to full analytical gate with stricter quality and decode-complexity checks.

## Next

1. Full analytical H54-B pass (all Q4 tensors, deterministic convergence limits).
2. Add explicit quality gate contract (NRMSE or bounded error target) to pass/fail logic.
3. Add decode-complexity proxy vs baseline Q4 decode path.
4. Re-run Ck-4/Ck-5 style feasibility framing for H54-B.
