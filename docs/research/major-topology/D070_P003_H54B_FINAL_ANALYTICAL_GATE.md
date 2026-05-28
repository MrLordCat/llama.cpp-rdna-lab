# D070 P003 H54-B final analytical gate

Date: 2026-05-28
Owner: Copilot/perf workspace
Mode: theory-only final package (entropy + quality + complexity)

## Context

D067/D068/D069 established stable entropy improvement for H54-B from sampled to
wide scopes. D070 closes the analytical package with explicit contracts and a
single pass/fail decision.

## Inputs

Primary artifact:

- `build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-wide-r3.q4_c2_value_aware_gate.json`

Final gate runner:

- `scripts/research/q4_c2_value_aware_final_gate.py`

Command:

- `python scripts/research/q4_c2_value_aware_final_gate.py --value-aware-json build_logs/agent-workload/q4c2-value-aware-qwen36-27b-q4ks-wide-r3.q4_c2_value_aware_gate.json --label q4c2-value-aware-final-qwen36-27b-q4ks-r1`

## Contracts

1. Payload corridor upper bound: `new_entropy_bpw <= 3.7701`
2. Quality budget: `weighted_nrmse <= 0.115`
3. Complexity budget: `complexity_index <= 1.35`

Complexity model is an analytical proxy aligned with D058 event weights.

## Results

From:

- `build_logs/agent-workload/q4c2-value-aware-final-qwen36-27b-q4ks-r1.q4_c2_value_aware_final_gate.json`
- `build_logs/agent-workload/q4c2-value-aware-final-qwen36-27b-q4ks-r1.q4_c2_value_aware_final_gate.md`

Observed:

- `new_entropy_bpw = 3.277495` -> PASS
- `weighted_nrmse = 0.101327` -> PASS
- `complexity_index = 1.127917` -> PASS
- Tensors analyzed: `348`
- Total sampled elements: `45,613,056`

Final decision:

- **authorize_guarded_prototype**

## Interpretation

H54-B now has a complete analytical pass package (compression, quality, and
complexity) with all contracts satisfied in wide scope.

## Decision

- Promote H54-B from theory-only candidate to **guarded prototype authorization**.
- This is not a default rollout decision.

## Required follow-up before default consideration

1. Runtime A/B on representative lanes (prefill-heavy + decode-focused).
2. Quality/task validation (same task set and strict comparability contract).
3. Fail-closed flags and rollback switch for prototype path.
