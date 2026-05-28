# D071 - P003 H54-B guarded prototype plan

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: guarded-prototype planning pass

## Scope

Convert D070 analytical authorization into a concrete guarded prototype plan
without changing runtime defaults.

## What Was Implemented

New script:

- `scripts/research/q4_metacomp_guarded_prototype.py`

Inputs:

- C1 phase1 plan artifact (`q4_metacomp_phase1_plan.json`)
- H54-B wide artifact (`q4_c2_value_aware_gate.json`)

Outputs:

- guarded prototype manifest (`.json`)
- compact summary (`.md`)

Guard contracts:

- per-tensor selection by entropy gain and quality budget,
- fail-closed fallback policy (non-selected tensors stay legacy Q4 path),
- safety factor for projected C2 savings.

## Run

- model: `models/Qwen3.6-27B-Q4_K_S.gguf`
- target: `13.0 GiB`
- quality budget: `nrmse <= 0.115`
- entropy gain threshold: `>= 0.45 bpw`
- C2 safety margin: `0.90`
- label: `q4metacomp-guarded-prototype-qwen36-27b-q4ks-r1`

## Results

Projection:

- total model size: `15.004 GiB`
- C1 projected save: `1.373 GiB`
- C2 projected save (raw): `1.668 GiB`
- C2 projected save (safe): `1.501 GiB`
- projected size after C1+C2-safe: `12.130 GiB`
- target: `13.000 GiB`
- target headroom: `0.870 GiB`

Selection:

- selected tensors: `348/348`
- fallback tensors: `0/348`

## Decision

Keep as guarded prototype planning artifact.

- This is not runtime rollout.
- Next stage remains mandatory: runtime A/B plus task-quality validation with
  explicit rollback switch.

## Artifacts

- `scripts/research/q4_metacomp_guarded_prototype.py`
- `build_logs/agent-workload/q4metacomp-guarded-prototype-qwen36-27b-q4ks-r1.q4_metacomp_guarded_prototype.json`
- `build_logs/agent-workload/q4metacomp-guarded-prototype-qwen36-27b-q4ks-r1.q4_metacomp_guarded_prototype.md`
