# D052 - P003 C2 requirements table for target13

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: measured analytical gate

## Scope

Convert D050/D051 gap into explicit payload-side requirements (C2) under
multiple metadata compaction levels.

## Method

New script:

- `scripts/research/q4_metacomp_c2_requirements.py`

Run:

- source target json:
  `build_logs/agent-workload/q4metacomp-target13-qwen36-27b-q4ks-r1.q4_metacomp_target.json`
- label: `q4metacomp-c2req-target13-qwen36-27b-q4ks-r1`

## Key table

| meta_save_frac | metadata saved | remaining gap | required payload bpw |
| ---: | ---: | ---: | ---: |
| 0.60 | 0.824 GiB | 1.180 GiB | 3.5701 |
| 0.70 | 0.961 GiB | 1.043 GiB | 3.6201 |
| 0.80 | 1.098 GiB | 0.906 GiB | 3.6701 |
| 0.90 | 1.236 GiB | 0.768 GiB | 3.7201 |
| 1.00 | 1.373 GiB | 0.631 GiB | 3.7701 |

Interpretation:

- Even with full metadata elimination, C2 still needs effective payload average
  about `3.7701 bpw`.
- If Phase1 metadata converter reaches only `~0.8` metadata save fraction, C2
  requirement tightens to about `3.6701 bpw`.

## Decision

Promote these numbers as hard engineering gates for C2 algorithm design:

1. C2 candidate must demonstrate target payload average in the
   `3.57-3.77 bpw` corridor (depending on achieved C1 fraction).
2. Any C2 prototype that cannot reach this corridor offline should be rejected
   before runtime integration.

## Artifacts

- `scripts/research/q4_metacomp_c2_requirements.py`
- `build_logs/agent-workload/q4metacomp-c2req-target13-qwen36-27b-q4ks-r1.q4_metacomp_c2_requirements.json`
- `build_logs/agent-workload/q4metacomp-c2req-target13-qwen36-27b-q4ks-r1.q4_metacomp_c2_requirements.md`
- `docs/research/major-topology/D050_P003_Q4_TARGET13_FEASIBILITY_GATE.md`
