# D060 - P003 Q4 C2 reopen admission gate (H49-H52)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only continuation gate

## Purpose

Continue P003 autonomously after D059 by defining a strict admission gate for a
new hypothesis family instead of reopening rejected H45-H47 routes.

## Method

New script:

- `scripts/research/q4_c2_reopen_admission_gate.py`

Run:

```bash
python scripts/research/q4_c2_reopen_admission_gate.py \
  --label q4c2-reopen-admission-qwen36-27b-q4ks-r1
```

Admission constraints inherited from prior closed gates:

- payload corridor upper bound from D052: `<= 3.7701 bpw`
- decode complexity budget from D058: `<= 1.35`

Important scope note:

- this gate is projection-only planning,
- values are not measured runtime/compression evidence,
- admission only means a hypothesis may enter the next analytical queue.

## Screened candidates

From
`build_logs/agent-workload/q4c2-reopen-admission-qwen36-27b-q4ks-r1.q4_c2_reopen_admission.json`:

| ID | projected bpw | projected complexity | Decision |
| --- | ---: | ---: | --- |
| H49 | 3.7400 | 1.3300 | admit_research |
| H50 | 3.7600 | 1.2900 | admit_research |
| H51 | 3.7900 | 1.2700 | park_compression |
| H52 | 3.7000 | 1.4800 | park_complexity |

## Gate decision

1. Reopen continuation is accepted only for H49 and H50.
2. H51 is parked until a stronger compression projection is shown.
3. H52 is parked until decode complexity is reduced under budget.
4. P003 remains in theory-only mode; no converter/runtime prototypes are
   authorized by D060.

## Next analytical queue

1. H49: conditional-entropy atlas and page-overhead proof with explicit random
   access contract.
2. H50: bounded-rANS micropage overhead model with deterministic decode-state
   bounds.
3. Re-run Ck-1..Ck-5 style closures for admitted candidates only.

## Related artifacts

- `docs/research/major-topology/D059_P003_Q4_CK5_MIXED_POLICY_OPTIMIZER_GATE.md`
- `scripts/research/q4_c2_reopen_admission_gate.py`
- `build_logs/agent-workload/q4c2-reopen-admission-qwen36-27b-q4ks-r1.q4_c2_reopen_admission.json`
- `build_logs/agent-workload/q4c2-reopen-admission-qwen36-27b-q4ks-r1.q4_c2_reopen_admission.md`
