# D058 - P003 Ck-4 decode complexity budget gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only checkpoint (analytical budget)

## Purpose

Close Ck-4 from
`docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md` by defining and
executing an analytical decode-complexity budget model for surviving C2
candidates.

## Method

New script:

- `scripts/research/q4_c2_decode_budget.py`

Run:

```bash
python scripts/research/q4_c2_decode_budget.py \
  --label q4c2-decode-budget-qwen36-27b-q4ks-r1
```

Inputs carried from prior full gates:

- D052 corridor upper bound: required payload `<= 3.7701 bpw`.
- D055 modeled payload signals:
  - H45 entropy floor: `3.864885 bpw`
  - H46 active-symbol fixed-bits expectation: `3.999999 bpw`
- D057 modeled payload signal:
  - H47 best full tuple proxy: `4.500000 bpw`

Complexity proxy (theory-only, not runtime benchmark):

- baseline legacy Q4 decode index = `1.0`
- weighted event terms:
  - extra bitstream ops (`0.18`)
  - table lookups (`0.22`)
  - branch events (`0.35`)
  - random reads (`0.30`)
  - state updates (`0.12`)
- hard primary budget: `complexity_index <= 1.35`

## Results

From
`build_logs/agent-workload/q4c2-decode-budget-qwen36-27b-q4ks-r1.q4_c2_decode_budget.json`:

| Candidate | Modeled bpw | Gap vs 3.7701 | Complexity index | Decision |
| --- | ---: | ---: | ---: | --- |
| H45 (EBNS) | 3.864885 | +0.094785 | 1.512500 | auxiliary_only |
| H46 (SPRE) | 3.999999 | +0.229899 | 1.257000 | reject_primary |
| H47 (PDNT) | 4.500000 | +0.729900 | 1.704000 | reject_primary |

## Interpretation

1. No candidate currently qualifies as a primary standalone C2 route under the
   combined compression+complexity gate.
2. H45 is the only near-corridor candidate; it remains admissible only as an
   auxiliary component for Ck-5 mixed-policy synthesis.
3. H46 and H47 are rejected as primary routes at Ck-4 because compression gaps
   are too large, and additional decode complexity is not justified by gain.

## Gate decision

1. Ck-4 status: closed.
2. Primary shortlist entering Ck-5: none (standalone).
3. Auxiliary shortlist entering Ck-5: H45 only.
4. H46/H47 remain closed as primary routes unless new evidence materially
   changes compression and/or complexity model.

## Next step

- Ck-5: build mixed-policy optimizer using H45 as the only surviving auxiliary
  component and quantify whether global policy can enter the `3.57-3.77 bpw`
  corridor under bounded complexity.

## Related artifacts

- `docs/research/major-topology/D052_P003_Q4_C2_REQUIREMENTS_TABLE.md`
- `docs/research/major-topology/D055_P003_Q4_CK1_CK2_SYMBOL_ESCAPE_FULL_GATE.md`
- `docs/research/major-topology/D057_P003_Q4_CK3_TUPLE_ATLAS_FULL_GATE.md`
- `scripts/research/q4_c2_decode_budget.py`
- `build_logs/agent-workload/q4c2-decode-budget-qwen36-27b-q4ks-r1.q4_c2_decode_budget.json`
- `build_logs/agent-workload/q4c2-decode-budget-qwen36-27b-q4ks-r1.q4_c2_decode_budget.md`
