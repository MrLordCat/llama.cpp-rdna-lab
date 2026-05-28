# D059 - P003 Ck-5 mixed-policy optimizer gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only checkpoint (analytical optimizer)

## Purpose

Close Ck-5 from
`docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md` by testing whether
any mixed C2 policy can satisfy both compression corridor and decode complexity
budget.

## Method

New script:

- `scripts/research/q4_c2_mixed_policy_optimizer.py`

Run:

```bash
python scripts/research/q4_c2_mixed_policy_optimizer.py \
  --label q4c2-mixed-policy-qwen36-27b-q4ks-r1 \
  --step 0.05
```

Route set (from prior full gates):

- H45 payload `3.864885 bpw`, complexity `1.512500`
- H46 payload `3.999999 bpw`, complexity `1.257000`
- H47 payload `4.500000 bpw`, complexity `1.704000`
- fallback Q4 payload `4.000000 bpw`, complexity `1.000000`

Constraints:

- payload corridor: `[3.57, 3.7701] bpw` (D052)
- complexity budget: `<= 1.35` (D058)

## Results

From
`build_logs/agent-workload/q4c2-mixed-policy-qwen36-27b-q4ks-r1.q4_c2_mixed_policy.json`:

- feasible policy count on 5% grid: `0`
- best payload among searched mixtures: `3.864885 bpw`
- best payload mixture: `H45=1.00, H46=0.00, H47=0.00, fallback=0.00`
- complexity at best payload mixture: `1.512500`

## Interpretation

1. No mixed policy can enter the required corridor because every available route
   already sits above corridor upper bound.
2. Convex mixtures cannot beat the minimum component payload bpw; current floor
   is H45 at `3.864885 bpw`, still above `3.7701`.
3. Even the best-payload point also violates complexity budget.

## Gate decision

1. Ck-5 status: closed.
2. Prototype unlock status for P003 current portfolio: denied.
3. No top-2 prototype shortlist is produced because no feasible candidate or
   mixture satisfies both compression and complexity gates.
4. Re-open condition: introduce materially new C2 hypothesis with modeled payload
   at or below corridor and bounded decode complexity proof.

## Related artifacts

- `docs/research/major-topology/D052_P003_Q4_C2_REQUIREMENTS_TABLE.md`
- `docs/research/major-topology/D058_P003_Q4_CK4_DECODE_COMPLEXITY_BUDGET_GATE.md`
- `scripts/research/q4_c2_mixed_policy_optimizer.py`
- `build_logs/agent-workload/q4c2-mixed-policy-qwen36-27b-q4ks-r1.q4_c2_mixed_policy.json`
- `build_logs/agent-workload/q4c2-mixed-policy-qwen36-27b-q4ks-r1.q4_c2_mixed_policy.md`
