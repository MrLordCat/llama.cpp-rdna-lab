# E043 Hypothesis Shortlist By Ceiling Gate

## Metadata

- Experiment ID: E043
- Date: 2026-05-17
- Owner: Copilot
- Type: process upgrade and candidate triage

## Objective

Apply the new Amdahl-based ceiling gate to `docs/research/HYPOTHESES.md` and produce an actionable shortlist for the current C01 context.

## Inputs

- Share used for C01 target center: `s = 0.39008672104`
- Wall goal for screen: `S_total = 1.02` (+2%)
- Required local speedup from gate: `S_local >= 1.0529` (local gain `>= 5.29%`)

Command:

```bash
python scripts/research/hypothesis_shortlist_by_ceiling.py --share 0.39008672104 --goal-total-speedup 1.02 --top 12
```

## Keep shortlist (by expected-impact ceiling)

- H08, H03, H01, H05, H02, H06, H07, H04, H13, H15, H12

## Drop/low-ceiling at this gate

- H17, H19
- H09/H10/H11/H14/H16/H18 are process/modeling hypotheses, not direct speedup candidates, and should be treated as enabling gates rather than runtime candidates.

## Integration result

`agent_workload_bench.py` now supports optional preflight ceiling checks before benchmark start:

- `--preflight-share`
- `--preflight-goal-total-speedup`
- `--preflight-candidate-local-speedup`
- `--preflight-enforce`

This allows hard-fail on non-viable candidates before server startup.

Smoke check:

```bash
python scripts/agent_workload_bench.py --label preflight-enforce-smoke --no-start --tasks quick --task-ids review_bug --preflight-share 0.39008672104 --preflight-goal-total-speedup 1.02 --preflight-candidate-local-speedup 1.03 --preflight-enforce
```

Observed:

- preflight verdict: FAIL
- process exits with code `6` before any server interaction

## Decision

- Keep this workflow upgrade.
- Use E043 shortlist as the primary queue for next runtime experiments.
- Keep low-ceiling/runtime-neutral hypotheses in enabling/modeling track only.
