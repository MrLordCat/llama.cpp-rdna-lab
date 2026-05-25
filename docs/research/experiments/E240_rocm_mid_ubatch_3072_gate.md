# E240 ROCm Mid-Ubatch 3072 Gate

## Metadata

- Experiment ID: E240
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `56acee5d3`
- Target lane: cold-first ROCm Qwen3.6-27B-Q3_K_S, `ctx=12288`, q4/q4 KV, FlashAttention on, full offload, no reuse, thinking on, `spec=none`

## Hypothesis

- Statement: `ubatch=3072` may reduce prompt chunk repetitions versus the active `ubatch=2048` lane without falling into the known `ubatch=4096` timeout/cliff.
- Mechanism: fewer prompt chunks can reduce repeated Q3_K staging/GEMM passes over the same layer weights, but larger per-ubatch compute buffers can trigger residency or kernel-shape cliffs.
- Why now: E224 rejected nearby outer-batch tuning and found `ubatch=4096` hard-timeout. Before deeper code work, one midpoint gate can test whether the cliff boundary leaves a usable high-ceiling no-code lane.

## Math / Theory

- Assumptions:
  - Current cold reference from E226: `7.8890 TPS`, prompt mean `5978.04 ms`, decode `30.45 tok/s`.
  - +20% target: `9.4668 TPS`.
  - E224 `ubatch=4096` hard-timed out with `batch_chunks max=7489`, while `ubatch=2048` remains stable.
- Expected speedup corridor:
  - If `ubatch=3072` is viable and reduces prompt chunks enough, a single r1 should show a clear prompt eval increase before any follow-up.
  - If r1 ties/regresses, do not sweep more ubatch sizes.
- Failure conditions:
  - hard timeout or no prompt completion;
  - prompt eval tie/regression versus current best;
  - decode degradation or server errors.

## Implementation Plan

1. Minimal code surface to change:
   - none.
2. Guard rails:
   - one candidate run only before deciding whether it deserves a paired control;
   - no reuse, no prime, `build-rocm-vec`, thinking on.
3. Rollback path:
   - no code rollback.

## Benchmark Plan

- Baseline command:
  - compare against E226 cold-control r3/current best for the same lane.
- Candidate command:
  - `batch=6144`, `ubatch=3072`, same cold quick `triage_diff`, `max_tokens=64`.
- Number of runs:
  - one run for gate; paired control only if candidate is materially positive.
- Artifacts path:
  - `build_logs/agent-workload/e240-*`.

## Metrics

- aggregate completion TPS
- prompt eval TPS / prompt eval ms
- decode tok/s
- errors/timeouts

## Result

- Outcome: rejected.
- Delta:
  - E226 current cold reference: `7.8890 TPS`, prompt mean `5978.04 ms`, decode `30.45 tok/s`.
  - Candidate `batch=6144,ubatch=3072`: `7.7050 TPS`, prompt mean `6196.89 ms`, decode `30.83 tok/s`, errors `0`.
  - The midpoint is viable but slower than the active `ubatch=2048` lane.
- Confidence: medium. Single-run gate is enough because the candidate is below the current reference and below the +20% target by a wide margin.
- Recommendation: do not continue mid-ubatch retuning. `ubatch=3072` avoids the `4096` timeout but does not improve cold-first TPS, so the route remains structural Q3_K/GDN/body work rather than more batch-size search.

## Notes

- This is intentionally a single midpoint test, not a new broad batch sweep.
- The prompt chunk metric stayed near the same effective outer split (`batch_chunks mean/max=3744.5/6144`), so the candidate did not produce a meaningful graph-level chunk reduction.
