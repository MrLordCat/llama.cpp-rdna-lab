# E047 H08 Prefill ubatch2048 GDN Chunk Sweep

## Metadata

- Experiment ID: E047
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: `GATED_DELTA_NET` chunk size may need to be retuned for the new `ubatch=2048` prefill baseline.
- Mechanism: E045 recentered the prefill lane from `ubatch=1024` to `ubatch=2048`. The RDNA4 GDN default still uses `chunk_size=128` for `n_tokens > 256`, which means 16 GDN kernel launches at `n_tokens=2048`. Larger chunks may reduce launch/state handoff overhead if the longer inner token loop does not create a register or residency cliff.
- Why now: GDN chunk/fast-exp probes were negative on the earlier `ubatch=1024` gate, but E045 `ubatch=2048` trace shows GDN is `14.50%` of prompt node time and was not independently swept at this new physical batch size.

## Math / Theory

- Assumptions:
  - E045 `ubatch=2048` prompt trace total: `13969.454 ms`.
  - `GATED_DELTA_NET`: `2024.980 ms`, `14.50%` of prompt trace.
  - Same-session E046 baseline total server time was about `10145 ms`, prompt about `6156 ms`.
- Expected speedup corridor:
  - If GDN local time improves by `10%`, total wall should improve by roughly `0.9%`.
  - If GDN local time improves by `20%`, total wall should improve by roughly `1.8%`.
  - Below `~0.5%` wall gain, treat r1 as noise and do not promote.
- Failure conditions:
  - Larger chunks increase per-launch token loop pressure and slow the kernel.
  - Single large chunk behaves like the known large-ubatch residency cliffs.
  - Fast-exp changes math/quality risk without enough speedup.

## Implementation Plan

1. Minimal code surface to change: none for the gate; use existing `GGML_GDN_CHUNK_SIZE` and `GGML_GDN_FAST_EXP` knobs.
2. Guard rails: only consider a default-policy code change if r1 is positive and r3 confirms both aggregate TPS and prompt eval.
3. Rollback path: unset env vars; no code rollback needed for the gate.

## Benchmark Plan

- Baseline command:
  - `python scripts/agent_workload_bench.py --label prefill-e047-gdn-base-r1 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"`
- Candidate command:
  - same command with `GGML_GDN_CHUNK_SIZE=256`, `512`, `1024`, `2048`, and optional fast-exp only if chunk results look promising.
- Number of runs:
  - r1 sweep, r3 only for a clear positive.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e047-gdn-*.csv`
  - `build_logs/agent-workload/prefill-e047-gdn-*.server.log`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- GDN path/timing trace if a candidate is near-positive

## Result

- Outcome: negative.
- Baseline/default: `prefill-e047-gdn-base-r1 = 11.82 TPS`, prompt eval `1208.66 tok/s`, prompt mean `6137.03 ms`, decode eval `30.11 tok/s`.
- Candidates:
  - `GGML_GDN_CHUNK_SIZE=256`: `11.73 TPS`, prompt eval `1199.82 tok/s`.
  - `GGML_GDN_CHUNK_SIZE=512`: `11.45 TPS`, prompt eval `1164.67 tok/s`.
  - `GGML_GDN_CHUNK_SIZE=1024`: `11.48 TPS`, prompt eval `1159.85 tok/s`.
  - `GGML_GDN_CHUNK_SIZE=2048`: `11.45 TPS`, prompt eval `1154.14 tok/s`.
- Delta: best candidate `256` is `-0.76%` aggregate and `-0.73%` prompt; larger chunks regress by roughly `-2.9%` to `-3.1%` aggregate.
- Confidence: r1 sweep is enough to reject because every candidate moved below the same-session default.
- Recommendation: keep RDNA4 default `chunk_size=128` for `n_tokens > 256`; do not promote larger GDN chunks for `ubatch=2048`.

## Notes

- Surprises: reducing launch count did not help; the longer per-chunk token loop likely hurts occupancy/residency more than it saves launch overhead.
- Follow-up action: close H08 for this prefill lane unless a future trace shows a different GDN shape or allocator regime.
