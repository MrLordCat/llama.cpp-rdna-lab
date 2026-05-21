# E114 ROCm Session Ngram Match Scout

## Metadata

- Experiment ID: E114
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 917eff1db
- Driver: AMD `32.0.31007.5012`
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, reuse enabled, thinking on

## Hypothesis

- Statement: reducing `ngram-mod` match length below `12` may further increase repeated-session draft coverage and improve after-first TPS.
- Mechanism: E113 `ngram-mod 12/16/32` won because it raised effective acceptance to `0.035028`; a shorter match may trigger more drafts on repeated repo tasks.
- Why now: post-driver best is now a speculative session route, not a kernel route. The cheapest next improvement is to tune the accepted-token burst coverage without changing backend code.

## Math / Theory

- Assumptions: E113 reuse-only is `17.8934 TPS`; E113 `12/16/32` best r3 is `19.5051 TPS`, after-first mean `23.9038 TPS`.
- Expected speedup corridor: `+0%` to `+5%` over `12/16/32` if coverage rises while local acceptance stays high enough.
- Failure conditions: local acceptance drops enough that decode slows below `12/16/32`, or first-task overhead dominates aggregate.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: compare only against E113 repeated/session ngram baseline, not cold-first.
3. Rollback path: no code changes.

## Benchmark Plan

- Candidate A: reuse + `ngram-mod n-match=8, n-min=16, n-max=32`, r3.
- Promote only if it beats E113 `12/16/32` after-first mean or aggregate by more than noise.

## Metrics

- aggregate completion TPS
- after-first mean/median task TPS
- generated/accepted drafts and tokens
- local acceptance, coverage, effective acceptance
- decode eval tok/s

## Result

- Outcome: reject.
- Delta: E113 reuse + `ngram-mod 12/16/32` best was `19.5051 TPS`; match-8 `16/32` measured `14.2479 TPS`.
- Confidence: high. The regression is large and visible in both wall TPS and decode metrics.
- Recommendation: keep `ngram-mod 12/16/32` as the current session preset; do not probe shorter match lengths on this task mix without a new acceptance model.

## Notes

- Match-8 increased draft attempts but destroyed acceptance quality: `gen_tokens=544`, `acc_tokens=74`, local acceptance `0.136029`, effective acceptance `0.004251`.
- Decode mean fell to `20.3283 tok/s`, far below reuse-only `28.88 tok/s` and `12/16/32` best `35.2867 tok/s`.
- Why it failed: lower match length raised false-positive drafts. Verification/bookkeeping then added work while accepting too few tokens to offset target-model decode.
- Workflow update: for session ngram tuning, coverage must not be optimized alone. Require local acceptance high enough to keep effective acceptance near or above the E113 `12/16/32` level (`0.035028`), and reject configs that lower decode eval below reuse-only.
- Follow-up action: keep `12/16/32` as current best; next speculative tuning should adjust draft length or adaptive policy around match 12, not reduce match length further.
