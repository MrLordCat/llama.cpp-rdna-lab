# E060 H03 ngram+MTP chain smoke

## Metadata

- Experiment ID: E060
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 baseline checkpoint
- Target lane: Qwen3.6-27B Q4_K_S, ROCm RX 9070 XT, ctx=12288, b=4096, ub=512, q4_0/q4_0, no reuse, no v2 prime

## Hypothesis

- Statement: A single `ngram-mtp` speculative mode can cheaply try `ngram-mod` first and fall back to MTP when ngram cannot draft.
- Mechanism: Reuse the existing `common_speculative` implementation list as a fixed chain ordered `ngram_mod -> mtp`.
- Why now: GUI autotune can already compare `ngram-mod` and `mtp` separately; the missing piece is one run that can exercise both under the same server context.

## Math / Theory

- Assumptions: ngram coverage and MTP coverage differ by text span, and fallback overhead stays below the acceptance gain.
- Expected speedup corridor: viability first; potential wall change from regression to low positive single digits on mini smoke.
- Failure conditions: MTP context desynchronizes after ngram drafts, ngram minimum blocks MTP fallback, or fallback overhead exceeds any coverage gain.

## Implementation Plan

1. Minimal code surface to change: add `COMMON_SPECULATIVE_TYPE_NGRAM_MTP`, parse `--spec-type ngram-mtp`, initialize both ngram and MTP implementations.
2. Guard rails: keep mode opt-in and expose it as experimental in GUI/autotune only when the model supports MTP.
3. Rollback path: remove the enum/parser value and Python GUI/harness additions; baseline checkpoint is `8c1195ab4`.

## Benchmark Plan

- Baseline command: `scripts/agent_workload_bench.py --autotune --label hybrid-spec-baseline ... --autotune-spec-values ngram-mod,mtp --runs 1 --max-tokens 20`
- Candidate command: same lane with `--autotune-spec-values ngram-mod,mtp,ngram-mtp` and `--max-tokens 64` so `ngram-mod` can pass its default `n_min=48` gate.
- Number of runs: 1 for mini viability.
- Artifacts path: `build_logs/agent-workload/`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- acceptance stats from server log

## Result

- Outcome: tie / experimental keep
- Delta: `ngram-mtp` 13.54 TPS vs `mtp` 13.53 TPS (`+0.07%`, noise), and vs `ngram-mod` 10.91 TPS (`+24.1%`).
- Confidence: low for speed, sufficient for viability smoke.
- Recommendation: keep `ngram-mtp` opt-in only; do not promote as default until a prompt/task with positive ngram coverage shows a real gain over pure MTP.

## Notes

- Baseline mini result before code changes: `ngram-mod` 8.64 TPS, `mtp` 8.21 TPS, best `ngram-mod`.
- First candidate smoke exposed a fallback bug: `common_speculative_n_min()` used the max `n_min` across implementations, so `ngram_mod`'s `n_min=48` blocked MTP fallback near the end of a short decode. After passing `n_draft_max` into `common_speculative_draft()` and skipping only implementations whose local minimum does not fit, MTP fallback covered the full mini decode.
- Final smoke artifacts: `hybrid-spec-ngram-mtp-fallback-smoke-autotune-summary.csv`, `hybrid-spec-ngram-mtp-fallback-smoke-cfg01.server.log`, `hybrid-spec-ngram-mtp-fallback-smoke-cfg02.server.log`, `hybrid-spec-ngram-mtp-fallback-smoke-cfg03.server.log`.
- Final server stats for `ngram-mtp`: ngram initialized but generated 0 draft tokens on `triage_diff`; MTP fallback generated 48 and accepted 46 (`0.95833`). This validates fallback, not positive ngram routing benefit.