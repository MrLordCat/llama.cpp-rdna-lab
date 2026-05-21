# E113 Driver Update Rebaseline

## Metadata

- Experiment ID: E113
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 9fdc32b0d plus local driver rebaseline notes
- Hardware: AMD Radeon RX 9070 XT
- Driver: AMD `32.0.31007.5012`, driver date `2026-05-12`
- Target lane: Qwen3.6-27B-Q3_K_S ROCm, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, thinking on

## Hypothesis

- Statement: after the video driver update, all ROCm/Vulkan performance baselines must be refreshed before new TPS claims.
- Mechanism: ROCm runtime scheduling, residency, compiler paths, and Vulkan shader/runtime behavior can shift after a driver update even with unchanged code.
- Why now: the previous accepted route metrics were recorded before the driver update. Comparing new candidates to those old values would mix software and driver effects.

## Math / Theory

- Assumptions: the active lane contract stays identical, only the driver/runtime layer changed.
- Expected speedup corridor: unknown; driver can improve or regress prompt, decode, or repeated-session cache behavior independently.
- Failure conditions: if new baseline variance is high, use r3/r6 before deciding whether a candidate is real.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: do not compare post-driver candidates to pre-driver baselines except as historical context.
3. Rollback path: no code changes.

## Benchmark Plan

- ROCm cold-first r3: no reuse, `--cache-ram 0 --ctx-checkpoints 0`, `spec=none`.
- ROCm repeated/session r3: reuse enabled, `spec=none`.
- ROCm stacked repeated/session r3: reuse enabled, `ngram-mod 24/48/64`.
- ROCm shorter stacked repeated/session r3/r3b: reuse enabled, `ngram-mod 12/16/32`.
- ROCm repeated/session negative control: reuse enabled, `ngram-simple n8/m16`.
- Vulkan sanity/probe: refresh only after ROCm anchors, with timeout-aware command selection.

## Metrics

- aggregate completion TPS
- per-task wall TPS
- prompt/decode eval tok/s
- prompt-cache/checkpoint log evidence
- ngram draft stats when enabled
- driver version fingerprint

## Result

- Outcome: post-driver baseline refreshed; new repeated/session best is reuse + `ngram-mod 12/16/32`.
- Delta:
  - Cold-first ROCm `spec=none`: `11.9858 TPS` r3, up about `+1.18%` vs E106 `11.8464` historical control.
  - Reuse ROCm `spec=none`: `17.8934 TPS` r3, after-first mean `20.2012 TPS`.
  - Reuse + `ngram-mod 24/48/64`: mixed after driver, `17.7270 TPS` r3 then `18.4637 TPS` r3b; keep as noisy but still plausible.
  - Reuse + `ngram-mod 12/16/32`: `19.0148 TPS` r3 and `19.5051 TPS` r3b; after-first means `23.1681` and `23.9038 TPS`.
  - Reuse + `ngram-simple n8/m16`: `15.3491 TPS`, reject.
  - Vulkan pp7488: first post-driver run `900.22 +/- 151.13 tok/s`, second warmed run `962.41 +/- 33.93 tok/s`; ROCm pp7488 `1159.49 +/- 73.80 tok/s`.
- Confidence: high that cold/reuse baselines moved only slightly; medium-high that `ngram-mod 12/16/32` is the new best practical repeated route; high that `ngram-simple` is bad on this lane.
- Recommendation: use post-driver E113 as the new baseline set. For practical GUI/agent sessions prefer prompt cache/checkpoints plus `ngram-mod 12/16/32`; for cold-first kernel claims keep `spec=none`, no reuse, and compare against `11.9858 TPS`.

## Notes

- `ngram-mod 12/16/32` changes the mechanism: effective acceptance rose to `0.035028` (`gen_tokens=484`, `acc_tokens=320`), enough to make repeated tasks run in the `23-26 TPS` range. The first cold task is slower than reuse-only because early low-quality drafts add overhead before the useful repeated pattern appears.
- `ngram-mod 24/48/64` kept the same sparse burst stats as E112 (`102/126` accepted), but post-driver r3 results were noisy. Treat it as superseded by `12/16/32` for this lane.
- `ngram-simple n8/m16` generated drafts but lowered decode to `22.70 tok/s`; local acceptance `0.359375` and effective acceptance `0.007855` were not enough to pay for verification/bookkeeping. This closes `ngram-simple` for the repeated q4-KV route.
- Vulkan did not get a driver breakthrough. The warmed pp7488 gate is still about `17%` behind ROCm on raw prompt throughput.
- `gui/model_autotune_best.json` was updated with the new post-driver cold entry (`active-ctx12k`) and a separate `active-ctx12k-session` entry so cold-first and practical session presets remain distinct.
- Follow-up action: use the separate session entry when launching practical repeated tasks; do not write session numbers into cold-first benchmark defaults.
