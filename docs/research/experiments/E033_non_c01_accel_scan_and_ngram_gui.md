# E033 - Non-C01 Acceleration Scan + Ngram GUI Preset

## Metadata

- Experiment ID: E033
- Date: 2026-05-16
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: non-C01 scan over H12 TurboKV, RDNA4 graph optimizer, MoE35B, and GUI ngram preset integration

## Hypothesis

- Statement: After closing the cheap C01 candidates, the next practical speed improvement may come from a broader runtime feature or from making an already-confirmed opt-in acceleration accessible through the GUI.
- Mechanism:
  - H12/TurboKV: if separate `TURBO_WHT` nodes are a meaningful cost, fusing or avoiding them could improve compressed-KV decode.
  - RDNA4 graph optimizer: if safe on the current lane, graph-level concurrency may reduce request wall time.
  - MoE/MMQ: MoE35B has a different topology and may expose non-C01 bottlenecks.
  - GUI ngram preset: confirmed `ngram-mod 24/48/64` should not be hidden behind stale GUI defaults or partial presets.
- Why now: C01 Q3/Q4/F32 cheap-route candidates are mostly closed, while E028/E029/E030 confirmed `ngram-mod 24/48/64` as an opt-in profile.

## Math / Theory

- Assumptions:
  - WHT fusion ceiling is bounded by measured `TURBO_WHT` time divided by total wall time.
  - Graph optimizer must beat the same no-trace lane, not a trace-instrumented run.
  - GUI preset integration is a productization win, not a new kernel speed claim.
- Expected speedup corridor:
  - WHT fusion: only promising if `TURBO_WHT` is several percent of wall time.
  - Graph optimizer: possible low-single-digit win if stable.
  - GUI ngram profile: inherits measured opt-in gains from E028/E029/E030, especially warm/session workloads.
- Failure conditions:
  - WHT share below practical threshold.
  - Graph optimizer equal to baseline or unstable.
  - MoE staging path no clear activation or no measured gain.

## Implementation Plan

1. Minimal code surface to change:
   - If no kernel/runtime candidate passes gate, update GUI ngram defaults and preset parsing only.
2. Guard rails:
   - Keep server default `spec=None`; ngram remains user opt-in.
   - Do not enable RDNA4 graph optimizer by default.
   - Do not change TurboKV or MoE defaults from single-run probes.
3. Rollback path:
   - Revert `gui/server_tab.py` if command generation changes are invalid.

## Benchmark Plan

- TurboKV trace:
  - q4 control: `h12-tkv-wht-q4-trace-r1`
  - mixed TurboKV: `h12-tkv-wht-turbo4q8-trace-r1`
- Graph optimizer A/B:
  - control: `g01-rdna4-graphopt-q4-control-r1`
  - candidate: `g01-rdna4-graphopt-q4-probe-r1`
- MoE smoke:
  - q4 control: `g02-moe-q4-control-r1.jsonl`
  - staging probe: `g02-moe-q4-staging-r1.log`
- Number of runs: `r1` only; this is a gate/scouting pass.

## Metrics

- aggregate completion TPS (wall)
- node timing share for `TURBO_WHT`
- MoE pp/tg throughput
- GUI command generation behavior for ngram presets

## Result

- Outcome: keep GUI feature; reject WHT fusion and graph optimizer as immediate speed candidates.
- Delta:
  - H12 trace: q4 `9.58 TPS`; `turbo4/q8_0` `9.23 TPS`.
  - `TURBO_WHT`: `48` nodes, `2.531 ms` total, max `0.386 ms`; too small to justify fusion as a near-term speed path.
  - Graph optimizer: candidate `10.98 TPS`; same-lane no-trace control `11.00 TPS`; no win.
  - MoE q4 smoke: pp512/2048/4096 `626.23/3527.23/3455.37 tok/s`, tg128 `101.01 tok/s`; staging probe tg128 `102.75 tok/s` but no clear activation trace or enough delta to keep pursuing from this pass.
- Confidence: medium for rejecting WHT fusion and graphopt; low-medium for MoE because it was a smoke/gate only.
- Recommendation:
  - Keep `ngram-mod 24/48/64` as the GUI opt-in default knobs.
  - When preset extra args contain only `--spec-type ngram-mod`, add the missing ngram knobs to the server command.
  - Continue future non-C01 search in MoE/MMQ only with a clearer route trace or a new staging design.

## Notes

- Code change:
  - `gui/server_tab.py` now uses `match=24`, `min=48`, `max=64` as ngram defaults.
  - It also fills missing `--spec-ngram-mod-*` arguments when Extra Args already contain `--spec-type ngram-mod`.
- This does not make speculative decoding the conservative default; it only makes the opt-in path match the measured profile.
