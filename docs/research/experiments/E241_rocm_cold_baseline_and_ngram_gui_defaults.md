# E241 ROCm Cold Baseline and Ngram GUI Defaults

## Metadata

- Experiment ID: E241
- Date: 2026-05-25
- Owner: Codex
- Branch/Commit: master after `735e7b512`
- Target lane: cold-first ROCm Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, no reuse, thinking on, `spec=none`

## Hypothesis

- Statement: the measured repeated/session `ngram-mod 12/16/32` profile should be the only GUI/server ngram profile, but the cold-first search must keep using `spec=none` and the current same-snapshot baseline.
- Mechanism: E226 showed `12/16/32` stacks with prompt reuse/checkpoints, while E227 showed ngram is a cold-first tie. Making GUI/server/bench/autotune defaults agree prevents accidental launches with the older `48/24/64` profile.
- Why now: the user asked to make ngram launches apply the same measured parameters and then continue the +20% cold search.

## Math / Theory

- Assumptions:
  - Current cold-control run: `7.6932 TPS`, prompt `6204.02 ms`, prompt eval `1207.12 tok/s`, decode `30.74 tok/s`, errors `0`.
  - Current +20% cold target: `7.6932 * 1.20 = 9.2318 TPS`.
  - With prefill share around `0.746`, a prefill-only route needs roughly `1.30x` local prefill speed to reach the +20% wall target if decode is unchanged.
- Expected speedup corridor:
  - GUI/default changes have no cold speed claim.
  - Future code candidates must move the Q3_K/GDN wall mix, not the ngram profile.
- Failure conditions:
  - any future cold claim measured with reuse, prime, mismatched task, or a non-`build-rocm-vec` binary;
  - any ngram cold claim without effective acceptance and same-lane control.

## Implementation Plan

1. Minimal code surface to change:
   - `gui/server_tab.py` already normalizes `ngram-mod` server launches to `12/16/32`.
   - propagate the same profile to `gui/benchmark_tab.py`, `scripts/agent_workload_bench.py` autotune defaults, `scripts/large_context_realworld_bench.py`, and ngram presets in `gui/model_presets.json` / `gui/model_autotune_best.json`.
2. Guard rails:
   - keep the profile as a repeated/session launch profile, not a cold-first speed claim.
   - keep cold perf validation on `build-rocm-vec`, `spec=none`, no reuse, no prime.
3. Rollback path:
   - profile constants and JSON args can be restored to previous values if a future repeated/session profile supersedes E226.

## Benchmark Plan

- Baseline command:
  - `python scripts\agent_workload_bench.py --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --label e241-rocm12k-cold-control-r1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --gpu-layers 999 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --parallel 1 --max-tokens 64 --tasks quick --task-ids triage_diff --real-context-mode repo-snapshot --no-disable-thinking --no-reuse --runs 1 --background-server-policy fail --server-extra "--spec-type none"`
- Candidate command:
  - none; this is a baseline/default alignment checkpoint.
- Number of runs:
  - one run for recentering the current snapshot before new code work.
- Artifacts path:
  - `build_logs/agent-workload/e241-rocm12k-cold-control-r1.*`

## Metrics

- aggregate completion TPS
- prompt eval TPS / prompt eval ms
- decode tok/s
- error rate

## Result

- Outcome: keep GUI/default alignment; use the new cold-control number as the immediate same-snapshot baseline.
- Delta:
  - cold-control: `7.6932 TPS`, prompt `6204.02 ms`, prompt eval `1207.12 tok/s`, decode `30.74 tok/s`, errors `0`.
  - no speed candidate was tested in this checkpoint.
- Confidence: medium for baseline recentering. A single run is enough to establish the next r1 gate reference, while any promotion candidate still needs a same-session control if the delta is close.
- Recommendation:
  - keep `ngram-mod 12/16/32` as the GUI/server/bench/autotune/large-context ngram profile;
  - keep cold-first performance work on `spec=none`;
  - continue with structural ROCm Q3_K/GDN route-body work because the cold +20% target is about `9.23 TPS`.

## Notes

- Analytic gates passed:
  - `python scripts/research/formula_sanity_checks.py`
  - `python scripts/research/speedup_model.py` at prefill speedups `1.10`, `1.20`, and `1.30`.
- Projected wall speedups from the current baseline:
  - `1.10x` prefill only: `8.2529 TPS` (`1.0728x`)
  - `1.20x` prefill only: `8.7855 TPS` (`1.1420x`)
  - `1.30x` prefill only: `9.2930 TPS` (`1.2080x`)
- This confirms that small post-H43 micro-wins are insufficient unless they stack across the dominant Q3_K route.
