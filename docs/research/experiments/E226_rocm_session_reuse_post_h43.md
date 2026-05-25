# E226 ROCm post-H43 repeated-session route

## Metadata

- Experiment ID: E226
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H36
- Target lane: ROCm Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FA on, thinking on
- Binary: `build-rocm-vec/bin/llama-server.exe`

## Hypothesis

- Statement: after H43 default-on and the driver refresh, prompt cache / context checkpoints should still be the highest-leverage practical route for repeated agent sessions with shared repo prompt prefix.
- Mechanism: repeated tasks reuse a large stable prefix and restore context checkpoints, reducing prompt work from about `7.5k` tokens to about `2.0k` tokens after the first cold request. `ngram-mod 12/16/32` may stack on the decode part when repeated outputs expose draftable spans.
- Failure condition: if the repeated/session gain disappears in r3, or if `ngram-mod` lowers aggregate TPS versus reuse-only, keep only prompt-cache/checkpoints and reject the speculative stack for this lane.

## Benchmark Plan

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Common args:
  - `--tasks quick --task-ids triage_diff,review_bug --runs 3`
  - `--ctx-size 12288 --batch-size 6144 --ubatch-size 2048`
  - `--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 999`
  - `--max-tokens 64 --real-context-mode repo-snapshot`
  - `--no-v2-prime-pass --no-disable-thinking`
- Cold-control adds `--no-reuse` and `--server-extra "--spec-type none"`.
- Reuse-only candidate keeps default prompt cache/checkpoints and uses `--server-extra "--spec-type none"`.
- Reuse+ngram candidate keeps default prompt cache/checkpoints and uses:
  - `--server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-max 32"`

## Measured Results

| Label | Reuse | Spec | Aggregate TPS | Mean task TPS | Decode tok/s mean | Prompt ms mean | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `e226-rocm12k-cold-two-task-specnone-r3-seq` | off | none | `7.8890` | `7.89` | `30.45` | `5978.04` | same taskset cold-control |
| `e226-rocm12k-session-reuse-specnone-r3` | on | none | `13.5774` | `14.81` | `30.605` | `2590.05` | prompt checkpoints carry the main win |
| `e226-rocm12k-session-reuse-ngram12-16-32-r3-seq` | on | ngram-mod | `14.1202` | `15.79` | `35.0967` | `2606.59` | best measured repeated/session stack |

## Deltas

- Reuse-only vs cold-control: `13.5774 / 7.8890 = 1.721x`, or `+72.11%`.
- Reuse+ngram vs cold-control: `14.1202 / 7.8890 = 1.790x`, or `+78.99%`.
- Reuse+ngram vs reuse-only: `14.1202 / 13.5774 = 1.040x`, or `+4.00%`.
- Against the E224 active cold single-task reference (`7.5575 TPS`), reuse+ngram is `1.868x`, or `+86.84%`.

## Route Evidence

- Cold-control keeps every task cold: prompt eval averages `5978.04 ms` and `7479.5` task prompt tokens.
- Reuse-only restores a `5437`-token context checkpoint after the first request; prompt eval mean drops to `2590.05 ms`.
- Reuse+ngram keeps the same checkpoint behavior and improves decode mean from `30.605` to `35.0967 tok/s`.
- Final ngram stats in the r3 run: `196` generated draft tokens, `155` accepted tokens, local acceptance `0.7908`. Coverage is bursty: the first cold task generated no drafts, while later repeated tasks produced the useful decode lift.

## Workflow Correction

- Two attempted r3 controls were accidentally launched in parallel:
  - `e226-rocm12k-cold-two-task-specnone-r3`
  - `e226-rocm12k-session-reuse-ngram12-16-32-r3`
- Both timed out and are invalid because the second run detected an already running `llama-server`. Heavy ROCm real-server benchmarks must be run sequentially; these artifacts are retained only as a workflow warning, not as evidence.

## Result

- Outcome: keep as practical repeated/session route.
- Decision:
  - keep prompt cache / checkpoints enabled for GUI and agent sessions;
  - keep `ngram-mod 12/16/32` as the current best measured ROCm repeated/session stack for this short two-task lane;
  - do not present this as a cold-first kernel speedup.
- Interpretation:
  - the +20% user target is met for repeated/session workflow (`+78.99%` vs same-task cold-control);
  - the first request remains cold and still needs H42/H39 style structural route work for true cold-first prefill/decode gains;
  - `ngram-mod` is useful here only after prompt reuse exposes repeated decode spans, so cold ngram/spec claims still need separate coverage checks.

## Artifacts

- `build_logs/agent-workload/e226-rocm12k-cold-two-task-specnone-r3-seq.diagnostics.md`
- `build_logs/agent-workload/e226-rocm12k-session-reuse-specnone-r3.diagnostics.md`
- `build_logs/agent-workload/e226-rocm12k-session-reuse-ngram12-16-32-r3-seq.diagnostics.md`
- `build_logs/agent-workload/e226-rocm12k-session-reuse-ngram12-16-32-r3-seq.server.log`
- Invalid parallel artifacts:
  - `build_logs/agent-workload/e226-rocm12k-cold-two-task-specnone-r3.diagnostics.md`
  - `build_logs/agent-workload/e226-rocm12k-session-reuse-ngram12-16-32-r3.diagnostics.md`
