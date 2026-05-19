# E070 - H30 RDNA4 Q4_K/Q5_K MMQ Selector

## Metadata

- Experiment ID: E070
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: `master` / `22a5430c7` plus experiment patch
- Target lane: `models/Qwen3.6-27B-Q4_K_S.gguf`, ROCm RX 9070 XT, `ctx=12288`, `b=4096`, `ub=1024`, `q4_0/q4_0` KV, FlashAttention, thinking ON, no reuse, no prime pass

## Hypothesis

- Statement: On RDNA4, Q4_K/Q5_K large-prefill matmul should use MMQ up to `ne11<=1024` instead of falling back to the dequant+hipBLAS path after `ne11>192`.
- Mechanism: The downloaded Q4_K_S model is dominated by Q4_K/Q5_K tensors. For prompt-heavy chunks around 512-1024 columns, the default backend route spent too much time in quant dequant/staging; forced MMQ was 3-4x faster in `llama-bench`.
- Why now: The user switched from the tuned Q3 model to `Qwen3.6-27B-Q4_K_S.gguf`; GUI/server startup looked usable, but generation was very slow because auto-fit offloaded layers and the remaining full-offload path still used the wrong route.

## Math / Theory

- Assumptions: Active Q4 prompt eval wall share is high enough that a prefill-only win dominates total wall time.
- Expected speedup corridor: `pp512`/`pp1024` should move from roughly `60-75 tok/s` to `240-330 tok/s`; full task wall should move from timeout-scale (`>100s`) to under the GUI/autotune 60s soft fail window.
- Failure conditions: Q3_K/Q6_K regressions, Q4 decode regressions, or threshold-sensitive instability above `ne11=1024`.

## Implementation Plan

1. Minimal code surface to change: `ggml/src/ggml-cuda/mmq.cu` selector only.
2. Guard rails: apply the new threshold only on RDNA4 and only to `GGML_TYPE_Q4_K` / `GGML_TYPE_Q5_K`; keep Q2_K/Q3_K/Q6_K at the old `ne11<=192` gate.
3. Rollback path: set `GGML_MMQ_RDNA4_Q4K_MAX_NE11=192` to restore the old route without rebuilding.

## Benchmark Plan

- Baseline command: Q4 active workload with default auto-fit, then `-fitt 0`, then `-fit off` to separate fit/offload from route selection.
- Candidate command: same workload after selector change with `-fit off`.
- Negative control command: same build with `GGML_MMQ_RDNA4_Q4K_MAX_NE11=192`.
- Number of runs: `r=1` for iteration per current protocol.
- Artifacts path: `build_logs/agent-workload/e070-*`.

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- decode eval tok/s
- layer offload count
- Q4 `llama-bench` pp512/pp1024 tok/s
- MTP acceptance stats for opt-in speculative check

## Result

- Outcome: win
- Delta: Q4 pp512 improved from `57.30 tok/s` old default to `246.60 tok/s` new default; old-threshold negative control returned `58.12 tok/s`. Clean full Q4 active lane improved from timeout-scale (`122.23s`, `64.39 prompt tok/s`, 60/66 offloaded) to `28.44s`, `2.25 TPS`, `330.42 prompt tok/s`, full 66/66 offload with `-fit off`.
- Confidence: good for Q4_K_S on RX 9070 XT / ROCm; scoped by type and architecture, with env rollback.
- Recommendation: keep code change and update GUI Q4 preset to use `ctx=12288`, `b=4096`, `ub=1024`, `q4_0/q4_0`, `--spec-type none`, `-fit off`, thinking ON.

## Notes

- `-fit off` is required for the practical GUI profile because auto-fit initially offloaded 60/66 layers and `-fitt 0` still offloaded 64/66. The route fix addresses the full-offload slow path; it does not change fit policy.
- Clean MTP check was slower overall on this prompt-heavy lane: `39.17s`, `1.63 TPS`, `219.37 prompt tok/s`, `12.80 decode tok/s`, acceptance `41/63`. Keep MTP as opt-in, not Q4 default.
- Q3 negative control remained healthy: `Qwen3.6-27B-Q3_K_S` pp512 measured `502.03 tok/s` after the patch. The selector keeps Q3_K on the old threshold.
