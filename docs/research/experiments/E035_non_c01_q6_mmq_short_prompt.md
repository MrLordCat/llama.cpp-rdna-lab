# E035 Non-C01 Q6_K Short-Prompt MMQ Gate

## Metadata

- Experiment ID: E035
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local dirty `master`
- Target lane: non-C01 MoE smoke, `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, ROCm/RDNA4, `b=1024`, `ub=1024`, q4 KV

## Hypothesis

- Statement: The E034 broad `GGML_CUDA_FORCE_MMQ_RUNTIME=1` spike at `pp512` may come from `Q6_K` prompt-batch matmuls. A scoped RDNA4 runtime selector for `Q6_K` only when `ne11 <= 512` can capture the short-prompt gain without touching `p2048+` forms that regressed or hung.
- Mechanism: current RDNA4 MFMA selector routes `Q6_K` prompt batches with `ne11 > 128` to the backend. For the MoE model at `p512`, shared expert and attention/linear `Q6_K` calls have `ne11=512`. For `p2048`, the same lane chunks at `ne11=1024`, so a max-ne11 gate should avoid the known bad zone.
- Why now: E034 route traces showed `Q6_K` backend calls as the remaining non-routed-MMQ bucket, while routed MoE staging did not produce a keep candidate.

## Math / Theory

- Assumptions:
  - `p512` control after rebuild: `674.33 tok/s`.
  - A useful opt-in needs at least `+3%`, or about `694.56 tok/s`.
  - The gate must not reduce `p2048` or `tg128`; those are negative controls.
- Expected speedup corridor: `0%` to `+10%` on `pp512` if backend route overhead dominates; near `0%` on `pp2048` because it should remain backend-routed.
- Failure conditions:
  - Candidate hangs or fails to complete.
  - `pp512` does not beat same-session control by at least `+3%`.
  - `pp2048` or `tg128` regress materially.

## Implementation Plan

1. Minimal code surface to change: `ggml_cuda_should_use_mmq()` runtime selector only.
2. Guard rails: env-gated, RDNA4-only, `GGML_TYPE_Q6_K` only, positive `ne11` max from env.
3. Rollback path: revert selector patch if benchmark is not clearly positive.

## Benchmark Plan

- Baseline command: `llama-bench` MoE lane with `-p 512,2048 -n 128 -r 1 --no-warmup`.
- Candidate command: same lane with `GGML_RDNA4_FORCE_MMQ_Q6_MAX_NE11=512`.
- Number of runs: `r1` gate; `r3` only if `r1` is positive and stable.
- Artifacts path: `build_logs/agent-workload/g035-*`.

## Metrics

- `pp512`, `pp2048`, `tg128` tok/s
- route activation trace for `Q6_K`
- completion/no hang

## Result

- Outcome: reject / code reverted
- Delta:
  - Broad `Q6_K max-ne11<=512` gate: control `pp512=670.38`, `pp2048=3539.89`, `tg128=103.07`; candidate `pp512=1327.10`, `pp2048=2214.66`, `tg128=103.59`. This confirms a real short-prompt route opportunity but fails the `p2048` negative control badly.
  - Shared-expert-name scoped gate (`ffn_gate`, `ffn_up`, `ffn_shexp` only): control `pp512=667.38`, `pp2048=3546.52`, `tg128=102.95`; candidate `pp512=606.46`, `pp2048=3510.79`, `tg128=101.89`.
- Confidence: high that the tested gates should not be kept.
- Recommendation: Revert both Q6 selector patches. Future work should trace the broad-Q6 `pp512` route delta directly to find which non-shared `Q6_K` nodes produce the spike, then test a narrower selector with `p2048` as a mandatory negative control.

## Notes

- This is deliberately narrower than `GGML_CUDA_FORCE_MMQ_RUNTIME=1`, which was marked unsafe in E034.
- Both selector prototypes were reverted after failing the gate.
