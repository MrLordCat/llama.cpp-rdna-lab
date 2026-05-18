# E057 H27 Q3_K Explicit Unroll4 Probe

## Metadata

- Experiment ID: E057
- Date: 2026-05-18
- Owner: Copilot
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

The current Q3_K fp16 conversion kernel emits four scalar stores through a short runtime-indexed loop. An explicit four-store variant may let HIP generate a slightly better schedule without changing the 64-thread geometry, block layout, or default code path.

## Candidate

- Code: templated Q3_K kernel with an env-gated explicit unroll4 path.
- Guard: `GGML_CUDA_Q3K_DEQUANT_UNROLL4=1`.
- Default path: unchanged loop instantiation.

## Acceptance Policy

- r1 can only screen for obvious regressions.
- A small positive signal must be confirmed with same-session r3 before keep.
- If r3 confirms about `0.5-1.5%` aggregate TPS without serious prompt/decode regression, keep only as opt-in stacking knob.

## Results

- Candidate r1: `prefill-e057-q3unroll4-r1 = 11.58 TPS`, stdev `0.1891` across the two quick tasks.
- Candidate r3: not run; r1 was below the same-session E056 control r3 (`11.67 TPS`) and below the recent E055 half2 r1 (`11.86 TPS`).
- Decision: reject and revert. Explicit unroll4 is not a promising stackable small-gain candidate.