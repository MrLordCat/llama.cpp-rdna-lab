# E022 H08 C05 GDN Chunk192 Probe

## Metadata

- Experiment ID: E022
- Date: 2026-05-14
- Owner: Codex
- Target lane: `Qwen3.6-27B-Q3_K_S`, `tasks=quick`, `task_ids=review_bug,patch_sim`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `spec=none`, no-reuse
- Hypothesis ID: H08

## Hypothesis

- Statement: For the current C01 prompt-heavy lane, GDN prefill uses `n_tokens=192` and default `chunk_size=96`, so forcing `chunk_size=192` may reduce launch/chunk overhead by processing each ubatch in one GDN chunk.
- Mechanism: Current trace shows the dominant GDN prompt work as two internal chunks (`96 + 96`) for each `n_tokens=192` call. A single `192` chunk may reduce per-chunk overhead if the kernel body does not slow down too much.
- Risk: Historical GDN sweeps showed `64/80/96/128` were flat or worse around the best `ub192` point. Larger chunks can increase inner-loop pressure.

## Math / Theory

- Current C05 trace: `GATED_DELTA_NET forward = 1467.855 ms` overall, `1174.486 ms` in prompt phase.
- Prompt GDN chunk histogram:
  - `n_tokens=192`, `chunk_size=96`, two chunks per call
  - chunk `96` sum about `1398 ms` in the diagnostic trace
- Best-case ceiling:
  - Even a large `10%` GDN gain would be only about `0.7%` wall on this lane.
  - The probe is only worth keeping if it gives an immediate clean r1 signal.

## Benchmark Plan

- Reference: current best `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
- Candidate:
  - `GGML_GDN_CHUNK_SIZE=192`
  - label `c05-gdn-chunk192-r1`

## Result

- Candidate: `c05-gdn-chunk192-r1 = 9.58 TPS`
- Reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`
- Decision: reject; do not change default GDN chunk policy.

## Notes

- Earlier same-session diagnostic `GGML_GDN_FAST_EXP=1` also failed to beat the reference (`c05-gdn-fast-exp-r1 = 9.59 TPS`).
- No code changes were made for this experiment.
