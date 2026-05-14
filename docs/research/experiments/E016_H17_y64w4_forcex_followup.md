# E016 H17 y64/w4 force-x follow-up

## Metadata

- Experiment ID: E016
- Date: 2026-05-14
- Owner: Codex
- Branch/Commit: post `46d97fb5f`
- Target lane: `Qwen3.6-27B-Q3_K_S`, `ctx=12288`, `batch=6144`, `ubatch=192`, `q4_0/q4_0`, `review_bug,patch_sim`, `no-reuse`, thinking on

## Hypothesis

- Statement: after E015 changes RDNA4 MMQ to `mmq_y=64/nwarps=4`, the best `mmq_x` might move away from the selected default `96`.
- Mechanism: smaller y tiles could change tile-count/resource balance enough that a different x tile wins.
- Why now: the pre-E015 force-x screen was measured under `mmq_y=128/nwarps=8`, so it was no longer sufficient.

## Result

- Outcome: regression; no keep candidate.
- Baseline reference: E015 default `mmq_x_best=96`, `9.6080 TPS` on r3.
- Candidate r1 force-x results:
  - `x64`: `9.02 TPS`
  - `x80`: `8.20 TPS`
  - `x112`: `9.06 TPS`
  - `x128`: `8.77 TPS`
- Decision: keep default selector (`mmq_x=96` on active bucket).

## Notes

- Artifacts:
  - `build_logs/agent-workload/c01-e016-y64w4-forcex64-r1.csv`
  - `build_logs/agent-workload/c01-e016-y64w4-forcex80-r1.csv`
  - `build_logs/agent-workload/c01-e016-y64w4-forcex112-r1.csv`
  - `build_logs/agent-workload/c01-e016-y64w4-forcex128-r1.csv`
- Follow-up action:
  - do not continue force-x sweeps on the active C01 y64/w4 geometry unless a later code change alters shared layout or tile selection.
