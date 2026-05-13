# Decode Speed - Trace Checklist (Current Focus)

## Baseline profile (current route)

- Lane: `tasks=quick`, `ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, `no-reuse`, `max_tokens=256`.
- Baseline run: `decode-trace-current-ctx12288-ub192-r1`.
- Aggregate TPS: `26.30`.
- Main artifacts:
  - `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`
  - `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.diagnostics.md`
  - `build_logs/agent-workload/decode-trace-smallk-compare.md`

## Main cost centers (by summed total_ms)

| Priority | Center | sum_ms | count | avg_ms | Status |
| --- | --- | ---: | ---: | ---: | --- |
| P1 | CUDA_NODE `MUL_MAT kind=forward` | 1717.322 | 8454 | 0.203 | in progress |
| P2 | MMVQ `type=11/q3_K ncols_dst=1` | 339.110 | 4618 | 0.073 | done: E013 kept RDNA4 Q3_K `nwarps=2` |
| P3 | CUDA_NODE `MUL_MAT kind=fused` | 326.936 | 2298 | 0.142 | queued |
| P4 | CUDA_NODE `RMS_NORM kind=fused` | 209.981 | 4389 | 0.048 | queued |
| P5 | CUDA_NODE `GATED_DELTA_NET kind=forward` | 149.095 | 1008 | 0.148 | queued |

## Conditional checklist (workflow)

- [x] Freeze baseline route and collect full kernel trace.
- [x] Build hotspot ranking by `sum(total_ms)`.
- [x] Create one document per center.
- [x] Start deep trace from `MUL_MAT forward`.
- [ ] Complete full sub-trace map for `MUL_MAT forward`:
  - top node names,
  - top tensor shapes (`ne`),
  - route deltas vs control A/B.
- [ ] Produce root-cause hypothesis set for `MUL_MAT forward` (memory bound, shape inefficiency, launch granularity, sync pressure).
- [x] Run micro A/B test for MMVQ Q3_K side center and keep reproducible gain (E013).
- [ ] Run micro A/B tests for top `MUL_MAT forward` hypothesis and keep only reproducible gains.
- [ ] Move to next center only after `MUL_MAT forward` has a closed trace + hypothesis verdict.

## Next-step runbook

1. Work inside `C01_mul_mat_forward.md` until trace is complete.
2. Promote only candidates with stable gain on same lane.
3. Re-run baseline control after every promising candidate.

## Latest Resume Checkpoint (2026-05-13)

- Return command executed: `c01-resume-r1-resources` (lane contract preserved).
- Artifact:
  - `build_logs/agent-workload/c01-resume-r1-resources.server.log`
- Mandatory gates on latest trace:
  - shape gate (`qtype=11`, `ncols_max=192`): PASS (`count=26524`),
  - cold/steady split: steady still dominated by `mul_mat_q_direct|q3_K`,
  - q3 path coarse split (steady): `compute_core_q3=84.25%`, `fallback_cublas=14.38%`.
- Comparison notes:
  - comparing against global decode baseline (`decode-trace-current-ctx12288-ub192-r1`) is methodologically invalid due different task mix,
  - apples-to-apples check vs previous C01 resource run (`e013-c01-two-tasks-trace-r1-resources`) shows `-5.4%` runtime and no major route flip,
  - control rerun (`c01-resume-r2-control`) is stable vs `c01-resume-r1-resources` (`+0.14%`, inconclusive/noise-level).

Latest artifacts from return sequence:
- `build_logs/agent-workload/c01-resume-r1-resources.server.log`
- `build_logs/agent-workload/c01-resume-r2-control.server.log`
- `build_logs/agent-workload/c01-resume-r2-control.csv`

## E013 MMVQ Q3_K closure

- Candidate: RDNA4 `GGML_TYPE_Q3_K`, `ncols_dst=1`, `nwarps=2`.
- Paired non-trace control: `9.1629 TPS`.
- Candidate non-trace: `9.3847 TPS`, `+2.42%`.
- Bootstrap CI: `[+0.2019, +0.2442]` TPS.
- Decision: keep; Q4_K unchanged pending a Q4-heavy lane.

## Pause/Resume workflow

Before switching to another problem:

1. Refresh `C01_RESUME_PLAYBOOK.md` with current lane, best point, and first resume command.
2. Ensure latest artifacts are in `build_logs/agent-workload/` and referenced in C01 notes.
3. Mark open vs done items in this checklist.

When returning to C01:

1. Read `C01_RESUME_PLAYBOOK.md` first.
2. Re-run one fresh resource trace and shape gate before any new code change.
3. Use decision stats for borderline runtime deltas.

Current return status:
- step 1: done,
- step 2: done (`c01-resume-r1-resources`),
- step 3: done (stats against e013 C01-compatible baseline).

## C01 diagnostics toolkit

- shape gate: `python scripts/research/c01_shape_presence_gate.py ...`
- cold/steady split: `python scripts/research/cold_steady_trace_split.py ...`
- q3 coarse component split: `python scripts/research/c01_q3_path_components.py ...`
- statistical verdict: `python scripts/research/decision_stats.py ...`
- trace compare: `python scripts/research/compare_kernel_traces.py ...`

## Center documents

- `docs/research/decode-hotspots/C01_mul_mat_forward.md`
- `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md`
- `docs/research/decode-hotspots/C02_mmvq_q3_ncols1.md`
- `docs/research/decode-hotspots/C03_mul_mat_fused.md`
- `docs/research/decode-hotspots/C04_rms_norm_fused.md`
- `docs/research/decode-hotspots/C05_gated_delta_net_forward.md`
