# E039 C01 Post-E038 Trace Recenter

## Metadata

- Experiment ID: E039
- Date: 2026-05-17
- Owner: Copilot
- Type: trace recenter after E038 rollback
- Lane: C01 (`ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, no-reuse)

## Goal

Reconfirm active hotspot center after E038 rejection and rollback.

## Run

- Label: `c01-e039-poste038-trace-r1`
- Command profile: `tasks=quick`, `runs=1`, `--trace-preset kernel-full`

## Runtime (trace-on)

- Aggregate TPS: `7.79`

Note: trace-enabled TPS is diagnostic-only and not compared to no-trace runtime baselines.

## Cold/Steady Split (`MUL_MAT forward`)

Source:

- `scripts/research/cold_steady_trace_split.py`
- `build_logs/agent-workload/c01-e039-poste038-trace-r1.server.log`

Steady (`<=5 ms`) route shares:

- `mul_mat_q_direct|q3_K`: `25008.892 ms` (`78.64%`)
- `cublas_backend|f32`: `4286.870 ms` (`13.48%`)
- `mul_mat_q_direct|q4_K`: `1972.103 ms` (`6.20%`)

## Q3 shape presence

From `scripts/research/c01_shape_presence_gate.py`:

- qtype `11` histogram top bucket:
  - `ncols_max=192` with `53048` hits
- next buckets are sparse (`94/75/63/74` each `349` hits)

## Decision

- Verdict: `keep as recenter checkpoint`
- Interpretation: active center remains unchanged after E038 rollback.
- Next direction: continue only Q3_K `ncols_max=192` focused probes; avoid broad graph-level fusion variants.

## Artifacts

- `build_logs/agent-workload/c01-e039-poste038-trace-r1.csv`
- `build_logs/agent-workload/c01-e039-poste038-trace-r1.server.log`
