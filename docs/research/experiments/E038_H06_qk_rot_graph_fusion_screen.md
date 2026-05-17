# E038 H06 Q/K Rotation Graph Fusion Screen

## Metadata

- Experiment ID: E038
- Date: 2026-05-17
- Owner: Copilot
- Branch/Commit: local `master` after E037 gate
- Lane: C01 (`ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, no-reuse)

## Hypothesis

- Statement: graph-level fusion of Q and K rotation (`ggml_mul_mat_aux`) into one concatenated op can reduce graph overhead and improve runtime.
- Scope: env-gated prototype only (`GGML_EXPERIMENTAL_QK_ROT_FUSION=1`) in `src/llama-graph.cpp`.

## Implementation

Temporary guarded prototype (later reverted):

- Added env knob `GGML_EXPERIMENTAL_QK_ROT_FUSION`.
- In attention paths with `self_k_rot`/`k_rot`, attempted:
  1. `ggml_concat(q_cur, k_cur, dim=1)`
  2. single `ggml_mul_mat_aux(...)`
  3. split back via `ggml_view_4d(...)`.
- Fallback to existing separate Q and K transforms when disabled or incompatible shapes.

## Benchmark Plan

- Control:
  - `c01-e038-h06-control-r1`
- Candidate:
  - `c01-e038-h06-fused-r1` with env `GGML_EXPERIMENTAL_QK_ROT_FUSION=1`

Common command lane:

- `scripts/agent_workload_bench.py`
- `tasks=quick`
- `runs=1` (screen)
- no-reuse
- repo-snapshot real-context

## Result

- Control: `11.2031 TPS`
- Candidate: `11.1688 TPS`
- Delta: `-0.0343 TPS` (`-0.31%`)

Diagnostics summary:

- prompt eval mean: `835.05 -> 832.29 tok/s`
- decode eval mean: `29.515 -> 29.44 tok/s`

## Decision

- Verdict: `reject`
- Reason: negative screen result on target lane.
- Code state: prototype reverted; `llama-server` rebuilt.

## Artifacts

- `build_logs/agent-workload/c01-e038-h06-control-r1.csv`
- `build_logs/agent-workload/c01-e038-h06-fused-r1.csv`
- `build_logs/agent-workload/c01-e038-h06-control-r1.server.log`
- `build_logs/agent-workload/c01-e038-h06-fused-r1.server.log`
