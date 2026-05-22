# E161 ROCm Q3_K MMVQ Microprobe Audit

## Metadata

- Experiment ID: E161
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `1cf415f0f`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Purpose

Review another agent's uncommitted E157-E160 work before continuing H39. The
work contained:

- `vecdotq.cuh` Q3_K MMVQ scale decode `%`/`/` to bit-op rewrite;
- a Q3_K-only `rows_per_block=1` policy while keeping `nwarps=2`;
- rejected probes for `nwarps=3` and a wrapper modulo-hoist variant.

The original E157/E158 measurements were on the older C01-style lane
(`ubatch=192`, six task runs), not the active H39 lane (`ubatch=2048`,
`triage_diff`, three runs). This audit checks whether the proposed keeps survive
on the active decode parity contract.

## Results

Active-lane r3 checks:

| Stack | Aggregate TPS | Decode eval | Decision |
| --- | ---: | ---: | --- |
| dirty stack: scale bit-ops + Q3_K `rows_per_block=1` | `28.3481` | `30.05 tok/s` | reject; this largely loses the E151 win |
| scale bit-ops + restored E151 `rows_per_block=2` | `30.1884` | `32.12 tok/s` | tied/noise versus clean |
| clean post-revert E151 policy | `30.1073` | `32.03 tok/s` | baseline for this audit |

References:

- E151 promoted speed claim: `30.3145 TPS` / `32.2467 tok/s` decode.
- E158 C01-only `rows_per_block=1` looked `+0.08%` decode, but active H39
  rejects it.
- E157 C01-only bit-op rewrite looked `+0.81%` decode, but active H39 only shows
  `32.03 -> 32.12 tok/s`, within run noise.

## Decision

- Do not keep Q3_K `rows_per_block=1` as a default. It contradicts the E151
  mechanism and regresses the active lane.
- Do not promote the scale bit-op rewrite now. It may be useful as a future
  microcandidate, but it did not show a significant active-lane win.
- Keep source code at the E151/E152 state before moving to the next larger
  Q3_K route branch.

## Artifacts

- `build_logs/agent-workload/e161-rocm-decode-q4-dirty-scalebitops-rpb1-r3.diagnostics.md`
- `build_logs/agent-workload/e161-rocm-decode-q4-scalebitops-rpb2-r3.diagnostics.md`
- `build_logs/agent-workload/e161-rocm-decode-q4-cleanpost-rpb2-r3.diagnostics.md`
- C01-only references:
  - `build_logs/agent-workload/c01-q3k-scalebitops-r3.diagnostics.md`
  - `build_logs/agent-workload/c01-e158-q3k-rpb1-r3.diagnostics.md`
  - `build_logs/agent-workload/c01-e159-q3k-nwarps3-r1.diagnostics.md`
  - `build_logs/agent-workload/c01-e160-q3k-wrapper-hoist-r3.diagnostics.md`
