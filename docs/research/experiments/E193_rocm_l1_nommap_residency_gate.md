# E193 ROCm L1 No-Mmap Residency Gate

## Metadata

- Experiment ID: E193
- Date: 2026-05-23
- Owner: Codex
- Hypothesis ID: H35 / residency negative-control
- Branch/Commit: local `master` after `624a8d163`
- Target lane: L1 ROCm route-chain, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on, `quick/triage_diff`, `max_tokens=128`

## Hypothesis

- Statement: `--no-mmap` might reduce cold-run host paging or residency interaction enough to improve the L1 ROCm Q3_K prefill/decode wall.
- Mechanism: E191/E192 show the current practical lane spends heavily in large Q3_K `cublas_backend` staging. If part of the wall is host-backed mapping pressure or Windows paging during repeated cold runs, forcing eager model loading may improve prompt eval without touching kernels.
- Why now: E104 rejected fp16 residency caches, and E192 shows a scheduler-wide repeated-staging pattern. This is a cheap negative-control before considering much larger graph-scheduling or fused-kernel work.

## Math / Theory

- Assumptions:
  - E190 paired control r3 is the closest clean L1 reference: `12.9580 TPS`, prompt eval `1284.85 tok/s`, decode `31.4433 tok/s`.
  - `--no-mmap` does not reduce Q3_K GEMM or conversion arithmetic directly, so any gain should appear as lower prompt wall or lower run variance.
- Expected speedup corridor:
  - Keep only if r1 beats same-session control clearly and r3 confirms.
  - Treat `<=1%` as noise unless prompt eval and diagnostics support it.
- Failure conditions:
  - slower prompt eval,
  - no wall gain,
  - higher memory/residency pressure,
  - server errors.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails: same-session baseline first; clear ROCm override env; ensure no background `llama-server`; keep `--cache-ram 0 --ctx-checkpoints 0`.
3. Rollback path: no code changes.

## Benchmark Plan

- Baseline command: `e193-l1-control-r1`.
- Candidate command: `e193-l1-nommap-r1` with `--server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap"`.
- Number of runs: `r1` gate, `r3` only if candidate is clearly positive.
- Artifacts path: `build_logs/agent-workload/e193-l1-*.{csv,jsonl,server.log,diagnostics.md}`.

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s and ms
- decode eval tok/s and ms
- error rate

## Result

- Outcome: reject; no r3. Same-session r1 gate was negative.
- Delta: aggregate `12.7743 -> 12.6940 TPS` (`-0.63%`), prompt eval `1243.96 -> 1234.43 tok/s` (`-0.77%`), decode eval `31.77 -> 31.65 tok/s` (`-0.38%`).
- Confidence: medium. This is only r1, but it is a clean same-session negative control and does not cross the promotion threshold.
- Recommendation: do not transfer the CPU `--no-mmap` lesson to the full-offload ROCm L1 lane. Keep mmap/default loading for this cold-first prompt-heavy ROCm profile unless a separate residency trace proves a new driver-specific paging issue.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e193-l1-control-r1.diagnostics.md`
- `build_logs/agent-workload/e193-l1-nommap-r1.diagnostics.md`
- `build_logs/agent-workload/e193-l1-control-r1.server.log`
- `build_logs/agent-workload/e193-l1-nommap-r1.server.log`

| Metric | Control | `--no-mmap` | Delta |
| --- | ---: | ---: | ---: |
| aggregate completion TPS | `12.7743` | `12.6940` | `-0.63%` |
| prompt eval TPS | `1243.96` | `1234.43` | `-0.77%` |
| decode eval TPS | `31.77` | `31.65` | `-0.38%` |
| prompt eval ms | `5959.17` | `6005.22` | `+46.05 ms` |
| decode eval ms | `4029.42` | `4044.73` | `+15.31 ms` |
| task prompt tokens | `7413` | `7413` | `0` |
| errors | `0` | `0` | `0` |

Command delta was only:

```text
--server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap"
```

## Notes

- Surprises: `--no-mmap` helps the CPU fallback lane in E125, but not this full-offload ROCm lane. That suggests the current L1 wall is not host paging during model access; E191/E192's large Q3_K cuBLAS staging/GEMM explanation still fits better.
- Follow-up action: close the `--no-mmap` residency control for ROCm L1. If residency returns as a suspect, use a trace that directly measures VRAM residency/transfer behavior rather than another no-code `--no-mmap` A/B.
