# E231 ROCm ubatch host timing gate

## Metadata

- Experiment ID: E231
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 scheduler/host-overhead gate
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse
- Binary: `build-rocm-vec/bin/llama-server.exe`

## Hypothesis

- Statement: after E228/E230, cold wall might still be limited by host-side graph build, allocation, input upload, or ubatch scheduling rather than GPU kernels.
- Mechanism: if per-ubatch build/alloc/input costs are large, a scheduler/graph route could produce a wall gain without changing GEMM kernels.
- Why now: GDN launch-count timing moved locally but did not move wall; the next gate is whether the wall is hiding in host runtime overhead.

## Benchmark Plan

- One cold `triage_diff` run with `--max-tokens 1`.
- Env:
  - `LLAMA_UBATCH_TIMING=1`
  - `LLAMA_UBATCH_TIMING_SYNC=1`
- Normal cold contract:
  - no reuse
  - no prime
  - `spec=none`
  - `build-rocm-vec`

## Result

Parsed `process_ubatch: ubatch timing` rows:

| n_tokens | Calls | Total ms | Build ms | Alloc ms | Inputs ms | Compute call ms | Sync ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2048` | `3` | `5078.48` | `2.42` | `1.81` | `29.23` | `475.37` | `4568.20` |
| `1382` | `1` | `1156.84` | `0.78` | `0.59` | `5.98` | `21.70` | `1127.41` |
| `2` | `1` | `289.51` | `0.77` | `0.52` | `0.03` | `207.91` | `80.28` |
| all rows | `5` | `6524.83` | `3.97` | `2.91` | `35.24` | `704.98` | `5775.89` |

- Outcome: diagnostic rejection of host/scheduler-first route.
- Delta: no candidate TPS claim.
- Confidence: medium. The sync instrumentation changes absolute timing, but the relative host slices are tiny versus GPU sync/compute.
- Recommendation:
  - Do not pursue graph build/allocation/input micro-optimizations as the next cold +20% path.
  - Continue with GPU route-body/layout changes, especially Q3_K GEMM-side work from E228.

## Notes

- Build + alloc + inputs are only about `42 ms` of `6525 ms` in this instrumented run.
- Most visible time is in GPU sync/compute, matching the E228 conclusion that Q3_K/GEMM-side work is the real blocker.

## Artifacts

- `build_logs/agent-workload/e231-rocm12k-ubatch-sync-timing-r1.server.log`
- `build_logs/agent-workload/e231-rocm12k-ubatch-sync-timing-r1.diagnostics.md`
