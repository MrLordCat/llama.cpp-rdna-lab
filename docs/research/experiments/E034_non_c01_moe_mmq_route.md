# E034 Non-C01 MoE MMQ Route Scan

## Metadata

- Experiment ID: E034
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local dirty `master`
- Target lane: non-C01 MoE smoke, `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, ROCm/RDNA4, `b=1024`, `ub=1024`, q4 KV

## Hypothesis

- Statement: Qwen3.6 A3B MoE prefill has a separate speedup opportunity outside the current C01 dense path: routed expert IQ matmuls may benefit from RDNA4 MoE LDS staging, and shared expert `q6_K` matmuls may benefit from MMQ routing instead of the current backend route.
- Mechanism: routed MoE expert tensors use `ids_dst` and currently launch `mul_mat_q_direct` at `mmq_x=128/mmq_y=64`; explicit RDNA4 staging can reduce repeated X loads if it fits LDS. Shared expert tensors are `q6_K` and route through the backend at prompt batches, so forced MMQ can test whether rocBLAS/hipBLAS is leaving throughput on the table for this MoE shape.
- Why now: E033 found MoE smoke throughput around `626 tok/s` at `pp512` and `101 tok/s` at `tg128`, but did not capture exact routes. The verbose trace now shows a concrete split between routed IQ MMQ and shared `q6_K` backend calls.

## Math / Theory

- Assumptions:
  - The main routed expert cases are `IQ2_S/IQ3_XXS/IQ4_XS`, `ncols_max=512`, `mmq_x=128`, `mmq_y=64`, `nwarps=4`.
  - Current base shared memory is `38400-40448` bytes with `smpbo=65536`; one block/SM is reported, so a staged variant must improve memory reuse enough to overcome any occupancy/register cost.
  - Shared expert `q6_K` calls occur once per layer for gate/up/shexp and currently use the generic backend.
- Expected speedup corridor:
  - RDNA4 MoE staging: `0%` to `+5%` prefill if staged X reuse is active and not LDS-bound.
  - Runtime forced MMQ for `q6_K`: `-10%` to `+5%`, because it is broad and may affect attention/linear `q6_K` calls beyond shared experts.
- Failure conditions:
  - `rdna4_staging_req=1` but `rdna4_staging_eff=0`, or staged route regresses throughput.
  - `GGML_CUDA_FORCE_MMQ_RUNTIME=1` moves too many non-target `q6_K` nodes to slower MMQ.
  - Any gain appears only in trace/no-warmup noise and does not survive same-session control.

## Implementation Plan

1. Minimal code surface to change: none for the first gate. Use existing env-gated `GGML_RDNA4_MOE_MMQ_STAGING=1` and `GGML_CUDA_FORCE_MMQ_RUNTIME=1`.
2. Guard rails: keep all probes opt-in; no default changes unless same-session A/B shows a clear prefill win.
3. Rollback path: unset env vars. If a later code knob is attempted, revert the patch unless it beats same-session control.

## Benchmark Plan

- Baseline command: `llama-bench` MoE lane with `-p 512,2048,4096 -n 128 -r 1 --no-warmup`.
- Candidate command: same lane plus one env knob at a time.
- Number of runs: `r1` for screening; `r3` only if a candidate is clearly positive.
- Artifacts path: `build_logs/agent-workload/g034-*`.

## Metrics

- prompt processing tok/s by prompt size
- token generation tok/s
- route histogram
- `rdna4_staging_req/eff`
- MMQ resource trace: `nbytes_shared`, regs, blocks/SM, occupancy

## Result

- Outcome: reject / no default code change
- Delta:
  - Same-session pre-reboot control: `pp512=688.04`, `pp2048=3517.32`, `pp4096=3443.60`, `tg128=101.37`.
  - Broad `GGML_CUDA_FORCE_MMQ_RUNTIME=1`: `pp512=1362.66` but `pp2048=3134.28`; the run did not complete and the machine later required reboot. Treat as unstable and unsafe as a broad runtime knob.
  - Post-reboot control pair: `pp512=637.88`, `pp2048=3534.41`, `tg128=101.52`.
  - Full MoE staging: `pp512=657.79` (`+3.12%` vs nearest control), `pp2048=3248.03` (`-8.10%`), `tg128=100.57` (`-0.93%`).
  - Scoped staging prototype with `MAX_NCOLS=512`: control `pp512=674.33`, `pp2048=3537.34`, `tg128=101.70`; candidate `pp512=655.51`, `pp2048=3455.64`, `tg128=99.78`.
- Confidence: medium for rejection of current staging/default promotion; high that broad forced-MMQ is not safe enough to keep as-is.
- Recommendation: Do not promote RDNA4 MoE staging or broad runtime MMQ. Keep the route evidence as a future guide: any useful MMQ force must be narrower than global `GGML_CUDA_FORCE_MMQ_RUNTIME=1` and must avoid the `p2048+` regression/hang zone.

## Notes

- Initial route trace artifact: `build_logs/agent-workload/g034-moe-route-q4-p512-r1-verbose.log`.
- Trace summary:
  - Routed experts: `IQ2_S` gate/up and `IQ3_XXS/IQ4_XS` down route through `mul_mat_q_direct`.
  - Shared experts and attention/linear `q6_K` routes use `cublas_backend`.
  - Routed MMQ cases use `ncols_max=512`, `mmq_x=128`, `mmq_y=64`, `nwarps=4`, with `38400-40448` bytes shared and one active block/SM.
- Staging trace:
  - At `p512`, staging is effective for routed MoE: `rdna4_staging_req=1`, `rdna4_staging_eff=1`, `nbytes_shared=57860-61956`, near the `65536` byte shared-memory ceiling.
  - At `p2048`, routed MoE uses `ncols_max=1024` with the same `mmq_x=128`; staging remains effective but regresses measured throughput.
  - A scoped `MAX_NCOLS=512` prototype was built and tested, then reverted because it did not beat the rebuilt same-session control.
- Artifacts:
  - `build_logs/agent-workload/g034-moe-control-r1.jsonl`
  - `build_logs/agent-workload/g034-moe-force-mmq-runtime-r1.jsonl`
  - `build_logs/agent-workload/g034-moe-control-postreboot-r1.jsonl`
  - `build_logs/agent-workload/g034-moe-staging-route-p512-r1-verbose.log`
  - `build_logs/agent-workload/g034-moe-staging-route-p2048-r1-verbose.log`
  - `build_logs/agent-workload/g034-moe-staging-postreboot-r1.jsonl`
  - `build_logs/agent-workload/g034-moe-control-poststaging-r1.jsonl`
  - `build_logs/agent-workload/g034-moe-control-scopedbuild-r1.jsonl`
  - `build_logs/agent-workload/g034-moe-staging-max512-r1.jsonl`
