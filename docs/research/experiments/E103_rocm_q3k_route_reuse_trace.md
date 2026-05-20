# E103 ROCm Q3_K Route Reuse Trace

## Metadata

- Experiment ID: E103
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: master @ b6f114650 plus local E103 trace prototype
- Target lane: Qwen3.6-27B-Q3_K_S cold-first ROCm prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a new ROCm route can outperform the current large Q3_K fp16 hipBLAS path only if it removes repeated Q3_K -> fp16 staging or fuses that staging into a shape-specific direct kernel.
- Mechanism: E049/E054 showed the current path spends real time in `src0_convert_ms`, not pool allocation. Before adding a persistent cache or a fused kernel, trace whether the same Q3_K `src0` tensor/range is staged multiple times within a prompt run.
- Why now: Vulkan's biggest acceleration came from selecting a better large-matmul route; ROCm already has the large hipBLAS route, so the next route-level opportunity is reducing or avoiding Q3_K staging.

## Math / Theory

- Assumptions:
  - E054 Q3_K `src0_convert_ms`: `3370.32 ms`.
  - E054 target `6144x5120@ncols2048` conversion: `1430.88 ms`.
  - Effective full-lane wall share for Q3_K conversion is about `10.1%` after prompt/decode discounting.
- Expected speedup corridor:
  - `10%` local Q3_K conversion gain projects about `+0.9%` wall.
  - `20%` local Q3_K conversion gain projects about `+1.7%` wall.
  - `25%` local Q3_K conversion gain projects about `+2.1%` wall.
  - Persistent cache is only worth implementing if repeated staging exists for the same `src0` tensor/range; otherwise use a direct fused route prototype.
- Failure conditions:
  - Trace shows almost every Q3_K `src0` key has one call, making cache route low ceiling.
  - Trace overhead changes scheduling; use it for structure, not TPS claims.
  - Tensor pointer/data lifetime is not stable enough to key a cache safely.

## Implementation Plan

1. Minimal code surface to change: add default-off `GGML_TRACE_CUBLAS_Q3K_ROUTE=1` logging in `ggml_cuda_op_mul_mat_cublas` for Q3_K fp16 staging.
2. Guard rails: no runtime behavior change unless env is set; optional `GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS` filter; no cache or fused route yet.
3. Rollback path: remove the trace helper or keep it default-off if useful.

## Benchmark Plan

- Baseline command: none for speed; this is a structure trace.
- Candidate command:
  - `GGML_TRACE_CUBLAS_Q3K_ROUTE=1 GGML_TRACE_CUBLAS_Q3K_ROUTE_MIN_NCOLS=1024 GGML_TRACE_CUBLAS_SPLIT_TIMING=1 GGML_TRACE_CUBLAS_SPLIT_DETAIL=1 GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024 python scripts/agent_workload_bench.py --label e103-rocm-q3k-route-reuse-r1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids triage_diff,review_bug --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --no-v2-prime-pass --no-disable-thinking --max-tokens 120`
- Number of runs: one diagnostic trace.
- Artifacts path:
  - `build_logs/agent-workload/e103-rocm-q3k-route-reuse-r1.*`

## Metrics

- number of Q3_K staging calls
- unique `src0` tensor/range keys
- repeated calls per key
- total staged elements per key
- target-shape `src0_convert_ms` from split-detail

## Result

- Outcome: structure-positive, no speed claim. Repeated Q3_K staging is real and large enough to justify a follow-up route probe, but the trace itself is sync-heavy diagnostic data.
- Delta: trace run aggregate was `11.19 TPS` and is not comparable to no-trace controls. Route trace produced `2792` Q3_K staging rows, `349` unique tensor/range keys, and all `349` keys repeated `8` times. Total traced `src0_convert_ms=3260.082 ms`, with estimated unlimited-cache saved conversion time `2852.549 ms` but a prohibitive fp16 footprint of `42.002 GiB`.
- Confidence: high for the reuse structure; medium for conversion-time totals because split-detail inserts stream synchronization.
- Recommendation: keep the default-off `GGML_TRACE_CUBLAS_Q3K_ROUTE` diagnostic. Do not implement an unlimited persistent cache. Test only bounded/targeted cache or direct fused-route candidates with strict A/B, then revert negative runtime probes.

## Notes

- Surprises: reuse is maximal in count (`8/8` calls per key), but the working set is very wide. Repeats for `blk.0.attn_gate.weight` are separated by roughly `900` Q3_K route rows, so a tiny LRU cannot capture the hot reuse. Full repeated-key cache would need far more VRAM than available.
- Follow-up action: E104 tested bounded persistent fp16 staging cache and rejected it. E105 tested a narrow existing-MMQ route override and rejected it. The remaining credible route is a shape-specific fused Q3_K x F16 path that avoids both repeated fp16 staging and large persistent fp16 residency.