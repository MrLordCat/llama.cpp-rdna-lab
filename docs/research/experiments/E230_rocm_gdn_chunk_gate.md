# E230 ROCm GDN chunk-size gate

## Metadata

- Experiment ID: E230
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 stack item / GDN second-hotspot gate
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, cold/no reuse
- Binary: `build-rocm-vec/bin/llama-server.exe`

## Hypothesis

- Statement: E228 full trace showed `GATED_DELTA_NET/f32` as the second-largest op (`998.759 ms`, `13.01%` of traced total). Increasing GDN prefill chunk size may reduce kernel launch count enough to produce a small cold wall gain.
- Mechanism: the current RDNA4 GDN prefill default chunks at `128` for large batches. Larger chunks reduce launch count and may lower dispatch/sync overhead, at the cost of longer per-kernel token loops.
- Why now: rocBLAS solution-index scouting in E229 did not justify runtime work, so the next measured stack item is the largest non-Q3_K op.

## Math / Theory

- E228 traced share: GDN is `13.01%`, so even a large `20%` local win only projects to about `2.6%` wall before overlap and bottleneck shift.
- Failure condition: if point timing moves but wall TPS does not improve, classify as bottleneck shift/non-critical trace cost and do not change defaults.

## Benchmark Plan

- Point timing:
  - one cold `triage_diff` run, `--max-tokens 1`
  - `GGML_TRACE_GDN_PATH=1`
  - `GGML_TRACE_GDN_TIMING=1`
  - `GGML_TRACE_GDN_TIMING_SYNC_HIP=1`
  - `GGML_TRACE_GDN_TIMING_PRE_SYNC_HIP=1`
  - `LLAMA_TRACE_DELTA_NET_CONTRACT=1`
- Candidates:
  - `GGML_GDN_FAST_EXP=1`
  - `GGML_GDN_CHUNK_SIZE=256`
  - `GGML_GDN_CHUNK_SIZE=512`
  - `GGML_GDN_CHUNK_SIZE=1024`
  - `GGML_GDN_CHUNK_SIZE=2048`
  - `GGML_GDN_CHUNK_SIZE=4096`
- Wall A/B:
  - two cold quick tasks, `triage_diff,review_bug`
  - `--max-tokens 64`
  - no trace, no reuse, no prime, `spec=none`

## Point Results

| Route | Calls | GDN total ms | Delta vs baseline | Notes |
| --- | ---: | ---: | ---: | --- |
| default chunk `128` | `2880` | `1017.705` | baseline | chunks `128`, tail `102`, decode `2` |
| `GGML_GDN_FAST_EXP=1` | `2880` | `1008.523` | `-0.90%` | too small |
| chunk `256` | `1488` | `895.252` | `-12.03%` | fewer launches |
| chunk `512` | `768` | `859.397` | `-15.56%` | better point |
| chunk `1024` | `432` | `833.785` | `-18.07%` | better point |
| chunk `2048` | `240` | `814.651` | `-19.95%` | best-ish point |
| chunk `4096` | `240` | `813.862` | `-20.03%` | same effective chunks as `2048` on this prompt |
| chunk `256` + fast exp | `1488` | `895.840` | `-11.98%` | no additive win |

## Wall Results

| Label | Route | Aggregate TPS | Prompt eval TPS mean | Prompt ms mean | Decode tok/s mean | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `e230-rocm12k-cold-gdn-control-r1` | default chunk policy | `7.8474` | `1241.045` | `6036.77` | `30.81` | control |
| `e230-rocm12k-cold-gdn-chunk4096-r1` | `GGML_GDN_CHUNK_SIZE=4096` | `7.7753` | `1223.3` | `6118.88` | `30.765` | regression/tie |

## Result

- Outcome: point win, wall no-win.
- Delta:
  - GDN sync point improved up to about `20%` local.
  - Cold wall regressed `-0.92%` in the paired r1 A/B.
- Confidence: medium. The local point move is real under synchronized GDN tracing, but the no-trace wall run does not convert it into TPS.
- Recommendation:
  - Reject larger GDN chunk as a default cold-first speed route.
  - Keep existing env override for future diagnostics; do not change default chunk policy.
  - Treat this as bottleneck shift / non-critical sync cost: without trace, Q3_K/FATTN and graph scheduling absorb the local GDN change.

## Notes

- This validates the route-chain rule: a local hotspot win is not sufficient if wall shifts elsewhere.
- `GGML_GDN_FAST_EXP=1` is below threshold and should not be promoted.
- If GDN is revisited, it needs a body/layout change that reduces unsynchronized critical path time, not only launch-count reduction seen through sync timing.

## Artifacts

- `build_logs/agent-workload/e230-rocm12k-gdn-point-baseline-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-gdn-point-fastexp-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-gdn-point-chunk256-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-gdn-point-chunk512-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-gdn-point-chunk1024-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-gdn-point-chunk2048-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-gdn-point-chunk4096-r1.server.log`
- `build_logs/agent-workload/e230-rocm12k-cold-gdn-control-r1.diagnostics.md`
- `build_logs/agent-workload/e230-rocm12k-cold-gdn-chunk4096-r1.diagnostics.md`
