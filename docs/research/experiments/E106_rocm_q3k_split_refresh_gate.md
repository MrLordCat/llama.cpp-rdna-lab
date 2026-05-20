# E106 ROCm Q3_K Split Refresh Gate

## Metadata

- Experiment ID: E106
- Date: 2026-05-20
- Owner: Codex
- Branch/Commit: master @ 533bb5ed2
- Target lane: Qwen3.6-27B-Q3_K_S cold-first ROCm prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: before writing another ROCm Q3_K route prototype, the current clean tree must still show a large enough Q3_K `src0` conversion/layout share to support a real TPS win.
- Mechanism: H35 only remains worth coding if the large hipBLAS route is still paying repeated Q3_K -> fp16 staging on hot shapes. E103/E104/E105 showed the issue, but recent documentation and route-map work should be followed by a fresh cold-first control and split trace.
- Why now: E104 rejected persistent fp16 cache and E105 rejected existing-MMQ selector overrides. The next viable route is either a fused Q3_K x F16 kernel or another non-persistent route; both need a stricter pre-code gate.

## Math / Theory

- Assumptions: effective Q3_K conversion/layout wall share is near the E053/E054 estimate of about `10.1%`, with hot traced shapes still dominated by `6144x5120@ncols2048` and tail chunks.
- Expected speedup corridor: a `10%` local conversion/layout win projects to about `+1%` aggregate wall; `24%` local projects to about `+2%`; `40%` local projects to about `+3%`.
- Failure conditions: if current traces show a materially lower conversion share, or if the trace mix moved away from Q3_K large-prefill staging, do not code another Q3_K conversion micro-probe.

## Implementation Plan

1. Minimal code surface to change: none for the gate; only run current-tree baseline and split trace.
2. Guard rails: do not repeat rejected toggles: `GGML_CUDA_FORCE_MMQ_RUNTIME`, compute16, hipBLASLt, half2 store, dequant128, Q3_K persistent fp16 cache, existing-MMQ selector override.
3. Rollback path: no code changes during gate; if a later prototype fails, revert prototype code and keep only the experiment note/workflow update.

## Benchmark Plan

- Baseline command: current-tree no-trace cold-first `agent_workload_bench.py` run on the active lane.
- Candidate command: current-tree split/route trace run with `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`, `GGML_TRACE_CUBLAS_SPLIT_DETAIL=1`, `GGML_TRACE_CUBLAS_Q3K_ROUTE=1`, and min ncols `1024`.
- Number of runs: one-run gate for trace; promote only a promising later code candidate to 3-run confirmation.
- Artifacts path:
  - `build_logs/agent-workload/e106-rocm-q3k-control-r1.*`
  - `build_logs/agent-workload/e106-rocm-q3k-split-r1.*`

## Metrics

- aggregate completion TPS (wall)
- prompt eval tok/s
- Q3_K `src0_convert_ms` and share in cublas split timing
- dominant Q3_K shape keys and repeated route count
- prefill/decode split if available

## Result

- Outcome: diagnostic gate kept H35 alive, but with a workflow correction.
- Delta: no-trace cold-first control `e106-rocm-q3k-control-r1` measured `11.8464 TPS`; split trace measured `11.2167 TPS` and is diagnostic only because sync/logging changes wall time.
- Confidence: medium for route mix, low for attributing each split bucket as pure kernel time without pre-stage sync.
- Recommendation: keep Q3_K fused/non-persistent route research open, but do not use the split `src0_convert_ms` numbers alone as proof that a simple conversion micro-kernel will speed wall TPS.

## Notes

- Split trace parsed `4456` split calls and `2792` Q3_K route lines, with `349` unique Q3_K tensor/shape keys and `2443` repeated-key lines.
- Aggregate traced split buckets were `src0_convert_ms=3407.599`, `src1_convert_ms=802.655`, `gemm_ms=8438.183`, `sum_ms=12657.655`.
- Q3_K rows dominated the trace: `calls=2792`, `sum_ms=10086.749`, `src0_convert_ms=3257.251`, `src1_ms=715.567`, `gemm_ms=6107.363`.
- Important workflow correction: the current split timing syncs after a stage, but not necessarily before it, so a bucket such as `attn_gate` conversion can include earlier queued GPU work. Future H35 probes need pre-stage sync or event timing before using bucket-level milliseconds as a pure local-speedup estimate.
- Follow-up action: any H35 code prototype must pass a `>=2%` modeled wall ceiling and include clean no-trace A/B; avoid repeating persistent fp16 cache, existing-MMQ routing, compute16, hipBLASLt, half2/dequant128, or route-wide selector toggles.
