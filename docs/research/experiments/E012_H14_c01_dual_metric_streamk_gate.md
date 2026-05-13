# E012: H14 C01 Dual-Metric Stream-k Gate (RDNA4)

Date: 2026-05-13
Owner: Copilot
Experiment ID: E012
Stage: measured lane A/B + trace hotspot analysis

## Goal

Continue C01 with a dual-metric decision policy:
- runtime metric: lane TPS
- hotspot metric: expensive-place trace time (`MUL_MAT forward` and target MMQ bucket)

User rule for this stage:
- hotspot-time improvement is a positive result even when TPS is neutral/noisy.

## Candidate

Add an env-gated RDNA4 stream-k threshold in `ggml/src/ggml-cuda/mmq.cu`:
- env: `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11`
- default remains `256` (no behavioral change without env)
- candidate run: set to `192`

## Commands

Build:

```bash
cmake --build build-rocm-vec --config Release -j 16
```

Lane A/B (`review_bug,patch_sim`, `r1`):

```bash
python scripts/agent_workload_bench.py --label e012-c01-two-tasks-r1-streamk-default --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 256 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0

GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=192 python scripts/agent_workload_bench.py --label e012-c01-two-tasks-r1-streamk-min192 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 256 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0
```

Trace A/B (`kernel-full` + MMQ timing):

```bash
GGML_TRACE_MMQ_TIMING=1 GGML_TRACE_MMQ_TIMING_SYNC=1 python scripts/agent_workload_bench.py --label e012-c01-two-tasks-trace-r1-streamk-default-mmqtiming --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 256 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full

GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=192 GGML_TRACE_MMQ_TIMING=1 GGML_TRACE_MMQ_TIMING_SYNC=1 python scripts/agent_workload_bench.py --label e012-c01-two-tasks-trace-r1-streamk-min192-mmqtiming --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 256 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full
```

Compare:

```bash
python scripts/research/compare_kernel_traces.py \
  build_logs/agent-workload/e012-c01-two-tasks-trace-r1-streamk-default-mmqtiming.server.log \
  build_logs/agent-workload/e012-c01-two-tasks-trace-r1-streamk-min192-mmqtiming.server.log \
  --baseline-name e012-default --candidate-name e012-min192 --top 80 \
  > build_logs/agent-workload/e012-c01-trace-compare-default-vs-min192.md
```

## Measured

TPS:
- baseline: `14.40`
- candidate: `14.42`

Trace deltas:
- `CUDA_NODE`: `22430.960 -> 22379.713 ms` (`-51.247 ms`)
- `CUDA_NODE op=MUL_MAT`: `14575.007 -> 14541.438 ms` (`-33.569 ms`)
- `CUDA_NODE op=MUL_MAT kind=forward`: `14417.721 -> 14384.724 ms` (`-32.997 ms`)
- `CUDA_NODE op=MUL_MAT ne=(10240,192,1,1)`: `1265.410 -> 1238.487 ms` (`-26.923 ms`)

Target MMQ bucket check:
- `mul_mat_q_case type=11, ncols_max=192`
- `8944.730 -> 8936.004 ms` (`-8.726 ms`), same `count=25128`

## E012-R1: Full Sweep Across Practical C01 Points

Runtime sweep over all practical RDNA4 stream-k thresholds:
- `128, 144, 160, 176, 192, 208, 224, 240, 256, 320, 9999`

Summary (agg TPS):
- best: `skmin=144` -> `14.4325`
- next: `160` -> `14.4230`, `128` -> `14.4029`, `176` -> `14.4008`
- baseline point: `256` -> `14.2724`

Trace confirmation for best point (fresh no-hard-timeout pair):
- baseline (`skmin=256`): `e012-c01-two-tasks-trace-r1-sweep-skmin-256-mmqtiming-nohard`
- candidate (`skmin=144`): `e012-c01-two-tasks-trace-r1-sweep-skmin-144-mmqtiming-nohard`

Hotspot deltas (`256 -> 144`):
- `CUDA_NODE`: `22625.835 -> 22440.438 ms` (`-185.397 ms`)
- `CUDA_NODE op=MUL_MAT`: `14639.932 -> 14576.019 ms` (`-63.913 ms`)
- `CUDA_NODE op=MUL_MAT kind=forward`: `14480.357 -> 14420.098 ms` (`-60.259 ms`)
- MMQ target bucket (`type=11, ncols_max=192`): `8968.599 -> 8959.550 ms` (`-9.049 ms`, same `count=25128`)

## Decision

- Runtime verdict: `positive` for `skmin=144` vs `256`.
- Hotspot verdict: `positive`.
- Overall: `keep-as-knob` with best-known point `skmin=144`.
- Default decision: keep default unchanged for now; require extra confirmation before promotion.

## Artifacts

- `build_logs/agent-workload/e012-c01-sweep-summary.md`
- `build_logs/agent-workload/e012-c01-two-tasks-r1-sweep-skmin-*.csv`
- `build_logs/agent-workload/e012-c01-two-tasks-trace-r1-sweep-skmin-256-mmqtiming-nohard.server.log`
- `build_logs/agent-workload/e012-c01-two-tasks-trace-r1-sweep-skmin-144-mmqtiming-nohard.server.log`
- `build_logs/agent-workload/e012-c01-trace-compare-skmin256-vs-144-nohard.md`
- `build_logs/agent-workload/e012-c01-two-tasks-r1-streamk-default.csv`
- `build_logs/agent-workload/e012-c01-two-tasks-r1-streamk-min192.csv`
- `build_logs/agent-workload/e012-c01-two-tasks-trace-r1-streamk-default-mmqtiming.server.log`
- `build_logs/agent-workload/e012-c01-two-tasks-trace-r1-streamk-min192-mmqtiming.server.log`
- `build_logs/agent-workload/e012-c01-trace-compare-default-vs-min192.md`
