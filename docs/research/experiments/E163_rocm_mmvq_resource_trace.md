# E163 ROCm MMVQ Resource Trace

## Metadata

- Experiment ID: E163
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `efd7af490`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: H39 needs MMVQ resource telemetry before larger fused/direct Q3_K route changes.
- Mechanism: E162 showed that branch-removal can regress despite looking cheaper. A trace for registers, static shared memory, occupancy, and waves per CU should explain whether a route is resource-limited or scheduling/grid-limited.
- Risk: Trace runs disable graphs and use sync timing, so they are topology/resource evidence only, not clean TPS claims.

## Method

Added `GGML_TRACE_MMVQ_RESOURCES=1` to `ggml/src/ggml-cuda/mmvq.cu`, analogous to existing MMQ resource tracing. The trace records:

- block threads;
- dynamic/static/total shared memory;
- HIP/CUDA reported registers;
- max active blocks per CU/SM;
- occupancy percentage;
- waves per CU/SM.

Trace command shape:

```powershell
$env:GGML_CUDA_DISABLE_GRAPHS = "1"
$env:GGML_TRACE_CUDA_NODE_TIMING = "1"
$env:GGML_TRACE_CUDA_NODE_TIMING_SYNC = "1"
$env:GGML_TRACE_CUDA_MUL_MAT_ROUTE = "1"
$env:GGML_TRACE_MMVQ_SMALL_K = "1"
$env:GGML_TRACE_MMVQ_TIMING = "1"
$env:GGML_TRACE_MMVQ_TIMING_SYNC = "1"
$env:GGML_TRACE_MMVQ_RESOURCES = "1"
python scripts\agent_workload_bench.py --label e163-rocm-decode-q4-mmvq-resources-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 16 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

## Results

Top decode `ncols_dst=1` Q3_K resource buckets:

| Bucket | Calls | Total ms | Avg ms | Regs | Shared | Blocks/CU | Occupancy | Waves/CU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fused `ncols_x=5120`, `grid.x=8704` | `962` | `341.640` | `0.355` | `84` | `512 B` | `28` | `87.5%` | `56` |
| fused `ncols_x=17408`, `grid.x=2560` | `962` | `211.049` | `0.219` | `84` | `512 B` | `28` | `87.5%` | `56` |
| direct `ncols_x=5120`, `grid.x=5120` | `720` | `112.252` | `0.156` | `88` | `256 B` | `28` | `87.5%` | `56` |
| direct `ncols_x=5120`, `grid.x=3072` | `720` | `89.043` | `0.124` | `88` | `256 B` | `28` | `87.5%` | `56` |

Interpretation:

- Q3_K MMVQ is not shared-memory limited; static shared is tiny.
- Fused/direct Q3_K both have high occupancy already.
- Future candidates must prove they reduce work without raising registers or cutting grid-level latency hiding.

## Decision

- Keep `GGML_TRACE_MMVQ_RESOURCES=1` as diagnostic infrastructure.
- Use this trace as the first gate for subsequent H39 fused/direct Q3_K route changes.

## Artifacts

- `build_logs/agent-workload/e163-rocm-decode-q4-mmvq-resources-r1.server.log`
- `build_logs/agent-workload/e163-rocm-decode-q4-mmvq-resources-r1.diagnostics.md`
