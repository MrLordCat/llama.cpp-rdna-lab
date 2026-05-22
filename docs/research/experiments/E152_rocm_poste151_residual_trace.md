# E152 ROCm Post-E151 Residual Decode Trace

## Metadata

- Experiment ID: E152
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `efcd58642`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: After E151, the remaining ROCm/Vulkan short-decode gap should still be dominated by Q3_K MMVQ, so the next work should be a larger Q3_K decode route branch rather than another nearby launch/fusion toggle.
- Mechanism: E151 improved clean server decode by moving RDNA4 Q3_K `ncols_dst=1` to `nwarps=2`, but Vulkan q4 still has about a `1.27x` decode-eval lead. A post-E151 trace should identify whether Q3_K fused/direct work still has enough residual share.
- Risk: The trace disables HIP graphs and uses synchronous timing; use it only for route structure and residual shares, not for speed claims.

## Method

Diagnostic command shape:

```powershell
$env:GGML_CUDA_DISABLE_GRAPHS = "1"
$env:GGML_TRACE_CUDA_NODE_TIMING = "1"
$env:GGML_TRACE_CUDA_NODE_TIMING_SYNC = "1"
$env:GGML_TRACE_CUDA_MUL_MAT_ROUTE = "1"
$env:GGML_TRACE_MMVQ_SMALL_K = "1"
$env:GGML_TRACE_MMVQ_TIMING = "1"
$env:GGML_TRACE_MMVQ_TIMING_SYNC = "1"
python scripts\agent_workload_bench.py --label e152-rocm-decode-q4-poste151-synctrace-mt16-r1 --out-dir build_logs\agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 16 --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --write-diagnostics
```

The route-delta parser was also corrected in this experiment: MMVQ timing logs report `grid.x`, not logical output rows. With E151 `rows_per_block=2`, the semantic shape must multiply `grid.x` by `block.y` for `small_k ncols_dst=1`. Without this fix, the post-E151 FFN shape appeared incorrectly as `m=8704` instead of `m=17408`.

## Result

Trace sanity:

- aggregate completion TPS: `5.1251`;
- decode eval: `6.16 tok/s`;
- errors: `0`;
- output preview remains normal `Thinking Process:` text.

These numbers are slow by design because graphs are disabled and sync timing is enabled.

Confirmed launch policy in the trace:

```text
type=11/q3_K ncols_dst=1 ncols_x=5120 blocks_per_row=20 nwarps=2 small_k=1
timing type=11/q3_K ncols_dst=1 small_k=1 fusion=1 ncols_x=5120 grid=(8704,1,1) block=(32,2,1)
```

Corrected post-E151 Q3_K route-delta against the E149 Vulkan comparator:

| ROCm bucket | Calls | Total ms | Share |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_q_fused q3_K->f32` | `2145` | `579.66` | `64.62%` |
| `mul_mat_vec_q_direct q3_K->f32` | `2175` | `317.33` | `35.38%` |

Top corrected normalized ROCm/Vulkan shapes:

| Shape | ROCm share | Vulkan share |
| --- | ---: | ---: |
| `q3_K m=17408 n=1 k=5120` | `37.08%` | `48.32%` |
| `q3_K m=5120 n=1 k=17408` | `24.20%` | `25.47%` |
| `q3_K m=10240 n=1 k=5120` | `13.61%` | `10.99%` |
| `q3_K m=6144 n=1 k=5120` | `11.01%` | `7.32%` |

E149 -> E152 sync comparison caveat:

- all CUDA-node sync timing is roughly tied: `3283.294 -> 3274.055 ms`;
- MMVQ timing rows alone are worse in this graph-disabled sync trace:
  `999.808 -> 1030.357 ms`;
- clean real-server r3 speed is still clearly better in E151:
  `29.77 -> 32.2467 tok/s` decode.

Interpretation: sync traces are good for topology and residual share, but they can conflict with clean replay behavior. Do not use E152 sync timing to reject E151.

## Decision

- Keep E151.
- H39 remains Q3_K-MMVQ-led after the first win: fused FFN Q3_K is still about two thirds of parsed Q3_K MMVQ time, with direct Q3_K the other third.
- The next branch should be larger than a single selector toggle. It should inspect the fused `mul_mat_vec_q<..., has_fusion=true>` Q3_K path for the FFN gate/up and down shapes, including resource pressure, row-pair scheduling, and whether a specialized RDNA4/Qwen FFN path can reduce work without the sync-trace penalties seen here.

## Artifacts

- `build_logs/agent-workload/e152-rocm-decode-q4-poste151-synctrace-mt16-r1.server.log`
- `build_logs/agent-workload/e152-rocm-decode-q4-poste151-synctrace-mt16-r1.diagnostics.md`
- `build_logs/agent-workload/e152-rocm-vulkan-decode-route-delta-q3k.md`
- `build_logs/agent-workload/e152-poste151-vs-e149-sync-compare.md`
