# E138 Vulkan 64k FlashAttention Forced Split-K Gate

## Metadata

- Experiment ID: E138
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E137 (`4e722e12e`)
- Hypothesis ID: H38
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: forcing Vulkan FlashAttention split-k on long-KV chunks might reduce per-workgroup loop length and improve the `KV=8k..57k` tail.
- Mechanism: the active FA route already has a split-k/reduce path, but heuristic leaves main prefill chunks at `split_k=1` because the graph has many row/head workgroups (`N=1024`, `24` heads). If long-KV cost is dominated by per-workgroup memory loop length rather than occupancy, a forced split could help.
- Why now: E129-E132 rejected nearby FA tile/mask/accumulation/staging toggles. This tests an existing whole-route topology rather than another tile-size microprobe.

## Implementation Probe

Temporary, reverted host-only env gate:

- `GGML_VK_FA_FORCE_SPLIT_K=2`
- `GGML_VK_FA_FORCE_SPLIT_K_MIN_KV=8192`

This did not change shaders or default pipeline state. It only forced the existing FA split-k route for main chunks with `KV >= 8192` and `gqa_ratio <= 1`.

## Benchmark Plan

Real server, full 64k-style prompt screen:

```powershell
$env:GGML_VK_ALLOW_GRAPHICS_QUEUE='1'
$env:GGML_VK_FA_ROUTE_TRACE='1'
python scripts\repo_snapshot_context_bench.py `
  --label-prefix e138-vulkan64k-fa-splitk-default `
  --server-bin build-vulkan\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --ctx-values 65536 --allow-ctx-above-16k `
  --batch-size 8192 --ubatch-size 1024 `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --gpu-layers 999 --max-tokens 1 `
  --base-ctx 65536 --base-char-budget 152000 --min-char-budget 152000 `
  --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap --flash-attn on" `
  --request-timeout 1200
```

Candidate adds:

```powershell
$env:GGML_VK_FA_FORCE_SPLIT_K='2'
$env:GGML_VK_FA_FORCE_SPLIT_K_MIN_KV='8192'
```

## Result

| Route | Prompt tokens | Elapsed | Prompt eval | Main FA split |
| --- | ---: | ---: | ---: | --- |
| Default | `57518` | `86.3639 s` | `666.87 tok/s` | `split_k=1` |
| Forced split-k2 from `KV>=8192` | `57518` | `597.4568 s` | `96.29 tok/s` | `split_k=2`, `split_kv=KV/2` |

Delta: `-85.56%` prompt eval.

## Interpretation

- The route topology explains the failure. The split-k branch writes a temporary output/L/M buffer, calls `ggml_vk_sync_buffers`, dispatches `pipeline_flash_attn_split_k_reduce`, and marks the split buffer dirty. That adds at least one extra dispatch plus a synchronization boundary per FA node.
- The default main route already has enough workgroups: route trace shows `N=1024`, `Br=16`, `Tr=64`, `wg_y=24`, so roughly `1536` row/head workgroups before considering KV loop work. Forcing more KV parallelism does not fix occupancy; it creates repeated global writes and graph serialization.
- This was predictable from the code before a full run: forced split-k should have been treated as high-risk because the existing reduce path is not a fused in-shader long-KV decomposition.
- A future FA route must avoid the sync/reduce topology for every chunk. Useful directions are inside the coopmat1 shader loop, mask/tail work that keeps one dispatch, or a redesigned reduce path that can stay graph-friendly.

## Decision

- Revert the env gate and reject forced split-k for the H38 lane.
- Do not repeat split-k forcing with different thresholds unless the reduce path is redesigned to avoid per-node synchronization.
- Workflow correction: FA long-KV candidates must state whether they add a new dispatch/sync per FA node. If yes, require an analytic overhead estimate before real-server time.

## Artifacts

- `build_logs/agent-workload/e138-vulkan64k-fa-splitk-default-repo-summary.md`
- `build_logs/agent-workload/e138-vulkan64k-fa-splitk-default-ctx64k.server.log`
- `build_logs/agent-workload/e138-vulkan64k-fa-splitk2-min8192-repo-summary.md`
- `build_logs/agent-workload/e138-vulkan64k-fa-splitk2-min8192-ctx64k.server.log`
