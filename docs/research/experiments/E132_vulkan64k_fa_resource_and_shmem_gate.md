# E132 Vulkan 64k FA Resource and SHMEM Gate

## Metadata

- Experiment ID: E132
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master after E131
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, real server repo-snapshot, no reuse

## Hypothesis

- Statement: after E131 identified the active 64k FA route, collect driver resource stats and test whether forcing FA shared-memory staging can reduce long-KV global traffic.
- Mechanism: the default coopmat1 route uses q4/q4 KV and long `KV=1024..57344` chunks. Staging K/V through shared memory might reduce repeated long-KV reads, but may exceed shared-memory limits or reduce occupancy.
- Why now: E131 rejected simple `Bc`, mask-opt, and f16acc toggles. SHMEM staging is a more structural FA route change.

## Diagnostic Setup

Default route diagnostics:

```powershell
$env:GGML_VK_ALLOW_GRAPHICS_QUEUE='1'
$env:GGML_VK_FA_ROUTE_TRACE='1'
$env:GGML_VK_PIPELINE_STATS='flash_attn_f32_f16_aligned_f32accq4_0'
python scripts\repo_snapshot_context_bench.py --label-prefix e132-vulkan64k-fa-pipeline-stats-c152k-b8192-ub1024-q4 --server-bin build-vulkan\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --ctx-values 65536 --allow-ctx-above-16k --base-char-budget 38000 --max-tokens 1 --gpu-layers 999 --batch-size 8192 --ubatch-size 1024 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0 --no-mmap"
```

Temporary code probe:

- Added and then reverted `GGML_VK_FA_FORCE_SHMEM_STAGING=1` inside `get_fa_tuning_params_coopmat1(...)`.
- Rebuilt `llama-server`.
- Ran the same 64k real-server screen with route trace and pipeline stats.

## Metrics

Default active FA resource stats:

| Route | VGPR | SGPR | LDS | Scratch | Prompt Eval TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup GQA coopmat1 route | `101` | `69` | `26112 B` | `0` | n/a |
| main 64k coopmat1 route | `98` | `76` | `26112 B` | `0` | `669.77` |

Forced SHMEM staging probe:

| Probe | Actual route | VGPR | SGPR | LDS | Scratch | Prompt Eval TPS | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `GGML_VK_FA_FORCE_SHMEM_STAGING=1` | scalar fallback, `Bc=32` | `192` | `78` | `6144 B` | `0` | `520.18` | reject/revert |

## Result

- Outcome: reject SHMEM staging for the current AMD coopmat1 q4/q4 FA route.
- Why it missed: the forced staging profile does not remain on the intended coopmat1 route. It fails the shared-memory support gate and falls back to scalar FA. The scalar fallback then uses `Bc=32`, much higher VGPR pressure (`192`), and loses about `22%` prompt eval versus the default diagnostic run.
- Confidence: high for rejecting this exact mechanism.
- Recommendation: do not add an env knob for SHMEM staging on this route. Future FA work needs either a valid lower-LDS staging design or a shader-level memory/coalescing change that keeps the coopmat1 route active.

## Q3_K Gate Note

The parallel Q3_K analytical gate was also re-run for the 64k long-K shape:

- `BK=64` halves K-loop/barrier rounds but leaves full-K B/dequant traffic unchanged and increases Q3 LDS to `36864 B`.
- Prebuild gate still classifies it as `needs-resource-proof` / `defer-for-higher-ceiling`.
- `bn256`/`bm256` families remain closed by E098: they had plausible static work reduction but regressed in driver pp7488 due LDS/occupancy/register effects.

## Workflow Correction

- Do not infer that a route is active from the requested tuning params alone. Route trace must confirm the final path after shared-memory support gates.
- For FA, a useful candidate must prove that it stays on `coopmat1` and does not silently fall back to scalar.

## Artifacts

- `build_logs/agent-workload/e132-vulkan64k-fa-pipeline-stats-c152k-b8192-ub1024-q4-repo-summary.md`
- `build_logs/agent-workload/e132-vulkan64k-fa-pipeline-stats-c152k-b8192-ub1024-q4-ctx64k.server.log`
- `build_logs/agent-workload/e132-vulkan64k-fa-shmem-staging-c152k-b8192-ub1024-q4-screen-repo-summary.md`
- `build_logs/agent-workload/e132-vulkan64k-fa-shmem-staging-c152k-b8192-ub1024-q4-screen-ctx64k.server.log`
