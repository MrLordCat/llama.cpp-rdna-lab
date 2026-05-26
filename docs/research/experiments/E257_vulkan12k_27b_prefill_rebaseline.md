# E257 Vulkan 12k Dense 27B Prefill Rebaseline

## Metadata

- Experiment ID: E257
- Date: 2026-05-26
- Owner: Copilot
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, cold-first 12k prompt-heavy
- User directive: focus only on dense Qwen3.6 27B and work on Vulkan because Vulkan's weak side is prefill.

## Contract

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Server: `build-vulkan/bin/llama-server.exe`
- Context: `ctx=12288`
- KV: `q4_0/q4_0`
- FlashAttention: on
- Prompt task: `quick:triage_diff`
- Real context: `repo-snapshot`
- Thinking: on (`--no-disable-thinking`)
- Reuse/prime: off (`--no-reuse`, `--no-v2-prime-pass`)
- Speculation: off (`--spec-type none`)

## Baselines

Fresh control with the previously active 12k shape:

| Label | Batch | UBatch | Runs | TPS | Prompt tok/s | Decode tok/s | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `e257-vulkan12k-fresh-control-r1` | 6144 | 2048 | 1 | 6.69 | - | - | initial fresh control |
| `e257-vulkan12k-control-b6144-ub2048-r3` | 6144 | 2048 | 3 | 6.6895 | 947.36 | 39.56 | confirmed control |
| `e257-vulkan12k-nommap-control-r1` | 6144 | 2048 | 1 | 6.64 | - | - | `--no-mmap` did not repeat E227's old win |

Prompt-only checks:

| Shape | pp7488 tok/s | Notes |
| --- | ---: | --- |
| b4096/ub1024 | 979.85 +/- 1.24 | accepted H31-style prompt gate |
| b6144/ub2048 | 953.25 +/- 11.20 | current control shape, lower prompt throughput |
| b6144/ub1024 | 979.67 +/- 0.87 | smaller ubatch recovers prompt throughput |
| b7168/ub1024 | 971.38 +/- 27.43 | wall winner despite pp variance |

## Perf Trace

Intrusive trace: `e257-vulkan12k-perflog-r1`, `max_tokens=1`, `GGML_VK_PERF_LOGGER=1`, `GGML_VK_MATMUL_ROUTE_TRACE=1`, `GGML_VK_FA_ROUTE_TRACE=1`.

Parsed total: `6447.72 ms`.

| Bucket | Calls | Total ms | Parsed share |
| --- | ---: | ---: | ---: |
| `MUL_MAT q3_K` | 1396 | 5333.04 | 82.71% |
| `FLASH_ATTN_EXT` | 80 | 618.94 | 9.60% |
| `MUL_MAT q4_K` | 192 | 359.19 | 5.57% |
| `MUL_MAT f32` | 688 | 90.15 | 1.40% |

Hot Q3_K shapes:

| Shape | Total ms | Parsed share |
| --- | ---: | ---: |
| `m=17408,n=2048,k=5120` | 2089.73 | 32.41% |
| `m=5120,n=2048,k=17408` | 1148.00 | 17.80% |
| `m=10240,n=2048,k=5120` | 493.37 | 7.65% |
| `m=17408,n=1382,k=5120` | 466.73 | 7.24% |

The route remains `matmul_q3_k_f32_f16acc_aligned_l`; H31 shader prebuild gate still blocks nearby helper/tile edits unless a genuinely new topology is proposed.

## Shape Gate

Single-run sweep:

| Shape | TPS | Decision |
| --- | ---: | --- |
| b4096/ub1024 | 6.69 | tie with control |
| b5120/ub1024 | 6.76 | positive but not best |
| b6144/ub1024 | 7.05 | promising |
| b5120/ub2048 | 6.55 | reject |
| b7168/ub1024 | 7.09 | best r1 |
| b8192/ub1024 | 6.70 | reject |
| b7168/ub768 | 6.71 | reject |
| b7168/ub512 | 6.93 | below b7168/ub1024 |
| b6144/ub768 | 6.76 | reject |

Confirmations:

| Label | Batch | UBatch | Runs | TPS | Prompt tok/s | Decode tok/s | Delta vs b6144/ub2048 r3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e257-vulkan12k-control-b6144-ub2048-r3` | 6144 | 2048 | 3 | 6.6895 | 947.36 | 39.56 | reference |
| `e257-vulkan12k-shape-b6144-ub1024-r3` | 6144 | 1024 | 3 | 6.8084 | 965.74 | 39.89 | +1.78% |
| `e257-vulkan12k-shape-b7168-ub1024-r3` | 7168 | 1024 | 3 | 7.0319 | 999.22 | 40.93 | +5.12% |

## Decision

Keep `ctx=12288,b=7168,ub=1024,q4_0/q4_0,spec=none` as the current Vulkan dense 27B cold 12k profile. This is not enough to reach the old 10 TPS cold target, but it is a measured `+5.12%` wall and `+5.47%` prompt eval improvement over the previous Vulkan control shape without source risk.

Update the GUI Qwen3.6-27B-Q3_K_S active preset to this Vulkan shape and expand the active autotune batch step to include `7168`.

## Artifacts

- `build_logs/agent-workload/e257-vulkan12k-fresh-control-r1.diagnostics.md`
- `build_logs/agent-workload/e257-vulkan12k-control-b6144-ub2048-r3.diagnostics.md`
- `build_logs/agent-workload/e257-vulkan12k-shape-b6144-ub1024-r3.diagnostics.md`
- `build_logs/agent-workload/e257-vulkan12k-shape-b7168-ub1024-r3.diagnostics.md`
- `build_logs/agent-workload/e257-vulkan12k-perflog-r1.server.log`
