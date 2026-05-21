# E125 Vulkan CPU 0-Offload Route Map

## Metadata

- Experiment ID: E125
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master, local working tree
- Hypothesis ID: H37
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan build, `-ngl 0`, `ctx=4096`, `batch=512`, `ubatch=128`, q4 KV, FlashAttention on, no reuse, thinking on, `max_tokens=32`

## Hypothesis

- Statement: the slow `-ngl 0` Vulkan route can be improved by separating residency/paging effects, Vulkan op-offload, CPU thread count, KV type, and the true CPU Q3_K decode kernel path.
- Mechanism: `-ngl 0` keeps model layers on CPU, but Vulkan build still schedules some graph work through Vulkan op-offload. Prompt/prefill and decode therefore have different bottlenecks. Decode is expected to be dominated by x86 `Q3_K -> Q8_K` dot products because Q3_K has no repacked/interleaved CPU route.
- Why now: user wants CPU fallback and future hybrid routes for constrained offload. CPU-only testing is much slower, so this lane uses a reduced real-server contract.

## Math / Theory

- Assumptions:
  - The local model is mostly Q3_K_S, so `MUL_MAT` against Q3_K weights dominates the CPU fallback route.
  - On Zen 3 / AVX2, Q3_K decode is memory/dequant heavy and does not benefit from many OpenMP threads.
  - If mmap paging is visible, `--no-mmap` can improve residency without changing kernels.
- Expected speedup corridor:
  - No-code residency/thread changes: low single digits to about `+8%`.
  - Partial Vulkan offload: proportional to number of layers removed from CPU.
  - Real code route: requires Q3_K repack/interleaved matvec or deeper x86 vec-dot work.
- Failure conditions:
  - A flag speeds only prompt but hurts decode, or only one noisy r1.
  - Disabling op-offload is mistaken for a cleaner CPU route while it destroys prompt TPS.
  - A CPU x86 micro-change is measured only against a stale or already-modified binary.

## Benchmark Plan

- Baseline command shape:

```powershell
python scripts\agent_workload_bench.py --label e125-cpu0offload-default32-r3 --server-bin build-vulkan\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug --runs 3 --ctx-size 4096 --batch-size 512 --ubatch-size 128 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 0 --max-tokens 32 --real-context-mode off --task-hard-timeout 90 --request-timeout 120 --startup-timeout 180 --server-extra "--spec-type none" --no-disable-thinking --no-reuse --write-diagnostics
```

- Candidate commands changed one variable at a time:
  - `--threads 6 --threads-batch 6`
  - `--no-op-offload`
  - f16/f16 KV
  - `--mlock`
  - `--no-mmap`
  - `--no-mmap --threads 6 --threads-batch 6`
  - partial offload with `ngl=8/16/32/48/65` and `--no-mmap`
- Number of runs:
  - r3 for default, thread, no-op-offload, no-mmap, no-mmap+t6.
  - r1 for clearly diagnostic or expensive partial-offload gates.
- Artifacts path: `build_logs/agent-workload/e125-*`.

## Metrics

| Label | Route | Runs | Aggregate TPS | Prompt eval | Decode eval | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `e125-cpu0offload-default32-r3` | `-ngl 0`, mmap, op-offload on | 3 | `1.7703` | `32.5033 tok/s` | `2.3267 tok/s` | baseline |
| `e125-cpu0offload-t6-32-r3` | `--threads 6 --threads-batch 6` | 3 | `1.7995` | `30.6367 tok/s` | `2.4267 tok/s` | tie/small decode skew |
| `e125-cpu0offload-noopoff32-r3` | `--no-op-offload` | 3 | `0.8900` | about `6.18 tok/s` | about `2.47 tok/s` | reject |
| `e125-cpu0offload-f16kv32-r1` | f16/f16 KV | 1 | `1.7617` | `27.69 tok/s` | `2.45 tok/s` | reject |
| `e125-cpu0offload-mlock32-r1` | `--mlock` | 1 | `1.7196` | `27.81 tok/s` | `2.36 tok/s` | reject |
| `e125-cpu0offload-nommap32-r3` | `--no-mmap` | 3 | `1.8815` | `33.9133 tok/s` | `2.4900 tok/s` | keep |
| `e125-cpu0offload-nommap-t6-32-r3` | `--no-mmap --threads 6 --threads-batch 6` | 3 | `1.8931` | `31.4133 tok/s` | `2.5767 tok/s` | optional decode skew, not a default |

Partial-offload scout with `--no-mmap`:

| Label | GPU layers | Aggregate TPS | Notes |
| --- | ---: | ---: | --- |
| `e125-vulkan-hybrid-ngl8-nommap32-r1` | 8 | `2.11` | still mostly CPU-bound |
| `e125-vulkan-hybrid-ngl16-nommap32-r1` | 16 | `2.32` | modest win |
| `e125-vulkan-hybrid-ngl32-nommap32-r1` | 32 | `3.46` | about 1.84x vs `--no-mmap -ngl 0` |
| `e125-vulkan-hybrid-ngl48-nommap32-r1` | 48 | `6.03` | strong hybrid route |
| `e125-vulkan-full-ngl65-nommap32-r1` | 65 | `28.93` | GPU route, not CPU fallback |

## Route Findings

- `-ngl 0` in the Vulkan build is not pure CPU. Server logs show `offloaded 0/65 layers to GPU`, but also Vulkan compute buffers and graph splits:
  - `CPU_Mapped model buffer size = 11775.72 MiB`
  - `Vulkan0 compute buffer = 83.69 MiB`
  - `Vulkan_Host compute buffer = 9.89 MiB`
  - `graph splits = 1023 (bs=128), 97 (bs=1)`
- `--no-op-offload` collapses the route to a mostly CPU scheduler split, but prompt eval drops by about 4-5x. It is a negative control, not an optimization.
- q4 V cache requires FlashAttention. `--no-flash-attn` failed at init with `V cache quantization requires flash_attn`.
- `--no-mmap` improves both wall TPS and decode/prompt split on this short real-server contract. The mechanism is residency/page behavior, not a kernel route change.
- Extra CPU threads are not a clean default. `t6` improves decode but hurts prompt; many more threads regress, consistent with memory/dequant pressure.
- Partial layer offload is the practical hybrid lever if VRAM is constrained. Every additional offloaded layer removes CPU Q3_K work; the curve is monotonic in this scout.

## CPU Code Route

Important files:

- `ggml/src/ggml-cpu/ggml-cpu.c`
  - `GGML_TYPE_Q3_K` uses `ggml_vec_dot_q3_K_q8_K`.
  - `GGML_TYPE_Q3_K` has `.nrows = 1`.
- `ggml/src/ggml-cpu/arch/x86/quants.c`
  - `ggml_vec_dot_q3_K_q8_K(...)` is the active x86 AVX2 dot route.
  - Current working tree contains a local Q3_K mask/shuffle preload micro-change. E125 does not isolate it as a separate speed claim because the built binary already included it.
- `ggml/src/ggml-cpu/repack.cpp`
  - Repack supports Q4_0, Q4_K, Q5_K, Q6_K, Q2_K, IQ4_NL, MXFP4, Q8_0.
  - There is no Q3_K repack/interleaved route in the checked code path.

Interpretation: small x86 arithmetic rewrites may not move wall TPS much. A larger CPU win likely needs a Q3_K repack format or a multi-row/interleaved matvec path so decode can use better cache locality and amortize dequant work.

## Result

- Outcome: keep `--no-mmap` as the first CPU fallback improvement and keep Vulkan op-offload enabled for `-ngl 0`.
- Delta:
  - `--no-mmap`: `1.7703 -> 1.8815 TPS`, `+6.28%`.
  - `--no-mmap --threads 6`: `1.7703 -> 1.8931 TPS`, `+6.94%`, but only `+0.62%` vs `--no-mmap` and less attractive for prompt.
  - `--no-op-offload`: `1.7703 -> 0.8900 TPS`, about `-49.7%`.
- Confidence:
  - High for `--no-mmap` as a route/profile improvement on this local machine because it was r3 and preserves the same graph route.
  - Medium for thread tuning because the prompt/decode tradeoff depends on output length.
  - High that code-level CPU work should start at Q3_K vec-dot/repack, not ngram/spec flags.
- Recommendation:
  - Add/choose `--no-mmap` for Vulkan `-ngl 0` CPU fallback.
  - Keep `--spec-type none` and q4 KV + FA.
  - Use partial `ngl` rather than pure CPU when any VRAM is available.
  - Start the next code experiment with a clean A/B around Q3_K x86 vec-dot or a Q3_K repack path; do not promote the current dirty `quants.c` micro-change without an isolated clean-vs-candidate rebuild.

## Artifacts

- `build_logs/agent-workload/e125-cpu0offload-default32-r3.csv`
- `build_logs/agent-workload/e125-cpu0offload-default32-r3.server.log`
- `build_logs/agent-workload/e125-cpu0offload-t6-32-r3.csv`
- `build_logs/agent-workload/e125-cpu0offload-noopoff32-r3.csv`
- `build_logs/agent-workload/e125-cpu0offload-f16kv32-r1.csv`
- `build_logs/agent-workload/e125-cpu0offload-mlock32-r1.csv`
- `build_logs/agent-workload/e125-cpu0offload-nommap32-r3.csv`
- `build_logs/agent-workload/e125-cpu0offload-nommap-t6-32-r3.csv`
- `build_logs/agent-workload/e125-vulkan-hybrid-ngl8-nommap32-r1.csv`
- `build_logs/agent-workload/e125-vulkan-hybrid-ngl16-nommap32-r1.csv`
- `build_logs/agent-workload/e125-vulkan-hybrid-ngl32-nommap32-r1.csv`
- `build_logs/agent-workload/e125-vulkan-hybrid-ngl48-nommap32-r1.csv`
- `build_logs/agent-workload/e125-vulkan-full-ngl65-nommap32-r1.csv`
