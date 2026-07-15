# E332: Qwen3.6-27B Q4_K_M performance and residency

Date: 2026-07-15

## Goal

Measure only `Qwen3.6-27B-Q4_K_M.gguf` on the reference dual-RX 9070 XT
machine and identify the context size at which the working set starts using
system RAM heavily enough to reduce throughput.

The GGUF is 17,106,773,120 bytes (15.932 GiB) and includes the Qwen NextN/MTP
tensors. No foreground GPU workload was active during these runs.

## Locked configuration

All rows use two GPUs, one server slot, full GPU offload, FlashAttention,
`b8192/ub1024`, q8_0 K/V cache, seed 42, no warmup, no prompt-cache reuse,
`--cache-ram 0`, `--ctx-checkpoints 0`, and `-fit off`.

ROCm uses `-dev ROCm1,ROCm0 -sm layer -ts 1,1`. Vulkan short runs use
`-dev Vulkan0,Vulkan1`; long runs use `-dev Vulkan1,Vulkan0` with
`LLAMA_OUTPUT_DEVICE=Vulkan1` and `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`.
MTP rows use `--spec-type draft-mtp --spec-draft-n-max 3`.

The performance builds were:

| Backend | Build |
| --- | --- |
| ROCm | `b9323-6daf9a9e8` |
| Vulkan | `b9322-255b8ab05` |

## Performance

### Short and matched-long lanes

| Backend | Mode | Context | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm | none | 12,288 | 6,393 / 256 | 1592.45 | 23.50 | 17.14 | - |
| ROCm | MTP n3 | 12,288 | 6,393 / 256 | 1545.58 | **43.56** | **25.46** | 68.40% |
| Vulkan | none | 12,288 | 6,393 / 128 | 1229.31 | 26.55 | 12.73 | - |
| Vulkan | MTP n3 | 12,288 | 6,393 / 128 | **1320.21** | **50.56** | **17.26** | 64.34% |
| ROCm | none, r2 mean | 49,152 | 29,561 / 128 | **1715.65** | 20.35 | 5.43 | - |
| ROCm | MTP n3 | 49,152 | 29,561 / 128 | 1656.03 | **39.90** | **6.06** | 77.19% |
| Vulkan | none | 49,152 | 29,561 / 128 | **1432.13** | 26.36 | 5.01 | - |
| Vulkan | MTP n3 | 49,152 | 29,561 / 128 | 1389.59 | **47.21** | **5.32** | 69.11% |

On the matched 29.5k-token lane, ROCm MTP costs 3.48% prompt throughput and
raises decode by 96.1%. Vulkan MTP costs 2.97% prompt throughput and raises
decode by 79.1%. Q4_K_M MTP is therefore useful while this lane remains
resident enough; it is not intrinsically slower than the Q4 baseline.

### Context and residency sweep, baseline decode

| Backend | Context | Actual prompt | Prompt TPS | Decode TPS | Aggregate TPS | Dedicated peak | Shared peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm | 49,152 | 29,561 | 1716.66 | 20.28 | 5.43 | 20.63 GiB | 1.51 GiB |
| Vulkan | 49,152 | 29,561 | 1432.13 | 26.36 | 5.01 | 18.09 GiB | 0.29 GiB |
| ROCm | 98,304 | 58,982 | 1466.27 | 18.77 | 2.71 | 24.01 GiB | 5.48 GiB |
| Vulkan | 98,304 | 58,982 | 1171.17 | 24.18 | 2.29 | 20.06 GiB | 0.54 GiB |
| ROCm | 131,072 | 75,979 | **447.59** | 17.19 | 0.72 | 25.97 GiB | **7.60 GiB** |
| Vulkan | 131,072 | 75,979 | 1051.67 | **23.02** | **1.64** | 21.38 GiB | 0.70 GiB |

`Context` is the allocated slot size; `Actual prompt` is the token count sent
to the model. The 98k and 131k rows therefore test both a larger KV allocation
and a longer prompt.

The ROCm 98k row already commits substantial Shared GPU Memory, but retains
good prompt throughput. The clear harmful-residency boundary is the 131k row:
Shared reaches 7.60 GiB, one ROCm device reports zero free local budget at
shutdown, and prompt throughput falls from 1466.27 to 447.59 tok/s. The same
131k workload remains healthy on Vulkan at 1051.67 tok/s with only 0.70 GiB
Shared.

## Per-device WDDM peaks

These are per-process WDDM counters for `llama-server.exe`, not total desktop
GPU usage. `Display` is the display-attached card and `Secondary` is the other
RX 9070 XT.

| Backend | Mode / context | Dedicated display / secondary | Shared display / secondary | Process private peak |
| --- | --- | ---: | ---: | ---: |
| ROCm | none / 12k | 8.88 / 9.50 GiB | 0.38 / 0.38 GiB | 17.92 GiB |
| ROCm | MTP n3 / 12k | 9.14 / 10.85 GiB | 0.38 / 0.59 GiB | 18.90 GiB |
| Vulkan | none / 12k | 7.99 / 8.70 GiB | 0.11 / 0.00 GiB | 18.33 GiB |
| Vulkan | MTP n3 / 12k | 8.24 / 8.99 GiB | 0.27 / 0.01 GiB | 20.31 GiB |
| ROCm | none / 49k | 10.15 / 10.48 GiB | 0.18 / 1.32 GiB | 18.11 GiB |
| ROCm | MTP n3 / 49k | 10.41 / 12.40 GiB | 1.00 / 1.72 GiB | 20.09 GiB |
| Vulkan | none / 49k | 8.38 / 9.71 GiB | 0.23 / 0.06 GiB | 20.01 GiB |
| Vulkan | MTP n3 / 49k | 8.81 / 9.97 GiB | 0.40 / 0.06 GiB | 22.94 GiB |
| ROCm | none / 98k | 12.17 / 11.84 GiB | 1.79 / 3.69 GiB | 21.29 GiB |
| Vulkan | none / 98k | 9.28 / 10.79 GiB | 0.42 / 0.12 GiB | 22.52 GiB |
| ROCm | none / 131k | 12.67 / 13.30 GiB | 3.40 / 4.22 GiB | 21.79 GiB |
| Vulkan | none / 131k | 9.87 / 11.51 GiB | 0.55 / 0.15 GiB | 23.96 GiB |

Process private memory includes the mapped GGUF and host runtime allocations;
it must not be interpreted as GPU spill. Shared GPU Memory is the more useful
pressure signal, but it can also include driver-managed or pinned allocations.
It is treated as harmful paging only when it coincides with a throughput cliff,
as it does for ROCm at 131k.

## Capacity conclusions

1. Q4_K_M cannot be fully resident on one 16 GiB card. The GPU-resident model
   tensors alone total about 15.25 GiB before KV, recurrent state, compute
   buffers, driver reservations, or desktop usage.
2. Dual-GPU Q4_K_M is healthy at `ctx=49152`. MTP is also viable there and
   almost doubles decode with only a 3% to 3.5% prompt cost.
3. `ctx=98304` is a pressure zone. ROCm uses 5.48 GiB Shared but has not yet
   reached a catastrophic prompt-eval cliff; Vulkan remains comfortably lower.
4. `ctx=131072` with a 75,979-token prompt is not a production-safe ROCm Q4_K_M
   configuration on this machine. Vulkan handles the same lane much better.
5. For Q4_K_M, use ROCm for the fastest prompt evaluation up to the measured
   49k lane. Use Vulkan when the allocated context or real prompt approaches
   the high-pressure region, unless later ROCm residency work changes this
   boundary.

## Reproduction and artifacts

The monitor is `scripts/research/windows_gpu_memory_monitor.py`. It samples the
WDDM `GPU Process Memory` Dedicated and Shared counters by process and LUID,
plus process private and working-set peaks. It also tags startup, prefill, and
decode phases from the server log.

Benchmark artifacts use the prefix `e332-q4km-` under
`build_logs/agent-workload`. The canonical command shape is:

```powershell
python scripts\agent_workload_bench.py `
  --server-bin <build>\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q4_K_M.gguf `
  --ctx-size <12288|49152|98304|131072> `
  --batch-size 8192 --ubatch-size 1024 `
  --cache-type-k q8_0 --cache-type-v q8_0 `
  --tasks quick --task-ids triage_diff --runs 1 --no-reuse `
  --request-timeout 300 --task-hard-timeout 0 --task-fail-timeout 0 `
  --server-extra "-dev <route> -sm layer -ts 1,1 --cache-ram 0 --ctx-checkpoints 0 -fit off"
```
