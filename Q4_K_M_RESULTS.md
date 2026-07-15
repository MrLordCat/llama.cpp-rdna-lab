# Qwen3.6-27B Q4_K_M Results

Measured on 2026-07-15 with no foreground GPU workload.

## Test system

- 2x AMD Radeon RX 9070 XT 16 GB
- `Qwen3.6-27B-Q4_K_M.gguf` (15.932 GiB)
- Windows 11, ROCm/HIP 7.1 and Vulkan
- dual-GPU layer split, one server slot, full GPU offload
- FlashAttention, `b8192/ub1024`, q8_0 K/V cache
- cold prompts, no cache reuse, no warmup, seed 42

## Performance

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

At 29.5k prompt tokens, ROCm MTP loses 3.48% prompt throughput and gains
96.1% decode throughput. Vulkan MTP loses 2.97% prompt throughput and gains
79.1% decode throughput.

## Long-context residency

| Backend | Context | Actual prompt | Prompt TPS | Decode TPS | Dedicated peak | Shared peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm | 49,152 | 29,561 | 1716.66 | 20.28 | 20.63 GiB | 1.51 GiB |
| Vulkan | 49,152 | 29,561 | 1432.13 | 26.36 | 18.09 GiB | 0.29 GiB |
| ROCm | 98,304 | 58,982 | 1466.27 | 18.77 | 24.01 GiB | 5.48 GiB |
| Vulkan | 98,304 | 58,982 | 1171.17 | 24.18 | 20.06 GiB | 0.54 GiB |
| ROCm | 131,072 | 75,979 | **447.59** | 17.19 | 25.97 GiB | **7.60 GiB** |
| Vulkan | 131,072 | 75,979 | 1051.67 | **23.02** | 21.38 GiB | 0.70 GiB |

ROCm remains the faster prompt-evaluation backend while its working set is
resident enough. At `ctx=131072`, ROCm reaches a real spill cliff: one device
reports no free local budget, Shared GPU Memory reaches 7.60 GiB, and prompt
throughput falls to 447.59 tok/s. Vulkan keeps a much smaller working set and
continues at 1051.67 tok/s on the same prompt.

Q4_K_M cannot be fully resident on one 16 GiB card. The GPU model tensors alone
use about 15.25 GiB before KV, recurrent state, compute buffers, driver
reservations, and desktop usage.

The complete methodology, per-device WDDM measurements, build IDs, and artifact
names are recorded in
[E332: Qwen3.6-27B Q4_K_M performance and residency](docs/research/experiments/E332_qwen36_q4km_performance_and_residency.md).
