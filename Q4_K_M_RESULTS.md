# Qwen3.6-27B Q4_K_M Results

ROCm short and 49K rows were refreshed on 2026-07-15 after the E334
quantized-KV scratch reservation change. The 98K production row was refreshed
on 2026-07-16 after E337 bounded Q8 WMMA and the E338 one-copy ROCm scheduler
default. Vulkan rows are the unchanged E332 reference measurements. No
foreground GPU workload was active.

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
| ROCm | none | 12,288 | 6,393 / 256 | 1522.70 | 22.23 | 16.23 | - |
| ROCm | MTP n3 | 12,288 | 6,393 / 256 | 1444.35 | **42.12** | **24.23** | 68.40% |
| Vulkan | none | 12,288 | 6,393 / 128 | 1229.31 | 26.55 | 12.73 | - |
| Vulkan | MTP n3 | 12,288 | 6,393 / 128 | **1320.21** | **50.56** | **17.26** | 64.34% |
| ROCm | none, r2 mean | 49,152 | 29,561 / 128 | **1716.16** | 20.00 | 5.39 | - |
| ROCm | MTP n3 | 49,152 | 29,561 / 128 | 1604.76 | **38.10** | **5.86** | 77.19% |
| Vulkan | none | 49,152 | 29,561 / 128 | **1432.13** | 26.36 | 5.01 | - |
| Vulkan | MTP n3 | 49,152 | 29,561 / 128 | 1389.59 | **47.21** | **5.32** | 69.11% |
| ROCm | none | 98,304 | 59,045 / 64 | **1493.21** | 19.15 | **1.4890** | - |
| ROCm | MTP n3 | 98,304 | 59,045 / 64 | 1435.97 | **35.44** | 1.4872 | **80.00%** |

At 29.5k prompt tokens, current ROCm MTP loses 6.49% prompt throughput, gains
90.5% decode throughput, and gains 8.52% aggregate throughput. Vulkan MTP loses
2.97% prompt throughput and gains 79.1% decode throughput in the archived E332
reference.

At 59k prompt tokens, ROCm MTP loses 3.83% prompt throughput and gains 85.1%
decode throughput. The 64-token request is exactly at the amortization
boundary: wall time changes only from 42.98 to 43.04 seconds. Longer generated
answers favor MTP.

## Long-context residency

| Backend | Context | Actual prompt | Prompt TPS | Decode TPS | Dedicated peak | Shared peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ROCm, current r2 | 49,152 | 29,561 | 1716.16 | 20.00 | 21.21 GiB | 2.60 GiB |
| Vulkan | 49,152 | 29,561 | 1432.13 | 26.36 | 18.09 GiB | 0.29 GiB |
| ROCm, pre-E337 spill | 98,304 | 59,004 | 553.50 | 17.64 | 24.45 GiB | 6.25 GiB |
| ROCm, E338 one copy, none | 98,304 | 59,045 | **1493.21** | 19.15 | 22.05 GiB | 3.20 GiB |
| ROCm, E338 one copy, MTP n3 | 98,304 | 59,045 | 1435.97 | **35.44** | 23.96 GiB | 3.26 GiB |
| Vulkan | 98,304 | 58,982 | 1171.17 | 24.18 | 20.06 GiB | 0.54 GiB |
| Vulkan | 131,072 | 75,979 | 1051.67 | **23.02** | 21.38 GiB | 0.70 GiB |

ROCm remains the faster prompt-evaluation backend at `ctx=49152`. The old
`ctx=98304` run reached a real residency cliff: Shared peaked at 6.25 GiB and
prompt throughput fell to 553.50 tok/s. E337 removed context-sized Q8
FlashAttention staging, and E338 reduced the single-request ROCm scheduler from
four graph copies to one. The matched 64-token 98K control now measures
1493.21 tok/s, 2.70x the old spill result and 27.5% above the recorded Vulkan
prompt row.

Windows still reports 3.20 GiB Shared. This is largely pageable WDDM backing
for HIP graph allocations rather than proof of active RAM reads: dedicated
residency remains high and prompt throughput no longer has the spill shape.
The KV cache itself is allocated at context creation and does not grow as the
prompt is processed. Enabling MTP n3 raises prefill Dedicated by 1.91 GiB but
Shared by only 0.057 GiB, confirming that its extra working set remains local.

The 131K ROCm rows were deliberately not repeated after this result. The older
pre-E334 controls were 447.59 tok/s with equal `1:1` placement and 1379.14
tok/s with the memory-aware `27:37` placement. They remain useful historical
evidence for placement sensitivity, but they are not current post-fix numbers.

E334 identified and fixed one ROCm-only growth source: TILE FlashAttention
converted quantized K/V to context-sized F16 scratch through the non-VMM HIP
pool, and old sizes could accumulate as the graph grew. Reserving the scratch
in the graph bounds that allocator behavior. E337 then replaced the active
context-sized staging with bounded 4096-token scratch for RDNA4 Q8 K/V and
recovered 216 MiB in a one-card Q3 control with neutral throughput. The Q4
98K row above includes both E337 and E338. Q4 still has much less headroom than
Q3, especially with MTP or vision, but it no longer enters the old 553 tok/s
residency cliff on this tested no-MTP lane.

Q4_K_M cannot be fully resident on one 16 GiB card. The GPU model tensors alone
use about 15.25 GiB before KV, recurrent state, compute buffers, driver
reservations, and desktop usage.

The old 49K artifacts use the `e335-rocm-q4km-` prefix; the current 98K
scheduler-residency artifacts use `e338-rocm-dual-q4km-`. The benchmark
registry identifies the configured build directory as `build-rocm-full`. The
complete original methodology and historical per-device measurements are
recorded in
[E332: Qwen3.6-27B Q4_K_M performance and residency](docs/research/experiments/E332_qwen36_q4km_performance_and_residency.md)
and the follow-up
[E333: ROCm Q4_K_M memory-aware split](docs/research/experiments/E333_rocm_q4km_memory_aware_split.md)
and
[E334: ROCm quantized-KV scratch reservation](docs/research/experiments/E334_rocm_quantized_kv_scratch_reservation.md).
The post-fix rebaseline and current per-device peaks are recorded in
[E335: ROCm post-reservation rebaseline](docs/research/experiments/E335_rocm_post_reservation_rebaseline.md).
The bounded active-staging follow-up is recorded in
[E337: bounded ROCm Q8 FlashAttention WMMA](docs/research/experiments/E337_rocm_q8_chunked_wmma.md).
The scheduler-residency follow-up is recorded in
[E338: ROCm dual-GPU long-context scheduler residency](docs/research/experiments/E338_rocm_dual_long_context_scheduler_residency.md).
