# E333: ROCm Q4_K_M memory-aware split (interim mitigation)

Date: 2026-07-15

## Goal

Remove the ROCm prompt-evaluation collapse observed in E332 for
`Qwen3.6-27B-Q4_K_M.gguf` at `ctx=131072` without reducing the allocated
context, moving KV to the CPU, or switching to Vulkan.

## Immediate spill trigger

The equal `-ts 1,1` split assumes equal usable VRAM, but the two identical
16 GiB cards have different WDDM budgets. During the failing run, ROCm1 had no
free local budget while ROCm0 still had several GiB available. The driver then
backed more allocations through Shared GPU Memory and prompt evaluation fell to
447.59 tok/s.

This was the immediate trigger for the throughput cliff, not the root cause of
ROCm's prompt-dependent memory growth. E334 subsequently traced that growth to
quantized-KV FlashAttention: the HIP TILE path expanded K and V to full F16
scratch buffers, while the non-VMM legacy pool retained successively larger
allocations. The `27:37` split remains a useful placement mitigation, but it
must not be described as the allocator fix.

ROCm on Windows uses the legacy ggml CUDA/HIP pool (`NO_VMM=1`). Instrumentation
added in this experiment measured about 1.8 GiB of reclaimable high-water cache
at 131K. Capping that cache reduced residency, but repeated `hipFree` calls
synchronized the prompt path and made throughput worse. Allocator trimming is
therefore retained only as an opt-in diagnostic, not enabled by default.

The effective fix is memory-aware layer placement. With devices ordered as
`ROCm1,ROCm0`, `-ts 27,37` puts fewer layers on the card with the smaller WDDM
budget and more on the card that has room.

## Results

All production-comparable rows use the E332 setup: dual RX 9070 XT, q8_0 K/V,
FlashAttention, `b8192/ub1024`, one slot, no reuse, no warmup, and a
75,979-token prompt. The final row generates 128 output tokens.

| Configuration | Prompt TPS | Decode TPS | Aggregate TPS | Dedicated peak | Shared peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal `1:1` control | 447.59 | 17.19 | 0.72 | 25.97 GiB | 7.60 GiB |
| Pool cap 256 MiB, equal `1:1` | 229.96 | 17.30 | 0.38 | 25.23 GiB | 7.60 GiB |
| Memory-aware `27:37` | **1379.14** | **17.47** | **2.04** | 26.87 GiB | 6.49 GiB |

The final profile improves prompt evaluation by 208.1% (3.08x), aggregate
throughput by 183%, and decode by 1.6% relative to the equal-split control. It
also exceeds the matched Vulkan result of 1051.67 prompt tok/s by 31.1%.

Two nearby placement probes confirmed that this is not merely a warm-run
effect. With 16 output tokens, `27:37` reached 1380.92 prompt tok/s, while
`26:38` reached 1306.61 and the coarser `7:9` split reached 1288.95.

## Production profile

```text
-dev ROCm1,ROCm0 -sm layer -ts 27,37
```

The GUI now exposes this profile explicitly and selects it automatically only
for Qwen3.6-27B Q4_K_M at `ctx >= 131072`. The balanced `1,1` profile remains
the default for Q3, other models, and contexts up to 98K. Explicit single-GPU
and reverse-order selections remain manual.

For Q4_K_M device sweeps, the autotuner compares the memory-aware and balanced
dual-GPU profiles and omits single-GPU runs, because the 15.25 GiB model tensor
set leaves insufficient headroom for this workload on one 16 GiB card.

## Diagnostics

The legacy HIP pool has opt-in telemetry:

```text
GGML_TRACE_CUDA_POOL=1
```

It reports per-stream and per-device allocated, active, cached, peak, cache-hit,
allocation, and driver-free counts. An experimental cache ceiling is available
on Windows HIP:

```text
GGML_HIP_POOL_CACHE_LIMIT_MB=256
```

The cache ceiling is intentionally disabled by default. In this workload it
freed memory but reduced prompt throughput because `hipFree` synchronizes work.

Artifacts use the `e333-q4km-` prefix under `build_logs/agent-workload`.
The allocator follow-up is
[E334](E334_rocm_quantized_kv_scratch_reservation.md).
