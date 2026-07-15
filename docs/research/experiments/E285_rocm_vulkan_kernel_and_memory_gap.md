# E285: ROCm/Vulkan Kernel and Memory Gap

Date: 2026-07-14

## Scope

This experiment investigates why the current ROCm backend trails Vulkan on
the same dual-RX-9070-XT Q3_K workload, and why ROCm appears to consume more
VRAM. It uses the current post-E284 code and keeps speculative decoding off.

## Matched Runtime Lane

- model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`;
- context: 12,288;
- actual prompt / output: 7,923 / 32 tokens;
- batch / ubatch: 8,192 / 1,024;
- K/V cache: q8_0 / q8_0;
- cold prompt, no reuse, no prime pass, no warmup;
- dual layer split: equal weights on both RX 9070 XT devices.

Clean and low-overhead diagnostic results:

| Backend / placement | Prompt TPS | Decode TPS |
| --- | ---: | ---: |
| Vulkan dual | 1,732.36 | 36.31 |
| ROCm dual | 1,573.92 | 25.92 |
| ROCm1 single | 1,048.28 | 29.64 |

Vulkan dual is `+10.1%` faster in prompt eval and `+40.1%` faster in decode
than ROCm dual. ROCm dual improves prompt throughput by `+50.1%` over ROCm1
single, but decode drops by `12.6%`. Layer splitting therefore helps the wide
prefill matrices while serial per-token work and the layer boundary make TG
slower.

## Prompt Trace

The full ROCm kernel trace is timing-distorted and is not a wall baseline. Its
steady `MUL_MAT` split still provides a strong route ranking:

| Route | Steady time | Share |
| --- | ---: | ---: |
| Q3_K rocBLAS backend | 4,073.834 ms | 79.44% |
| F32 rocBLAS backend | 509.982 ms | 9.95% |
| Q4_K direct MMQ | 338.964 ms | 6.61% |
| Q3_K direct MMQ | 86.770 ms | 1.69% |
| Q3_K direct MMVQ | 59.161 ms | 1.15% |

Broad Q3_K prompt matrices still use Q3_K-to-F16 staging followed by rocBLAS.
The GEMM-side route, not MMQ or scheduler copy, is the prompt bottleneck.

## Decode Trace

HIP graph-state tracing found stable graph replay after warmup: 27 steady rows
per device used the existing graph without update or recapture. Missing graph
reuse is not the decode cause.

Scheduler split timing on the 159-token/16-output diagnostic lane measured
about 36 ms per token. Cross-device copy accounted for about 2 ms, while GPU
compute/synchronization accounted for about 33 ms. Host-staged PCIe transfer is
only about 5-6% of the token and cannot explain the Vulkan gap.

Same-shape kernel timestamps from the ROCm and Vulkan diagnostic traces put
common Q3_K N=1 ROCm matvec forms at approximately 1.4-3.4x the Vulkan kernel
time. Small F32 and Q3_K forms show an even larger fixed-overhead ratio. The
remaining decode gap is therefore in the kernel body/route, not MTP acceptance,
PCIe copy, or HIP graph recapture.

## Direct Q3_K x F32 Probe

An opt-in HIP prototype bypassed temporary Q8_1 activation quantization and
performed direct Q3_K x F32 dequant/FMA. Both tested geometries generated sane
text but regressed the exact single-GPU control:

| Kernel | Prompt TPS | Decode TPS | Decode delta |
| --- | ---: | ---: | ---: |
| Current Q8_1 + DP4A | 750.40 | 32.78 | baseline |
| Direct F32, 32 threads/row | 748.25 | 18.73 | -42.9% |
| Vulkan-like 128-thread workgroup | 744.72 | 6.02 | -81.6% |

The experiment was rejected and all prototype code was removed. Vulkan's
advantage is not explained by avoiding Q8_1 alone; its shader compiler,
superblock scheduling, scale cache, reduction topology, and command model work
together. Do not repeat a direct-F32 HIP matvec without a new resource model.

## VRAM Accounting Root Cause and Fix

The old Windows WDDM budget path selected a DXGI adapter by display name and
VRAM size. With two identical cards, both HIP device ordinals resolved to the
same DXGI LUID (`0x1185f`), so per-device free/used memory logs were duplicated
and placement decisions could consume the wrong free-memory value.

The runtime now maps:

1. HIP device ordinal to PCI BDF with `hipDeviceGetPCIBusId`;
2. DXGI LUID to physical bus/device/function through D3DKMT;
3. HIP and DXGI adapters by the exact PCI address, with the old name match only
   as a fallback.

Validation after the fix:

| Device | HIP PCI | LUID | Self | Unaccounted |
| --- | --- | --- | ---: | ---: |
| ROCm1 | `0000:0b:00.0` | `0x1185f` | 5,577 MiB | 281 MiB |
| ROCm0 | `0000:0e:00.0` | `0x138c3` | 6,437 MiB | 283 MiB |

This fixes reporting and free-memory planning. It does not remove real ROCm
overhead. The corrected pre-workload smoke reduces the unaccounted amount from
the previously duplicated 1-2 GiB reading to about 0.28 GiB per GPU. After the
full prompt/decode workload, the matched rows were:

| Backend / model share | Self | Model | Context | Compute | Unaccounted |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROCm, 5.38 GiB share | 5,671 MiB | 5,380 MiB | 281 MiB | 9 MiB | 1,220 MiB |
| ROCm, 6.25 GiB share | 6,544 MiB | 6,252 MiB | 275 MiB | 15 MiB | 1,038 MiB |
| Vulkan, 5.43 GiB share | 5,725 MiB | 5,433 MiB | 281 MiB | 9 MiB | 989 MiB |
| Vulkan, 6.21 GiB share | 6,500 MiB | 6,209 MiB | 275 MiB | 15 MiB | 987 MiB |

The llama-owned allocations are effectively matched. The remaining ROCm-only
amount is dynamic runtime/library state, not an extra copy of model or KV data.
rocBLAS owns persistent per-handle temporary device memory and grows it on
demand, so HIP runtime/code objects and rocBLAS workspace can remain outside
llama.cpp's model/context/compute allocation breakdown. Task Manager can
additionally show global WDDM/display use that is not owned by the server
process.

The fast HIP Q3_K storage layout was also isolated. Disabling it reduced the
single-GPU model buffer from `11,633.26` to `11,460.49 MiB` (`172.77 MiB`) but
reduced prompt/decode from `743.72/32.84` to `736.94/32.34`. Vulkan also uses a
padded Q3_K block representation, so this is not the backend memory delta. Keep
the HIP layout enabled.

## Result

- Keep the PCI-exact WDDM adapter mapping.
- Keep the current Q8_1 + DP4A MMVQ implementation.
- Prompt work should target the dominant Q3_K staging + rocBLAS GEMM route.
- The follow-up E289 experiment found the decode cause inside Q3_K packed
  subtraction code generation; graph, PCIe, and direct-F32-only explanations
  remain closed by measurement.

Primary artifacts:

- `e285-rocm-dual-graphstate-12k-none-r1`;
- `e285-rocm-dual-kernelfull-12k-none-r1`;
- `e285-rocm-dual-schedsplit-short-none-r1`;
- `e285-rocm1-single-12k-none-r1`;
- `e285-vulkan-dual-12k-none-r1`;
- `e285-vulkan-dual-perftrace-12k-none-r1`;
- `e285-rocm1-q8dp4a-short-none-r1`;
- `e285-rocm1-q3f32-short-none-r1`;
- `e285-rocm1-q3f32-wg128-short-none-r1`;
- `e285-rocm-dual-pcimapped-wddm-smoke-r1`.
