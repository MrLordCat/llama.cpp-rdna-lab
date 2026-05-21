# Vulkan CPU 0-Offload Route Map

Updated: 2026-05-21.

This document maps the local Vulkan build when it is launched with
`-ngl 0` / `--gpu-layers 0`. It is a CPU fallback route, but not a pure CPU
route unless op-offload is disabled.

## Lane

| Field | Value |
| --- | --- |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Build | `build-vulkan/bin/llama-server.exe` |
| CPU | AMD Ryzen 7 5800X3D, AVX2/F16C/FMA/BMI2/OpenMP |
| GPU backend | Vulkan on AMD Radeon RX 9070 XT |
| Context | `ctx=4096` for quick CPU experiments |
| Batch | `batch=512`, `ubatch=128` |
| KV | `q4_0/q4_0` |
| Attention | FlashAttention on; required for q4 V cache |
| Speculation | `--spec-type none` |
| Reuse | disabled for route claims |
| Output | short real-server `max_tokens=32` by default |

The short lane is intentional. Full Qwen CPU fallback is slow enough that every
code edit should first pass a small real-server gate before any longer run.

## Launch Route

1. `llama-server.exe` receives `--gpu-layers 0`.
2. Model tensors load to the CPU model buffer. With mmap enabled the log shows
   `CPU_Mapped`; with `--no-mmap` it uses a resident CPU buffer.
3. Context creation enables q4 KV and FlashAttention.
4. The scheduler still creates Vulkan compute buffers unless
   `--no-op-offload` is passed.
5. Prompt graph pieces can execute through Vulkan op-offload even though no
   transformer layers are offloaded.
6. Decode remains dominated by CPU matvec work against Q3_K weights.

Important negative control:

- `--no-op-offload` is the way to inspect a more CPU-only graph, but it is much
  slower for prompt. It is not the practical fallback route.

## Runtime Evidence

E125 baseline log, `-ngl 0` with op-offload enabled:

| Log signal | Value |
| --- | --- |
| Layer offload | `offloaded 0/65 layers to GPU` |
| Model buffer | `CPU_Mapped model buffer size = 11775.72 MiB` |
| Context | `n_ctx=4096`, `n_batch=512`, `n_ubatch=128` |
| KV | q4 K/V, FlashAttention enabled |
| Vulkan compute buffer | `83.69 MiB` |
| Vulkan_Host compute buffer | `9.89 MiB` |
| Graph splits | `1023 (bs=128), 97 (bs=1)` |

With `--no-op-offload`, graph splits collapse to `1`, but prompt eval drops to
about `6 tok/s` on the same short server lane.

## Metrics

| Route | Aggregate TPS | Prompt eval | Decode eval | Decision |
| --- | ---: | ---: | ---: | --- |
| `-ngl 0`, mmap, op-offload on | `1.7703` | `32.5033 tok/s` | `2.3267 tok/s` | baseline |
| `-ngl 0 --threads 6 --threads-batch 6` | `1.7995` | `30.6367 tok/s` | `2.4267 tok/s` | small/tie |
| `-ngl 0 --no-op-offload` | `0.8900` | about `6.18 tok/s` | about `2.47 tok/s` | reject |
| `-ngl 0` f16/f16 KV | `1.7617` | `27.69 tok/s` | `2.45 tok/s` | reject |
| `-ngl 0 --mlock` | `1.7196` | `27.81 tok/s` | `2.36 tok/s` | reject |
| `-ngl 0 --no-mmap` | `1.8815` | `33.9133 tok/s` | `2.4900 tok/s` | keep |
| `-ngl 0 --no-mmap --threads 6 --threads-batch 6` | `1.8931` | `31.4133 tok/s` | `2.5767 tok/s` | optional |

Partial offload, all with `--no-mmap`:

| GPU layers | Aggregate TPS | Read |
| ---: | ---: | --- |
| 0 | `1.8815` | CPU fallback best r3 |
| 8 | `2.11` | small hybrid gain |
| 16 | `2.32` | still CPU-heavy |
| 32 | `3.46` | useful constrained-VRAM route |
| 48 | `6.03` | strong hybrid route |
| 65 | `28.93` | full Vulkan GPU route |

## CPU Compute Route

Active files:

| Layer | File | Route detail |
| --- | --- | --- |
| Type traits | `ggml/src/ggml-cpu/ggml-cpu.c` | `GGML_TYPE_Q3_K` selects `ggml_vec_dot_q3_K_q8_K`, dot type `GGML_TYPE_Q8_K`, `.nrows = 1` |
| x86 vec-dot | `ggml/src/ggml-cpu/arch/x86/quants.c` | AVX2 `ggml_vec_dot_q3_K_q8_K` implementation; active decode hot path |
| generic fallback | `ggml/src/ggml-cpu/quants.c` | scalar/generic Q3_K route for non-x86 or fallback |
| repack | `ggml/src/ggml-cpu/repack.cpp` | Q3_K is absent from the current x86 repack-supported list |
| ops | `ggml/src/ggml-cpu/ops.cpp` | routes `MUL_MAT`/matvec through type traits and thread slicing |

Current bottleneck interpretation:

- Decode is around `2.3-2.6 tok/s` because every token must run many CPU
  Q3_K matvecs.
- The active Q3_K route is single-row (`nrows = 1`) and not repacked, so it
  repeatedly pays decode/dequant and memory traffic costs.
- OpenMP thread scaling is poor beyond roughly 6-8 threads, consistent with a
  memory/dequant-limited loop rather than a wide compute-saturated kernel.

## FlashAttention and KV Route

- q4 K/V with FlashAttention works and is the practical CPU fallback KV route.
- `--no-flash-attn` with q4 V cache fails at init:
  `V cache quantization requires flash_attn`.
- f16 KV did not improve this lane and costs more memory; keep it as a
  diagnostic only.

## Speculative and Ngram Route

- Keep `--spec-type none` for this lane.
- Earlier CPU fallback ngram r1 did not improve the short real-server gate.
- Vulkan session ngram E124 is also rejected for the nearby Vulkan q4 route
  because effective acceptance was too low.

## Practical Settings

Recommended CPU fallback profile:

```powershell
--gpu-layers 0 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --spec-type none
```

Optional long-decode probe:

```powershell
--gpu-layers 0 --no-mmap --threads 6 --threads-batch 6 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --spec-type none
```

Hybrid constrained-VRAM profile:

```powershell
--gpu-layers 32 --no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --spec-type none
```

The hybrid profile is not a CPU-kernel win; it is a route choice that removes
some Q3_K layers from CPU.

## Next Code Work

High-value candidates:

1. Isolated clean-vs-candidate A/B for the current `quants.c` Q3_K
   mask/shuffle preload change.
2. Q3_K x86 repack design, starting from existing Q4_K/Q5_K/Q6_K interleaved
   repack patterns in `repack.cpp`.
3. Multi-row Q3_K vec-dot route if shapes allow batching several rows and
   amortizing q8/scales work.
4. Per-op timing for CPU `MUL_MAT` in the `-ngl 0 --no-mmap` lane, so changes
   can be attributed to Q3_K rather than mmap/session noise.

Rejected or low-value repeats:

- Disabling op-offload for speed.
- f16 KV as a CPU fallback improvement.
- `--mlock` as a substitute for `--no-mmap`.
- More large thread-count sweeps without a new mechanism.
- Ngram/speculative decoding without measured effective acceptance.
