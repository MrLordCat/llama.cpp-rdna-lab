# Supported Backends, Models and Vision

Detailed reference moved out of the README.

## Backends

| Backend | Role | Status |
| --- | --- | --- |
| ROCm/HIP | Primary prompt-eval, long-context MTP, and RDNA4 runtime | Supported and preferred for prompt-heavy MTP work |
| Vulkan | General AMD runtime and backend comparison | Supported; competitive for decode-heavy work; q8/MTP path fixed (D094) |
| CPU | Fallback, conversion, sanity checks, and tests | Supported |

ROCm still builds HIP-compatible kernels from `ggml/src/ggml-cuda`. That is an
internal HIP implementation detail and does not mean that this fork supports
NVIDIA hardware. See [Supported Backends](SUPPORTED_BACKENDS.md).

## Model and Format Matrix

| Model / feature | CPU | ROCm/HIP | Vulkan | Notes |
| --- | --- | --- | --- | --- |
| Qwen3.8 GGUF (Q4_K_M primary; Qwen3.6 family also supported) | Yes | Yes | Yes | Primary supported family |
| Qwen3.8 NextN MTP | Yes | Yes | Yes | Requires an MTP-enabled GGUF |
| Ternary Bonsai 27B `PQ2_0` | Yes | Yes | Not yet | Native loader, CPU kernels, and HIP MMQ/MMVQ path |
| Qwen3.5/3.6/3.8 vision projector | Yes | Yes | Yes | Use a matching `mmproj-*.gguf` |

D094 (2026-08-07, `Qwen3.6-27B-Q4_K_M.gguf` and
`Qwen3.6-27B-Q3_K_S_mtp.gguf`, 2x RX 9070 XT): the Vulkan q8_0 vec/mmq
numerical divergence vs ROCm was root-caused and fixed (CUDA-style dp4a
accumulation, round-half-away q8_1 quantize, mmq variant-B math). MTP
acceptance recovered from 0.33 to 0.80+ on 52k-token drafts (target 0.53).
Benchmark numbers are archived in [BENCHMARKS.md](../BENCHMARKS.md) and
[Q4_K_M_RESULTS.md](../Q4_K_M_RESULTS.md); Windows re-runs are planned.

Qwen3.8-27B-Q4_K_M is the primary practical Qwen model on this 2x16 GB
machine (rebased 2026-08-14; it shares the qwen35 architecture family with
Qwen3.6 and runs the same MTP/vision paths). The one-copy ROCm scheduler and
bounded Q8 Flash Attention route make its measured 49K and 98K lanes viable.
Q3_K_S (Qwen3.6) remains the secondary choice for maximum
context/VRAM headroom, vision, and Q3-specific kernel research. `PQ2_0` is an
experimental Prism format and should not be confused with conventional `Q2_0`
quantization.

## Vision

Qwen3.6 vision requires a projector that matches the text model architecture
and embedding dimension. In the GUI, enable Vision and select
`models/mmproj-F16.gguf`. The equivalent server argument is:

```text
--mmproj models/mmproj-F16.gguf
```

Use `Spec: None` for the first image request so vision-pipeline issues can be
separated from speculative decoding.
