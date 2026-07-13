# D084 P003 Vulkan Native Tensor All-Reduce

## Status

Kept as an experimental opt-in implementation; rejected as the P003 route to
beat dual-layer prompt evaluation on the current Windows Vulkan driver.

## Implementation

- Added a Vulkan backend communicator so Meta tensor split no longer falls back
  to the generic scheduler reduction.
- Queues both device readbacks/uploads before concurrent waits.
- Added GPU F32-to-BF16 and BF16-to-F32 conversion pipelines for large partials.
- Added AVX2 two-device BF16 reduction on pinned host buffers.
- Extended the Q3 quad route to tensor-split projection shapes.
- `GGML_META_PARTIAL_TRACE=1` records exact partial boundaries.

Opt-ins:

- `GGML_VK_TENSOR_ALLREDUCE=1`
- `GGML_VK_TENSOR_ALLREDUCE_BF16=1`

## Measurements

Same 12k prompt lane, `b=ub=1024`, two RX 9070 XT devices:

| Route | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: |
| Original tensor/generic reduction | about `540` | not promoted |
| Native FP32, 16-token check | `809.03` | `5.74` |
| Native BF16, 16-token check | `1042.67` | `7.22` |
| Native BF16 after upstream submission port | `1031.99` | `7.17` |
| Layer `-ts 5,6`, q8 KV | `1826.47` | `12.39` |

The BF16 response was coherent and began identically to FP32; no NaN/error was
observed. BF16 cuts each large transfer from 20 MiB to 10 MiB. Measured large
collective phases were about `1.2 ms` read, `0.9 ms` AVX2 reduce, and `1.3 ms`
write per boundary.

## Root Cause

The Qwen3.6 graph has the standard two mathematically required partial outputs
per layer: attention/linear-attention output and FFN output. Across 64 layers,
the final graph boundary needs no following reduction, producing 127
all-reduces per ubatch. This is already model-aware column/row TP, not generic
per-matmul splitting.

The Windows driver exposes the two GPUs as two singleton Vulkan device groups,
so a device-group peer collective is unavailable. Host mediation therefore
adds roughly 0.4 seconds per full 127-boundary ubatch even after BF16 compression.
Parallel GEMMs cannot recover that serialized cost.

## Negative Gate

Direct compute access to coherent system memory was tested to remove DMA
staging. It measured about `4.5 ms` for the read phase versus about `1.2 ms`
with device-local compression plus DMA, and produced `1040.21 tok/s`, a tie.
That branch was removed.

## Decision

Keep the native BF16 communicator as opt-in infrastructure and correctness
evidence. Do not make tensor split the default on this driver. A target-closing
implementation requires a true cross-device primitive (device-group, external
device-memory interoperability, or a driver-supported peer collective), not
more host-copy tuning.
