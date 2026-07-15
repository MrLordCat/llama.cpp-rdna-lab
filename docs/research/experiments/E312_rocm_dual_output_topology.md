# E312: ROCm dual-GPU output topology correction

Date: 2026-07-14

## Root cause

For `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, forcing
`LLAMA_OUTPUT_DEVICE=ROCm1` puts the output tensors back on the first device.
The repeating layers still flow from ROCm1 to ROCm0, so prompt and decode gain
a second device boundary: `ROCm1 -> ROCm0 -> ROCm1`.

The normal model placement uses `get_layer_buft_list(n_layer)` for output, so
without the override it remains on ROCm0 after the last repeating layer. This
keeps the graph monotonic and requires only one cross-device boundary.

## 30K result

All runs used both GPUs, q8 KV, ctx 49,152, batch/ubatch 8192/1024, and a
30,097-token repository prompt.

| Topology | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: |
| Forced output on ROCm1 | 1024.06 | 26.48 |
| Default output on last device, ROCm0 | 1770.25 | 26.72 |

Removing the override improved prompt evaluation by 72.9%. The bad run looked
similar to a single-GPU spill but was actually a dual-GPU graph with an
avoidable return transfer.

At equal split with correct output placement, model buffers were 5380.55 MiB
on ROCm1 and 6252.72 MiB on ROCm0; KV was 816 MiB per GPU. No shared-RAM spill
was observed on this lane.

## Split and order sweep

- `ROCm1,ROCm0`, `17,16` moved one layer to ROCm1 and balanced reported free
  VRAM, but repeated decode results did not beat the drifting equal control;
  prompt stayed about 2% lower.
- `9,8` and `6,5` lost 4-7% prompt throughput and were rejected.
- `ROCm0,ROCm1`, `1,1` reached 1811.25 prompt tok/s but only 25.12 decode
  tok/s. Its `16,17` variant reached 1848.45 prompt and 24.54 decode tok/s.

## Decision

Use `-dev ROCm1,ROCm0 -sm layer -ts 1,1` for the production ROCm profile and
do not set `LLAMA_OUTPUT_DEVICE`. Long performance claims must use both GPUs;
single-GPU runs are limited to kernel isolation because the long lane can spill
into system RAM.

Primary artifacts use `e312-rocm-dual-*` prefixes.
