# D087 P003 Vulkan RDNA4 Compact KV Gate

Date: 2026-07-20

Status: completed/kept. The algebraically equivalent RDNA4-oriented Q5
dequantization path passed correctness, improved the exact long-prompt A/B,
and matched q8_0 on the paired BFCL-lite smoke gate. Use q5_1/q5_1 as an
opt-in compact-KV profile; keep q8_0/q8_0 as the maximum-quality reference.

## Problem

The active P003 Qwen3.6-27B lane uses q8_0/q8_0 KV. A q8_0 block stores 32
signed bytes plus one FP16 scale, so it is 34 bytes for 32 values, or 8.5 bpw.
At `ctx=131072`, 16 attention layers and 1024 K/V elements per layer, the full
cache is therefore 4352 MiB, not the 4096 MiB implied by the name. The 256 MiB
difference is format overhead.

Even an ideal scale-free eight-bit repack can save at most those 256 MiB. D037
needed about 1967 MiB of residency relief for its single-device-heavy profile,
so repacking q8_0 alone cannot close that fit gap. It may improve aligned loads,
but it is not a complete memory solution.

## Current Evidence

Fresh dual-Vulkan `llama-bench` at pp7488, `b8192/ub1024`, `-ts 5,6`:

| KV | Prompt tok/s | Notes |
| --- | ---: | --- |
| q8_0/q8_0 first | 1817.00 | noisy first member of bracket |
| q4_0/q4_0 | 1836.16 | adjacent control |
| q8_0/q8_0 second | 1837.59 | q8 is not generally slower at short KV |
| q5_0/q5_0 | 1800.90 | homogeneous coopmat1 |
| q5_1/q5_1 | 1809.89 | homogeneous coopmat1 |

Fresh server runs on the same 43,081-token prompt and full 131072-cell cache:

| KV | Cache MiB | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| q8_0/q8_0 | 4352 | 1491.39 | 14.19 |
| q5_0/q5_0 | 2816 | 1356.78 | 14.14 |
| q5_1/q5_1 | 3072 | 1383.21 | 14.80 |

Q5 fits entirely on the two GPUs and avoids the D037 host-KV/PCIe collapse, but
the current q5_1 long-prefill cost is 7.25% relative to q8_0.

## Upstream Audit

- Fresh `upstream/master` has the same logical q8_0 AoS payload/scale format and
  no compact Vulkan KV type.
- `upstream/0cc4m/vulkan-repack` separates q8_0 quants and scales for static
  matmul tensors. It preserves the 34-byte payload and does not wire the layout
  into FlashAttention, so it cannot solve KV residency as-is.
- `upstream/gg/kv-compress` is a 2024 experiment predating the current cache and
  Vulkan FA architecture.
- Local TurboKV formats have CUDA/HIP FA support but no Vulkan kernels.

## Candidate A: Q5 Coopmat1 Dequantization

The current `dequantize4` builds the four q5 high bits as four scalar shifts,
converts them to a floating vector, multiplies by 16, and then adds the low
nibbles. The MMQ helper already reconstructs the same four values with one
packed high-bit expression:

`(qh_bits * 0x02040810u) & 0x10101010u`.

Use that expression plus `unpack8` for both K and V. This is algebraically
identical and changes neither the cache bytes nor quantization quality.

Gates:

1. Vulkan build and backend-op correctness pass.
2. q8 route and performance remain unchanged.
3. q5_1 improves the adjacent pp7488 and 43k-prompt controls. Reject and revert
   if the long-prompt center does not improve outside noise.

## Candidate A Results

The shader change reconstructs the four Q5 high bits in one packed integer,
merges them with the low nibbles, and calls `unpack8`. It removes four scalar
shift-to-float conversions and the floating `* 16` vector while preserving the
exact quantized values. Both K and V paths use the same expression.

Correctness and build gates:

- `llama-server`, `llama-bench`, and `test-backend-ops` build successfully;
- filtered `FLASH_ATTN_EXT` covers 58 q5_0 and 56 q5_1 cases;
- result: `Backend Vulkan0: OK`, `3/3 backends passed`, no failed case;
- q8_0 is a separate shader variant and remained unchanged; its adjacent
  long-prompt control/recontrol was `1491.39 -> 1498.00 prompt tok/s`.

Performance on the exact 43,081-token P003 request, `ctx=131072`,
`b8192/ub1024`, `-dev Vulkan1,Vulkan0 -sm layer -ts 5,6`, cold cache and no
warmup/reuse/prime:

| Run | Original q5_1 | Packed-bit q5_1 | Delta |
| --- | ---: | ---: | ---: |
| cold first | 1386.56 | 1454.44 | +4.90% |
| steady 2 | 1359.29 | 1391.58 | +2.38% |
| steady 3 | 1358.22 | 1388.79 | +2.25% |
| r3 mean | 1368.02 | 1411.60 | +3.19% |

The separate candidate r1 was `1448.73 prompt tok/s`, consistent with the r3
cold-first result. Short pp7488 also moved from `1809.89` to `1829.28`
(`+1.07%`). The q8_0 cold-control center is `1494.70`, leaving packed q5_1
about `2.69%` behind q8_0 on comparable cold-first runs instead of the original
`7.25%` gap.

Memory is unchanged by the shader optimization but substantially lower than
q8_0 because q5_1 is 24 bytes per 32 values (6 bpw):

| KV profile | Full-cache MiB | Saving vs q8_0 |
| --- | ---: | ---: |
| q8_0/q8_0 | 4352 | - |
| q5_1/q5_1 | 3072 | 1280 MiB (29.4%) |
| q5_0/q5_0 | 2816 | 1536 MiB (35.3%) |

The allocation model also matches runtime observation at ctx16384. Across both
GPUs, the reported context allocation was 1141 MiB for q8_0 and 981 MiB for
q5_1, an exact 160 MiB delta. Scaling that eightfold to ctx131072 gives the
predicted 1280 MiB saving; the q8 excess is format metadata, not a leak.

Paired deterministic BFCL-lite smoke, same model/topology, seed 42,
temperature 0.001 and eight default cases:

| KV | Result |
| --- | ---: |
| q8_0/q8_0 reference | 8/8 |
| packed q5_1/q5_1 | 8/8 |

This smoke gate covers simple, multiple, parallel and irrelevance tool-call
classes. It does not prove parity on every long-context quality workload, so
q8 remains the conservative quality reference rather than being removed.

## Decision

Keep Candidate A. It is a narrow Vulkan shader-body improvement, bit-exact
relative to the existing Q5 cache representation, and clears the long-lane
performance gate. Recommend q5_1/q5_1 as the RDNA4 compact-KV opt-in when the
1.25 GiB residency saving matters. The paired 8/8 BFCL-lite smoke supports
practical use, but do not silently replace q8_0 for maximum-quality profiles.

## Next Format Gate

Candidate A closes the comparable cold-first q8/q5 gap below 5%, so a new cache
type is not justified yet. Standard Q6_K exists in Vulkan matrix kernels but is
not an accepted KV-cache type and has no Vulkan FlashAttention dequant path;
wiring it in would be a new format project, not an upstream toggle. Reopen a
regular 6-bit KV design only if q5_1 fails the quality gate. Any such design
must first prove quality against q8_0 and a static shader-resource gate before
receiving a new GGML type ID.
