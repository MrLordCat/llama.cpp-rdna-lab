# E337: bounded ROCm Q8 FlashAttention WMMA

Date: 2026-07-15

## Goal

Remove the remaining context-sized F16 FlashAttention staging allocation for
Q8 K/V on RDNA4 without replacing the fast WMMA prompt kernel and without
moving work to host or shared memory.

The validation deliberately uses one physical RX 9070 XT. ROCm calls this
card `ROCm0` (PCI `0000:0e:00.0`); Vulkan calls the same card `Vulkan1`.

## Root cause after E334

E334 moved temporary F16 K/V conversion storage from the HIP legacy pool into
the graph allocation. That stopped old allocation sizes accumulating as the
prompt grew, but the active allocation still scaled with configured context.

For Qwen3.6 D=256, four K/V heads, and `ctx=49152`, the old Q8 WMMA route
reserved two full-context F16 copies:

```text
K: 256 * 49152 * 4 * 2 bytes = 96 MiB
V: 256 * 49152 * 4 * 2 bytes = 96 MiB
total context-sized staging  = 192 MiB
```

Together with the 24 MiB FA output, this produced a 216 MiB graph arena.
Vulkan does not need this allocation because it dequantizes K/V while loading
shader tiles.

## Rejected probes

The following routes removed some or all full-context staging but were not
acceptable production defaults:

| Probe | Representative prompt result | Decision |
| --- | ---: | --- |
| Existing direct Q8 VEC/TILE route | about 730 TPS short | Rejected, about 38% slower |
| Direct K in WMMA | 816.89 TPS long | Rejected, about 22% slower |
| Direct V in WMMA | 947.17 TPS long | Kept only for small-Q decode inside the long-context policy |
| 64-row direct-V variants | 1081-1126 TPS short | Rejected, still slower and not a complete bounded route |
| Q8 block dequant micro-optimization | 1107 TPS short | Rejected, no sufficient recovery |

The useful conclusion was that direct dequantization inside every WMMA tile
repeats too much work during prompt evaluation. K/V must be converted once per
bounded context chunk and then reused by the existing fast WMMA kernel.

## Implementation

The RDNA4 Q8 path now has a bounded chunked-WMMA specialization:

1. Split visible K/V into 4096-token chunks.
2. Convert one Q8 K chunk and one Q8 V chunk to fixed-size F16 scratch.
3. Run the existing fast F16 WMMA kernel for that chunk.
4. Merge each chunk's normalized output online using its softmax maximum and
   row sum. The merge uses the stable rescaling formula and does not retain the
   attention matrix or all chunk outputs.

At `ubatch=1024`, the maximum extra graph allocation is 40.38 MiB:

- 16 MiB for bounded K/V F16 scratch;
- 24 MiB for one partial output;
- about 0.38 MiB for partial and accumulated softmax metadata.

The total FA allocation is therefore 64.38 MiB instead of 216 MiB, and it no
longer grows with context length.

Small-Q decode under the same long-context policy uses direct Q8 V loads while
staging K. This avoids reintroducing the full V copy for each decode graph.

## Stable automatic policy

The reservation graph exposes configured KV capacity before prompt execution.
When it observes at least 16,384 KV positions, the process latches the bounded
route for its lifetime (server contexts use the same configured capacity).
This latch is important: selecting a route from the
currently filled KV length switched graph topology at 1K, 16K, and later
prompt chunks. The HIP allocator retained those graph-resize histories and a
first automatic prototype ended at 0 MiB free and 2215 MiB unaccounted.

After latching, the same request contained only `wmma_f16_chunked_q8` prompt
graphs. It made the initial six backend allocations and no prompt-time graph
reallocation sequence.

The policy is limited to HIP RDNA4, D=256, and Q8 K plus Q8 V. Other devices,
head sizes, and KV types keep their existing path. The diagnostic override is:

- unset: automatic at configured KV capacity >= 16,384;
- `GGML_ROCM_FATTN_Q8_CHUNKED_WMMA=0`: disable;
- any other value: force for eligible tensors.

## One-GPU ROCm result

Common lane:

- model: `Qwen3.6-27B-Q3_K_S.gguf`;
- context: 49,152;
- actual prompt/output: 29,561/16 tokens;
- `b8192/ub1024`, q8_0 K/V, FlashAttention, no MTP;
- no warmup, prompt reuse, cache RAM, or context checkpoints;
- one physical card: `ROCm0 -sm none`.

| ROCm route | Prompt TPS | Decode TPS | Self | Unaccounted | Free |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old full-context F16 staging | 1044.47 | 31.07 | 13213 MiB | 1282 MiB | 933 MiB |
| Bounded chunked WMMA | 1045.61 | 31.31 | 13213 MiB | 1066 MiB | 1149 MiB |

The bounded route recovers exactly 216 MiB while prompt and decode throughput
remain within normal run variance (`+0.11%` and `+0.77%` in this matched pair).

The 12K automatic control stayed on standard `wmma_f16` and measured 1133.48
prompt TPS and 35.16 decode TPS. The policy therefore does not impose the
chunked route on the normal short lane.

## Same-card Vulkan comparison

The saved Vulkan control uses the same physical card, model, prompt, context,
batch sizes, KV types, and 16-token output.

| Backend | Prompt TPS | Decode TPS | Self | Unaccounted | Backend-reported total |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROCm bounded chunked WMMA | 1045.61 | 31.31 | 13213 MiB | 1066 MiB | 15428 MiB |
| Vulkan | 893.30 | 36.39 | 13222 MiB | 1003 MiB | 16304 MiB |

ROCm is 17.0% faster for this long prompt and remains 14.0% slower in decode.
The backend-reported total budgets differ, so absolute `free` values are not
cross-backend comparable. The useful comparison is that ROCm's backend-only
unaccounted difference is now 63 MiB instead of 279 MiB.

## Artifacts

- old ROCm: `e337-rocm0-q3ks-49k-auto-r1`
- final ROCm: `e337-rocm0-q3ks-49k-auto-production-r2-latched`
- short ROCm control: `e337-rocm0-q3ks-12k-auto-production-r2-latched`
- Vulkan control: `e337-vulkan1-q3ks-49k-none-r2`
- rejected probes and intermediate automatic-policy runs use the `e337-`
  prefix under `build_logs/agent-workload`

## Verification

- ROCm `llama-server` build: pass
- automatic 49K/30K one-card request: pass
- automatic 12K/6K one-card control: pass
- bounded allocation and route traces: pass
- graceful server cleanup: pass
- no hard kill and no direct `hipMemGetInfo` query were used

## Dual-GPU follow-up

E338 separates the remaining Windows Shared commitment from the fixed KV and
bounded FlashAttention storage. The dominant remainder was four copies of the
pipeline scheduler graph. Reducing the ROCm single-request default to one copy
saved up to 1.8 GiB dedicated and 2.3 GiB Shared in the Q4 98K lane without a
prompt-throughput regression. See
[E338: ROCm dual-GPU long-context scheduler residency](E338_rocm_dual_long_context_scheduler_residency.md).
