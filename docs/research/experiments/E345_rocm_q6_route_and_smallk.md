# E345: ROCm RDNA4 Q6_K route and MMVQ small-k policy

Date: 2026-07-16

## Goal

Determine whether Q6_K is an active optimization target for Qwen3.6 Q4_K_M,
separate its prompt and decode routes, and improve the hot decode path without
regressing prompt evaluation. All runtime measurements used only the secondary,
non-display GPU through `-dev ROCm0 -sm none`.

## Why Q6 matters

The local Qwen3.6-27B-Q4_K_M tensor inventory contains 294 Q4_K, 48 Q5_K, and
67 Q6_K tensors. Standard mixed Q4_K_M quantization assigns Q6_K to selected
attention-value and FFN-down weights. The earlier Q4_K_S proxy has only one
Q6_K tensor, so its 0.23% Q6 trace share was not representative of Q4_K_M.

`Qwen3.5-9B-Q6_K.gguf` was used as a resident one-GPU route proxy. It occupies
about 6.95 GiB on disk and avoids WDDM spill, making Q6 kernel comparisons
possible without involving the primary GPU.

## Locked lane

- Backend/build: `build-rocm-full/bin/llama-server.exe`
- Device: `ROCm0` only, `-sm none`
- Model: `Qwen3.5-9B-Q6_K.gguf`
- Context: 4096; actual prompt: 1825 tokens; output: 32 tokens
- Batch/ubatch: 4096/1024
- KV: q4_0/q4_0; FlashAttention on; speculative decoding off
- `--cache-ram 0 --ctx-checkpoints 0 -fit off`; no prompt reuse
- Three sequential runs for clean A/B points

## Route split

The route trace counted 173 Q6_K operations at N=1024 and another 173 at N=801
on `cublas_backend`. Decode used `mul_mat_vec_q_direct`: 173 operations at N=2
and 150 at N=1. No Q6_K MMQ call appeared in the normal prompt/decode lane.

This rules out applying the E344 Q4/Q5 MMQ geometry to normal Q6 prefill. A
forced-MMQ control confirmed the selector is correct:

| Prompt route | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: |
| normal hipBLAS, r3 mean | 3219.19 | 65.93 |
| normal hipBLAS, warm r2/r3 mean | 3785.75 | - |
| forced MMQ, r3 mean | 2045.80 | 66.13 |

Forcing Q6 prompt N=801/1024 onto current MMQ removes the warm hipBLAS benefit
and is about 45.9% slower than the warm control. Keep the existing Q6 selector.

## Decode trace and candidate

The pre-change RDNA4 Q6_K decode policy inherited Qwen-hot `small_k=true` with
`nwarps=8`. This batches rows inside each block. Resource tracing showed the
fused FFN Q6 shapes using 84 registers and 14336 bytes of shared memory with
only 50% measured occupancy, while the direct shape used 94 registers and 7168
bytes of shared memory.

An environment A/B first showed that disabling small-k improved decode. The
source change then made `small_k=false` the RDNA4 Q6_K default while retaining
the existing Q3_K/Q4_K policies. The force and disable environment overrides
remain available.

## Same-build A/B

| Metric | Forced old Q6 `small_k=1` | New Q6 `small_k=0` | Delta |
| --- | ---: | ---: | ---: |
| Prompt tok/s, r3 mean | 3268.21 | 3263.85 | -0.13% |
| Decode tok/s, r3 mean | 66.74 | 68.51 | **+2.65%** |
| Aggregate completion TPS | 29.04 | 29.08 | +0.14% |

The 4.36 tok/s prompt difference is noise-scale and expected because prompt
uses hipBLAS rather than this MMVQ branch. Aggregate movement is small because
the lane contains a long prompt and only 32 generated tokens; decode itself is
the relevant metric.

A Q6 `nwarps=4` follow-up was neutral/slightly lower (`68.45` decode versus
`68.51`, prompt `3259.15` versus `3263.85`) and was reverted. Q6 therefore keeps
`nwarps=8` and changes only row batching.

## Decision

Keep the RDNA4 Q6_K `small_k=false` default. It is a narrow, same-build,
repeatable decode improvement with no meaningful prompt tax. Keep Q3_K/Q4_K
small-k behavior unchanged, keep Q6 prompt on hipBLAS, and retain
`GGML_MMVQ_QWEN_FORCE_SMALL_K=1` as the rollback control.

## Production Q4_K_M validation

After dual-GPU testing became available, the Q6 policy was compared on the
actual Q4_K_M file with the same binary. Only
`GGML_MMVQ_QWEN_FORCE_SMALL_K=1` changed between old and new Q6 behavior.

| Lane | Metric | Forced old Q6 small-k | New Q6 one-row default | Delta |
| --- | --- | ---: | ---: | ---: |
| 12K ctx, 6,393 prompt, 256 output | Prompt tok/s | 1800.22 | 1798.03 | -0.12% |
|  | Decode tok/s | 22.66 | 24.00 | **+5.91%** |
|  | Aggregate TPS | 17.1514 | 17.9169 | **+4.46%** |
| 49K ctx, 29,561 prompt, 128 output | Prompt tok/s | 1777.285 | 1778.590 | +0.07% |
|  | Decode tok/s | 21.045 | 21.975 | **+4.42%** |
|  | Aggregate TPS | 5.6159 | 5.6829 | **+1.19%** |

The whole-model gain is larger than the pure-Q6 proxy percentage on both
decode lanes, while prompt remains neutral. This is the decisive production
validation for keeping the default.

The final MTP n3 profile also remains healthy. At 49K it measures 1731.71
prompt tok/s, 39.575 decode tok/s, 6.2802 aggregate TPS, and 74.36% acceptance.
Relative to the new baseline this is a 2.64% prompt cost, 80.11% decode gain,
and 10.51% aggregate gain. The matched 12K MTP result is 1671.69/47.1733
prompt/decode tok/s, 27.4421 aggregate TPS, and 79.06% acceptance.

## Rejected Q5 follow-ups

The production trace also exposed 48 Q5_K tensors. Q5_K N=1 uses `nwarps=8`,
one row per block, 29 registers, and 100% reported occupancy. Two isolated
follow-ups were negative and were fully removed from runtime code:

- forcing eight rows per block reduced decode `24.34 -> 23.64 tok/s` (-2.88%);
- reducing Q5_K to `nwarps=4` reduced decode `24.34 -> 23.98 tok/s` (-1.48%).

Keep Q5_K at `small_k=false,nwarps=8` on this RDNA4 lane.

## Artifacts

- Control: `e345-rocm0-q6k-4k-ub1024-prompt-control-r3.*`
- Route trace: `e346-rocm0-q6k-4k-ub1024-route-mmq-r1.*`
- Forced MMQ: `e347-rocm0-q6k-4k-ub1024-force-mmq-r3.*`
- MMVQ resources: `e348-rocm0-q6k-decode-mmvq-trace-r1.*`
- Environment screen: `e349-rocm0-q6k-4k-ub1024-*.{csv,server.log}`
- Same-build source A/B: `e350-rocm0-q6k-*.{csv,server.log}`
- Rejected nwarps=4: `e351-rocm0-q6k-nosmallk-nwarps4-r3.*`
- Production Q4_K_M Q6 A/B: `e352-q4km-rocm-dual-short-q6-*.{csv,server.log}`,
  `e353-q4km-rocm-dual-long30k-q6-*.{csv,server.log}`
- Q5 resource trace and rejected probes: `e354-q4km-rocm-dual-q5-*`,
  `e355-q4km-rocm-dual-short-q5-*`, `e356-q4km-rocm-dual-short-q5-*`
- Current MTP: `e357-q4km-rocm-dual-*.{csv,server.log}`
