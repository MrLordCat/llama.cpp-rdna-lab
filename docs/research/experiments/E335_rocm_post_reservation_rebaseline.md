# E335: ROCm post-reservation rebaseline

Date: 2026-07-15

## Goal

Rebaseline the published long-prompt Qwen3.6 Q3_K_S ROCm lanes and the Q4_K_M
ROCm result set after reserving quantized-KV FlashAttention scratch in the
graph. Determine whether bounding the non-VMM HIP pool also removes the Q4
long-context WDDM residency cliff.

## Locked configuration

All production rows use `build-rocm-full` (`b9326-18196cf16`), two RX 9070 XT
GPUs, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, q8_0 K/V, FlashAttention,
`b8192/ub1024`, one slot, full GPU offload, seed 42, no warmup, no prompt reuse,
`--cache-ram 0`, `--ctx-checkpoints 0`, `-fit off`, and direct HIP peer copy
disabled. Long rows use temperature 0.0 and 128 output tokens. No foreground
GPU workload was active.

## Q3_K_S long rebaseline

Both modes use `Qwen3.6-27B-Q3_K_S_mtp.gguf` so only speculative execution
changes between paired rows.

| Context | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 49,152 | none | 29,561 / 128 | 1734.14 | 25.77 | 5.7934 | - |
| 49,152 | MTP n3 | 29,561 / 128 | 1672.05 | 35.42 | 5.9877 | 63.08% |
| 65,536 | none | 41,058 / 128 | 1630.59 | 24.96 | 4.2096 | - |
| 65,536 | MTP n3 | 41,058 / 128 | 1546.88 | 33.92 | 4.2062 | 68.00% |

At 49K, MTP changes prompt/decode/aggregate throughput by
`-3.58% / +37.45% / +3.35%`. At 65K, the changes are
`-5.13% / +35.90% / -0.08%`. The 65K repository snapshot requested 147,456
characters but reached the current 144,287-character safety cap.

Acceptance differs from the older E315 output because the repository snapshot
and generated answer changed. Acceptance is not a backend invariant; compare
it only within an identical prompt and output sample.

## Q4_K_M performance rebaseline

| Context | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 12,288 | none | 6,393 / 256 | 1522.70 | 22.23 | 16.2304 | - |
| 12,288 | MTP n3 | 6,393 / 256 | 1444.35 | 42.12 | 24.2335 | 68.40% |
| 49,152 | none, r2 mean | 29,561 / 128 | 1716.16 | 20.00 | 5.3936 | - |
| 49,152 | MTP n3 | 29,561 / 128 | 1604.76 | 38.10 | 5.8580 | 77.19% |
| 98,304 | none | 59,004 / 128 | 553.50 | 17.64 | 1.1227 | - |

Q4 MTP remains useful at 49K: prompt evaluation falls 6.49%, decode rises
90.5%, and aggregate throughput rises 8.52%. The 98K row is not healthy and
must not be presented as a production result.

## Current WDDM peaks

These counters belong to the benchmark `llama-server.exe` process. Device
columns retain the monitor's display/secondary ordering. The 49K baseline peak
covers both cold requests in the r2 run.

| Lane | Dedicated per device | Dedicated total | Shared per device | Shared total | Private peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 none, 12K | 8.868 / 9.535 GiB | 18.403 GiB | 0.376 / 0.375 GiB | 0.751 GiB | 17.922 GiB |
| Q4 MTP n3, 12K | 9.109 / 10.866 GiB | 19.975 GiB | 0.378 / 0.590 GiB | 0.968 GiB | 18.893 GiB |
| Q4 none, 49K r2 | 10.437 / 10.771 GiB | 21.208 GiB | 0.974 / 1.624 GiB | 2.598 GiB | 19.119 GiB |
| Q4 MTP n3, 49K | 10.680 / 11.722 GiB | 22.402 GiB | 0.973 / 1.767 GiB | 2.740 GiB | 20.963 GiB |
| Q4 none, 98K | 11.949 / 12.498 GiB | 24.447 GiB | 2.538 / 3.708 GiB | 6.246 GiB | 21.298 GiB |

A separate 98K diagnostic used only 192,000 injected characters and produced
52,388 prompt tokens. It still reached 564.80 prompt TPS with 6.212 GiB Shared.
The comparable maximum-safe-fill run produced 59,004 tokens and 553.50 prompt
TPS. This isolates the cliff to the allocated 98K working set and WDDM budget
pressure rather than the last 6,616 prompt tokens.

## Conclusion

The E334 allocator change is valid but was incomplete by itself. It prevents
old converted K/V scratch sizes from accumulating as a graph grows, which was
verified on a one-GPU Q3 lane. E337 subsequently replaced the context-sized
active F16 staging with bounded 4096-token chunk scratch for the RDNA4 Q8 K/V
WMMA path. That one-card result recovered 216 MiB without a measurable
throughput regression.

The measurements in this document predate E337 and remain historical. Q4_K_M
must be rebaselined before drawing a new 49K or 98K residency conclusion: the
new route removes FA staging growth, but it does not reduce model tensors, KV
cache, recurrent state, general compute buffers, or host-staged split
allocations. See
[E337: bounded ROCm Q8 FlashAttention WMMA](E337_rocm_q8_chunked_wmma.md).

The 131K ROCm rows were not repeated after the 98K cliff was confirmed. A new
dual-GPU Q4 rebaseline is the next step; changing tensor split alone remains a
placement mitigation rather than a backend memory fix.

## Artifacts

- Q3 labels: `e335-rocm-q3ks-*`
- Q4 labels: `e335-rocm-q4km-*`
- WDDM captures: matching `*.memory.json` files under
  `build_logs/agent-workload`
- The attempted stock refresh ended after two generated tokens and is excluded
  from all decode and aggregate comparisons.
