# E344: ROCm RDNA4 Q4_K/Q5_K type-specific MMQ geometry

Date: 2026-07-16

## Goal

Improve the active ROCm Q4 prompt path without changing the established Q3_K
geometry. All runtime measurements in this experiment used only the secondary,
non-display GPU through `-dev ROCm0 -sm none`.

## Residency gate

`Qwen3.6-27B-Q4_K_M.gguf` does not fit resident on one 16 GiB RX 9070 XT. At
`ctx=4096` its server process reached about 15.99 GiB dedicated plus 1.04 GiB
shared GPU memory, and the spill-bound lane produced only 67.71 prompt tok/s.

The kernel experiment therefore used `Qwen3.6-27B-Q4_K_S.gguf` as a resident
Q4/Q5-path proxy. It reached about 15.50 GiB dedicated and 0.21 GiB shared GPU
memory while retaining healthy throughput. This isolates kernel behavior, but
it is not a final dual-GPU Q4_K_M performance claim.

## Locked lane

- Backend/build: `build-rocm-full/bin/llama-server.exe`
- Device: `ROCm0` only, `-sm none`
- Model: `Qwen3.6-27B-Q4_K_S.gguf`
- Context: 4096; actual prompt: 1930 tokens; output: 32 tokens
- Batch/ubatch: 4096/1024
- KV: q4_0/q4_0; FlashAttention on; speculative decoding off
- `--cache-ram 0 --ctx-checkpoints 0 -fit off`; no prompt reuse
- Three sequential runs per clean A/B point

## Hot-path evidence

The control full-kernel trace showed that MMQ consumed 1155.728 ms, or 38.94%
of traced wall time. Q4_K accounted for 1020.730 ms (88.32% of MMQ), Q5_K for
134.998 ms (11.68%), and Q6_K for only 0.23% of steady `MUL_MAT` time in this
Q4_K_S proxy. The hot Q4_K/Q5_K shapes use 5120-17408 rows and N=906 or N=1024,
so the existing RDNA4 Q4/Q5 MMQ selector is the correct route to tune.

That Q6 percentage must not be generalized to the production Q4_K_M file. The
local Q4_K_M tensor inventory contains 67 Q6_K tensors (versus one in Q4_K_S),
including selected attention-value and FFN-down weights. E345 therefore studies
the Q6 route separately with a resident pure-Q6 proxy.

The previous common RDNA4 geometry was `mmq_x=128`, `mmq_y=64`, `nwarps=4`.
Forcing narrower `mmq_x` values did not help:

| Forced `mmq_x` | Prompt tok/s | Decode tok/s |
| ---: | ---: | ---: |
| control 128 | 1151.42 | 28.70 |
| 64 | 1073.79 | 28.52 |
| 80 | 1113.47 | 28.73 |
| 96 | 1084.70 | 28.40 |
| 112 | 1101.54 | 28.68 |

Low nominal occupancy therefore was not evidence that a narrower X tile would
be faster. The positive route was to retain X=128 and double only the Q4_K and
Q5_K Y dimension and wavefront count.

## Implementation

`hip-source-bundles.cmake` defines `GGML_MMQ_RDNA4_Q4Q5_Y128_W8=1` only while
compiling `mmq-instance-q4_k.cu` and `mmq-instance-q5_k.cu`. Under that source
gate, `mmq.cuh` selects `mmq_y=128,nwarps=8` on RDNA4. All other MMQ template
instances retain `mmq_y=64,nwarps=4`.

`GGML_MMQ_RDNA4_Q4Q5_FORCE_MMQ_X=<8..128, multiple of 8>` remains available
as a diagnostic override; the measured sub-128 points are rejected defaults.

## Clean A/B result

| Metric | Q4/Q5 y64/w4 control | Q4/Q5 y128/w8 | Delta |
| --- | ---: | ---: | ---: |
| Prompt tok/s, r3 mean | 1126.11 | 1178.91 | **+4.69%** |
| Decode tok/s, r3 mean | 28.7167 | 29.0533 | +1.17% |
| Aggregate completion TPS | 11.1575 | 11.5169 | **+3.22%** |
| Prompt time | 1713.89 ms | 1637.13 ms | -4.48% |
| Decode time | 1114.41 ms | 1101.46 ms | -1.16% |
| Total measured time | 2828.30 ms | 2738.59 ms | **-3.17%** |

The candidate full trace provides causal support:

| Traced component | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| CUDA node total | 2746.223 ms | 2619.018 ms | -4.63% |
| `MUL_MAT` forward | 1799.487 ms | 1718.031 ms | -4.53% |
| MMQ total | 1155.728 ms | 1127.807 ms | -2.42% |
| Q4_K MMQ | 1020.730 ms | 1000.070 ms | -2.02% |
| Q5_K MMQ | 134.998 ms | 127.737 ms | -5.38% |

A Q3_K negative-control trace reports `mmq_y=64,nwarps=4`, confirming that the
source-specific build gate does not alter the Q3 path.

## Decision

Keep the type-specific Q4_K/Q5_K RDNA4 geometry. It gives a repeatable prompt
gain, improves total wall time, and does not regress decode or Q3 geometry.

The production dual-GPU Q4_K_M follow-up used the same 49K lane as the earlier
E335 result. With the current Q4/Q5 geometry and the old Q6 small-k behavior
forced, prompt evaluation measured 1777.285 tok/s. The older pre-E344 lane was
1716.16 tok/s, so the production result is directionally consistent with the
resident proxy gain. This historical comparison is not a compile-isolated A/B;
the strict causal result remains the resident E342/E343 pair above.

The final current profile, including E345's Q6 decode change, measures 1778.59
prompt tok/s and 21.975 decode tok/s on the 29,561-token/128-output lane. MTP n3
measures 1731.71/39.575 prompt/decode tok/s and 6.2802 aggregate TPS.

## Artifacts

- Residency: `e339-rocm0-q4km-4k-ub1024-none-r1-wddm.*`,
  `e339-rocm0-q4ks-4k-ub1024-none-r1-wddm.*`
- Control trace: `e340-rocm0-q4ks-4k-ub1024-kernelfull-r1.*`
- X sweep: `e341-rocm0-q4ks-4k-ub1024-*.{csv,server.log}`
- Clean control: `e342-rocm0-q4ks-4k-ub1024-y64w4-control-r3.*`
- Candidate: `e343-rocm0-q4ks-4k-ub1024-q4q5-wide-r3.*`
- Candidate trace: `e343-rocm0-q4ks-4k-ub1024-q4q5-wide-kernelfull-r1.*`
- Q3 negative control: `e343-rocm0-q3ks-4k-ub1024-geometry-negative-control-r1.*`
- Production Q4_K_M: `e353-q4km-rocm-dual-long30k-*.{csv,server.log}`,
  `e357-q4km-rocm-dual-long30k-mtp3-current-r2.*`
