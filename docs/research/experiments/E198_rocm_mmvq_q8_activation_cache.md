# E198 ROCm MMVQ Q8 Activation Cache

## Metadata

- Experiment ID: E198
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E197 rollback
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: ROCm MMVQ loses part of the Vulkan decode advantage because it repeatedly quantizes the same activation tensor to transient q8_1 inside one graph compute, while Vulkan caches `prealloc_y` by tensor/pipeline within the command-buffer build.
- Mechanism: add an env-gated per-context q8_1 activation cache for CUDA/HIP MMVQ. If consecutive or near-consecutive quantized matvec nodes use the same `src1` tensor, strides, and padded shape, reuse the already quantized q8_1 buffer instead of launching another `quantize_q8_1` kernel.
- Why now: E197 rejected pure wave/topology transfer, and the E196 trace shows repeated `src1` names such as `attn_norm-*` across Q3_K/Q4_K matvec routes. This is a larger route-level Vulkan-to-ROCm difference than vec-dot instruction microforms.

## Math / Theory

- Assumptions:
  - E196 steady forward split puts Q3_K at about `59.5%` of measured forward MUL_MAT time when `mul_mat_vec_q_direct` and `mul_mat_q_direct` are considered together;
  - q8_1 activation quantization is not directly separated in the current node timing, so this candidate must be treated as a route-overhead gate rather than a proven Q3_K body win;
  - cache validity is per graph compute, not across generated tokens, because tensor pointers can be reused with new contents.
- Expected speedup corridor:
  - `required_local_speedup.py --share 0.595 --goals 1.02,1.05,1.10,1.278` says `+2%` wall needs `1.034x` local, `+5%` needs `1.087x`, `+10%` needs `1.180x`, and Vulkan parity needs `1.576x`;
  - this route is only worth keeping if clean wall improves, because q8 quantization may be a small share compared with Q3_K dot/dequant work.
- Failure conditions:
  - no cache hits on the active lane;
  - cache hits appear but clean wall is tied/regressed;
  - graph capture correctness fails or real server text becomes corrupt;
  - time simply moves into MMVQ/FA/norm routes with no aggregate TPS gain.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/common.cuh`: add per-context q8_1 cache metadata and reset/free helpers;
   - `ggml/src/ggml-cuda/ggml-cuda.cu`: reset cache at graph-compute start and reuse it in the generic `mul_mat` q8_1 quantization path;
   - `ggml/src/ggml-cuda/mmvq-dispatch.cu`: reuse it in the fused `ggml_cuda_mul_mat_vec_q` path.
2. Guard rails:
   - env gate `GGML_CUDA_MMVQ_Q8_CACHE=1`;
   - optional trace `GGML_TRACE_MMVQ_Q8_CACHE=1`;
   - key includes tensor pointer, data pointer, q8 buffer size, padded shape, and strides;
   - reset cache metadata at each graph compute.
3. Rollback path:
   - revert the small helper and call-site hunks; rebuild ROCm.

## Benchmark Plan

- Baseline command: E196 clean ROCm r3 reference `31.9233 TPS`, decode `32.3833 tok/s`.
- Candidate resource command: short real-server run with `GGML_CUDA_MMVQ_Q8_CACHE=1`, `GGML_TRACE_MMVQ_Q8_CACHE=1`, `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_RESOURCES=1`, `max_tokens=16`.
- Candidate clean command: real-server run with only `GGML_CUDA_MMVQ_Q8_CACHE=1`, `runs=1`, `max_tokens=512`; r3 only if r1 is promising.
- Number of runs: resource `r1`, clean `r1` first.
- Artifacts path: `build_logs/agent-workload/e198-rocm-mmvq-q8-cache-*`.

## Metrics

- aggregate completion TPS (wall)
- decode/prompt split
- error rate
- q8 cache hit/miss counts
- Q3_K MMVQ timing/resource buckets
- next-bottleneck split if candidate lowers one measured route without wall gain

## Result

- Outcome: rejected and reverted.
- Delta:
  - same-binary clean baseline r1: `31.9368 TPS`, decode mean `32.50 tok/s`, prompt mean `634.43 tok/s`, errors `0`;
  - candidate r1 with `GGML_CUDA_MMVQ_Q8_CACHE=1`: `31.8573 TPS`, decode mean `32.405 tok/s`, prompt mean `632.375 tok/s`, errors `0`;
  - wall delta `-0.0795 TPS` (`-0.25%`), decode eval delta `-0.095 tok/s` (`-0.29%`).
- Trace signal:
  - short real-server trace with cache and route logging produced `303` cache hits and `777` misses (`28.1%` hit rate);
  - examples confirm the intended mechanism: `attn_norm-*` and `attn_post_norm-*` hit on repeated Q3_K MMVQ users within the same graph compute;
  - the logged q8 activation buffers are small on this lane, commonly `11520`, `13824`, or `39168` bytes.
- Confidence: medium. The route activation was real, the build passed, and the clean A/B was same-binary, but only r1 was needed because the first clean result was non-positive and below the keep threshold.
- Recommendation: do not keep or repeat standalone MMVQ q8 activation caching. Revisit only if a later trace proves `quantize_row_q8_1_cuda` itself is a large synchronized node-time share, or if caching is part of a broader graph-scheduling/layout branch that also reduces Q3_K dot/dequant work.

## Notes

- Surprise: the Vulkan-like reuse premise was true, but the expected payoff was too small. In the active graph path, removing repeated q8 activation quantization does not remove a dominant cost; the real Q3_K dot/dequant body remains the limiter.
- Why the prediction failed: the initial hypothesis over-weighted host/kernel-launch savings. HIP graph capture already amortizes launch overhead, and the cache hit buffers are tiny compared with the Q3_K weight traffic and vec-dot work. The cache also adds key checks and persistent-buffer plumbing, which can consume the small saving.
- Follow-up action: move back to structural Q3_K route-body work. Avoid q8 activation-cache-only variants unless a fresh synchronized trace shows quantization, not Q3_K dequant/dot, became the bottleneck.
