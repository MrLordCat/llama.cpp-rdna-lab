# W14: GATED_DELTA_NET decode per-token cost audit (C3)

Date: 2026-09-07

Scope: C3 from W13/W12 - why GATED_DELTA_NET shows `13.5%` of the traced decode
token and where the per-token cost actually comes from. Source audit only: no
GPU launches (GPUs reserved by user; measurement plan below, not executed).

## Model facts (Qwen3.8-27B-UD, qwen35)

From the GGUF metadata (read from
`models/Qwen3.8-27B-UD-Q4_K_M.gguf` with a byte scanner - the file has a
non-standard `general.tags` value, so a normal sequential parser desyncs):

| key | value |
|---|---|
| `qwen35.block_count` | 65 |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.ssm.state_size` (S_v) | 128 |
| `qwen35.ssm.time_step_rank` (H_v = num_v_heads) | 48 |
| `qwen35.ssm.group_count` (H_k = num_k_heads) | 16 |
| `qwen35.ssm.inner_size` (d_inner) | 6144 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.attention.head_count` | 24 |
| `qwen35.attention.key/value_length` | 256 |
| `qwen35.embedding_length` | 5120 |

Recurrent layer selection (`src/models/qwen35.cpp:30`):
`recurrent_layer_arr[i] = (i < n_main) && ((i + 1) % 4 != 0)` -> 49 GDN layers
out of 65. W12 observed 48 GDN nodes/token; the delta is one boundary layer
(post-norm / non-attn), so the graph count matches within one.

State per GDN layer: `S_v^2 x H_v` f32 = 128 x 128 x 48 x 4 B = **3.0 MiB**.
Total across all GDN layers = **144 MiB** (matches the live server log
`S (f32): 144.00 MiB` at the 49K lane).

## Kernel geometry (decode, n_tokens == 1, non-KDA, keep_intermediates=false)

`launch_gated_delta_net` (`ggml/src/ggml-cuda/gated_delta_net.cu`):

- grid = `(H_v, n_seqs, ceil(S_v / 4))` = `(48, 1, 32)` = **1536 blocks**
- block = `(min(warp, S_v), 4, 1)` = `(32, 4, 1)` = **128 threads**
- per block: 4 output columns (threadIdx.y), one warp per column
- per thread: `rows_per_lane = 128 / 32 = 4`; holds 4 `s_shard` + 4 `k_reg` +
  4 `q_reg` + partials (~18-24 VGPRs); `__launch_bounds__(128, 2)` -> 2 CTAs/SM
- shared memory: none (pure register recurrence + warp reduce)
- per launch memory: each (h, col) slice reads S_v f32 (512 B); total read
  6144 cols x 512 B = **3.0 MiB**, write the same transposed state at the end
  = **6 MiB per launch per token**
- per launch compute: 6144 cols x 128 rows x ~5 flops = ~3.9 MFLOP
- **49 such launches per decode token** (one per GDN layer, serialized through
  the layer pipeline because each layer's GDN consumes the previous layer's
  output)

## Where the 13.5% comes from

- W12's census is sync-inflated (57.2 ms/token vs ~44 ms production; every
  traced node's med includes a sync round). The GDN med-sum (147.4 ms for 48
  nodes = 3.07 ms/node) is dominated by that per-node sync overhead, not by
  GPU kernel time: C05 on Qwen3.6 measured decode GDN chunks at ~45 µs each
  with ~16 µs-level kernel times.
- State traffic is not the limit: 6 MiB x 49 = ~294 MiB/token full model,
  ~144 MiB per GPU after layer split -> at ~44 ms/token that is ~3.3 GB/s per
  GPU, trivial next to the 3 GiB KV stream (tens of GB/s).
- The real production structure is: ~49 small kernels (1536 blocks each, tiny
  flops), one per GDN layer, on the layer critical path, already overlapped
  with the previous layer's other ops by the graph scheduler (2 CTAs/SM).
  Modeled real cost is roughly `49 x ~45 µs = ~2.2 ms/token` -> about
  **5-8% of a decode token**, a plausible upper bound for the sync-adjusted
  GDN share (below the inflated 13.5%).

## Candidate directions (no code yet, ordered by cheapness/expected value)

1. **Kernel-time verification first (cheapest, no code change)**. The source
   already has env-gated device timing:
   `GGML_TRACE_GDN_TIMING=1` (+ `GGML_TRACE_GDN_TIMING_SYNC_HIP=1` for a
   non-captured run), `GGML_TRACE_GDN_PATH=1`,
   `LLAMA_TRACE_DELTA_NET_CONTRACT=1`. One short 49K decode run with these
   flags separates per-launch device time from dispatch/sync and decides
   whether any GDN candidate is worth touching.
2. **State dtype (f16)**: halves the 6 MiB/layer read-modify-write traffic.
   GDN is not bandwidth-bound, so expected gain is small; would need a
   quality/perplexity gate. Low priority unless the step-1 trace shows GDN is
   memory-latency bound after all.
3. **Launch geometry flattens**: E108 tested `num_warps=1/2` (rejected,
   ~-0.1%), so the 4-warp, 32-col-group geometry is already near its
   optimum. No new geometry probe without a resource argument.
4. **Overlap**: the 49 launches are serialized by layer dependencies; graph
   capture already overlaps them with FFN work. A scheduler/GDN-specific
   overlap claim needs the step-1 trace; do not assume a win.

## Verdict for the track

GDN's `13.5%` traced share is a sync-inflation artifact; the modeled real
share is `5-8%`. Because GDN is launch/latency-bound (not bandwidth-bound) and
any local win needs >30% GDN cut to clear the `>=3%` whole-lane decode gate,
GDN remains a **second-order target**: do not prototype before the step-1
device-trace run. The W12/W13 direction (MMVQ weight-stream first) is
unchanged.

## Measurement plan (when GPUs are free)

1. Same 49K lane (`ROCm1,ROCm0 -sm layer`, f8_e4m3 KV, spec none, 128 out),
   one short run with `GGML_TRACE_GDN_TIMING=1 GGML_TRACE_GDN_TIMING_SYNC_HIP=1
   LLAMA_TRACE_DELTA_NET_CONTRACT=1`; capture a clean `.server.log`.
2. Report per-layer `enqueue_ms`/`sync_ms`/`total_ms`, n_chunks and contract
   (chunked_prefill=0 at decode, chunk_size N/A, fast_exp=0).
3. Only if measured GDN device time is a large share (>10% of a token):
   open a candidate for state dtype or layer-batching; otherwise keep GDN on
   the shelf and continue with MMVQ.
