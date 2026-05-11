# Runtime Performance Research Notes

This file maps the llama.cpp host/runtime areas that shape the active Qwen3.6 ROCm prompt-heavy lane.

## Key Files

- `src/llama-context.cpp`
  - `llama_context::process_ubatch()` switches between PP and TG schedulers, builds or reuses graphs, sets inputs, and launches compute.
  - `llama_context::decode()` drives memory context preparation, ubatch iteration, logits extraction, backend sampling copies, and output mapping.
  - `llama_context::graph_reserve()` determines reserve graph shapes and therefore compute buffer pressure.
  - `sched_reserve()` probes fused Gated Delta Net support and chooses AR/chunked fused paths.
- `src/llama-batch.cpp`
  - `llama_batch_allocr::split_simple()` creates contiguous ubatches up to `n_ubatch`.
  - `llama_batch_allocr::ubatch_add()` materializes tokens, positions, seq ids, outputs, and output row mapping.
- `src/models/delta-net-base.cpp`
  - `build_delta_net()` routes between autoregressive, chunking, and fused GDN graph variants.
  - `build_delta_net_fused()` emits `GGML_OP_GATED_DELTA_NET` with `keep_intermediates=false` for normal inference.
  - `build_delta_net_chunking()` is the fallback graph path and has a diagnostic `LLAMA_DELTA_NET_CHUNK_SIZE` override.

## Current Shape Facts

- The active lane uses one sequence and a prompt around `8030` tokens.
- `batch=6144`, `ubatch=192` is the current best narrow point.
- `ub194` regresses sharply because the split produces worse tail shapes; trace showed GDN token histogram changing from `{192, 158, 2, 1}` to `{194, 140, 130, 2, 1}`.
- `ubatch > 256` is outside the current search space by user constraint and prior cliff behavior.

## Aggressive Runtime Patch Ideas

### Shape-Aware Split Planner

Goal: keep `ubatch <= 256` but avoid tails that route through slow GDN/FATTN shapes.

Initial experimental controls:

- `LLAMA_UBATCH_SPLIT_POLICY=tail-avoid` keeps the legacy tail-only heuristic.
- `LLAMA_UBATCH_SPLIT_POLICY=shape-score` enables deterministic scoring over candidate split sizes for single-sequence prefill.
- `LLAMA_UBATCH_SHAPE_PREFERRED=<N>` caps the planned single-sequence prefill shape under the requested `-ub` value.
- `LLAMA_UBATCH_SHAPE_MIN_TAIL=<N>` avoids final tails smaller than this value when possible; default is `144`.
- `LLAMA_UBATCH_SHAPE_MIN_STEP=<N>` sets the lower bound for candidate split sizes in `shape-score` mode.
- `LLAMA_UBATCH_SHAPE_CHUNK_HINT=<N>` sets the chunk-tail hint used by `shape-score` penalties (default `96`).
- `LLAMA_UBATCH_SHAPE_MIN_CHUNK_TAIL=<N>` penalizes candidate split sizes with non-zero chunk tails smaller than this value.
- `LLAMA_UBATCH_TRACE=1` logs planned shapes at split time.
- Default behavior is unchanged when the policy env var is not set.

Current intended test:

- Run with `-ub 256` or `-ub 194`, plus `LLAMA_UBATCH_SPLIT_POLICY=shape-score LLAMA_UBATCH_SHAPE_PREFERRED=192`, to decouple scheduler reserve size from actual GDN/FATTN prompt shapes.
- If this only reproduces `ub192`, it is still useful as proof that the cliff is shape-driven rather than a generic `n_ubatch` parameter effect.

Validation:

- Trace `ub192` and the new planner with `GGML_TRACE_GDN_PATH=1 GGML_TRACE_FATTN_SELECTED=1`.
- Compare prompt eval TPS and wall TPS on `v2-review --runs 1`; confirm with 3 runs only if improvement is above the threshold.

### Prefill Graph Reserve Slimming

Goal: reduce PP compute buffer pressure without harming decode.

Possible design:

- Keep the existing TG scheduler for `ubatch.n_tokens == 1`.
- Add a dedicated PP-reserve profile for the active prompt-heavy lane that reserves only the shapes actually used by the split planner.
- Avoid reserving a maximal PP graph when a smaller set of shapes is enough.

Validation:

- Log compute buffer sizes and graph reuse counts.
- Compare `prompt_eval_ms` and `decode_eval_ms` separately.

Runtime timing controls:

- `LLAMA_UBATCH_TIMING=1` logs per-ubatch host-side timings for memory apply, scheduler switch, graph build/allocation, input setting, and compute enqueue.
- `LLAMA_UBATCH_TIMING_SYNC=1` additionally synchronizes after each graph compute to estimate device-side ubatch time. This perturbs performance and should be used only for diagnosis, not for final TPS claims.

Observed on the active lane:

- Graph build/allocation/input setup are sub-millisecond to low-single-millisecond per prompt chunk.
- Synchronized prompt chunks at `n_tokens=192` cost about `232-240 ms`; decode tokens cost about `36 ms` each.
- Scheduler bookkeeping is not large enough for a 10% wall breakthrough by itself.

### Logits and Sampling Path

Goal: remove host overhead only if it is measurable.

Current finding:

- `--backend-sampling` regressed on `ub192`, so raw sampling overhead is not the main bottleneck.

Next useful instrumentation:

- Time `ggml_backend_tensor_get_async()` for logits and any sampler copies.
- Keep this disabled by default to avoid perturbing normal runs.

### MMVQ Decode Observability (P2)

Goal: keep MMVQ dispatch/kernels tunable without returning to heavy single-TU edit loops.

Runtime controls (all optional):

- `GGML_TRACE_MMVQ_PATH=1`
  - Logs MMVQ route decisions (`qwen-hot` vs `rest`) with type/shape fields.
- `GGML_TRACE_MMVQ_SMALL_K=1`
  - Logs small-k decision points for RDNA4 Qwen-hot decode-side calls.
- `GGML_MMVQ_QWEN_FORCE_SMALL_K=1`
  - Forces small-k path for RDNA4 Qwen-hot MMVQ calls (`Q3_K/Q4_K/Q6_K`).
- `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1`
  - Forces non-small-k path for the same RDNA4 Qwen-hot set.

Notes:

- These knobs are experimental and intended for decode-biased A/B lanes.
- Default behavior is unchanged when no MMVQ env var is set.

## Safety Rules

- Do not disable fused GDN chunked prefill; it hung on the active lane.
- Do not count changes below 1%.
- Do not benchmark with prompt cache/checkpoints unless the label explicitly says so. Use `scripts/agent_workload_bench.py --no-reuse` for cold prompt-heavy runs.
- Keep experimental routing behind env flags or local benchmark-only options until validated.
