# Qwen TPS Route Atlas

Updated: 2026-05-22.

This is the top-level route map for future TPS work on the local
`llama.cpp-with-GUI` fork. It connects GUI/CLI/server parameters, Qwen graph
construction, KV cache behavior, speculative decoding, ROCm/Vulkan backend
dispatch, and the measured bottlenecks.

Detailed companions:

- ROCm backend routes: `docs/research/ROCM_ROUTE_MAP.md`
- Vulkan backend routes: `docs/research/VULKAN_ROUTE_MAP.md`
- Vulkan `-ngl 0` CPU fallback route: `docs/research/VULKAN_CPU_0OFFLOAD_ROUTE_MAP.md`
- Metrics and bottleneck table: `docs/research/ROUTE_METRICS_AND_GAPS.md`

## Coverage Contract

This document is complete for the active local Qwen TPS planning surface:

- launch/config route;
- server request and slot route;
- batch/ubatch and KV memory route;
- Qwen graph families;
- FlashAttention and quantized KV interaction;
- speculative/ngram/MTP route;
- ROCm and Vulkan compute dispatch;
- metrics, trace points, and current bottleneck reasons;
- code-removal risk boundaries.

It is not a full map of every model architecture in llama.cpp. Routes that are
not active for the local Qwen/RDNA4 profile are included only when they can
interfere with cleanup, backend dispatch, or future upstream sync.

## Active Lane Definitions

Main cold-first lane for code-speed claims:

| Field | Value |
| --- | --- |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | Vulkan on RX 9070 XT / AMD proprietary driver for the current post-E257 focus; ROCm remains the preferred production backend and comparison route |
| Context | `ctx=12288` |
| Batch | `batch=7168`, `ubatch=1024` for the active Vulkan 12k profile |
| KV | `q4_0/q4_0` |
| Speculation | `--spec-type none` |
| Reuse | `--cache-ram 0 --ctx-checkpoints 0`, no reuse |
| Thinking | on / `--no-disable-thinking` |
| Current best | E257 r3 `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s` |
| Metric type | cold-first baseline only; do not compare with repeated/session or A3B practical profiles |

Active ROCm decode parity lane:

| Field | Value |
| --- | --- |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend baseline | ROCm/HIP SDK 7.1 on RX 9070 XT / `gfx1201` |
| Backend comparator | Vulkan on RX 9070 XT / AMD proprietary driver |
| Context | `ctx=12288` |
| Batch | `batch=6144`, `ubatch=2048` |
| KV | first parity gate `q4_0/q4_0`; f16 Vulkan kept as decode-only upper comparator |
| Speculation | `--spec-type none` |
| Reuse | `--cache-ram 0 --ctx-checkpoints 0`, no reuse |
| Thinking | on / `--no-disable-thinking` |
| Current evidence | E116 ROCm q4 `29.1685 TPS` / `29.625 tok/s`; Vulkan q4/f16 `39.8801-40.2753 TPS` / `40.8683-41.2283 tok/s`; E151 raises current ROCm q4 decode to `32.2467 tok/s` on the r3 short-decode gate |
| First action | E149 ROCm/Vulkan diagnostic traces and Q3_K route-delta table are complete; E150 rejects disabling fusion; E151 keeps RDNA4 Q3_K MMVQ `nwarps=2`; E152 post trace confirms residual Q3_K MMVQ remains fused/direct dominated. Next action is a larger Q3_K decode branch plan, not another nearby fusion-disable or launch-overhead toggle |
| Metric type | decode-focused backend parity; do not mix with prompt-heavy cold-first or 64k prefill headlines |

Important session/long-context lane:

| Field | Value |
| --- | --- |
| Model | `Qwen3.6-27B-Q3_K_S` |
| Context | `ctx=32768` |
| Batch | `batch=5120`, `ubatch=1024` |
| KV | `q4_0/q4_0` |
| Speculation | `ngram-mod` only as opt-in repeated/steady behavior |
| Metric type | keep cold-first and repeated/steady separate |

Active Vulkan 64k investigation lane:

| Field | Value |
| --- | --- |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | Vulkan on RX 9070 XT / AMD proprietary driver |
| Context | `ctx=65536` |
| Batch | `batch=8192`, `ubatch=1024` |
| KV | `q4_0/q4_0` |
| FlashAttention | on |
| Speculation | `--spec-type none` |
| Reuse | `--cache-ram 0 --ctx-checkpoints 0`, no reuse |
| Required env/args | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `--no-mmap` |
| Current best | E128 `1.3406 TPS`, prompt `666.62`, decode `36.58` |
| Comparator | ROCm same lane E128 `1.5545 TPS`, prompt `799.09`, decode `22.83` |
| Metric type | cold-first long-context real-server lane |

Confirmed 12k repeated/session lane:

| Field | Value |
| --- | --- |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | ROCm/HIP on RX 9070 XT / `gfx1201` |
| Context | `ctx=12288` |
| Batch | `batch=6144`, `ubatch=2048` |
| KV | `q4_0/q4_0` |
| Speculation | `--spec-type none` |
| Reuse | prompt cache/checkpoints enabled |
| Metric type | repeated/steady only; post-driver E113 reuse `17.8934 TPS`, reuse + `ngram-mod 12/16/32` best r3 `19.5051 TPS`, after-first mean `23.9038 TPS` |

Historical trace lane:

| Field | Value |
| --- | --- |
| Context | `ctx=12288` |
| Batch | `batch=6144`, `ubatch=192` |
| KV | `q4_0/q4_0` |
| Use | C01 route shares, FA trace, decode/medium-shape hotspot evidence |

Vulkan CPU fallback lane:

| Field | Value |
| --- | --- |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | Vulkan build with `--gpu-layers 0` |
| Context | `ctx=4096` for quick CPU experiments |
| Batch | `batch=512`, `ubatch=128` |
| KV | `q4_0/q4_0` |
| Speculation | `--spec-type none` |
| Reuse | disabled |
| Metric type | short real-server CPU fallback gate; E125 default `1.7703 TPS`, `--no-mmap` `1.8815 TPS` |

Do not compare headline TPS across these lanes without marking the lane.

## End-to-End Route

The active request path is:

1. GUI or CLI creates server args.
2. `common/arg.cpp` parses backend, batch, KV, FlashAttention, and speculative
   options into `common_params`.
3. `common/common.cpp` copies these into `llama_model_params` and
   `llama_context_params`.
4. The server initializes model/context in `tools/server/server-context.cpp`.
5. Each request enters a slot, fills `llama_batch`, and calls `llama_decode()`.
6. `src/llama-context.cpp` allocates/splits the batch into `llama_ubatch`
   objects and prepares the memory/KV context.
7. `model.build_graph(...)` builds the Qwen graph for the current ubatch.
8. `ggml_backend_sched_graph_compute_async(...)` schedules the graph onto ROCm,
   Vulkan, CPU, or split backends.
9. Backend `supports_op(...)` and compute switches select actual kernels.
10. Logits/sampling data come back to server code, which samples, accepts
    speculative drafts when enabled, and repeats.

## Launch and Parameter Route

| Layer | Code | TPS relevance |
| --- | --- | --- |
| GUI server tab | `gui/server_tab.py` | Current GUI launch route; default FA on in newer server tab; emits `--cache-type-k/v` and `--flash-attn on` |
| Legacy/main GUI route | `gui/llama_gui.py` | Exposes KV combo, FA checkbox, TurboQuant warnings, and command construction |
| Benchmark tab | `gui/benchmark_tab.py` | Emits repeatable benchmark commands and saves presets/history |
| Profiles | `gui/model_presets.json`, `gui/optimization_profiles.py` | Can silently change batch, ubatch, KV, FA, backend, and spec mode |
| CLI parser | `common/arg.cpp` | Authoritative route for `--flash-attn`, `--cache-type-k/v`, `--spec-type`, `--batch-size`, `--ubatch-size` |
| Common conversion | `common/common.cpp` | Converts parsed params into model/context params used by runtime |

Launch parameters that can change backend routes:

- `-ngl` / `--gpu-layers`: offload depth and scheduler splits.
- `-c`: KV size and FA pressure.
- `-b`: batch size, especially prompt chunks.
- `-ub`: ubatch size, directly changes matmul and FA shapes.
- `--flash-attn on|off|auto`: switches attention graph route.
- `--cache-type-k`, `--cache-type-v`: changes KV storage and FA constraints.
- `--spec-type`: changes decode loop and sampling route.
- `--cache-ram`, `--ctx-checkpoints`: affects server reuse/checkpoint route.

## Build and Backend Selection Route

| Backend | Build route | Runtime route |
| --- | --- | --- |
| ROCm/HIP | `GGML_HIP=ON`, `ggml/src/ggml-hip/CMakeLists.txt`, `ggml/src/ggml-hip/hip-source-bundles.cmake` | HIP build of `ggml/src/ggml-cuda/*`; names still say CUDA but compile through ROCm |
| Vulkan | `GGML_VULKAN=ON`, `ggml/src/ggml-vulkan/CMakeLists.txt`, `vulkan-shaders-gen` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` plus generated shader pipelines |
| CPU fallback | `ggml/src/ggml-cpu/*` | Used for unsupported ops, host work, graph inputs, tokenizer/sampling, or split fallback |

Local ROCm build-risk note:

- Windows ROCm builds must use Ninja and ROCm clang/clang++; Visual Studio
  generators are not the local route.
- `GGML_HIP_EXPERIMENT_PROFILE` can reduce source families for experiments:
  `default`, `qwen-fa-reduced`, `mmvq-focused`, `mmvq-isolated`.

## Server and Slot Route

Primary file: `tools/server/server-context.cpp`.

| Phase | Code area | TPS relevance |
| --- | --- | --- |
| Slot state | `server_slot` fields | Holds prompt tokens, sampler, speculative draft, checkpoints, timing, accepted draft stats |
| Prompt batching | main server loop before `llama_decode()` | Builds `llama_batch` until `n_batch`; prompt-heavy TPS is dominated here |
| Context reuse | prompt cache, cache-ram, checkpoints | Useful for sessions, disabled for cold-first speed claims |
| Decode call | `llama_decode(ctx, batch_view)` | Enters core runtime; server can retry with smaller `n_batch` if KV slot fails |
| Sampling | `common_sampler_sample`, `common_sampler_accept` | Host-side or backend-assisted sampling; synchronizes when logits are needed |
| Speculative accept | `common_sampler_sample_and_accept_n` | Tests and accepts draft tokens; can restore checkpoints after partial acceptance |
| Metrics | server timings and Prometheus metrics | Separates prompt eval, token generation, decode calls, draft counts |

Server-route bottleneck rules:

- If prompt eval is slow and kernel traces point to `MUL_MAT`, fix backend
  compute first.
- If repeated tasks share a large prefix, keep prompt cache/checkpoints enabled
  for practical throughput; disable them only for cold-first kernel claims.
- If prompt eval is fast but generation stalls, inspect sampling, draft
  coverage, TG scheduler, and small-batch kernels.
- If failures shrink `n_batch`, inspect KV slot pressure and reuse/checkpoint
  behavior before changing kernels.

## Batch, Ubatch, and Scheduler Route

Primary files:

- `src/llama-batch.cpp`
- `src/llama-context.cpp`
- `src/llama-memory*.cpp`
- `src/llama-kv-cache.cpp`
- `ggml/src/ggml-backend-sched.cpp`

Runtime route:

1. `llama_decode()` validates the input `llama_batch`.
2. `llama_batch_allocr::init(...)` converts it to internal layout.
3. `memory->init_batch(...)` finds KV/recurrent slots and creates a memory
   context with one or more ubatches.
4. `llama_context::process_ubatch(...)` applies memory context changes.
5. A TG scheduler can be swapped in for single-token decode.
6. Graph reuse is attempted if graph parameters match.
7. Graph is built or reused, inputs are set, then scheduler compute is launched.

Important route gates:

- `ubatch.n_tokens > 1`: prompt/prefill path; large matmul shapes dominate.
- `ubatch.n_tokens == 1`: token-generation path; TG scheduler and MMVQ/MMVF
  shape choices matter more.
- `cparams.n_ubatch`: changes Q3_K route shapes and can trigger cliffs.
- `graph_reuse_disable`: changes graph build/alloc overhead, not kernel math.
- `LLAMA_UBATCH_TIMING`: route-level runtime timing at `process_ubatch`.

## KV and Memory Route

Primary files:

- `common/arg.cpp`
- `src/llama-context.cpp`
- `src/llama-model.cpp`
- `src/llama-kv-cache.cpp`
- `src/llama-memory-hybrid.cpp`
- `src/llama-memory-recurrent.cpp`

KV type route:

- CLI types: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`,
  `q5_0`, `q5_1`, local `TBQ*`, `TQ*`, `TKV*`, and aliases such as `turbo4`.
- Active lane: `q4_0/q4_0`.
- Qwen hybrid/recurrent layers use both attention KV and recurrent state memory.

FlashAttention interaction:

- Quantized V cache without FA can force graph dequant or be rejected.
- TurboKV direct routes require FA for intended acceleration.
- FA masks are converted to F16 in the graph.

Current metric decision:

- `q4_0/q4_0` is the practical default because q8/f16/TurboKV alternatives
  regressed in existing local A/B.
- q4 KV is a memory/enabler route, not the top measured 12k prompt bottleneck.

## Qwen Graph Route

Main Qwen3.6 graph family in this tree:

- `src/models/qwen35moe.cpp`
- MTP graph companions: `src/models/qwen35_mtp.cpp`,
  `src/models/qwen35moe_mtp.cpp`
- Common graph helpers: `src/llama-graph.cpp`, `src/llama-graph.h`

Qwen graph skeleton:

1. `build_inp_embd(...)`
2. per-layer `attn_norm`
3. either full attention or recurrent/linear attention:
   - full attention: Q projection, Q norm, K/V projection, K norm, RoPE,
     KV copy/update, FA or KQ/softmax/KQV, gate, output projection;
   - recurrent/linear attention: qkvz projection, beta/alpha projections,
     convolution state, `SSM_CONV`, `GATED_DELTA_NET`, state update,
     gated norm, output projection.
4. residual add
5. post-attention norm
6. MoE FFN:
   - gating/top-k;
   - expert projections through `MUL_MAT_ID`;
   - optional shared expert route;
   - residual add.
7. final norm
8. output head `MUL_MAT`

Active hot op families for Qwen TPS:

| Graph family | Typical ggml ops | Backend route pressure |
| --- | --- | --- |
| Q/K/V/output projections | `MUL_MAT` | Main Q3_K/Q4_K/F32 matrix route pressure |
| MoE experts | `MUL_MAT_ID`, top-k, gather/scatter | Expert route and fusion pressure |
| Full attention | `FLASH_ATTN_EXT` or KQ/softmax/KQV | FA route, KV bandwidth, mask handling |
| Linear/recurrent attention | `SSM_CONV`, `GATED_DELTA_NET`, copy/state ops | Secondary hotspot and state-memory pressure |
| Norm/residual | `RMS_NORM`, `ADD`, `MUL`, `SILU`, fused variants | Smaller but frequent fused routes |
| Output head | `MUL_MAT`, logits copy/sampling | Small decode cost; can force sync for sampling |

## FlashAttention Route

Graph route:

- `common/arg.cpp` parses `--flash-attn`.
- `src/llama-context.cpp` sets `cparams.flash_attn`.
- `src/llama-graph.cpp` chooses `ggml_flash_attn_ext(...)` when enabled and
  eligible.

ROCm route:

- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/fattn-qwen-reduced.cpp`
- `ggml/src/ggml-cuda/fattn-vec.cuh`
- `ggml/src/ggml-cuda/fattn-wmma-f16.cu`
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh`

Vulkan route:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn*.comp`
- generated shader pipelines from `vulkan-shaders-gen`

Measured status:

- E026: `FLASH_ATTN_EXT forward = 638.004 ms` out of
  `24758.198 ms` sync CUDA_NODE total on the C01 trace, about `2.58%`.
- Existing data makes FA a correctness/long-context/KV enabler, not the first
  12k cold-first prompt-heavy TPS lever.

## Speculative, Ngram, and MTP Route

Primary files:

- `common/common.h`
- `common/arg.cpp`
- `common/speculative.cpp`
- `common/ngram-mod.cpp`
- `common/ngram-cache.cpp`
- `tools/server/server-context.cpp`
- `src/llama-context.cpp`
- `src/llama-mtp.h`

Route families:

| Spec type | Route | Current use |
| --- | --- | --- |
| `none` | normal sampling only | Cold-first baseline |
| `ngram-mod` | local ngram table, no draft model | Opt-in repeated/steady accelerator |
| `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-cache` | alternate self-spec variants | Mapped, not current default |
| `draft`, `eagle3` | external draft model style | Supported route, not current Qwen target |
| `mtp` | MTP head/context | Experimental; requires MTP-enabled GGUF |
| `ngram-mtp` | ngram first, MTP fallback | Experimental; promising only with compatible MTP GGUF |

Server behavior:

- `server_slot::update_batch(...)` can add draft tokens to the next batch.
- `common_speculative_draft(...)` generates draft candidates.
- Main model verifies the draft in `common_sampler_sample_and_accept_n(...)`.
- Partial acceptance can restore checkpoints and trim KV state.
- MTP context is mirrored through `llama_set_mtp(...)` and the ubatch hook.

Metric interpretation:

- `ngram-mod` has measured repeated/steady upside in older lanes, but E107
  rejects the tested ngram settings for the current 12k q4-KV cold-first lane:
  coverage/effective acceptance were near zero and wall TPS regressed.
- E113 post-driver retunes the stacked route: `ngram-mod 12/16/32` on top of
  prompt cache/checkpoints measured `19.0148` and `19.5051 TPS`, with
  effective acceptance `0.035028`.
- `ngram-simple n8/m16` is rejected for this route (`15.3491 TPS`) because
  decode slowed despite draft generation.
- E114 rejects going shorter to `ngram-mod 8/16/32`: `14.2479 TPS`, local
  acceptance `0.136029`, effective acceptance `0.004251`.
- Future speculative claims must report local acceptance, coverage, and
  effective acceptance together, plus per-task accepted-token bursts when
  coverage is sparse.
- MTP smoke results are promising, but not a default route without the right
  GGUF and compatibility validation.

## Prompt Cache and Checkpoint Route

Primary file: `tools/server/server-context.cpp`.

Route mechanics:

1. The server keeps slot prompt state after a request.
2. Before the next request, prompt cache update and LCP similarity pick the best
   reusable slot/prefix.
3. Context checkpoints restore a shared prefix when the exact current prompt
   tail differs.
4. The server reprocesses only the changed tail, then continues normal decode.

E111 measured this route on the active 12k ROCm q4-KV lane with `spec=none`:

| Run | Aggregate TPS | Interpretation |
| --- | ---: | --- |
| cold-first reference `e106-rocm-q3k-control-r1` | `11.8464` | no reuse, cache/checkpoints disabled |
| `e111-rocm-q3k-reuse-steady-r1` | `14.6132` | first task cold, second task reused prefix |
| `e111-rocm-q3k-reuse-steady-r3` | `17.7984` | six tasks, five reused-prefix tasks |
| `e112-rocm-q3k-reuse-ngram244864-r3` | `18.7194` | prompt reuse plus opt-in ngram bursts |
| `e113-driver5012-rocm-reuse-specnone-r3` | `17.8934` | post-driver reuse baseline |
| `e113-driver5012-rocm-reuse-ngram121632-r3b` | `19.5051` | post-driver best repeated/session route |
| after-first tasks in E111 r3 | `~20.00` | practical repeated-session throughput |
| after-first tasks in E113 `12/16/32` r3b | `~23.90` | stacked ngram repeated-session throughput |

Log evidence:

- prompt cache enabled with `8192 MiB` limit;
- LCP similarity `0.982-0.984`;
- restored checkpoint at `5370` tokens;
- reused tasks processed about `2033-2052` prompt tokens instead of full
  `7403-7422` token prompts.

Interpretation:

- This is a real practical route for GUI/agent sessions.
- It is not a cold-first kernel/default improvement.
- `ngram-mod 12/16/32` can stack on this route, but only as an opt-in repeated
  session profile because the first cold request can slow down and the benefit
  depends on accepted-token bursts.
- Do not disable server reuse in user-facing presets unless the goal is a clean
  first-response benchmark.

## ROCm Compute Route

Full details: `docs/research/ROCM_ROUTE_MAP.md`.

Main ROCm dispatch path:

1. Scheduler asks ROCm backend whether each op is supported.
2. `ggml_cuda_compute_forward(...)` switches on op.
3. `MUL_MAT` enters `ggml_cuda_mul_mat(...)`.
4. Route candidates are selected in priority order:
   - `mul_mat_vec_f`
   - `mul_mat_f`
   - `mul_mat_vec_q`
   - `mul_mat_q`
   - `batched_cublas`
   - backend/split wrappers
   - final `cublas_backend` / hipBLAS fallback

Current ROCm hot route:

- Large Q3_K prompt prefill falls to hipBLAS staging.
- The route stages Q3_K `src0` to fp16 and runs rocBLAS/hipBLAS.
- E049/E054/E103 show conversion/staging is large and repeated.
- E104 persistent fp16 cache regressed.
- E105 existing-MMQ route forcing regressed or tied.

Current ROCm acceleration thesis:

- Avoid or fuse Q3_K staging for the hot large-prefill shapes.
- Do not solve this by broad selector forcing.
- A credible future route is a shape-specific fused Q3_K x F16 RDNA4 kernel.

## Vulkan Compute Route

Full details: `docs/research/VULKAN_ROUTE_MAP.md`.

Main Vulkan dispatch path:

1. Scheduler asks Vulkan backend whether each op is supported.
2. `ggml_vk_compute_forward(...)` switches on op.
3. Matmul routes through `ggml_vk_mul_mat(...)` and shader pipeline selection.
4. Shaders come from `vulkan-shaders/*.comp` and generated SPIR-V wrappers.

Current Vulkan hot route:

- Decode can outperform ROCm on some lanes.
- Prompt-heavy Q3_K remains limited by the active large matmul shader route,
  and at `ctx=65536` q4 FlashAttention becomes a co-primary blocker.
- Accepted route after E082/E086/E102:
  `matmul_q3_k_f32_f16acc_aligned_l`.
- E128/E131 64k route:
  - best safe Vulkan stack: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `--no-mmap`,
    `b8192/ub1024`, q4/q4 KV, FA on, `spec=none`, no reuse;
  - `MUL_MAT q3_K` is `47.79%` and `FLASH_ATTN_EXT` is `38.03%` of traced
    time;
  - active FA route is `flash_attn_f32_f16_aligned_f32accq4_0`,
    `coopmat1`, q4/q4, `Br=16,Bc=64,D_split=8,row_split=4`,
    main `N=1024`, growing `KV`, `split_k=1`, `use_mask_opt=1`.
  - E132 resource stats for the main FA route: `98 VGPR`, `76 SGPR`,
    `26112 B LDS`, `0 scratch`.
  - E133 shape summary: parsed hot rows are `MUL_MAT q3_K 42684.45 ms`
    and `FLASH_ATTN_EXT 33965.16 ms`; the top Q3_K forms are
    `m=17408,n=1024,k=5120` (`20338.69 ms`) and
    `m=5120,n=1024,k=17408` (`11289.87 ms`), together `74.1%` of parsed
    Q3_K time.
  - E134 route ceiling: to match the ROCm 64k wall alone, FFN gate/up would
    need `2.234x` local speedup, all Q3_K would need `1.357x`, FA would need
    `1.494x`, while the combined Q3_K+FA prefill core needs only `1.172x`.
  - E135 graph proof: default-off `GGML_VK_FFN_ROUTE_TRACE=1` on the real
    64k server lane found `63 x q3_K SWIGLU` prefill candidates with
    `m=17408,n=1024,k=5120` in each active graph.
  - E136 FFN route model: base dual-A/same-B fusion would have `29696 B`
    dual-A LDS and `16 -> 32` accumulator fragments. With unchanged A-dequant
    proxy included, the local ceiling is about `1.417x`, projecting
    `1.4466 TPS`, below the ROCm `1.5545 TPS` target.
  - E137 Q3_K dual-N gate: a temporary `NITER=2` `mul_mm.comp` prototype kept
    LDS at `20480 B` and scratch at `0`, but raised the candidate to `120 VGPR`
    and regressed pp7488 to `855.29`; the clean restored default is `974.92`
    with `113 VGPR`.
  - E138 FA split-k gate: forcing existing FA split/reduce from `KV>=8192`
    routed correctly but dropped prompt eval from `666.87` to `96.29 tok/s`,
    because it adds temp writes, sync, and reduce dispatch per FA node.
  - E139 Q3_K predequant route gate: forcing hot Q3_K matmuls through the
    existing `Q3_K -> fp16 prealloc_x -> f16 matmul` fallback routed correctly
    but dropped pp7488 from `969.61` to `743.65`. Even the narrow
    `k>=17000` hot shape regressed to `929.40`, while the f16 matmul pipeline
    itself used only `77 VGPR`; the route cost is the fp16 temp, sync, and
    extra global traffic.
  - E140 Q3_K matmul split-K gate: forcing existing split-K on
    `m=5120,n=1024,k=17408` routed correctly but measured `966.21` for
    split-K2 and `964.46` for split-K4 vs `968.74` direct. The hot reverse
    shape already exposes enough workgroups, so split/reduce overhead does not
    buy useful parallelism.
  - E141 KV dtype upper-bound gate: f16/f16 KV improved pp7488 only
    `970.03 -> 996.00 tok/s` and failed real 64k server fit
    (`16183 MiB` projected Vulkan device use vs `15221 MiB` free); q8_0/q8_0
    regressed to `940.03 tok/s`. q4/q4 remains the H38 route.
  - E142 FA `Br32/Bc32` route gate: the candidate stayed single-dispatch
    coopmat1 q4/q4 but regressed pp7488 `971.09 -> 896.97` and raised the main
    FA route to `133 VGPR`; a f16acc companion still measured only `922.22`.
    Larger query-row cm1 is closed unless a new design cuts per-row live state.
  - E143 Q3_K large-N warptile gate: `BN192/WN96` and `BN256` variants reduced
    static A-dequant/workgroup proxies but regressed badly. Default pp7488 was
    `974.19`; `bn192-wn96` measured `760.78` with `139 VGPR`, `bn256-*`
    measured about `660` with `165 VGPR / 29696 B LDS`, and
    `bn192-wm128-wn96` spilled scratch and fell to `137.71`. Larger N tiles are
    closed for the current `mul_mm.comp` topology.
  - E144 Q3_K BK-depth gate: `BK16` lowered resources from
    `113 VGPR / 20480 B LDS` to `70 VGPR / 12288 B LDS`, but pp7488 fell
    `972.77 -> 587.52`; `BK64` exceeds the 32 KiB shared-memory budget.
    Shrinking K depth is closed unless a new design reduces barrier cadence.
  - E146 Q3_K BM256 gate: static model halved workgroups and B reload proxy
    while leaving A-pair dequant flat, but runtime fell `972.84 -> 916.62`.
    Resources changed to `94 VGPR / 31744 B LDS`; larger M tiles are closed in
    the current shader because near-limit LDS/occupancy dominates.
  - E147 Q3_K layout route gate: broad persistent fp16/int8 FFN layouts are
    rejected on memory (`+25.03 GiB` / `+9.09 GiB` over current Q3), while
    signed-nibble is memory-plausible (`+1.12 GiB`) but low-confidence because
    current Q3_K SPIR-V is only modestly heavier than f16/Q4_K and E088/E090
    already rejected nearby unpack/scale simplifications.
  - E145 FA D-split gate: `D_split=4` and `D_split=16` kept the same q4/q4
    coopmat1 `Br16/Bc64,row_split=4` route and the same reported
    `98 VGPR / 76 SGPR / 26112 B LDS`, but pp7488 regressed from `978.88` to
    `953.24` and `951.54`. Simple head-dimension split retuning is closed.
- Rejected route families include old corrupt tile profiles, Q8_1/int-dot
  Q3_K route, expression-only dequant cleanup, aligned-store cleanup, and
  invalid warptiles. For 64k FA, E129 rejects `Bc=32/128`, E131 rejects
  mask-opt disable and forced FA f16acc, and E132 rejects SHMEM staging
  because it falls back to scalar FA.

Current Vulkan acceleration thesis:

- Continue in the active Q3_K coopmat prefill path and the now-measured q4
  long-KV FA path.
- Treat the next work as a route stack, not a single tweak. E135 proved the
  dense FFN graph hook, E136 says dual-A/same-B alone is below target unless it
  also reduces A-side Q3_K work, and E137 says current dual-N/same-A loses to
  accumulator/VGPR pressure. E139 says existing per-node predequant is the
  wrong repack implementation because it adds a large fp16 temp and sync. E143
  says bigger current warptiles are also the wrong A-reuse implementation
  because they trade dequant reduction for VGPR/LDS/scratch pressure. E144
  says simply shrinking BK is not the answer either: the shader gets lighter
  but spends too much on doubled K-loop/barriers. E146 shows that larger M
  tiles also lose when B/workgroup savings push LDS near the device limit. E147
  says broad persistent fp16/int8 layout is a VRAM dead end, and compact
  signed-nibble layout needs a stronger instruction/resource gate before code.
  E145 closes the simple FA
  `D_split` retune because it changes runtime but not the reported resource
  fingerprint in a positive direction.
  Prioritize a direct/single-dispatch shape-specific Q3_K shader or a
  backend-private layout that avoids fp16 temp/sync/reduce, or single-dispatch
  FA long-KV work inside the existing q4/q4 route, but not by simply increasing
  Br or flipping `D_split`. Use FFN fusion only as a stack component if resource proof stays
  coopmat/no-scratch.
- Do not spend time on speculative decode, nearby ubatch sweeps, FA `Bc`
  retuning, mask-opt disable, f16acc forcing, SHMEM staging, forced FA
  split-k for the 64k lane. Do not repeat the existing Q3_K predequant fallback
  or existing matmul split-K as route candidates; do not repeat larger-N
  warptile, BM/BK-depth, broad Q3_K fp16/int8 layout, or FA `D_split` retunes
  without a new topology that reduces live-state pressure, LDS pressure, memory
  residency, and barrier cadence together.

## Backend Scheduler and Op Coverage

Backend scheduling lives in the generic ggml scheduler and backend-specific
`supports_op(...)` functions.

Coverage by active op class:

| Op class | ROCm | Vulkan | TPS priority |
| --- | --- | --- | --- |
| `MUL_MAT` | Covered deeply | Covered deeply | Highest |
| `MUL_MAT_ID` | Covered | Covered | High for MoE/expert route |
| `FLASH_ATTN_EXT` | Covered | Covered | Medium/low for 12k, higher for long context |
| `SSM_CONV`, `GATED_DELTA_NET` | Covered | Covered/partial by op family | Medium |
| `RMS_NORM`, `ADD`, `MUL`, activation fusions | Covered | Covered | Medium/low but frequent |
| KV copy/update/set rows | Covered as memory route | Covered as memory route | Medium |
| Sampling/logits copies | Covered as server/runtime route | Covered as server/runtime route | Decode-sensitive |
| Non-Qwen training/optimizer/image ops | Mapped only as cleanup risk | Mapped only as cleanup risk | Low for current TPS |

## Metrics and Trace Points

Use these to keep future acceleration plans evidence-based:

| Question | Tooling/route | Notes |
| --- | --- | --- |
| Where is wall time by server phase? | server timings, `scripts/agent_workload_bench.py` output | Separates prompt eval and generation |
| Does prompt reuse help? | same-lane run without `--no-reuse`, cache/checkpoint log evidence | E111 confirms this is the main repeated/session route; E113 shows shorter opt-in ngram can stack after driver update |
| How does ubatch split behave? | `LLAMA_UBATCH_TIMING=1`, optional `LLAMA_UBATCH_TIMING_SYNC=1` | Use sync only for diagnostic shares, not headline TPS |
| Which ROCm matmul route is active? | route traces around `ggml_cuda_mul_mat` and `GGML_TRACE_CUBLAS_Q3K_ROUTE` | Default off; trace changes timing |
| How much Q3_K staging repeats? | E103-style Q3_K route reuse trace | Already shows maximal repeat count but too much fp16 footprint |
| Which Vulkan shader route is active? | `GGML_VK_MATMUL_ROUTE_TRACE=1` | Use with perf logger only for diagnostics |
| Which Vulkan FA route is active? | `GGML_VK_FA_ROUTE_TRACE=1` | Shows path, q/k/v types, `Br/Bc`, split-k, mask-opt, and workgroup geometry |
| Which Vulkan kernels dominate? | `GGML_VK_PERF_LOGGER=1` | Intrusive; not a speed claim |
| Does ngram help cold or warm? | cold/warm split, draft stats | Report coverage and effective acceptance |
| Does KV type help? | same-lane q4/q8/f16 A/B | Compare prompt and decode separately |
| Is FA worth targeting? | FA on/off A/B and `FLASH_ATTN_EXT` share | Existing 12k share is low |

## Bottleneck Ranking For Planning

| Rank | Route | Why it blocks TPS | Planning implication |
| --- | --- | --- | --- |
| P0 | ROCm large Q3_K prefill via hipBLAS staging | Large share, repeated Q3_K -> fp16 conversion, current alternatives rejected | First serious code-design target |
| P1 | ROCm Q3_K MMQ/MMVQ decode/medium shapes | Sustained Q3 direct route pressure in C01 traces | Tune only with exact bucket evidence |
| P2 | Vulkan Q3_K prompt shader | Vulkan decode is strong but prompt Q3_K route trails ROCm; at 64k it is `47.79%` of traced time; E133 shows top forms `17408x1024x5120` and `5120x1024x17408` are `74.1%` of parsed Q3_K time; E134 says all-Q3_K needs `1.357x` local to close the lane alone; E135 proves the real 64k graph exposes `63 x q3_K SWIGLU` FFN gate/up candidates; E136 says dual-A/same-B FFN fusion alone projects below target unless it reduces A-side work; E143 rejects larger-N warptiles because A-reuse lost to VGPR/LDS/scratch; E144 rejects BK shrink because barriers dominate; E146 rejects BM256 because B/workgroup savings lose to near-limit LDS; E147 rejects broad persistent fp16/int8 layout and marks signed-nibble low-confidence | Active Vulkan 64k code target, but require route-ceiling, graph-pattern, resource, shape-level perf, and A-dequant/layout proof. Next Q3_K implementation should wait for a topology that reduces repeated A work without larger-tile live-state growth, near-limit LDS, fp16 temp, broad extra VRAM, or extra K-loop/reduce cadence; FFN fusion remains a stack component |
| P3 | Vulkan q4 FA long-context route | `FLASH_ATTN_EXT` is `38.03%` of traced 64k Vulkan time; main route is `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch`; easy FA toggles have regressed; E133 shows tail KV chunks dominate the FA series; E134 says FA alone needs `1.494x` local; E141 says f16/f16 KV is only a small pp upper bound and does not fit the real 64k server lane; E142 says `Br32/Bc32` loses to VGPR pressure; E145 says `D_split=4/16` keeps the same resource fingerprint but regresses runtime | Keep q4/FA; optimize only with shader-body/resource evidence, per-KV tail timing, and same-lane A/B. Treat it as the second half of the combined Q3_K+FA stack; do not pivot to f16/q8 KV, larger-Br cm1, or `D_split` flips for H38 |
| P4 | Prompt cache/checkpoint session route | Strong repeated/session gain by avoiding shared-prefix prefill | Keep enabled for practical sessions; do not mix with cold baseline |
| P5 | `ngram-mod` session route | Can stack on prompt cache via accepted-token bursts; current 12k cold-first coverage is near zero; match-8 is too noisy | Keep `12/16/32` opt-in; require coverage/effective acceptance and burst evidence |
| P6 | GDN/SSM/RMS/fusions | Visible but smaller; past simple probes negative | Revisit if a trace shows shared memory/residency slowdown |

## Cleanup and Deletion Boundaries

Safe to consider:

- default-off diagnostics;
- build-profile gates;
- removal of local rejected experimental knobs after checking GUI/scripts/docs;
- pruning only after a dedicated build profile proves no active route uses it.

Not safe to delete from route maps alone:

- broad `ggml/src/ggml-cuda` source families;
- Vulkan shader files without generator/support checks;
- FlashAttention code;
- q4/TurboKV code paths;
- speculative/ngram/MTP code paths;
- server checkpoint/reuse code paths;
- MoE or recurrent Qwen route code.

Reason: many of these are low-share for the current 12k prompt-heavy bottleneck
but required for long context, fallback correctness, GUI features, or future
MTP/speculative work.

## Done Criteria For This Map

The route map is considered complete enough for the next TPS plan when:

- the active Qwen launch/config route is mapped;
- q4 KV and FA interactions are mapped;
- ngram/MTP/speculative routes are mapped and separated from cold-first claims;
- prompt cache/checkpoint reuse route is mapped and separated from cold-first
  claims;
- Qwen graph layer families are mapped to ggml ops;
- ROCm and Vulkan compute routes are cross-linked;
- the current measured bottlenecks are ranked;
- remaining metric gaps are explicit and bounded.

This document meets those criteria. The next phase should be a TPS plan that
starts from P0/P1, not another broad map pass.
