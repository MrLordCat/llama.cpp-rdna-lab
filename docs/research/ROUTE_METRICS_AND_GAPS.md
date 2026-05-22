# Route Metrics and Bottleneck Map

Updated: 2026-05-22.

This document connects the ROCm and Vulkan route maps to the measured Qwen/RDNA4
performance evidence. It answers a narrower question than the backend atlases:
which active routes matter for TPS, why they are limiting, and which routes are
already low-ceiling or rejected.

Companion route maps:

- `docs/research/QWEN_TPS_ROUTE_ATLAS.md`
- `docs/research/ROCM_ROUTE_MAP.md`
- `docs/research/VULKAN_ROUTE_MAP.md`

## Completeness Snapshot

The maps now cover the active Qwen TPS route surface. The remaining work is
metric refresh as lanes change, not route discovery for the current profile.

| Area | Coverage now | Metric refresh needed |
| --- | --- | --- |
| End-to-end active Qwen TPS route | High | Fresh metric updates as lanes change; not every non-Qwen architecture |
| ROCm build/backend/dispatch routes | High | Fresh line-by-line timing for every supported op on the current `ubatch=2048` cold-first lane |
| ROCm `MUL_MAT`/`MUL_MAT_ID` hot routes | High | A new fused large-Q3_K design is not mapped because it does not exist yet |
| Vulkan build/backend/shader routes | Medium-high | Same-lane route timings for every non-matmul op; more FA/KV-specific Vulkan traces |
| Q4 KV cache route | Medium | Fresh ROCm same-lane q4/q8/f16 A/B at `ctx=12288` and `ctx=32768`; existing evidence is enough to keep q4 as default |
| FlashAttention | Medium | Fresh FA on/off A/B for the active `ubatch=2048` lane; existing trace says FA is not the main 12k bottleneck |
| Prompt cache/checkpoint session route | High | Refreshed in E113 after driver update; refresh again when prompt templates, driver/runtime, or server cache defaults change |
| `ngram-mod` and speculative routes | High for current same-lane ngram; medium for MTP | E107 covered cold-first same-lane failure; E113 post-driver retuned repeated-session ngram to `12/16/32`. MTP still needs an MTP-enabled GGUF before speed claims |
| GUI/server command route | Medium | Server launch generation and GUI knobs are identified, but not fully mapped node-by-node |
| Cleanup/pruning guidance | Conservative | Any deletion still needs build-profile proof; current docs are not a deletion authorization |

Short answer: the map is now strong enough to guide the next ROCm/Vulkan
performance search, but not complete enough to delete backend source families.
The highest-confidence performance target is still large Q3_K prefill on ROCm,
not FlashAttention or ngram.

## Active Local Lanes

Main cold-first prompt-heavy lane:

- Model: `Qwen3.6-27B-Q3_K_S`.
- Backend: ROCm/HIP on RX 9070 XT / `gfx1201`.
- Context: `ctx=12288`.
- Batch: `batch=6144`, current prompt-heavy target `ubatch=2048`.
- KV: `q4_0/q4_0`.
- Speculative decoding: `spec=none`.
- Reuse: off for cold-first claims.
- Thinking: on.

Important long-context/session lane:

- `ctx=32768`, `batch=5120`, `ubatch=1024`.
- KV: `q4_0/q4_0`.
- Speculative mode used in practical profiles: `ngram-mod`, but this should be
  reported as warm/session behavior unless the cold split proves otherwise.

Historical decode/C01 lane used by several trace documents:

- `ctx=12288`, `batch=6144`, `ubatch=192`, KV `q4_0/q4_0`, `spec=none`.
- Keep its per-route timing data, but do not mix its TPS headline with the
  newer `ubatch=2048` prompt-heavy cold-first lane.

Repeated/steady session lane:

- Same model/backend/context/batch/KV as the main 12k lane.
- Reuse enabled; do not pass `--no-reuse`, `--cache-ram 0`, or
  `--ctx-checkpoints 0`.
- `spec=none` for the confirmed E111 route.
- Metric type: repeated/steady only; never use as a cold-first kernel/default
  headline.

## Route Metric Table

| Route | Evidence | Bottleneck reason | Current decision |
| --- | --- | --- | --- |
| ROCm large Q3_K prefill through `cublas_backend` / hipBLAS | E045/E053/E056/E058 baseline cluster around `11.65-11.77 TPS`; E049 split: `src0 32.29%`, `src1 6.74%`, GEMM `60.97%`; E054: `src0_convert_ms=3370.32`, target convert `1430.88`; E103: `2792` Q3_K route rows, `349` keys, each repeated `8x` | Current route repeatedly stages Q3_K weights to fp16 before rocBLAS. GEMM is useful, but conversion/store and memory traffic are large enough to cap prompt TPS. Persistent fp16 cache would save about `2852.549 ms` conversion but costs about `42.002 GiB` and regressed in practice | Main ROCm performance target. Do not force existing MMQ for large Q3_K; design a shape-specific fused Q3_K x F16 route instead |
| ROCm Q3_K MMQ/MMVQ decode and medium shapes | C01 split: `mul_mat_q_direct|q3_K 386.811 ms`, `mul_mat_vec_q_direct|q3_K 214.295 ms`, `cublas_backend|f32 205.952 ms`; steady small-slice share is about `71%` Q3 direct routes | Quantized unpack, tile shape, and RDNA4 occupancy dominate repeated decode/medium work. Local nwarps/tile tuning gave small real gains, but route remains a sustained cost center | Keep. Incremental tuning only if it targets observed Q3 buckets and has cold/warm split |
| FlashAttention on ROCm | E026 C01 trace: sync CUDA_NODE total `24758.198 ms`; `FLASH_ATTN_EXT forward = 638.004 ms`, about `2.58%`; dominant shape `ne=(256,24,192,1)`, sum `607.121 ms`, count `1216`, avg `0.4993 ms`; active reduced route was WMMA F16 with `D=256`, `q_rows=192`, `selected_cols=16` | On the 12k Qwen lane FA is not large enough to move wall TPS much. A 10% local FA win is only about `0.25-0.30%` wall in that trace | Covered as route and metric. Keep FA, but it is not the first TPS lever for 12k prompt-heavy |
| FlashAttention on Vulkan 64k | E128 perf trace: `FLASH_ATTN_EXT 33965.16 ms`, `38.03%` of traced Vulkan time. E131 route trace: `flash_attn_f32_f16_aligned_f32accq4_0`, `coopmat1`, q4/q4, `Br=16,Bc=64,D_split=8,row_split=4`, main `N=1024`, `KV=1024..57344`, `split_k=1`, `use_mask_opt=1`. E132 resource stats: main route `98 VGPR`, `76 SGPR`, `26112 B LDS`, `0 scratch`; forced SHMEM staging fell back to scalar and prompt dropped to `520.18`. E133 shape summary shows largest tail chunks around `KV=55k-57k` are `1.12-1.17 s` each across 16 calls. E134 route ceiling says FA alone would need `1.494x` local speedup to match the ROCm 64k wall alone. E138 forced existing split-k from `KV>=8192`; route trace showed `split_k=2`, but prompt fell `666.87 -> 96.29 tok/s` | At 64k FA becomes a first-class bottleneck, but the easy toggles are already negative. Mask optimization pays for itself, split-k is not active on main chunks because the default already has enough row/head workgroups, f16acc does not move full wall TPS, SHMEM staging does not fit as a valid coopmat1 route on the current driver, and existing split-k adds temp writes plus sync/reduce per FA node. The remaining cost is likely shader memory/resource behavior over long KV, not a trivial tile/accumulation/split flag | Keep FA route trace diagnostic. Future FA work needs shader/resource instrumentation, per-KV tail timing, or structural long-KV work that remains on coopmat1 and stays single-dispatch/graph-friendly; do not repeat mask-opt disable, f16acc, SHMEM staging, forced split-k, or nearby `Bc` probes |
| Q4 KV cache (`q4_0/q4_0`) | E009 TurboKV probes: q4 baseline `11.15-11.17 TPS`; TKV4 direct/hybrid regressed `-7%` to `-10%`; mixed TKV/Q8 narrowed gap but still lost to q4. E076 Vulkan 32k KV gate: q4 safe-force baseline `9.8493`; q8 `9.1102`; f16 `8.8361` | Q4 KV reduces memory footprint and bandwidth enough to fit long context and keep attention viable. Higher precision KV increases memory pressure; local TurboKV direct paths remain slower than q4 | Current preferred KV route. Treat q4 as part of the active lane, not as an optional side note |
| Server prompt cache / context checkpoints | E111 same-lane reuse route: cold-first reference `11.8464 TPS`; reuse r3 `17.7984 TPS`; E113 post-driver cold `11.9858`, reuse `17.8934`, after-first mean `20.2012 TPS`; logs show prompt cache enabled, LCP similarity, and restored `5370`-token checkpoints | Sequential repo tasks share a large prompt prefix. The route does not make kernels faster; it avoids reprocessing most of the shared prompt and turns repeated prompt-heavy tasks into shorter prefill + same decode | Keep enabled for practical GUI/agent sessions. Disable only when collecting cold-first kernel/default claims |
| `ngram-mod` speculative route | E107 cold-first q4-KV variants lost with effective acceptance max `0.004908`; E112 stacked reuse+`24/48/64` `17.7984 -> 18.7194`; E113 post-driver `24/48/64` was noisy, while `12/16/32` measured `19.0148` and `19.5051 TPS`, after-first means `23.1681` and `23.9038`, effective acceptance `0.035028`; `ngram-simple n8/m16` regressed to `15.3491`; E114 `8/16/32` regressed to `14.2479` | It helps only when a session/task pattern produces accepted-token bursts. Cold-first prompt-heavy TPS barely moves because draft coverage is low at the start; after prompt-cache reuse, decode share is larger and match 12 balances coverage and acceptance. Match 8 creates too many false-positive drafts | Keep `ngram-mod 12/16/32` as current opt-in warm/session accelerator on top of prompt cache/checkpoints. Do not make it a cold-first default; reject `ngram-simple` and match-8 for this lane |
| MTP / `ngram-mtp` | E060 smoke: MTP accepted `46/48` draft tokens, `0.958` local acceptance, `13.53 TPS`; `ngram-mtp` `13.54 TPS`; `ngram-mod` generated zero drafts in that triage | MTP can be strong only with an MTP-enabled GGUF and compatible server route. It is not a generic replacement for ngram or prompt prefill work | Documented, guarded/experimental. No default GUI claim until compatible model and server path are verified |
| Vulkan Q3_K prompt route | E061 12k prompt-heavy: Vulkan `4.2206 TPS` vs ROCm `6.3327 TPS`; decode-biased Vulkan was faster (`35.2850` vs ROCm `27.9781`). E100/E102 32k valid spec-none: Vulkan `10.5230`, ROCm `10.8879`. E128 64k best Vulkan is `1.3406 TPS` vs ROCm `1.5545`, with `MUL_MAT q3_K` at `47.79%`; active pipeline remains `matmul_q3_k_f32_f16acc_aligned_l`. E133 shape summary: `MUL_MAT q3_K` parsed total `42684.45 ms`; top shapes `m=17408,n=1024,k=5120` (`20338.69 ms`) and `m=5120,n=1024,k=17408` (`11289.87 ms`) are `74.1%` of parsed Q3_K time. E134 route ceiling: FFN gate/up alone needs `2.234x` local to close the lane, all Q3_K needs `1.357x`, and Q3_K+FA needs `1.172x`. E135 real-server trace proves the active graph exposes `63 x q3_K SWIGLU` FFN gate/up candidates with `m=17408,n=1024,k=5120`. E136 models dual-A/same-B FFN fusion: base dual-A LDS `29696 B`, accumulators `16 -> 32`, local ceiling with unchanged A proxy `1.417x`, projected `1.4466 TPS`. E137 rejects current dual-N/same-A: clean restored default pp7488 `974.92`, `113 VGPR`; `niter2` pp7488 `855.29`, `120 VGPR`, no scratch. E139 rejects existing per-node predequant: direct pp7488 `969.61`; all-large `Q3_K -> fp16 prealloc_x -> f16 matmul` `743.65`; only `m>=17000` `832.27`; only `k>=17000` `929.40`; f16 route resources `77 VGPR`, `44 SGPR`, `22528 B LDS`, `0 scratch`. E140 rejects existing matmul split-K for `m=5120,n=1024,k=17408`: direct `968.74`, split-K2 `966.21`, split-K4 `964.46` | Vulkan decode is competitive, but prompt-heavy Q3_K matmul remains one of the two 64k blockers alongside FA. Prior Q3_K shader/tile families around this route are heavily screened and many are negative. The evidence now says to target a whole Q3_K route branch rather than global tile churn, launch-only fusion, FFN fusion that only saves B/intermediate traffic, dual-N accumulation that buys A reuse with VGPR pressure, existing predequant fallback that buys lower VGPR but pays temp/sync/global traffic, or existing split-K where the hot shape already has enough workgroups | Keep Vulkan as active 64k target. New Q3_K work must pass route-ceiling, graph-pattern, resource, shape-level perf, and A-side work-reduction gates before server A/B. First complex branch should be a direct/single-dispatch shape-specific Q3_K shader or backend-private layout that avoids fp16 temp/sync/reduce; dual-A/same-B FFN fusion is a stack component only if resource proof is clean |
| GDN / SSM / RMS/fused elementwise | C01: `GATED_DELTA_NET forward 149.095 ms`, `RMS_NORM kind=fused 209.981 ms`; probes around GDN chunking did not produce a better default | These are visible but smaller than Q3_K matmul/prefill. Several candidate changes were negative or ambiguous | Keep mapped as secondary. Revisit only if a full trace shows they rise together with memory/residency symptoms |

## Q4 KV Cache Route

CLI and GUI route:

- CLI accepts `--cache-type-k` / `--cache-type-v` in `common/arg.cpp`.
- Supported cache types include `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`,
  `iq4_nl`, `q5_0`, `q5_1`, and local TurboKV/TBQ aliases.
- Runtime context copies those into `llama_context_params.type_k/type_v`, then
  KV tensors are allocated in `src/llama-kv-cache.cpp`.
- GUI exposes the KV cache combo in `gui/llama_gui.py`.

Attention route interaction:

- `src/llama-context.cpp` rejects some quantized or TurboKV cases without
  FlashAttention and warns when V must be dequantized in the non-FA graph.
- `src/llama-graph.cpp` routes FA through `ggml_flash_attn_ext(...)` when
  `cparams.flash_attn` is active and the attention graph can use it.
- This makes KV q4 and FA coupled for practical long-context performance.

Why q4 is not the current TPS limiter:

- The measured alternatives are worse for the local long-context lanes.
- q4 mainly lowers memory pressure and preserves fit; it does not explain the
  large Q3_K prefill conversion/staging wall.
- If q4 is changed, compare both cold-first and repeated/steady metrics because
  decode and prefill can move in opposite directions.

## FlashAttention Route

Graph entry:

- `src/llama-graph.cpp` builds `GGML_OP_FLASH_ATTN_EXT` when FA is enabled.
- Masks are cast to F16 for FA paths.
- Without FA, the graph falls back to explicit KQ, softmax, and KQV routes.

ROCm backend entry:

- `ggml/src/ggml-cuda/ggml-cuda.cu` dispatches `GGML_OP_FLASH_ATTN_EXT` to
  `ggml_cuda_flash_attn_ext(...)`.
- Normal source families include tile, vec, WMMA F16, and MMA F16 routes.
- The local Qwen reduced profile routes through `fattn-qwen-reduced.cpp` and
  keeps the Qwen-relevant vec/WMMA surface smaller.

Vulkan backend entry:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` dispatches
  `GGML_OP_FLASH_ATTN_EXT` to `ggml_vk_flash_attn(...)`.
- Shader variants include scalar, coopmat, coopmat2, split-k reduce, and mask
  optimization routes.
- `GGML_VK_FA_ROUTE_TRACE=1` prints unique Vulkan FA route keys including
  path, q/k/v types, `Br/Bc`, split-k, mask-opt, and workgroup geometry.

Performance interpretation:

- FA is covered in the code map; refresh it on the current `ubatch=2048`
  cold-first lane before making new FA speed claims.
- Existing 12k trace says FA is a low-share route, so it is a correctness and
  long-context enabler more than the first prompt-heavy TPS target.
- E128/E131 show FA has moved up for the 64k Vulkan lane: `FLASH_ATTN_EXT`
  is `38.03%` of traced time and routes through aligned coopmat1 q4/q4 with
  `Br=16,Bc=64`, `split_k=1`, and `use_mask_opt=1` on main chunks.
- The first easy FA gates are negative: disabling mask-opt regresses, forced
  f16acc does not beat the full-run best, and forced SHMEM staging falls back
  to scalar FA. Future FA speed work should start from shader/resource
  evidence, not another trivial env toggle.

## Speculative / Ngram Route

CLI/server route:

- `--spec-type` accepts `none`, `mtp`, `draft-mtp`, `ngram-mtp`,
  `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, and
  `ngram-mod`.
- `common/speculative.cpp` owns speculative state creation and routing.
- `common/ngram-mod.cpp` implements the local ngram lookup table.
- `tools/server/server-context.cpp` includes extra care around KV reuse/shift
  because speculative/MTP state can desynchronize KV if reused incorrectly.

Performance interpretation:

- E107 showed the current q4-KV 12k cold-first lane does not benefit from the
  tested ngram settings: `24/48/64` generated zero drafts, and smaller
  settings reached only `0.001428-0.004908` effective acceptance while
  regressing wall TPS.
- E113 post-driver retuned the stacked route: `ngram-mod 12/16/32` measured
  `19.0148` and `19.5051 TPS` vs reuse-only `17.8934`, with effective
  acceptance `0.035028`.
- `ngram-mod 24/48/64` became noisy after the driver update; `ngram-simple`
  regressed badly (`15.3491 TPS`) despite draft generation.
- `ngram-mod` can still have repeated/steady upside, but only after useful
  accepted-token bursts appear; report local acceptance, coverage, effective
  acceptance, and per-task burst locations together.
- It does not attack prompt prefill, so it cannot solve the current main
  prompt-heavy Q3_K bottleneck alone.
- Report it separately as `repeated/steady` unless the split says otherwise.

## Prompt Cache / Checkpoint Route

Server route:

- `tools/server/server-context.cpp` updates the prompt cache when slots become
  available.
- Similar prompt prefixes are selected by LCP similarity.
- Context checkpoints can restore a shared prefix and then reprocess only the
  changed tail.

Measured E111 behavior on the active 12k ROCm q4-KV lane:

- cold-first reference: `11.8464 TPS`;
- reuse r1: `14.6132 TPS`;
- reuse r3: `17.7984 TPS`;
- reuse + `ngram-mod 24/48/64` pre-driver r3: `18.7194 TPS`;
- reuse + `ngram-mod 12/16/32` post-driver r3/r3b: `19.0148` / `19.5051 TPS`;
- after-first repeated tasks: about `20.00 TPS`;
- full prompt tokens stayed around `7403-7422`, but reused tasks processed only
  about `2033-2052` prompt tokens after restoring the `5370`-token checkpoint.

Performance interpretation:

- This is the strongest route gain found in the current cycle, and it can stack
  with ngram on repeated tasks, but it is a session/reuse gain rather than a
  kernel gain.
- Keep it enabled in practical GUI/agent sessions.
- Keep disabling it for cold-first kernel work so route changes are compared
  against the same baseline.

## Why The Current TPS Wall Is Not One Thing

For the local Qwen/RDNA4 setup, TPS is bounded by different routes depending on
the scenario:

| Scenario | Dominant pressure |
| --- | --- |
| 12k cold-first prompt-heavy ROCm | Large Q3_K prefill through hipBLAS staging; Q3_K -> fp16 conversion and memory movement are the route ceiling |
| 12k decode/medium ROCm | Q3_K MMQ/MMVQ direct routes and smaller fused/runtime kernels |
| 32k long-context | KV memory pressure and attention viability matter more; q4 is the practical default, FA is required for some quantized KV cases |
| Warm/repeated sessions | Prompt cache/checkpoints can remove most shared-prefix prefill; ngram can help decode only after coverage appears |
| Vulkan fallback | Decode can be strong; prompt-heavy Q3_K shader path is the limiter |

## Measurement Gaps

Before deleting code or changing defaults, collect these in order:

1. Fresh ROCm cold-first trace on the active lane:
   `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `spec=none`,
   no reuse, thinking on.
2. Same-lane FA on/off A/B, only to confirm ceiling; do not over-prioritize it
   unless FA share rises sharply.
3. Same-lane KV A/B: `q4_0/q4_0` vs `q8_0/q8_0` and `f16/f16` at 12k and 32k.
4. Broader repeated-session `ngram-mod` validation on longer/more varied task
   mixes. E112 covers the current two-task lane; keep cold-first and
   repeated/steady headlines separate.
5. MTP or `ngram-mtp` only with a known MTP-enabled GGUF and an explicit
   compatibility note.
6. Vulkan 32k route trace focused on active Q3_K shader pipeline, not broad
   shader churn.

## Cleanup Guidance

Safe cleanup direction today:

- Add build-profile gates for local ROCm reduced profiles.
- Keep rejected experiments documented and out of default route selection.
- Remove dead experimental knobs only after proving they are not used by GUI,
  scripts, server command generation, or benchmark replay.

Unsafe cleanup today:

- Deleting broad `ggml-cuda` source families because ROCm shares them with CUDA
  naming, HIP aliases, templates, and upstream sync.
- Deleting Vulkan shaders without checking generator fields and
  `supports_op(...)`.
- Treating FA, q4 KV, or ngram as unused because they are not the primary
  12k prompt-heavy bottleneck.
