# D136: Stock b10666 Vulkan decode beats the fork on short contexts — RESOLVED

2026-08-28 · author: coordinator · Status: RESOLVED for short-context decode
(measured root cause: missing upstream MTP/NextN load gating; see §10).
Dual-GPU layer-split overlap still ~2% behind stock (open, see §7/§10);
pre-existing single-GPU prefill anomaly noted in §10.

Related: D105 (fork Vulkan decode bandwidth ceiling), D094/D096/D098 (fork
FP8/FA local paths), D129 (P2 workgroup lever rejected). Note:
`docs/local/UPSTOCK_STOCK_b10666_NOTES.md` (now English) has the full
bench-methodology writeup.

## 1. Observation

bench2, rdna-lab profile, Qwen3.8-27B-Q4_K_M, q8_0/q8_0 KV, FA on, spec none,
`-sm layer -ts 1,1`, seed 42, warmup on, 1 run/level. Stock b10666 built with
winlibs GCC 16.2; fork built with Strawberry GCC 13 (GPU shaders are
driver-compiled from GLSL, so the C-compiler difference only affects host
dispatch). Decode (tps):

| L | ctx | stock-vk | fork-vk | delta (fork vs stock) |
|---|---|---|---|---|
| 0 | 8K   | 30.01 | 27.12 | **-9.6%** |
| 1 | 16K  | 29.67 | 27.95 | **-5.8%** |
| 2 | 49K  | 27.77 | 27.09 | -2.5% |
| 3 | 98K  | 25.03 | 24.99 | ~0% |
| 4 | 131K | 23.92 | 24.90 | +4.1% |

The fork is *faster* on prefill at every level (Vulkan +14..+35%), so this is
specifically a **decode / short-KV** effect that shrinks as KV grows.

## 2. What is different in the Vulkan backend

Tree shapes differ: fork keeps `ggml-vulkan.cpp` as an 18-line wrapper over
`runtime/*.inc`; stock has a single 20k-line `ggml-vulkan.cpp`. Shader trees
live in `ggml/src/ggml-vulkan/vulkan-shaders` in both.

### Stock-only additions relevant to decode (first-pass survey)
- `flash_attn_dequant.glsl` — asymmetric K/V dequant4() evaluated per
  K/V type via spec-constant specialization, included by `flash_attn.comp`
  and `flash_attn_cm1.comp` (NOT `cm2`).
- `dot_product_funcs.glsl` — SPIR-V intrinsic `v_dot2_f32_f16`
  (`SPV_VALVE_mixed_float_dot_product`) plus dot products; used by
  cooperative-matrix paths.
- `SPV_NV_cooperative_matrix_decode_vector` support
  (ggml-vulkan.cpp:24-28, 2676-2705): `coopmat2_decode_vector` device flag,
  SPIR-V strip pass for the NV decode-vector extension. **Not active on this
  AMD rig** (no extension in logs / no fork equivalent) — NV-only candidate,
  ignore for RDNA4.
- `mul_mat_vecq.comp` / `mul_mat_vec_base.glsl` / `types.glsl` differ from
  the fork (small edits; K_PER_ITER layouts, block_q8_1_x4 indexing).
- `flash_attn_base.glsl`: fork = 457 lines (local FP8/scalar/MMQ patches from
  D094/D096) vs stock = 223 lines (clean). Large divergence.

### Fork-only (local) tune points that may cost decode
- Fork `runtime/vk_pipeline.inc` `get_fa_tuning_params()` has local branches:
  `GGML_VK_FA_F8_NATIVE` (fp8 native, D096-A2, opt-in), F8/H16 hybrid forced
  to scalar, `GGML_VK_FA_FORCE_SCALAR`; plus FP8 scalar staging/direct
  branches (`GGML_VK_FA_F8_DIRECT`, `GGML_VK_FA_SCALAR_*`).
- Main path decision is the same in both: `n_rows == 1` -> FA_SCALAR
  (comment "scalar is faster than coopmat when N==1").

## 3. Hypotheses (ranked; first-pass code survey updated)

Survey result (head-pass, no A/B yet):
- `get_fa_tuning_params()` main path is IDENTICAL for our case: n_rows==1 ->
  FA_SCALAR in both; subgroup_size=32, workgroup_size, row_split, block_rows=1,
  block_cols=64, shmem_staging=0 for q8_0 KV are the same in both.
- Stock-only `limit_occupancy_shmem` (ggml-vulkan.cpp:3808-3821) targets
  RDNA n_rows>=64/hsk<=128 and GCN n_rows<=8/hsk>=256 — **does NOT apply**
  to our n_rows==1 RDNA4 decode; not the cause.
- Stock-only env-free scalar tuning additions: `scalar_shmem_support` check
  (block_rows /= 2 when unsupported) and `GGML_VK_FA_*` override envs exist
  only in the FORK (unset in our runs -> default path; not the cause).

1. **New q8_1 dequant-MMVQ + MMQ-in-FA machinery (stock-only), STRONGEST**:
   - `mul_mat_vec_max_cols = 8` + NEW arrays
     `pipeline_dequant_mul_mat_vec_q8_1_f32`
     (ggml-vulkan.cpp:936, DMMV_WG_SIZE_COUNT×GGML_TYPE_COUNT×8) — a
     dequant-style MMVQ variant for q8_1 that the fork does not have
     (fork only has f32/f16 DMMV).
   - `ggml_vk_fa_scalar_uses_mmq()` (ggml-vulkan.cpp:4175) + `mul_mmq.comp`
     q8_1/integer-dot path; "Separate flags for the q8_1 (integer dot) mmq
     path" (line 898). The fork's local D094 scalar-FA comment mentions a
     `MMQ int8 dot products` variant too, but the pipeline sets differ.
   - Difference in `mul_mat_vecq.comp` / `types.glsl` / `mul_mat_vec_base.glsl`
     (fork tree diverged locally; stock 141/1938/230 lines).
2. **Host-side batching/dispatch**: low probability (fork wins prefill, so
   host overhead would likely show there too).
3. **Compiler**: winlibs GCC 16 vs Strawberry GCC 13 for host hot paths —
   cheap to isolate (N2).

## 4. Original execution plan

- N1: add a trace (or read code + `GGML_VK_DEBUG`) that prints the selected
  mul-mat pipeline (MMVQ vs MMQ vs DMMV-q8_1) and FA scalar params for the
  L0 decode graph in both builds.
- N2: build fork with winlibs GCC 16 (same toolchain as stock) and re-run
  decode L0-L2 — isolates hypothesis 3.
- N3: A/B by importing stock `flash_attn*`/`mul_mat_vecq*`/`mul_mmq*` +
  pipeline registration into the fork build ONLY for q8_0 K/V (leave FP8
  paths intact), behind an opt-in env gate, bench decode L0-L2, revert if no
  win.
- N4: if N3 wins, diff the winning knob, port the smallest change, and re-run
  fork-vs-stock L0-L4 with `--runs 2-3` for a final claim.

## 5. Artifacts

- bench2 runs: `build_logs/bench/{stock,fork}-vk-l0-l4` (2026-08-28).
- Full method + flags: `docs/local/UPSTREAM_STOCK_b10666_NOTES.md` §5b.
- Stock build recipes: same file §1.

## 6. Execution results

### GCC 16 control

Rebuilding the fork with the same winlibs GCC 16.2 toolchain closed roughly
half of the short-context gap. Three-run decode averages were:

| L | stock GCC 16 | fork GCC 16 | fork delta |
|---|---:|---:|---:|
| 0 | 29.72 | 28.48 | -4.2% |
| 1 | 29.44 | 28.68 | -2.6% |
| 2 | 27.26 | 27.28 | parity |

### Vulkan timestamp isolation

`GGML_VK_PERF_LOGGER` L0 traces disproved the FA hypothesis. The comparable
4096-KV FA work was 50.2 ms / 1008 calls in the fork and 51.3 ms / 1008
calls in stock. Small per-call deltas remained in several MMVQ and auxiliary
nodes (for example Q5_K +3.4%, Q4_K m=10240 +2.2%, GET_ROWS +3.5%). Total
timestamped GPU work per decode block was about 18.14 ms in the fork versus
17.70 ms in stock (~2.5%).

### Manual-port gate: `f9f33654a` is already present

The proposed upstream change `f9f33654a` (coalesced Q4_K/Q5_K scale loads)
must not be ported again. The fork already contains the equivalent commit
`90488bd1b`, which is an ancestor of the current `master`; both diffs have
the same stable patch-id `2c55c5c95691a57306bbf70fdfbe61a29377027d`.
`git blame` also attributes the active packed-scale code to `90488bd1b`.
Therefore all GCC 16 fork baselines above already include this optimization.

The later Q3_K/Q6_K block-load change `19620004f` is likewise already
represented semantically by the fork's newer packed-32-bit loads and
bit-twiddle subtraction. No Vulkan shader source was changed and no redundant
benchmark was run. The next useful isolated candidate is the genuinely
different submission-batching behavior around upstream `803b7fcae`, followed
by a new adjacent fork-only L0 control.

## 7. Wide decode diagnostics (2026-08-28, L0, runs=1, all layers)

Full-layer isolation using `GGML_VK_PERF_LOGGER` + `LLAMA_UBATCH_TIMING` +
`GGML_SCHED_SPLIT_TIMING`, plus a one-off `STOCKSPLITCOPY` timer backported
to the stock scheduler (since reverted). Both builds were rerun back-to-back
in identical sessions.

- Graph count, split count, and inter-GPU copy counts are identical
  (156/156 decode graphs, 168 sync boundaries). Graph reuse is fully active
  in the fork: 76/78 decode graphs `reused=1`, build/alloc = 0.
- Inter-GPU copies are NOT the difference: the fork's 40 KB `l_out`
  Vulkan1->Vulkan0 copy is 10.77 ms; the stock copy of the same 40 KB is
  9.58 ms, and steady-state 20 KB copies are 12.8-16.4 ms in stock.
  Both builds serialize through the same sync copy path
  (`cpy_tensor_async` returns false for cross-device VK->VK).
- GPU kernel time per decode graph is close: 23.81 ms (fork) vs 23.46 ms
  (stock). The fork is FASTER on the large MMVQ nodes (q4_K m=17408: 7.27
  vs 8.16 ms/graph; fused q4_K/q6_K: -0.22/-0.39 ms) and FA, but SLOWER
  on small nodes (MUL 1.43 vs 0.22 ms/graph, SET_ROWS 0.45 vs 0.05,
  MUL_MAT_VEC f32 m=48 1.07 vs 0.69, q6_K m=10240 1.22 vs 0.91,
  q6_K m=1024 0.32 vs 0.04), netting +0.35 ms.
- The remaining ~1.0 ms/graph is host-side pipeline overlap. Wall per
  decode graph is 35.11 ms (fork, 28.48 tps) vs 33.66 ms (stock, 29.72
  tps). With GPU 23.8 and copy 13.1, the fork overlaps only ~1.8 ms,
  stock ~3.6 ms.
- Decisive concurrent check: `GGML_VK_PERF_LOGGER_CONCURRENT=1` leaves
  stock unchanged (29.73 tps vs 29.72 baseline) but costs the fork
  2.9% (27.67 tps vs 28.48 baseline). The stock already overlaps the two
  GPU pipelines; the fork's inter-submit/event scheduling does not.

### Verdict

- Not FA, not MMVQ kernels, not copy cost, not graph/split count.
- The fork delays: (1) a small kernel-time penalty on narrow/host-shaped
  ops (MUL/SET_ROWS/f32 m=48/q6_K m=1024/10240), and (2) primarily a worse
  two-GPU pipeline overlap between the layer-split halves.
- The current fork (master 2026-08-28) and the pre-RPC fork (276121b7e,
  2026-08-20) decode within 2% of each other, so the RPC/D132 series is not
  the cause; the overlap difference predates it (local D094-D131 base or
  the upstream drift).
- Next concrete candidates: event/fence sequencing in
  `ggml-backend.cpp` `compute_splits` (fork has extra async/guard branches
  vs stock's simple sync path), fork `vk_backend_execution.inc` submit
  batching and `almost_ready` fence handling, and per-op dispatch of small
  shapes (`vk_dispatch.inc`).

## 8. Single-GPU isolation (2026-08-28) - real location of the delay

The split-trace experiment revealed that the stock `-dev Vulkan1,Vulkan0
-sm layer` 4K decode actually runs only ONE GPU split behind a CPU split
(two splits total, single `l_out` copy per graph). The fork runs three
splits (CPU + Vulkan1 + Vulkan0) with 13 inter-device copies. To remove
the scheduling variable entirely, both builds were rerun on a single GPU
(`-dev Vulkan1`), L0, 64 tokens:

| build        | decode tps | wall/dec | GPU/dec graph |
|--------------|-----------:|---------:|--------------:|
| fork 1 GPU   | 19.15      | 52.2 ms  | 53.38 ms      |
| stock 1 GPU  | 31.76      | 31.5 ms  | 32.60 ms      |

PERF node breakdown on the identical single-GPU path:

- Every MMVQ op is EQUAL or FASTER in the fork (q4_K m=17408: 84.3 us vs
  88.1 us; q5_K/q6_K/fused -0.2..-0.4 ms per graph).
- FA 4096 is dramatically FASTER in the fork (31.1 us vs 90.1 us).
- But `GET_ROWS` is 121.3 us vs 4.77 us (+2444%) and `CPY` is 115.1 us vs
  3.67 us (+3035%). Combined they are 22.98 ms per graph in the fork vs
  0.66 ms in stock - exactly the 20.8 ms/graph gap (43% of fork GPU time).

These are the GDN conv-state `GET_ROWS conv_states-<il>` and
`CPY state_update_target-<il>` nodes (ne0=4096 f32, one per layer pair).
The pipeline/prologue code (`ggml_vk_op_f32`, `ggml_vk_op_get_pipeline`,
`get_rows.comp`, `cpy` path) is textually identical upstream between fork
and stock, and the fork covers these nodes with the same count; the cost
difference therefore comes from either the fork's local dispatch/sync
changes or from local fork kernels/pipeline selection in this small-op
path (local D094-D131 Vulkan work). On the dual-GPU split these ops were
hidden on the CPU split, which is why earlier dual-GPU PERF runs could not
see them.

Next step: instrument/count `ggml_vk_sync_buffers` invocations around
these nodes and compare the fork `ggml_vk_op_f32`/`ggml_vk_dispatch_pipeline`
with the upstream implementation, then port the minimal difference from
b10666 (or revert the local small-op dispatch change). Fixing this one
path should bring the fork single-GPU decode to ~31 tps, and with the
dual-GPU split already at 28.5 tps, the two-GPU lane would then exceed the
stock single-GPU result.

## 9. Deep-dive into the 30x CPY/GET_ROWS gap (2026-08-28 evening)

Single-GPU full-layer isolation  completed. Starting from the two clean
1-GPU baselines (fork 19.15 tps / 53.38 ms GPU, stock 31.76 tps /
32.60 ms GPU), the fork loses exactly 22.1 ms/graph to two node types:
`GET_ROWS` (98 x 121 us vs 97 x 5 us) and `CPY` (97 x 114 us vs
96 x 3.3 us). Everything else is equal or faster in the fork.

A per-op shape trace (`GGML_VK_SHAPE_TRACE`, both builds, reverted after
use) proved the shapes are identical: 3 MiB `node_XXX` GET_ROWS
`(786432,1,1,1)`, 120 KB `conv_states-XX` GET_ROWS `(30720,1,1,1)`,
3 MiB `cache_s_lXX` CPY `(128,128,48,1)->(786432,1,1,1)` and small
`state_update_target` CPY. All buffers are Vulkan device buffers
(`Vulkan1`) on both sides. Buffer counts and per-graph submit counts are
also similar (fork 2633 vs stock 3303 submissions over the run; same
~180-node batches), so submission batching is not the cause either.

Exhaustive source comparison found the dispatch layer functionally
identical: `ggml_vk_op_f32`, `ggml_vk_op_get_pipeline`,
`ggml_vk_dispatch_pipeline` (byte-identical), `ggml_vk_build_graph`
(only case-removals/trace diffs), `ggml_vk_tensor_subbuffer` (only
host-handling/size helpers, same for F32), shaders `get_rows*.comp`
(byte-identical after CRLF normalisation), `copy_from_quant`/
`copy_transpose` (identical). Micro A/B of the CPY kernel itself via
`test-backend-ops` on the 402 MB bf16 case is also identical
(455.4 GB/s fork vs 457.8 GB/s stock), i.e. the raw kernel is not slower.

The one hard difference found: the fork's `vk_op_unary_push_constants`
layout and `generic_unary_head.glsl` predate upstream commit
`1a7718b4c "vulkan: support non-contig unary/glu ops (#24215)"`
(2026-06-13, not present in the fork). The fork uses six full-width
fastdiv `L` fields (param1/param2); the (newer) stock packs them into
two `Ls` words (param1..param4). Consequently the generated
`cpy_f32_f32.spv` and `contig_cpy_f32_f32.spv` differ between builds,
while `get_rows_f32_f32.spv` is identical. This is the single observable
runtime divergence on this path, but the 402 MB micro-AB is bandwidth
bound and showed no difference, so the layout is only a candidate, not
yet a proven cause of the 30x.

A full port of `1a7718b4c` is the next concrete action: it touches ~18
shaders that include `generic_unary_head.glsl` (clamp, cos, log, sin,
sqrt, square, scale, diag, roll, tri, repeat/back, l2_norm, copy*,
contig_copy, copy_transpose_02), the `vk_op_unary_push_constants` C++
struct + init functions (fastdiv/offsets), `ggml_vk_cpy_to_contiguous`/
`to_strided`, and the unary pipeline creation macro. It also replaces the
fork's 20 separate unary `.comp` files with upstream's single
`unary.comp`, so this is a full upstream-merge of the unary/glu path,
not a one-liner. Everything measured above (traces) has been reverted;
both trees are `git diff --check` clean.

## 10. Outcome: root cause was missing MTP/NextN load gating

### Final A/B (single-GPU isolation, `-dev Vulkan1`, L0/L1, bench2)

| build | L0 decode | L1 decode | L0 prefill | L1 prefill |
|---|---:|---:|---:|---:|
| stock b10666 (GCC16) | 31.68 tps (r3: 31.64/31.70/31.68) | 19.63 tps | 760 tps | 711 tps |
| fork pre-RPC GCC16 (pre-port, pre-fix) | 19.25 tps | – | 132 tps | – |
| fork after `1a7718b4c` port, before MTP fix | 19.24 tps | – | 133 tps | – |
| fork after MTP fix (`82dbc4f01`) | **32.90 tps (r3: 32.92/32.84/32.95)** | 19.66 tps | 134 tps | 134 tps |

So the original short-context gap is closed and reversed: L0 +3.9% vs stock,
L1 parity (+0.2%). The full `1a7718b4c` unary/GLU port by itself changed
nothing (19.24 vs 19.25 tps); the decisive change was the MTP/NextN fix.

### Root cause

The fork was missing upstream `82dbc4f01` "llama : load MTP tensors only if
they are really used (#26296)" (b10212). The fork's qwen35 loader created the
whole appended MTP decoder block (`blk.64`: attn_norm/post_norm, q/k/v, wo,
q/k norms, FFN, six `nextn.*` tensors ≈ 276 MiB) even with `--spec-type none`.
`load_tensors` offloaded 66/66 layers and the Vulkan model buffer was
15662.89 MiB (of 16304), 0 MiB truly free; after the fix the same 15 tensors
are reported `unused tensor ... -- ignoring` (matching stock exactly) and the
model buffer drops to 15386.07 MiB. The decode graph itself does not execute
`blk.64`, but the loaded tensors occupied the single device's VRAM and
coincided with the previously measured 30x `GET_ROWS`/`CPY` costs and the
19.2 tps decode; freeing them restored the 22+ ms/graph of those costs to
stock-like levels, moving decode to 32.9 tps. The GET_ROWS/CPY "host
dispatch" analysis in §8/§9 remains a description of the effect, not the
cause.

Changes ported (accepted, upstream-minimal): `load_mtp` in
`llama_model_params`/loader (wired from `common.cpp` speculative types),
`TENSOR_SKIP` on MTP layers when `!load_mtp` for `qwen35`, `qwen35moe` and
`glm-dsa` (glm-dsa previously skipped NextN unconditionally, now only when
the MTP sidecar is not requested); `llama-quant.cpp` passes `load_mtp=true`.
`cohere2moe`/`hy-v3`/`step35` upstream hunks do not apply (files absent or
already unconditional-skip in the fork). Validated: `--spec-type none` skips
all `blk.64` tensors + decode improves; `--spec-type draft-mtp --spec-draft-
n-max 3` still loads both GPUs (Vulkan0 8207.60 / Vulkan1 7455.29 MiB) and
runs the speculative pipeline. `git diff --check` clean; server build
(winlibs GCC16, Vulkan) passes.

### Remaining open items

- Dual-GPU production lane (`-dev Vulkan1,Vulkan0 -sm layer -ts 1,1`) is
  STILL behind stock after the MTP fix. Repeated r3 (2026-08-28, commit
  572cdc0f7, one server per run, `-c 131072`, decode tps mean):

  | L | fork dual (r3) | stock dual (§1) | stock dual §6 GCC16 r3 | delta vs §1 |
  |---|---|---:|---:|---:|
  | 0 | 26.79 | 30.01 | 29.72 | -10.7% |
  | 1 | 27.47 | 29.67 | 29.44 | -7.4% |
  | 2 | 25.49 | 27.77 | 27.26 | -8.2% |
  | 3 | 23.61 | 25.03 | – | -5.7% |
  | 4 | 22.29 | 23.92 | – | -6.8% |

  (L2 shot 1 decoded only 13 tokens and is excluded; valid shots 2-3.)
  A fresh same-day dual-L0 r1 (server `-c 8192`) gave 29.30 (-2.4% vs §1):
  dual numbers swing with session state. Critically, the fixed fork is now
  SLOWER on dual-GPU (26.8) than on single-GPU (32.9) at L0, while stock is
  only 30.01 dual vs 31.68 single — i.e. the fork loses ~18% enabling the
  second GPU, stock ~5%. This isolates the remaining loss to the two-GPU
  pipeline overlap, and supports the already-listed candidates: fork
  `ggml_vk_submit` holds `queue_mutex` (stock does not), `almost_ready` /
  event sequencing, and small-shape dispatch in `vk_dispatch.inc`
  (§7 `GGML_VK_PERF_LOGGER_CONCURRENT` result).

  Same-day stock r3 comparison (2026-08-28, `dual-stock-l0-l4-r3`,
  server crashed at L4, only L0/L1 usable; `stock-vk-r3` 13:39 L0-L2):

  | L | fork r3 | stock r3 same-day | delta | stock r3 13:39 | delta |
  |---|---|---:|---:|---:|---:|
  | 0 | 26.79 | 28.88 (20:30 run) | -7.2% | 29.72 | -9.9% |
  | 1 | 27.47 | 28.54 (20:30 run) | -3.7% | 29.44 | -6.7% |
  | 2 | 25.49 | – (crashed) | – | 27.26 | -6.5% |

  For L2-L4 fork r3 still compares vs §1 1-run stock: -8.2% / -5.7% / -6.8%.
  Fork prefill dual stays well ahead of stock L2-L4: 1575/1391/1243 vs §1
  1204/1002/860 tps (+31%/+39%/+45%).

### Dual-GPU serialization identified (split timing)

`GGML_SCHED_SPLIT_TIMING=1` on dual-L0 decode shows 3 splits per graph:
CPU (1 node), Vulkan1 (1986 nodes), Vulkan0 (1862 nodes). Per token
steady-state: Vulkan1 `compute_async` (CPU) ~2.4 ms; Vulkan0 `copy`
= **13.1-16.0 ms on a 20 KB `l_out-32` transfer** (mostly
`ggml_backend_synchronize(input_backend)` waiting for GPU1's full
execution; GPU1 is done only after its ~13 ms GPU run), Vulkan0
`compute_async` ~2.4 ms, then GPU0 executes ~13 ms. Result: the two GPUs
run STRICTLY SERIALLY (GPU1 13 ms → host copy → GPU0 13 ms) with zero
token-level overlap; the whole 26-28 ms of GPU work is the same as
single-GPU (~30 ms), and the mismatch is confirmed from both
`cpy_tensor_async` implementations: both the fork and stock refuse
cross-device async copies (`src_buf_ctx->dev_buffer->device !=
dst_buf->device → return false`), so both take the same sync fallback
(`ggml_backend_synchronize(input_backend)` + host copy). The residual
fork-vs-stock delta on dual (~2-3 ms/token) therefore is NOT the copy
path but the surrounding per-token overhead (cmdbuffer build, submit
chain, final sync), not yet bisected; the §7 candidates (queue_mutex,
almost_ready, small-shape dispatch) remain the working hypotheses.

Conclusion: with `-sm layer` the dual-GPU decode is structurally
serial, so no -sm-layer tuning can beat single-GPU decode at L0/L1;
the only real split-mode lever is token pipelining (PP across devices,
currently only speculative/MTP), or accepting layer-split for the
long-context/KV-bound lanes where prefill already wins.

### Update (2026-08-28, E274 out of date): pipeline parallelism re-enabled

The fork had an E274-era guard disabling `pipeline_parallel` for any
Vulkan model with `nextn_predict_layers > 0` (i.e. EVERY `-sm layer`
run on the Qwen3.8 GGUF, including `spec=none`), because in July it
produced `sched copies=4` and was slower. Stock b10666 has NO such
guard, so stock dual-GPU ran with the scheduler pipeline parallelism
enabled — this is the reference behavior the fork was missing. The
guard was removed; `LLAMA_MTP_PIPELINE_PARALLEL=0` keeps the opt-out.

A/B (same-day, dual `-dev Vulkan1,Vulkan0 -sm layer`, r3):

| L | PP off | PP on | stock same-day | stock 13:39 |
|---|---:|---:|---:|---:|
| 0 | 26.79 | **28.66 (+7.0%)** | 28.88 (parity) | 29.72 (-3.6%) |
| 1 | 27.47 | **28.11 (+2.3%)** | 28.54 (parity) | 29.44 (-4.5%) |
| 2 | 25.49 | 25.40 (ns) | – | 27.26 (-6.8%) |

Prefill unchanged/improved (L0 1505, L1 1648, L2 1600 tps); VRAM fine
(Vulkan1 7178 / Vulkan0 6054 MiB free at startup, PP copies are small).
PP closes the short-context decode gap to parity with same-day stock
and keeps the 49K lane unchanged — so the remaining L2 -6.8% is GPU
compute (attention/MMVQ at 49K), not submission overhead.

MTP smoke with PP on runs both contexts with `pipeline parallelism
enabled` and does NOT crash, but `draft acceptance rate = 0.00000`
(0/27, 0/123) — identical to the PP-off run of the same session
(0/39, 0/183), i.e. a PRE-EXISTING MTP regression on the
Qwen3.8-27B-Q4_K_M + current build (worked 2026-08-14 at 81.8%
acceptance on VK 49K), unrelated to this change; needs a separate
diagnosis (suspects: load-mtp gating / a40af6dd2 guard / draft path).
- Single-GPU prefill is pre-existing and NOT touched by this work: fork 134
  tps vs stock 711-760 tps at L0/L1 (pre-RPC binary shows the same 132 tps),
  while dual-GPU fork prefill is healthy (1495 tps) and, per §1, faster than
  stock. Needs a separate investigation (single-GPU diagnostic-only per
  AGENTS.md) — likely related to fork-local GDN/fused shape handling, not
  to 1a7718b4c or the MTP fix.

Artifacts: `build_logs/bench/{fix-skip-mtp-fork-l0-r3,fix-stock-adjacent-l0-r3,
fix-skip-mtp-fork-l1-r1,fix-stock-adjacent-l1-r1,fix-skip-mtp-fork-dual-l0-r1,
diag-fork-prerpc-l0-r1}`, binaries in `bench2-bins/{fork_vk_gcc16,
stock_vk,fork_vk_pre_rpc_gcc16}`.
