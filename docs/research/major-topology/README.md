# Major Topology Program Board

This directory is for post-E264 architecture work. It is intentionally upstream
of normal E### benchmarking: use it to rank designs before editing backend code.

## Primary Project Baseline

| Field | Value |
| --- | --- |
| Decision | D089 Q4_K_M primary baseline promotion |
| Latest gate | [D091 Q4_K_M ROCm 98K WDDM placement](D091_Q4_K_M_ROCM98K_WDDM_PLACEMENT_GATE.md), closed: corrected device order recovered MTP n2 prompt `947.46 -> 1426.54 tok/s` (`+50.56%`) |
| Vulkan recovery | [D093 current-driver Q4_K_M wn32 recovery](D093_VULKAN_Q4KM_AMD_WN32_RECOVERY.md), accepted: backend-default `wn32` restored 59K prompt evaluation `401.19 -> 1171.94 tok/s` (`2.92x`) on driver `32.0.31035.1003` |
| Prompt-eval supremacy program | [D094 Q4_K_M Vulkan-vs-ROCm prompt-eval program](D094_Q4KM_VULKAN_PROMPT_SUPREMACY_PROGRAM.md), open, cycle 1 done: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` accepted (`1241.3` vs `1201.5`, +3.31% r3); all other launch-level gates closed negative (ts, low-tile, KV types, FA geometry); FA = 40% not KV-byte-bound, matmul ~46% at dequant ceiling; remaining gap `1.1375x`; cycle 2 = source work (T205 fused GDN/CONCAT first); Gate A (98k/49k re-measure with gfxq) still open |
| FP8 MTP polish | [D095](D095_Q4KM_VULKAN_FP8_MTP_POLISH.md) R1-R9 prebuild complete; [D097](D097_Q4KM_VULKAN_FP8_LONG_ACCEPTANCE.md) fixes the 98K acceptance regression with context-scoped FP8 last12: `1510.95/41.79`, 73.79% acceptance versus q8-center `1422.71/41.96`, 72.60%. It costs 5376 vs 4704 MiB KV. q8 and shorter FP8 remain last8; M6 q8 bridge is default-off generation research. |
| Model | `models/Qwen3.6-27B-Q4_K_M.gguf` (MTP-enabled) |
| Safe lane | ROCm `ctx=49152,b=8192,ub=1024,q8_0/q8_0,-dev ROCm1,ROCm0,-sm layer,-ts 1,1`, cold/no-reuse/no-warmup |
| Control | spec-none `1778.59 prompt / 21.98 decode tok/s`, `5.6829` aggregate TPS |
| Agent profile | MTP n3 `1731.71 / 39.58`, `6.2802` aggregate TPS, `74.36%` acceptance |
| Extended lane | ROCm `ctx=98304` one-copy scheduler with `-dev ROCm1,ROCm0 -sm layer -ts 1,1`; 131K remains a `27:37` placement/residency stress lane |
| KV policy | q8 primary; alternatives require matched Q4 quality and performance gates |

## Secondary Q3-Specific Program

| Field | Value |
| --- | --- |
| Program | P003 dual-Vulkan Q3_K_S 2000 tok/s long-prompt route |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` (non-MTP) |
| Backend | Vulkan primary; ROCm is a later control only |
| Lane | `ctx=131072,b=8192,ub=1024,q8_0/q8_0,FlashAttention,spec=none,real-context-chars=152000,max_tokens=16`, 56,456 measured prompt tokens |
| Reuse/prime | off / off |
| Thinking | on |
| Current best | D080 `-ts 5,6`: cold run 1 `1350.01 prompt tok/s`, r3 mean `1327.82`; target `2000` requires `1.4815x` from the cold baseline |
| Current trace | D079: Q3_K 46.6%, FA 46.4%, Q4_K 4.0%, GLU 1.8%; tensor split rejected at 540.18 vs 1809.02 small-layer control |
| Primary risk | target needs a topology-level gain; do not reopen D029-D033 helper/layout-only Q3 probes |
| Residency | startup reports 6,685 MiB free on Vulkan1 and 7,958 MiB free on Vulkan0; WDDM runtime observation remains mandatory |

## 2000 tok/s Scaling Verdict

Applying the Candidate Selection Rubric to the intuitive "one GPU is about
`1100`, so two should reach `2000`" model. Verdict: the naive parallelism route
is rejected by gate 4 on already-measured evidence, without a new GPU run.

Measured topology curve on the P003 lane:

| Topology | Prompt tok/s | Data-parallel within a ubatch? |
| --- | ---: | --- |
| `-sm layer -ts 5,6` (production, D080) | `1350` (56k) / `1826` (12k) | no; layers are pipelined |
| `-sm tensor`, native BF16 all-reduce (D084) | `1042` (12k) | yes, but communication-bound |
| `-sm tensor`, generic reduction (D084) | `540` (12k) | yes, worst |

Why `2x` cannot come from adding the second GPU here:

- Layer split is not data-parallel; it pipelines layers, so it cannot be `2x`
  single-GPU. It already exceeds single-GPU via pipeline overlap and better
  residency.
- Tensor split is the only data-parallel topology, but Qwen3.6 needs 127
  all-reduce boundaries per 1024-token ubatch, the Windows AMD driver exposes
  the two cards as two singleton device groups (no peer collective), and every
  boundary is host-mediated at about `3.4 ms` (`~0.43 s/ubatch`). D084/D085 also
  exclude `VK_KHR_external_memory_win32`, D3D12 cross-adapter, and Windows
  RCCL/NCCL.
- Gate 4 on the best remaining transport candidate (Q8 all-reduce compression,
  D085) is already modeled at `1250-1450` tok/s at 12k, still below the `1826`
  layer result, and "unlikely to reach 2000 by itself". So it does not close the
  target and does not even beat production layer split.

Rubric-selected route to `2000` instead: reduce per-GPU compute in the two
dominant D079 buckets (Q3_K `46.6%`, FA `46.4%`). From layer `1350` the target
is `1.4815x`; only two open families have a real ceiling, and both must clear a
cheap measurement gate before any shader:

1. T5a attention sparsity / K-compression. Scout now exists. Gate 1
   (concentration) PASSED at mid-context (frac75 `0.044`; 99% of mass in 35% of
   keys), but gate 2 (cheap block selector) FAILED at the realistic `Bc=64`
   FA-tile granularity: block-max top-25% recovers only `79.7%` mass and needs
   ~75% of blocks for 99%. block-max ≈ oracle, so the keys are scattered, not
   block-clustered (confirmed at both `Bc=64` and `Bc=16`). Block-sparse FA at
   contiguous tiles is therefore Amdahl-rejected (`<=1.18x`). Only a key-level
   top-k gather kernel could approach the target (~`1.43x` at 99% mass), and it
   needs its own design note plus a GPU-instrumented 56k confirmation.
2. T2/T3a true Q3_K compressed-dot body that reduces matrix work itself, not
   helper arithmetic (nearby Q3 body/layout routes already rejected in
   D030-D033).

Do not spend a GPU run re-confirming the tensor-parallel negative; it is closed
by model, not by a missing measurement.

## Previous P002 Program

| Field | Value |
| --- | --- |
| Program | P002 130k dense Qwen3.6 Vulkan/ROCm residency route |
| Model | `models/Qwen3.6-27B-Q3_K_S.gguf` |
| Backend | Vulkan and ROCm on RX 9070 XT / AMD proprietary driver + HIP SDK 7.1 |
| Lane | `ctx=131072,b=512,q4_0/q4_0,FlashAttention,spec=none,real-context-chars=24576,max_tokens=16`; Vulkan `ub=256`, ROCm `ub=128` until rechecked |
| Reuse/prime | off / off |
| Thinking | on |
| Current best | P002 quick target reached by D012 Vulkan opt-in stack: `b512/ub256` `2.0013 TPS` r3 with `bn256 + lowtile3 + q3quad + GLU fast path`; D036 hardens the D035 host-KV guard with direct pinned host-KV and restores default decode to `40.2033 tok/s` (`1.9410 TPS` r3), but remains below D012; D037 rejects q8 KV as a default speed route and keeps q8/q8 only behind `LLAMA_VK_KV_HOST_AUTO_Q8=1` for stability/offline testing; D038 makes the q4/q3 tool-call reliability guard default-on, disableable with `LLAMA_SERVER_TOOL_CALL_THINKING_GUARD=0`, improving D038 quality from `2/4` to `4/4` without changing the speed lane; D039 transfers that guard to a public BFCL-lite subset (`24/25` default vs `16/25` explicit thinking) and leaves repeated parallel-call undercoverage as the next quality failure class; new active Vulkan target is `2.4 TPS`; D005 default split-K anchor remains `1.7898 TPS` r3; ROCm `b512/ub128` `1.5200 TPS` r3 and paused after D013-D027 |
| Current trace | Vulkan: [P002_VULKAN_130K_EVIDENCE.md](P002_VULKAN_130K_EVIDENCE.md), D003 cliff gate: [D003_P002_VULKAN_UBATCH_CLIFF_GATE.md](D003_P002_VULKAN_UBATCH_CLIFF_GATE.md), D004 FFN ceiling: [D004_P002_VULKAN_FFN_ROUTE_CEILING.md](D004_P002_VULKAN_FFN_ROUTE_CEILING.md), D005 split-K: [D005_P002_VULKAN_FFN_DOWN_SPLITK.md](D005_P002_VULKAN_FFN_DOWN_SPLITK.md), D006 output/residency: [D006_P002_VULKAN_RESIDENCY_OUTPUT_PLACEMENT.md](D006_P002_VULKAN_RESIDENCY_OUTPUT_PLACEMENT.md), D007 FFN block gate: [D007_P002_VULKAN_FFN_BLOCK_ROUTE_GATE.md](D007_P002_VULKAN_FFN_BLOCK_ROUTE_GATE.md), D012 2 TPS stack: [D012_P002_VULKAN_Q3QUAD_GLU_2TPS_STACK.md](D012_P002_VULKAN_Q3QUAD_GLU_2TPS_STACK.md), D028 2.4 target gate: [D028_P002_VULKAN_2P4_TARGET_GATE.md](D028_P002_VULKAN_2P4_TARGET_GATE.md), D029 whole-FFN gate: [D029_P002_VULKAN_WHOLE_FFN_2P4_GATE.md](D029_P002_VULKAN_WHOLE_FFN_2P4_GATE.md), D030 all-Q3 gate: [D030_P002_VULKAN_ALLQ3_2P4_GATE.md](D030_P002_VULKAN_ALLQ3_2P4_GATE.md), D031 Q3S layout-body gate: [D031_P002_VULKAN_Q3S_LAYOUT_BODY_2P4_GATE.md](D031_P002_VULKAN_Q3S_LAYOUT_BODY_2P4_GATE.md), D032 Q3+FA stack gate: [D032_P002_VULKAN_Q3_FA_STACK_2P4_GATE.md](D032_P002_VULKAN_Q3_FA_STACK_2P4_GATE.md), D033 q3-octa gate: [D033_P002_VULKAN_Q3_OCTA_PREBUILD_GATE.md](D033_P002_VULKAN_Q3_OCTA_PREBUILD_GATE.md), D034 residency recheck: [D034_P002_VULKAN_130K_RESIDENCY_RECHECK.md](D034_P002_VULKAN_130K_RESIDENCY_RECHECK.md), D035 default guard hardening: [D035_P002_VULKAN_D012_DEFAULT_GUARD_HARDENING.md](D035_P002_VULKAN_D012_DEFAULT_GUARD_HARDENING.md), D036 direct host-KV decode recovery: [D036_P002_VULKAN_DIRECT_HOST_KV_DECODE_RECOVERY.md](D036_P002_VULKAN_DIRECT_HOST_KV_DECODE_RECOVERY.md), D037 q8 KV stability gate: [D037_P002_VULKAN_Q8_KV_STABILITY_GATE.md](D037_P002_VULKAN_Q8_KV_STABILITY_GATE.md), D038 tool-call guard: [D038_P002_Q4_TOOL_CALL_THINKING_GUARD.md](D038_P002_Q4_TOOL_CALL_THINKING_GUARD.md), D039 BFCL pilot: [D039_P002_BFCL_Q3_TOOL_CALL_PILOT.md](D039_P002_BFCL_Q3_TOOL_CALL_PILOT.md); ROCm D002 paused: [D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md](D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md) |
| Primary risk | 130k KV/context/working set spills beyond 16 GB VRAM into system RAM |
| Pause checkpoint | Paused by user on 2026-05-27 before a public `llama-bench` comparison track. On resume, rerun the D012 same-lane control first; do not compare new candidates against the D034 `~0.37 TPS` slow pocket. |

Previous program P001 (`ctx=12288` Vulkan Q3_K dense prompt route) is historical.
Its kept best is E257 r3 `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s`; E258/E259/E260/E264 closed nearby transfers.

## Current Rejection Fence

Do not reopen these for the 130k program without a new mechanism and a design note:

- Q3_K transpose-A as a single layout for both prompt and decode.
- f16 KV as the archived 12k dense default.
- `batch=7680`, `batch=8192`, graphics queue, `--no-mmap` transfer gates from 12k. For 130k, queue/mmap must be re-tested because residency pressure changed.
- `GGML_VK_DISABLE_F16=1` or broad f32acc/f16-disable pivots.
- Per-layer FFN activation casts to F16.
- Helper-only Q3_K arithmetic rewrites, pair-scale reuse, packed32 helper-only
  rewrites, nearby stride tweaks, and large current-tile variants already
  rejected by H31 history.
- Output-layer placement as a launch/profile fix. D006 showed it can recover
  prompt eval from a 130k residency cliff, but decode falls to about `22 tok/s`,
  so it is diagnostic evidence rather than a route to `2 TPS`.
- Broad backend-host KV placement as a launch/profile fix. D034 showed partial
  `Vulkan_Host` KV can recover the current slow pocket up to `1.9826 TPS`, but
  ordinary host-KV placement pays decode back. D036 keeps only a narrow
  Qwen35-like direct pinned host-KV auto guard for default stability and decode
  recovery; do not treat host-KV sweeps as a speed route.
- q8 KV as a default Vulkan 130k speed profile, and mixed q4/q8 or q8/q4 KV on
  the current RDNA Vulkan lane. D037 showed q8/q8 can fit only with direct
  host-KV relief and falls to about `0.36 TPS` / `185-188` prompt tok/s, while
  mixed K/V creates `34` graph splits without coopmat2 mixed-KV FA support.
  Keep q8/q8 only as explicit stability/offline opt-in via
  `LLAMA_VK_KV_HOST_AUTO_Q8=1`.

## P003 Candidate Queue

| ID | Candidate | Status | Next required artifact |
| --- | --- | --- | --- |
| T11 | Same-lane Q3_K + FA wall decomposition | completed D079 | Q3_K 46.6%, FA 46.4%; neither route can close alone |
| T12 | True Q3_K + long-KV FA body/dataflow route | D081 rejected | `Br32/Bc64` exceeds the exposed 32 KiB LDS limit; compact `Br32/Bc32` passed at 65 VGPR / 32,256 B LDS / zero scratch but changed the two-run prompt center by `-0.98%`. Reopen only with sparsity/KV compression or a non-nearby dataflow mechanism |
| T13 | Vulkan tensor-parallel prefill rehabilitation | D084 opt-in infrastructure kept; rejected as default | native BF16 all-reduce raises tensor from about 540 to `1032-1043`, but remains far below layer `1826`; 127 required host-mediated collectives per ubatch |
| T14 | Hybrid PP tensor / TG layer topology | design-only | prove phase-specific graph/device ownership can avoid duplicate model residency and preserve decode |
| T15 | Reopened 12k Q3_K `BN512` route | rejected D082 | over-LDS route was not selected; `950.35 tok/s`; prototype removed |
| T16 | Vulkan native tensor collective | completed D084 | keep BF16 communicator opt-in; require true peer/device-group primitive before reopening tensor as a target-closing route |
| T17 | RDNA4 compact KV and Q5 FA dequant | D087 packed-bit Q5 path kept | q8_0 is 8.5 bpw and correctly consumes 4352 MiB at ctx131072. q5_1 consumes 3072 MiB, and the bit-exact packed high-bit dequant improves the exact 43k r3 mean `1368.02 -> 1411.60 prompt tok/s` (`+3.19%`); comparable cold-first q5_1 is within `2.69%` of q8 and paired BFCL-lite smoke is `8/8` for both. Keep q5_1 as the compact-KV opt-in and q8 as the maximum-quality reference; only design a new Q6 KV type if broader q5 quality fails |

## P002 Candidate Queue

| ID | Candidate | Status | Next required artifact |
| --- | --- | --- | --- |
| T1 | Dual-layout Q3_K storage: raw decode layout plus prefill layout for matrix routes | design-needed | VRAM/storage model and fail-closed tensor movement plan |
| T2 | Shader-native Q3_K prompt layout: compact signed/scale-expanded block for coopmat matmul | D031 rejects compact Q3S/signed-nibble plus scale-expanded layout-body as target-closing route | Reopen only with a compute body that reduces matrix work itself, not just unpack metadata |
| T3 | Fused dense FFN block: gate/up/swiglu/down tiled route | D029 rejects activation-only and naive streaming whole-FFN routes; D007 still proves the graph surface exists | reopen only if the design reduces Q3_K matmul work itself or becomes part of a broader all-Q3 dataflow; launch/hidden-materialization-only fusion is below the `2.4 TPS` bar |
| T3a | Vulkan all-Q3/body-layout target for `2.4 TPS` | D030 rejects current q3quad extension, scale-only metadata/helper reuse, signed-nibble-only storage, Q8_1/int-dot, expanded layouts, and neighboring tile tweaks; D031 rejects compact Q3S layout-body; D032 shows FA cannot carry alone; D033 rejects q3-octa/`LOAD_VEC_A=8` repeat | First implementation gate remains a true Q3_K body/compressed-dot route; a Q3+FA stack is only useful after Q3 reaches about `1.18-1.20x` local point/static proof |
| T4 | FA shader-body work for long-KV and 130k RAM-spill lane | D076 rejects Bc scaling; the active Windows Vulkan driver exposes a 32 KiB compute-shared-memory limit, and D081 closes exact two-query nearby geometry. D077 concludes wider matmul instructions do not help this memory-bound route; sparsity/K-compression/architectural approaches remain open | rerun evidence if attention sparsity scout shows >50% K/V can be skipped |
| T5 | Vulkan/ROCm differential harness | first Vulkan pack complete | run `bench: vulkan q3 130k route trace`, `bench: vulkan q3 130k q3 stats`, then `research: vulkan evidence pack` |
| T5a | FA memory-bandwidth research: sparse FA, K-compression, or attention sparsity | scout implemented (`llama-attn-sparsity-scout` + `scripts/research/attention_sparsity_scout.py`). Gate 1 (mass concentration) PASSED on Qwen3.6-27B-Q3_K_S at 3,509-token/1,753-valid mid-context: global frac75 `0.044`, frac90 `0.107`, frac95 `0.171`, frac99 `0.350`. Gate 2 (cheap block-max top-k recovers the mass) FAILED at every tested contiguous-block granularity: top-25% blocks recover only `79.7%` mass at `Bc=64` (`99%` needs ~65-75% of blocks). block-max ≈ oracle — the important keys are scattered ~1 per block, not clustered. Gate 3 (gather-FA viability: per-query key-level recovery + tile-union shared-key test) FAILED on both models at ctx=4096/tile=32. 9B: `pq25=0.9603`, `tu25=0.8911` (penalty `+6.9%`). 27B at full 3,509‑token context: `pq25=0.9621`, `tu25=0.8755` (penalty `+8.7%`). 99% gate needs ~50% key budget on both (`pq50=0.9906` 27B, `pq50=0.9908` 9B) — no advantage over dense FA at that ratio | All three cheap sparse-FA families rejected: block-sparse at `Bc>=16` Amdahl-limited (`<=1.18x`), per-query key-level gather fails 99%@25%, tile-union shared-key fails worse. The attention mass is too diffuse at key granularity. Remaining non-matrix FA route: true K-compression (2-3× fewer VRAM bytes per key). Reopen sparse/gather-FA only on a model where pq25 ≥ 0.99. Evidence: `build_logs/agent-workload/attn-scout-{27b-blk64,27b-blk16,gather-9b,gather-27b}.csv` |
| T6 | ROCm low-level Q3_K MMQ/FFN body | D078 keeps a dedicated RDNA4 Q3_K small-N DP4A MMQ route: MTP n3 decode `34.92 -> 41.25 tok/s` and `1.65x` the same-build spec-none short baseline; 131k/56k-token prompt decode `19.02 -> 26.85` (`1.41x`) | next route must target the long-context FA/KV share or the remaining draft overhead; keep N=5 DP4A opt-in because n4 did not improve materially |
| T7 | Vulkan `ub>=320` residency cliff | D003 closed as non-speed recovery | do not promote `GGML_VK_ENABLE_MEMORY_PRIORITY=1`; revisit only with a shader/body/lifetime change that beats `ub256` |
| T8 | Vulkan output-layer residency relief | D006 closed as diagnostic, not speed route | next work must be a source/topology design with a residency model; do not continue ubatch/output-placement sweeps |
| T8a | Vulkan backend-host KV residency relief | D036 keeps a narrow Qwen35-like direct host-KV auto guard for q4/q4 default stability and restores decode to `40.2033 tok/s` (`1.9410 TPS` r3). D037 keeps q8/q8 only as `LLAMA_VK_KV_HOST_AUTO_Q8=1` stability/offline opt-in and rejects mixed q4/q8/q8/q4 on this Vulkan lane; broad D034-D037 KV sweeps remain below D012 | reopen only with a lifetime/migration or mixed-FA design that beats D012, not as a launch/KV placement sweep |
| T9 | Vulkan q3quad/bn256/lowtile promotion hardening | D035 promotes guarded source defaults for AMD `bn256`, Q3_K quad dequant, and Q3 low-tile split-K; D012 `2.0013 TPS` r3 remains the speed comparator | run confirm3 before final default-promotion wording; keep exact D012 r3 as baseline until a matching confirmation beats or ties it |
| T10 | Q4/Q3 tool-call reliability guard | D038 promotes the guard to default-on after improving the D038 q4 tool-call workload from `2/4` to `4/4`; D039 BFCL-lite public subset measured `24/25` default versus `16/25` with explicit thinking; disable with `LLAMA_SERVER_TOOL_CALL_THINKING_GUARD=0` for server A/B or explicit request `enable_thinking=true` for per-request A/B | next implementation should try fallback/retry after no-tool-call/length failures and repeated parallel-call undercoverage, while normal thinking remains available where explicitly requested |

## Candidate Selection Rubric

Derived from the E344/E345 wins versus the E347/D081 loss series. The negative
series was selected top-down (idea or analogy first, then "does it fit"); the
wins were selected bottom-up (measured bottleneck, then measured waste, then a
minimal compile-gated change). Every new candidate must pass all gates below
before a prototype is written.

1. Bottleneck gate. Start from a fresh hot-path trace on the exact target
   lane and backend. The candidate must attack an op that is a documented
   top-share consumer of that trace. No trace, no candidate.
2. Waste gate. Provide a measured waste signal for the proposed mechanism:
   low occupancy, register/LDS spill, redundant traffic counter, or a
   forced-route A/B proving the current selector is suboptimal. A design story
   or analogy is not evidence.
3. Transfer gate. A cross-backend transplant (for example a ROCm win moved to
   Vulkan) requires target-backend evidence that the same bottleneck exists
   there. Do not port because it worked elsewhere.
4. Upside gate. Compute the expected upside by Amdahl:
   `expected = op_share * plausible_per_op_improvement`. Reject if the expected
   upside is below the lane's measured order-swing noise (about `3%` on the
   current two-order cold-first harness) unless the noise is first reduced.
5. Work-volume preference. Prefer changes that reduce matrix/memory work volume
   over helper-arithmetic or tile micro-reshapes on an already-tuned automatic
   route. Reshaping dataflow on a good coopmat path usually trades one overhead
   for another.
6. Feasibility is a filter, not a reason. "Fits in LDS and compiles as
   coopmat1" gates admission only; it never justifies selecting a candidate.

If a candidate cannot pass gates 1, 2, and 4 from existing or cheap-diagnostic
evidence, write a scout/design rejection instead of a benchmark-shaped E### note.

## Required Evidence Pack

Before any source prototype, attach or link:

- paired Vulkan and ROCm 130k lane controls on the current binaries;
- route trace for Q3_K and FA;
- server diagnostics covering startup, mmap/no-mmap, allocator/residency, and RAM pressure;
- `vulkan_perf_shape_summary.py` output for the active trace;
- relevant SPIR-V opcode/resource summaries;
- Amdahl/ceiling estimate for the touched route;
- correctness plan for prompt and decode routes;
- rollback path.

Vulkan helper: `scripts/research/vulkan_evidence_pack.py` converts a benchmark
label into `build_logs/agent-workload/<label>.vulkan-evidence.md`. It summarizes
diagnostics, residency/startup signals, Vulkan route/perf traces, pipeline
resource lines, SPIR-V opcode fingerprints, and optional Amdahl ceiling sketches
via `--baseline-tps`. If route/perf traces are absent, it marks the missing
gates instead of pretending the pack is complete. Use
`--extra-log <q3-stats.server.log>` to merge a separate `--trace-preset
vulkan-q3-stats` resource run into the same pack; the matching VS Code task is
`bench: vulkan q3 130k q3 stats`.

## Branch Discipline

- Use one branch/worktree per candidate topology.
- Keep negative prototypes out of `master` unless they are default-off diagnostic
  tools with a documented reason to keep them.
- If a candidate fails before lane A/B, write a design/scout rejection instead
  of creating a benchmark-shaped E### note.
