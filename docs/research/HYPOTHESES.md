# Candidate Hypotheses Beyond Current ngram + FlashAttention

This file tracks candidate changes that could unlock additional efficiency.

## Core Equation Used For Screening

Approximate decode speedup from speculative decoding:

`S_spec ~= (1 + a * (D - 1)) / (1 + o)`

Coverage-aware variant (for sparse draft coverage):

`S_spec_cov ~= (1 + (c * a_local) * (D - 1)) / (1 + o)`

Where:

- a: accepted draft token ratio
- D: drafted tokens per verification step
- o: relative overhead of draft generation + bookkeeping
- c: coverage (share of verify steps where draft was present)
- a_local: accepted/attempted inside draft-enabled steps

Approximate total wall speedup:

S_total ~= 1 / (p / S_prefill + (1 - p) / (S_spec * S_decode))

Where:

- p: baseline wall-time share spent in prefill
- S_prefill: prefill speedup (for example from attention/kernel work)
- S_decode: decode kernel speedup not covered by speculation

## Hypothesis Backlog

| ID | Idea | Why It Might Work | Expected Impact | Main Risk | First Check |
| --- | --- | --- | --- | --- | --- |
| H01 | Adaptive ngram length by local entropy | Fixed n can be too short in repetitive spans and too long in noisy spans | +5% to +20% decode | extra control overhead | acceptance rate vs entropy bucket |
| H02 | Dynamic draft length policy | Keep D small when mismatch risk is high, increase D in stable segments | +3% to +15% wall | oscillation and instability | accepted tokens per verify call |
| H03 | Hybrid router: ngram or mtp per step | Different spans favor different draft methods | +5% to +25% wall | routing overhead kills gain | net TPS with router on/off |
| H04 | Early reject bound in verify stage | Fast reject without full path when mismatch is obvious | +2% to +10% wall | quality regressions if bound is unsafe | token agreement and exact output diff |
| H05 | Flash-attn tile retarget for current RDNA path | Better tile mapping can reduce memory stalls | +5% to +20% prefill | compile/runtime complexity | prefill tok/s and occupancy |
| H06 | QKV and RoPE fusion in prefill hot path | Fewer launches and less memory traffic | +3% to +15% prefill | register pressure | kernel time breakdown |
| H07 | KV cache layout tuned for decode locality | Better cache-line behavior in long decode loops | +3% to +12% decode | migration complexity | decode tok/s and memory counters |
| H08 | Chunk-size contract alignment (model op + runtime) | Avoid pathological boundaries that trigger slow paths; current bad zone is physical n_ubatch >480 | +5% to +30% at cliff zones | model-specific behavior and boundary drift | current-best ubatch boundary sweep + physical context cap |
| H09 | Coverage-aware speculative acceptance model | Local draft acceptance overestimates global speedup if draft coverage is low | improves prediction fidelity | extra instrumentation complexity | compare implied vs effective acceptance |
| H10 | Overhead-aware speculative model by mode/config | Fixed overhead term misses severe regressions in some high-coverage MTP cases | improves prediction fidelity | more parameters and overfitting risk | backsolve implied overhead across measured cases |
| H11 | ROCm compute vbuffer chunking | A single large ROCm graph compute allocation can land in a bad RDNA4/Windows residency pocket even when kernel routes are unchanged | removes 3x+ prefill cliffs at large native ubatch | extra backend buffer chunks may add allocator fragmentation or affect non-ROCm backends if scoped incorrectly | single-chunk A/B with full `PP reserve` |
| H12 | Direct/hybrid compressed-KV FlashAttention for local TurboKV | Full graph-dequant fallback is slow; full direct prefill is also slower at large ubatch; Turbo4 currently wants F16/WMMA prefill plus direct TKV decode | implemented; corrected ub1024 Turbo4 gap vs q4 is ~7% for `turbo4/turbo4`, ~5% for opt-in `turbo4/q8_0` | complex graph/backend integration and output equivalence risk | keep hybrid default for Turbo4, keep mixed TKV/Q8 opt-in, tune decode vec-dot and F16 dequant/prefill overhead, continue equivalence validation |
| H13 | RDNA4 MoE/MMQ LDS staging adaptation | Stormrage's RDNA2 MoE accelerator suggests MMQ prefill can benefit from explicit LDS staging, padding, and occupancy tuning; RDNA4 needs a separate gated variant rather than a direct RDNA2 port | +3% to +10% MoE prefill if current RX 9070 XT path is LDS/occupancy limited | wrong kernel route or RDNA4 occupancy regression; dense path regressions | opt-in RDNA4-only A/B on MoE `b=1024,ub=1024` with dense negative control |
| H30 | RDNA4 Q4_K/Q5_K large-prefill MMQ selector | Q4_K/Q5_K dequant+hipBLAS is much slower than MMQ on RX 9070 XT for Qwen3.6-27B-Q4_K_S at `ne11<=1024`, while the old RDNA4 K-quant MMQ gate was capped at `ne11<=192` | +3x to +4x Q4 prompt eval on affected prompt-heavy lanes | Q3_K/Q6_K regressions if broadened too far; Q4/Q5 shape-specific cliffs | Q4 pp512/pp1024 forced-MMQ A/B, default-threshold rerun, old-threshold negative control, Q3 negative control |
| H31 | RDNA4 Vulkan Q3_K coopmat prefill | Kept code is E082 stride18 + E086 corrected Q3_K `LOAD_VEC_A=4`; E102 now auto-enables AMD large matmul on the local eligible AMD proprietary coopmat device. E097-E101 closed invalid/negative nearby routes: `wn48/wn96`, `bn256` large tiles, Q8_1/int-dot, aligned-store cleanup, and Q3_K arithmetic microforms | +2% to +8% prompt eval for stacked structural wins; target-closing work still needs about +20% local Q3_K hotspot speedup vs accepted E086/E102 baseline | invalid tile shapes can look fast by skipping output work; large LDS/register variants and expression-only helper rewrites are mostly exhausted | run `python scripts/research/vulkan_q3k_prebuild_gate.py --candidate "..."`, `python scripts/research/vulkan_warptile_static_scout.py`, and `python scripts/research/spirv_op_summary.py <spv>` before any new shader/tile probe |
| H32 | Vulkan Q3_K MMQ byte-wise sub4 | Q3_K MMVQ already subtracts the offset with a 32-bit byte-wise bit trick, but Q3_K MMQ still does unpack/subtract/repack in the hot dot loop that dominates prefill | +1% to +4% prompt eval if ALU/register overhead matters | compiler already optimizes it away, or bit-twiddle increases pressure | active-lane E072 `wm32-wn32` A/B with perf logger if promising |
| H33 | Vulkan Q3_K MMQ packed32 loads | Q3_K MMQ still loads four packed16 pairs for each repack even though Q3_K packed32 layout exists and MMVQ already uses it | +1% to +5% prompt eval if load/repack overhead is material | added temporaries/register pressure, or old packed16 loads already coalesce well | active-lane E073 `wm32-wn32` A/B |
| H34 | Q3 12k f16 KV decode profile | At ctx=12288 the Q3_K_S model fits with f16 KV; post-driver E116 shows this is useful on Vulkan decode-heavy generation, while E117 rejects it as a ROCm prompt-heavy replacement | +30% decode-route win vs ROCm q4 control when switching backend/KV together; small q4->f16 Vulkan edge | more VRAM, output-length/wall-TPS comparability changes, not viable for all prompt-heavy contexts | E116 Vulkan f16 decode r3 + E118 live-server sanity; E117 ROCm prompt-heavy negative control |
| H35 | ROCm Q3_K direct/persistent prefill route | Current large ROCm prefill already uses hipBLAS after Q3_K -> fp16 staging; E103 proves repeated staging exists, but E104 rejected persistent fp16 cache due VRAM residency pressure and E105 rejected selector-only routing to existing MMQ. A useful new route now likely requires a shape-specific fused Q3_K x F16 RDNA4 GEMM kernel | +1% to +3% wall for `20-30%` local Q3_K conversion/layout win; higher only if fused route also improves GEMM-side locality | fused kernel losing to hipBLAS GEMM, RDNA4 tiling complexity, or insufficient local share after conversion removal | Start from E103 trace and E104/E105 negatives; do not repeat persistent fp16 cache or existing-MMQ selector probes |
| H36 | Prompt-cache/checkpoint repeated-session route | Sequential repo tasks share a large prefix; server prompt cache and context checkpoints can restore the common prefix and avoid most repeated prefill work; E113 shows shorter ngram can stack when repeated-session coverage appears | +20% to +70% repeated/session TPS, zero cold-first gain; post-driver stacked `ngram-mod 12/16/32` reached `19.5051 TPS` | misleading if compared to cold-first claims; depends on prompt similarity and cache memory; ngram benefit is bursty and first cold request can slow down | E111 same-lane reuse A/B; E112 reuse+ngram 24/48/64; E113 post-driver reuse+ngram 12/16/32 |

## Priority (Start Here)

1. H11 is completed and kept: E008 confirms ROCm compute vbuffer chunking fixes the native `ub904/1024` residency cliff.
2. H08 remains useful for symptom triage, but caps/planners are now diagnostic tools rather than the preferred final fix when allocator layout can be repaired.
3. H02 because it can be prototyped quickly in scheduler logic.
4. H05 because prefill IO is still dominant in prompt-heavy scenarios.
5. H12 implemented as default Turbo4 hybrid path, but remains a performance-tuning track until the remaining `~7%` active-lane q4 gap is closed.
6. H13 is the next Stormrage-derived performance idea, but must stay opt-in and RDNA4-gated until MoE A/B proves a win.
7. H30 is completed and kept for Q4_K/Q5_K only: E070 confirms the `ne11<=1024` MMQ selector fixes the downloaded Q4_K_S slow path.
8. H31 remains a Vulkan Q3_K performance track. Kept code is E082 stride18 + E086 corrected Q3_K `LOAD_VEC_A=4`, and E102 makes the fast AMD large-matmul route the default for the local eligible AMD proprietary coopmat device (`GGML_VK_DISABLE_AMD_LARGE_MATMUL=1` is the rollback). Current no-env pp7488 is `983.48`; disable-control is `708.19`. Current 32k controls are Vulkan `10.5230` / prompt `993.94` and ROCm `10.8879` / prompt `1132.44`, so aggregate is close but prompt/prefill still trails. Use `vulkan_q3k_prebuild_gate.py`, `vulkan_warptile_static_scout.py`, and `spirv_op_summary.py` before new probes; E097-E101 already rejected invalid tiles, large LDS/register tiles, Q8_1/int-dot, aligned-store cleanup, and arithmetic-only Q3_K helper rewrites.
9. H32 was tested in E072 and rejected as a no-code-kept micro-optimization; Q3_K MMQ remains the dominant target, but this specific subtract rewrite is not useful.
10. H33 was tested in E073 and rejected as a no-code-kept packed32 load-side rewrite for active Q3_K MMQ.
11. H34 is kept as a 12k Q3 decode-heavy profile, not a ROCm prompt-heavy default. Post-driver E116 measured Vulkan f16 decode r3 at `40.2753 TPS` aggregate / `41.2283 tok/s` decode, and E118 live-server sanity found normal thinking/answers with no `wm32-wn32`-style corruption. E117 rejects f16/q8 KV for the cold prompt-heavy ROCm default (`q4 11.9858`, f16 `11.9028`, q8 `11.6392`).
12. H35 is the ROCm Q3_K route track. E103 proved repeated staging, E106 refreshed the same-lane split, but E104 rejected persistent fp16 cache and E105 rejected existing-MMQ selector overrides. Continue only with a fused Q3_K x F16 kernel design or another route that avoids persistent fp16 residency; do not repeat rejected route toggles (`GGML_CUDA_FORCE_MMQ_RUNTIME`, compute16, hipBLASLt, half2 store, dequant128, Q3_K persistent fp16 cache, existing-MMQ selector override). E106 also adds a workflow correction: split buckets need pre-stage sync/event timing before pure local attribution.
13. H36 is confirmed as a practical repeated/steady route: keep server prompt cache/checkpoints enabled for GUI/agent sessions, but never compare it as a cold-first kernel speedup. After driver `32.0.31007.5012`, E113 supersedes E112's `24/48/64` ngram with `ngram-mod 12/16/32` (`17.8934 -> 19.5051 TPS`, after-first mean `23.9038 TPS`). E114 rejects going shorter to match `8`: local acceptance collapses and decode regresses.
14. H09 to avoid misleading speculative projections in low-coverage runs. E107 shows local acceptance without coverage is not enough: tested cold-first ngram variants had effective acceptance `0.000000` to `0.004908` and lost wall TPS.
15. H10 to explain cross-mode speculative regressions with measured overhead.
16. H01 as a low-risk extension of existing ngram flow.

## Evidence Snapshot (E006 Retest)

- Supported by measured evidence: H11 as the allocator/residency root cause for the native `ub904/1024` cliff, H08 as a boundary/cliff symptom class, H09.
- Supported as modeling-next-step: H10.
- Analytic-only so far: H02.
- Plausible but not measured yet: H01, H03, H04, H05, H06, H07, H13.
- Active Vulkan Q3_K track: H31 / E082 + E086 accepted, E102 defaulted the fast AMD large-matmul path for the local AMD proprietary coopmat device. Current no-env pp7488 is `983.48` on `matmul_q3_k_f32_f16acc_aligned_l`; disabling auto-large drops to `708.19`, proving the rollback gate. E100 fresh 32k controls show the broader gap is now smaller in aggregate but still prompt/prefill-side (`993.94` Vulkan vs `1132.44` ROCm prompt eval), while Vulkan decode is faster (`32.93` vs `28.49`). E095 shows AMD RX 9070 XT runtime has KHR coopmat but not NV coopmat2, so `mul_mm_cm2` is not an active AMD route on this driver. E096 adds SPIR-V opcode fingerprints for generated Q3_K shaders; E097 corrects the warptile static scout to subgroup `64`, coopmat `16x16x16`, and base Q3 LDS `20480 B`. Prebuild screening is required before more H31 probes because many nearby ideas are already measured negative or invalid: `LOAD_VEC_A=8`, pair-scale/helper-only dequant reuse, packed32 pair helper, stride16/19/20/22, f16 dequant, unsigned scale, `wn48/wn96`, `bm256`/`bn256*`, Q8_1/int-dot, aligned-store cleanup, Q3_K shift/mask arithmetic, Q3_K scale-int arithmetic, BK-depth tweaks without resource proof, and old invalid `wm32-wn32`.
- Active ROCm new-route track: H35 starts from the E049/E054 finding that the large Q3_K fp16 hipBLAS path is conversion-dominated for hot prompt shapes: Q3_K `src0_convert_ms=3370.32 ms`, target `6144x5120@ncols2048` convert `1430.88 ms`, allocation effectively zero. E103 proved repeated Q3_K staging (`2792` rows, `349` keys, all repeated `8` times), but the working set is too wide for a small cache and full fp16 cache is too large (`42.002 GiB`). E104 rejected persistent `attn_gate` cache (`11.74 -> 9.56 TPS`, 480 MiB variant `11.59 TPS`) despite real conversion reduction, and E105 rejected existing-MMQ route overrides (`11.74 -> 11.54/11.68/11.44 TPS`). E106 refreshed the same-lane control at `11.8464 TPS` and split trace (`2792` Q3_K rows, `src0_convert_ms=3257.251`, `gemm_ms=6107.363`) but also showed a tracing caveat: without pre-stage sync/event timing, split-stage ms may include earlier queued GPU work. Route toggles and simple conversion kernels are already rejected (`GGML_CUDA_FORCE_MMQ_RUNTIME`, compute16, hipBLASLt, dequant128, half2 store, persistent fp16 cache, existing-MMQ selector override). Only a shape-specific fused Q3_K x F16 kernel still has enough structural ceiling.
- Confirmed repeated/steady route: H36 / E111 keeps prompt cache/checkpoints as a practical session route. Same `ctx=12288,b=6144,ub=2048,q4_0/spec=none` workload measured `14.6132 TPS` r1 and `17.7984 TPS` r3 with reuse enabled; after the first task, repeated tasks ran about `20.00 TPS` because checkpoints restored the shared `5370`-token prefix. E112 stacked `ngram-mod 24/48/64` on that route and measured `18.7194 TPS`, but post-driver E113 retuned the stack: `ngram-mod 12/16/32` measured `19.0148` and `19.5051 TPS`, with after-first mean `23.1681-23.9038 TPS` and effective acceptance `0.035028`. This is a real GUI/agent-session gain, not a cold-first kernel/default claim.
- Rejected narrow probe: H32 / E072, Q3_K MMQ offset subtract micro-optimization based on the existing MMVQ bit-twiddle; first active gate was a small regression, so the shader patch was reverted.
- Rejected narrow probe: H33 / E073, Q3_K MMQ packed32 load-side rewrite; first active gate was a small regression, so the shader patch was reverted.
- Kept profile change: H34 / E116+E118, Q3_K_S ctx=12288 Vulkan f16 decode-heavy route; r3 decode gate `40.2753 TPS` aggregate / `41.2283 tok/s` decode, with live-server output sanity accepted by the user. E117 keeps ROCm prompt-heavy default on q4 KV.
- Prototype measured and promoted to default for eligible TKV lanes. Smoke `pp64/tg8` improved `turbo4_0` from `186.69/17.09` fallback to `227.88/24.82` direct. Corrected active-lane `v2-review` at `ub=1024` shows Turbo4 hybrid below q4 but much closer; after specialized `TKV4 set_rows`, `q4_0=11.17 TPS`, `turbo4=10.38 TPS` (`-7.1%`). Mixed opt-in `turbo4/q8_0` measured `10.60 TPS` (`-5.1%`) with larger KV. Diagnostic `ub=192` remains useful only for direct-vs-fallback (`turbo4 direct=6.68`, fallback=3.10 TPS).
