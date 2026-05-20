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
| H31 | RDNA4 Vulkan Q3_K large-MMQ tile retarget | E075 found the original `wm32-wn32` speedup was invalid coopmat undercoverage, and E076 found no 32k win from valid no-code tiles/shape/KV gates; remaining work must be source-level Q3_K prefill profiling | +2% to +8% prompt eval only if a valid Q3_K large-matmul/dequant/layout change reduces pressure without corrupting logits | invalid tile shapes can look fast by skipping output work; every candidate needs layout validation plus real generation smoke | deeper Q3_K Vulkan prefill trace before any new shader/code probe |
| H32 | Vulkan Q3_K MMQ byte-wise sub4 | Q3_K MMVQ already subtracts the offset with a 32-bit byte-wise bit trick, but Q3_K MMQ still does unpack/subtract/repack in the hot dot loop that dominates prefill | +1% to +4% prompt eval if ALU/register overhead matters | compiler already optimizes it away, or bit-twiddle increases pressure | active-lane E072 `wm32-wn32` A/B with perf logger if promising |
| H33 | Vulkan Q3_K MMQ packed32 loads | Q3_K MMQ still loads four packed16 pairs for each repack even though Q3_K packed32 layout exists and MMVQ already uses it | +1% to +5% prompt eval if load/repack overhead is material | added temporaries/register pressure, or old packed16 loads already coalesce well | active-lane E073 `wm32-wn32` A/B |
| H34 | Q3 12k f16 KV prefill profile | At ctx=12288 the Q3_K_S model fits with f16 KV, avoiding compressed-KV attention overhead that shows up as a large FlashAttention block in Vulkan perf logs | +3% to +7% raw prompt eval | more VRAM, output-length/wall-TPS comparability changes, not viable for long context | E074 Vulkan/ROCm f16 KV A/B |

## Priority (Start Here)

1. H11 is completed and kept: E008 confirms ROCm compute vbuffer chunking fixes the native `ub904/1024` residency cliff.
2. H08 remains useful for symptom triage, but caps/planners are now diagnostic tools rather than the preferred final fix when allocator layout can be repaired.
3. H02 because it can be prototyped quickly in scheduler logic.
4. H05 because prefill IO is still dominant in prompt-heavy scenarios.
5. H12 implemented as default Turbo4 hybrid path, but remains a performance-tuning track until the remaining `~7%` active-lane q4 gap is closed.
6. H13 is the next Stormrage-derived performance idea, but must stay opt-in and RDNA4-gated until MoE A/B proves a win.
7. H30 is completed and kept for Q4_K/Q5_K only: E070 confirms the `ne11<=1024` MMQ selector fixes the downloaded Q4_K_S slow path.
8. H31 remains a Vulkan Q3_K performance track, but exact `wm32-wn32` is closed for promotion: E075 traced its old win to coopmat undercoverage (`BLOCK_SIZE=256` where `1024` invocations were required). E076 then rejected valid no-code follow-ups (`wm128-wn32`, `block128-*`, MMVQ routing, batch/ubatch, `q8_0`/`f16` KV) against the safe 32k force-only baseline. GUI keeps only `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`; next H31 work should start from source-level Q3_K prefill tracing.
9. H32 was tested in E072 and rejected as a no-code-kept micro-optimization; Q3_K MMQ remains the dominant target, but this specific subtract rewrite is not useful.
10. H33 was tested in E073 and rejected as a no-code-kept packed32 load-side rewrite for active Q3_K MMQ.
11. H34 is kept as a 12k Q3 speed profile: f16 KV gives a measured Vulkan raw prefill win, with q4_0 KV retained as the memory fallback.
12. H09 to avoid misleading speculative projections in low-coverage runs.
13. H10 to explain cross-mode speculative regressions with measured overhead.
14. H01 as a low-risk extension of existing ngram flow.

## Evidence Snapshot (E006 Retest)

- Supported by measured evidence: H11 as the allocator/residency root cause for the native `ub904/1024` cliff, H08 as a boundary/cliff symptom class, H09.
- Supported as modeling-next-step: H10.
- Analytic-only so far: H02.
- Plausible but not measured yet: H01, H03, H04, H05, H06, H07, H13.
- Rejected default runtime variant: H31 / E075, Vulkan `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32` recovered GUI 32k wall TPS above ROCm only because the coopmat tile launched 4 subgroups for a 16-subgroup coverage shape. The corrected exact tile passes real output smoke but is slow. E076 rejects the valid 32k no-code follow-ups as well: safe force-only Vulkan stays about `9.85 TPS` / `907 tok/s` prompt, while fresh ROCm control remains `11.06 TPS` / `1155 tok/s` prompt. Keep only `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` as the GUI Vulkan override and use generalized backend tile validation for manual variants.
- Rejected narrow probe: H32 / E072, Q3_K MMQ offset subtract micro-optimization based on the existing MMVQ bit-twiddle; first active gate was a small regression, so the shader patch was reverted.
- Rejected narrow probe: H33 / E073, Q3_K MMQ packed32 load-side rewrite; first active gate was a small regression, so the shader patch was reverted.
- Kept profile change: H34 / E074, Q3_K_S ctx=12288 f16 KV speed preset; raw Vulkan prompt `1230.7333 tok/s` over 3 runs, above same-KV ROCm `1194.22 tok/s`.
- Prototype measured and promoted to default for eligible TKV lanes. Smoke `pp64/tg8` improved `turbo4_0` from `186.69/17.09` fallback to `227.88/24.82` direct. Corrected active-lane `v2-review` at `ub=1024` shows Turbo4 hybrid below q4 but much closer; after specialized `TKV4 set_rows`, `q4_0=11.17 TPS`, `turbo4=10.38 TPS` (`-7.1%`). Mixed opt-in `turbo4/q8_0` measured `10.60 TPS` (`-5.1%`) with larger KV. Diagnostic `ub=192` remains useful only for direct-vs-fallback (`turbo4 direct=6.68`, fallback=3.10 TPS).
