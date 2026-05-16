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
| H14 | C01 q3_K shape-presence + narrow stream-k gating | Shape-scoped kernel toggles are useful only if target `ncols` buckets are present in the exact lane; pre-check can avoid false A/B conclusions | higher experiment yield; fewer non-activated probes | lane/task drift can silently change shape histogram and invalidate targeted knobs | confirm target `ncols` distribution in trace first, then run env-gated micro A/B |
| H15 | RDNA4 MMVQ Q3_K/Q4_K decode fast path | The active Qwen lane still spends steady decode time in q3/q4 matvec routes; an RDNA4-scoped MMVQ launch policy or kernel specialization may reduce per-token decode cost without extra VRAM | +2% to +8% decode if the target bucket is active | extra template variants can increase compile time or regress occupancy | C02 resource trace, `MMVQ type=11 ncols_dst=1` timing, then env-gated A/B |
| H16 | C01 Q3_K MMQ selector/resource pressure | After MMVQ is closed, the active C01 cost is still `mul_mat_q_direct|q3_K`; route-local selector/resource levers may expose a cheap win before deeper kernel work | 0% to +3% if selector-bound; larger only if it points to a deeper q3 compute/load issue | runtime noise and cold-start redistribution can look like gains without target hotspot improvement | fresh post-E013 trace, force/selectors A/B, target bucket timing |
| H17 | RDNA4 MMQ smaller y tile with fewer warps | `mmq_y=128/nwarps=8` is LDS-heavy on RDNA4 Q3_K; pairing `mmq_y=64` with `nwarps=4` preserves write-back geometry while reducing shared pressure | +1% to +4% on C01 if resource placement dominates | lower occupancy/waves could regress other MMQ-heavy lanes | paired C01 r3 plus target trace |
| H18 | C01 Q3_K theory gate before code probes | Q3_K MMQ ideas can be cheaply screened by shared-memory split, tile-count ratio, and loop-structure changes before rebuilding kernels | improves experiment yield; avoids low-ceiling probes | simple model can miss compiler/register effects | run `c01_mmq_q3_theory_gate.py`, then only test candidates with a plausible limiting-term change |
| H19 | RDNA4 F32 cuBLAS GemmEx route | Qwen SSM alpha/beta prompt GEMMs are small F32 `cublas_backend` calls; `GemmEx` can choose a different rocBLAS route than `Sgemm` without changing math | 0% to +1% wall if SSM small-GEMM route-bound | affects all F32 cuBLAS backend GEMMs under the env; route may regress larger F32 shapes | env-gated C01 r1 plus target `MUL_MAT f32 ne=(48,192)` trace |

## Priority (Start Here)

1. H11 is completed and kept: E008 confirms ROCm compute vbuffer chunking fixes the native `ub904/1024` residency cliff.
2. H08 remains useful for symptom triage, but caps/planners are now diagnostic tools rather than the preferred final fix when allocator layout can be repaired. E022/E024 rejected C05 `chunk_size=192` and `chunk_size=128` probes on the current `ub192` C01 lane.
3. H02 because it can be prototyped quickly in scheduler logic.
4. H05 is low-priority on the current C01 lane after E026: FATTN is only `~2.58%` of sync CUDA_NODE time, so a selector/tile probe needs a longer-context or FATTN-heavy lane to have meaningful wall ceiling.
5. H12 implemented as default Turbo4 hybrid path, but remains a performance-tuning track until the remaining `~7%` active-lane q4 gap is closed.
6. H13 remains opt-in and RDNA4-gated. E021 showed that extending the existing staging loop to dense C01 Q3_K is negative (`9.6080 -> 8.6216 TPS`, target avg `+25.9%` slower), so dense Q3 staging should not be promoted without a different staged layout.
7. H14 to reduce C01 trial noise: verify shape presence before shape-scoped kernel prototypes.
8. H09 remains important: E026 showed `ngram-mod 24/48/64` can report `+3.31%` aggregate while coverage is only `0.0167` and the bootstrap verdict is inconclusive. E028 confirmed the same preset as an opt-in C01 win with `10.3689 TPS` vs `9.4890 TPS` (`+9.27%`, positive CI), but coverage is still sparse (`0.040580`) and workload-dependent.
9. H10 to explain cross-mode speculative regressions with measured overhead.
10. H01 should be scoped to repeated/steady workloads, not cold-first C01 default. E026 rejected `ngram-simple`, found `n_match=12` neutral, and E028 confirmed `ngram-mod 24/48/64` only as an opt-in repeated/steady preset.
11. H15 is a narrow follow-up to C02: attempt only env-gated MMVQ Q3/Q4 decode variants and keep/revert by paired runtime + hotspot evidence.
12. H16 is completed as a negative selector/resource screen: simple force-x, stream-k, launch-bounds, and `mmq_y` probes did not produce a target-positive keep candidate; next C01 step should be deeper Q3_K compute/load specialization.
13. H17 is completed and kept for RDNA4: `mmq_y=64/nwarps=4` improves C01 paired r3 by `+2.24%` with target hotspot improvement.
14. H18 is now the required C01 screen for new Q3_K MMQ ideas. Initial gate rejected padded half-scale as low-ceiling and rejected k-pair8 after r1 (`9.59 TPS` vs E015 `9.6080`). E020 compact half-scale confirmed the shared/occupancy theory (`35712 -> 32640`, `1 -> 2` blocks/SM) and improved target MMQ timing, but r3 runtime was inconclusive (`9.6080 -> 9.6017`), so no default code was kept.
15. H19 was rejected in E023: `GemmEx` for RDNA4 F32 SSM calls reduced runtime (`9.6080 -> 9.42 TPS`) and worsened target `MUL_MAT f32 ne=(48,192)` avg timing (`0.1712 -> 0.1850 ms`), so keep `cublasSgemm`.

## Evidence Snapshot (E006 Retest)

- Supported by measured evidence: H11 as the allocator/residency root cause for the native `ub904/1024` cliff, H08 as a boundary/cliff symptom class, H09.
- Supported as modeling-next-step: H10.
- Analytic-only so far: H02.
- Plausible but not measured yet: H03, H04, H06, H07, H13.
- Measured but not a current cold-first default direction: H01 and H05 after E026.
- Prototype measured and promoted to default for eligible TKV lanes. Smoke `pp64/tg8` improved `turbo4_0` from `186.69/17.09` fallback to `227.88/24.82` direct. Corrected active-lane `v2-review` at `ub=1024` shows Turbo4 hybrid below q4 but much closer; after specialized `TKV4 set_rows`, `q4_0=11.17 TPS`, `turbo4=10.38 TPS` (`-7.1%`). Mixed opt-in `turbo4/q8_0` measured `10.60 TPS` (`-5.1%`) with larger KV. Diagnostic `ub=192` remains useful only for direct-vs-fallback (`turbo4 direct=6.68`, fallback=3.10 TPS).
