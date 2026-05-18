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
| H20 | RDNA4 large Q3_K cuBLAS compute16 route | The current `ubatch=2048` Qwen prefill route dequantizes Q3_K weights to fp16 and then forces `CUBLAS_COMPUTE_32F` on RDNA4; `CUBLAS_COMPUTE_16F` may improve large GEMM throughput if the extra fp16 output conversion is cheaper than fp32 accumulation | 0% to +5% prompt, about 0% to +3% wall if GEMM-bound | precision/quality risk and extra dst fp16->fp32 conversion may erase the gain | env-gated `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` A/B on current `ubatch=2048` cold lane |
| H21 | RDNA4 large-prefill hipBLASLt route | E045 moved the prefill hotspot to large `cublas_backend` Q3_K/Q4_K GEMMs; `ROCBLAS_USE_HIPBLASLT=1` may select better large GEMM kernels even though it was neutral on the old `ubatch=192` MMQ-heavy lane | 0% to +4% wall if large GEMM route-bound | rocBLASLt route may be slower or increase startup/plan overhead on Windows | same-session `ROCBLAS_USE_HIPBLASLT=1` A/B on `ubatch=2048` cold lane |
| H22 | RDNA4 large-prefill cuBLAS split timing | E046/E048 rejected broad compute/library toggles, so the next useful evidence is to split the `cublas_backend` path into src0 dequant, src1 conversion, GEMM, and output conversion costs | higher experiment yield; identifies whether a future speedup should target dequant, GEMM selection, or activation staging | timing syncs perturb runtime and are diagnostic-only; no speed claim from trace timing alone | env-gated `GGML_TRACE_CUBLAS_SPLIT_TIMING=1` on `ubatch=2048` trace lane, then choose the next code candidate by measured share |
| H23 | RDNA4 Q3_K dequant-dominated prefill shape MMQ route | E049 shows the Q3_K `row_diff=6144, ne10=5120, ncols=2048` shape spends about `76-78%` of local time in src0 dequant on the cuBLAS route; a narrow MMQ route might beat dequant+GEMM even though broad forced-MMQ was negative | 0% to +2% wall if this one shape improves enough | broad MMQ already regressed, and MMQ may be slower than the saved dequant; shape-specific route must not touch GEMM-dominant FFN/QKV shapes | broad forced-MMQ timing for target shape, then env-gated shape-only route if local timing passes |
| H24 | RDNA4 Q3_K fp16 dequant thread geometry | E049 shows Q3_K src0 dequant is a large repeated prefill cost; the current Q3_K dequant kernel uses 64 threads/block and each thread writes four values, so a 128-thread/two-value variant may expose more parallelism on RDNA4 | 0% to +2% wall if Q3_K dequant is arithmetic/latency limited | more threads may reduce occupancy or add overhead; correctness must remain bit-equivalent enough for fp16 dequant | env-gated `GGML_CUDA_Q3K_DEQUANT_128=1` prototype, r1 gate, revert if not positive |
| H25 | RDNA4 Q3_K cuBLAS staging decomposition | E053 confirmed the large-prefill Q3_K src0 stage is still the highest remaining measured target, but current `src0_ms` includes both temporary fp16 buffer allocation and Q3_K conversion/store | improves experiment selection; only code a dequant/layout candidate if the conversion kernel, not allocation/staging, is the real local bottleneck | extra timing syncs are diagnostic-only; allocation timing may be CPU-side and should not be mixed with kernel wall claims | env-gated split-detail trace that reports `src0_alloc_ms` vs `src0_convert_ms` and the same for `src1`, then choose allocation/reuse vs dequant-layout work |
| H26 | RDNA4 Q3_K fp16 half2 store conversion | E054 shows Q3_K `src0` time is almost entirely conversion/store, and the current 64-thread kernel emits four scalar fp16 stores per thread; preserving the 64-thread geometry but writing two `half2` pairs per thread may reduce store/conversion overhead without repeating the rejected 128-thread geometry | 0% to +2% aggregate if store instruction count is material; must show a large local conversion drop to be kept | compiler may already coalesce scalar stores, half2 arithmetic may increase instruction pressure, or local gain may be below the E053 `>=25%` gate | env-gated `GGML_CUDA_Q3K_DEQUANT_HALF2=1` prototype, r1 full-lane gate, then split-detail trace if runtime is not clearly negative |
| H27 | RDNA4 Q3_K explicit scalar unroll4 | E056 established that r3-confirmed small wins should be allowed as stackable opt-in knobs, so a minimal explicit four-store variant is worth screening even if it cannot clear the old `>=2%` standalone gate | 0% to +1% aggregate if the runtime loop inhibits scheduling | compiler may already unroll the loop or explicit stores may increase register/instruction pressure | env-gated `GGML_CUDA_Q3K_DEQUANT_UNROLL4=1`, r1 screen, r3 only if r1 is promising |

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
16. H20 was rejected in E046: `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1` slowed the large-prefill lane (`11.7908 -> 11.4146 TPS`, prompt eval `1205.145 -> 1145.945 tok/s`), so the current fp32-accumulation cuBLAS path stays.
17. H21 remains watchlist-only after E058: `ROCBLAS_USE_HIPBLASLT=1` improved aggregate r3 by `+0.42%` on `ubatch=2048`, but median task TPS was below control and the control run had a slow outlier. Do not promote as default/profile yet; re-evaluate only as part of a broader GEMM-route stack.
18. H22 completed in E049: keep env-gated split timing as a diagnostic. It shows Q3_K traced calls are roughly `32% src0 dequant / 7% src1 convert / 61% GEMM`, with one dequant-heavy `6144x5120@ncols2048` shape at `78% src0 dequant`.
19. H23 was rejected in E050: forced-MMQ for the dequant-heavy target shape was slower (`1839.27 -> 2529.35 ms`, `+37.52%` local), so do not add a shape-specific large-prefill Q3_K MMQ route.
20. H24 was rejected in E051: the env-gated 128-thread Q3_K fp16 dequant variant scored `11.46 TPS`, below both `11.6534` r3 baseline and `11.92` same-session default smoke, so the code was reverted.
21. H25 completed in E054: Q3_K `src0` cost is conversion/store, not allocation. Q3_K `src0_alloc_ms` was only `6.12 ms` (`0.18%` of src0), while `src0_convert_ms` was `3370.32 ms` (`99.80%` of src0). The target `6144x5120@ncols2048` shape had effectively zero allocation cost and `1430.88 ms` conversion time, so the next branch should target guarded Q3_K fp16 conversion/layout in `convert.cu`.
22. H26 was rechecked in E056 after adopting the small-gain policy. E055 r1 looked mildly positive (`11.86 TPS`), but same-session r3 did not confirm it: control `11.6726 TPS`, half2 `11.6375 TPS` (`-0.30%`). The prototype code was reverted.
23. H27 was rejected in E057: explicit scalar unroll4 scored `11.58 TPS` in r1, below the same-session E056 control `11.67 TPS`, so no r3 was run and the code was reverted.

## Evidence Snapshot (E006 Retest)

- Supported by measured evidence: H11 as the allocator/residency root cause for the native `ub904/1024` cliff, H08 as a boundary/cliff symptom class, H09.
- Supported as modeling-next-step: H10.
- Analytic-only so far: H02.
- Plausible but not measured yet: H03, H04, H06, H07, H13.
- Measured but not a current cold-first default direction: H01 and H05 after E026.
- Prototype measured and promoted to default for eligible TKV lanes. Smoke `pp64/tg8` improved `turbo4_0` from `186.69/17.09` fallback to `227.88/24.82` direct. Corrected active-lane `v2-review` at `ub=1024` shows Turbo4 hybrid below q4 but much closer; after specialized `TKV4 set_rows`, `q4_0=11.17 TPS`, `turbo4=10.38 TPS` (`-7.1%`). Mixed opt-in `turbo4/q8_0` measured `10.60 TPS` (`-5.1%`) with larger KV. Diagnostic `ub=192` remains useful only for direct-vs-fallback (`turbo4 direct=6.68`, fallback=3.10 TPS).
