# R001: RDNA4 architecture exploitation for the ROCm backend

Date: 2026-08-14

Status: active research track. Phase 1 (research-only) is closed; phase 2
(builds + gated experiments) is planned item-by-item in
[PHASE2_PLAN.md](PHASE2_PLAN.md) and executed strictly in that order.

## Objective

`ggml-cuda` was designed around NVIDIA hardware: 32-thread warps, NVIDIA
shared-memory bank/padding rules, `cp.async` pipelining, NVIDIA tensor-core
fragment layouts, and Ampere-style occupancy. The ROCm/HIP path compiles the
same sources, so every one of those assumptions maps onto gfx1201 through a
compatibility shim — and shims cost cycles.

This track studies RDNA4 (gfx1201, RX 9070 XT) on its own terms: catalogue
the architectural features the current HIP kernels do not exploit, verify
each suspicion against the actual ISA of the hot kernels, and only then
admit bounded experiments. The goal is a set of evidence-backed,
gfx1201-native optimizations that do not regress correctness and that keep
the CPU/Vulkan/ROCm-only backend policy intact.

This track supersedes the D102/D103 single-phase gate: the 49K decode token
is dominated by non-FA work (~55% unknown remainder), so per-kernel
micro-gates on the FA phase alone cannot carry the program. Architecture
work is evaluated on the whole locked lane.

## Known budget (why the program is worth it)

- FA kernel is ~20% of a 49K decode token; the weight stream (Q4_K_M,
  ~17 GB read per token) and the unmeasured remainder dominate.
- The largest unexploited levers are structural, not instruction-level:
  occupancy, wave mode, scalar-load streaming, LDS layout, barrier overlap,
  cache policy for KV.
- Any candidate must clear the existing `>=3%` decode gate on the locked
  lane; ISA-level wins below the gate are documented as negative results.

## CUDA-centric assumptions to audit (work items)

| # | Assumption in ggml-cuda | RDNA4 reality to verify in ISA | Probe |
| --- | --- | --- | --- |
| W1 | 8 warps of 32 threads per block (`threadIdx.y` = warp id) | wave32 native, wave64 mode available on gfx12; wave64 halves barrier count and enables 64-wide reductions | ISA + occupancy |
| W2 | Shared layout padded for NVIDIA 32x4B banks (`D_padded = 264`) | LDS 128 B/clk, swizzle 4/8/16 B modes, different conflict pattern | bank-conflict model on the census kernel |
| W3 | `cp.async` double-buffered tile loading | no async copy on gfx12; the idiom is `s_load_dwordx16` (scalar path, no VGPR) + `s_barrier_signal` | what the K/V tile loader compiles to |
| W4 | Tensor-core fragment layouts (m16n8k16 row-major) | `v_wmma` wave32 with KAB/KBA layouts and gfx12 swizzle; check permute cost at fragment store | disassembly of the PV/merge store |
| W5 | `__syncthreads` for every pipeline stage | `s_barrier_signal/wait` split barriers overlap the merge with the next tile loads | ISA + timing |
| W6 | Softmax rowmax/rowsum through shared memory | `v_permlane`/`bpermute`/`ds_swizzle` in-register reductions | softmax-phase ISA |
| W7 | Scalar requant/masks as generic ALU | VOPD dual-issue makes paired VALU nearly free; gfx12 packed `v_cvt_pk_f32_f8` / `v_cvt_pk_f8_f32` | check whether requant already uses cvt_pk |
| W8 | Block size / regs-per-thread from Ampere occupancy tables | 16 wave slots, 64 KiB LDS, 256 KiB VGPR per CU; 2 CTAs at 29568 B LDS today | occupancy spreadsheet from ISA vgpr count |
| W9 | L2 is a dumb cache | 64 MB Infinity Cache; decode re-reads the KV tail every step — residency/streaming hints (`SLC`, cache policy) | KV read pattern vs L2 size at 49K/98K |
| W10 | hipBLAS gemm as a black box | RDNA4 fp8/bf16 WMMA rates vs hipBLASLt path; MMVQ gemm size mismatch | benchmark the MMVQ gemm shape |

## Method

1. Extract the gfx1201 ISA of the hot kernels from `ggml-hip.dll`
   (`llvm-objcopy` the embedded `hip_fatbin` section, then
   `llvm-objdump -d --mcpu=gfx1201`) and keep the dumps as artifacts.
2. For each W# above: verify the assumption against the ISA and count the
   shim cost (extra instructions, barriers, permutes, spills).
3. Admit only candidates with a modeled decode win `>=3%` on the locked lane;
   run A-B-A with adjacent controls and 98K confirmation.
4. Phase 1 is research-only (no builds, no benchmarks - GPUs are also
   reserved for the subproject): ISA dumps of the existing binary, official
   AMD documentation, and source reading only.
5. Phase 2 (GPUs free, 2026-08-14): follow `PHASE2_PLAN.md` in order - each
   experiment clears a focused correctness gate, an A-B-A on the locked
   lane, and a 98K confirmation before acceptance.

## Progress (phase 1)

| Item | Status | Findings |
| --- | --- | --- |
| W00 | done | extraction recipe + kernel roster; production decode kernel = `flash_attn_ext_f16<256,16,8,128,float,...,native=1,1>` (8 warps x **wave32**) at 156 VGPR / 46 SGPR |
| W01 | done | [gfx1201 constants verified](W01_GFX12_VERIFIED_CONSTANTS.md): both wave sizes supported, VOPD wave32-only; this build is wave32 (empirical: 45 VOPD in the production kernel, 5-step reductions); wave32 WMMA C/D = 8 VGPR/lane; D-matrix hazard serializes WMMA chains; LLVM occupancy model (1024 VGPR units/SIMD, granule 16, 16 waves/EU) |
| W02 | done | [kernel ISA audit](W02_KERNEL_ISA_AUDIT.md): production kernel is ALU-dense (70% VALU, 17.6% waits) with 45 VOPD pairs in the softmax/merge math; K/V tiles load as scalar per-lane `global_load_u8` (no async copy, no vectorization); barriers already split; reductions via `ds_bpermute_b32` |
| W03 | done | [phase map + PV cost model](W03_PHASE_MAP_AND_PV_COST.md): PV = 2x KQ per WMMA because its B-fragments (P_f8) arrive from LDS inside the chain; merge fp32 roundtrip is only 2.0%; accumulators are 24 of 156 VGPR, working set dominates; **2 CTAs per CU already achieved, LDS is the binding constraint (3 CTAs need <= 21,845 B)** |
| W04 | done | [softmax + requant stream](W04_SOFTMAX_AND_REQUANT_STREAM.md): phase already fully optimized at instruction level (VOPD, `v_max3_num_f32`, `v_exp_f32`, packed `v_cvt_pk_fp8_f32`); cvt_pk already in use; only untried requant op = stochastic `V_CVT_SR_FP8_F32` |
| W09 | done | [KV vs Infinity Cache](W09_KV_VS_INFINITY_CACHE.md): KV = 128 KiB/token (64 layers, 4 KV heads x 256, fp8) -> 3 GiB per GPU at 49K vs 64 MB L2; KV stream is use-once per token; H80 = streaming cache-policy hints -> CLOSED-REJECTED 2026-09-07 (toolchain can now express TH, measured per-element NT regression -28.2% prefill / -22.8% decode) |
| W08 | done | occupancy closed in W03: wave32, 156 VGPR -> 6 waves/SIMD = 24/CU; 2 CTAs per CU already, LDS-bound, 3 CTAs need <= 21,845 B |
| W10 | done | [MMVQ gemm audit](W10_MMVQ_GEMM_SHAPES.md): decode = MMVQ M<=4/K=5120/N<=17408; prefill = MMQ stream-k; hipBLAS off the hot path; follow-ups: Q3_K batch cap 1, small-K toggle consolidation |
| W11 | done | [backend debt audit](W11_BACKEND_DEBT_AUDIT.md): dead diagnostics (Vulkan FA P2-P5/NATIVE_DECODE/HALF_CMP, census), live fallbacks (do not remove), removal order for phase 3 |
| W12 | done | [decode-token census](W12_DECODE_TOKEN_CENSUS.md): MUL_MAT >= 50% of a 49K decode token (weight stream IS the bottleneck), FA ~10-20%, GDN ~13.5%; next candidates = MMVQ/MMQ weight-stream, then GDN; FA-level micro-opts demoted |
| W13 | C1 measured; C1b confirmed, lever exhausted | [decode MUL_MAT/MMVQ weight-stream audit](W13_DECODE_MUL_MAT_WEIGHT_STREAM.md): Q4_K ncols==1 decode = 8 warps x 8 rows (small_k auto policy), 3 K-iterations, ~40-45% of peak BW. C1 (small_k=0, opt-in): 49K +2.7-4.1% vs 98K noise -> not default. C1b (staged x/gate reduce, default): shared 14336->7168 B, occupancy 50->100%; fresh Linux A-B-A 2026-09-07 +1.10% decode (22.53 base vs 22.78 cand, prefill neutral); shared/occupancy lever fully exhausted - residual gap is weight-stream BW (next candidate), no C1b-family variant remains |
| W16 | K-stream prefetch REJECTED | [MMVQ K-stream prefetch](W16_MMVQ_K_STREAM_PREFETCH.md): software prefetch of the next K-block in `mul_mat_vec_q` (MLP hypothesis) - `__builtin_prefetch` expressible, smoke byte-identical, but A-B-A = -0.64% decode (22.6221 vs 22.7676 controls); 3-iteration loop at 100% occupancy is not MLP-limited; reverted. Lesson: next MMVQ candidate must change memory system or geometry, not add instructions |
| W17 | ROCm 10 feature scan + profiler tooling ADOPTED | [ROCm 10.0 feature scan](W17_ROCm10_FEATURE_SCAN.md): vector `th:TH_LOAD_NT` (b64/b128) expressible on LLVM 23 (confirmed via probes) but the only measured use (H80) regressed - closed; no prefetch ISA on gfx1201; `hipEventDisableTiming`/`hipMemGetDefaultMemPool`/SM-partition/CK-a8w8 not applicable to our decode. Adopted `rocprofv3` per-graph/kernel trace + `scripts/research/rocprof_summary_top.py`; first device profile: MMVQ family ~72.6% of kernel time (fused q4_K 32.15%), GDN 1.51% (W14 confirmed below model), FATTN 0.56% - weight-stream direction confirmed as primary target; copy/quantize cluster next |
| W19 | Decode bottleneck DIAGNOSED (SPM) | [decoder bottleneck](W19_DECODE_BOTTLENECK_DIAGNOSIS.md): rocprofv3 PMC on decode burst: MMVQ = 86.7% of kernel time (fused q4_K 45.9%, 110 calls/token); SQ_BUSY_CYCLES ~470k at ~3.2-3.4 GHz boost = 93-100% of cycles -> suspected compute/issue-bound (explains 59% peak, W16 prefetch and C1b occupancy exhaustion); ~19% wall is outside kernels (~700 launches/token). W18 follow-up: removing ~50% of DP4A (dot2 hoist) had ~0 effect -> memory-system/issue limited, not ALU |
| W18 | INT4 dot8 / dot2-hoist measured (no-loss) REJECTED | [exact INT4 dot8](W18_INT4_DOT8_EXACT_PLAN.md): dot8 = 2x MAC/instr at same IPC, but exact 8-bit activation decomposition (lo+16*hi) needs same instruction count; dot2-hoist (dmin sum out of row loop, bit-identical) A-B-A = +0.08% decode (noise) -> DP4A is NOT the limiter, memory-system/geometry is. Next: GL2C_EA_RDREQ* request-size profile, then weight-stream geometry |
| W20 | MXFP4 decode MEASURED +15.3% (first R9700-path win) | [MXFP4 decode](W20_MXFP4_DECODE_MEASURED.md): in-tree MXFP4 (4.25 BPW vs 5.01) on Qwen3.8-27B, L2 A-B-A (30609/256, f8 KV, spec none, ROCm dual): decode 25.7168 vs 22.3044 control = **+15.30%**; prefill -0.76% noise (MMQ DP4A density same); gain matches -15.2% weight bytes -> CONFIRMS W19 revised: decode is bandwidth-limited (W16/W18 already rejected instruction/memory-parallelism fixes; MXFP4 simply streams fewer bytes). Model `models/Qwen3.8-27B-MXFP4-requant.gguf` (13.85GB, requant Q4_K caution). Next: NVFP4 A/B, native BF16->MXFP4 quality check, MTP+MXFP4 combo |
| W21 | Quality + matrix audit: WMMA-I8 = 5.9x DP4A, MXFP4 PPL +0.41 | [matrix WMMA-I8](W21_MATRIX_WMMA_I8.md): (1) MXFP4-requant PPL 7.1937 vs Q4_K 6.7802 (+0.41 / +6.1%, requant+no-imatrix worst case; native BF16->MXFP4 needed for a quality claim). (2) gfx1201 has `wmma_i32_16x16x16_iu8_w32_gfx12` (int8 x int8 -> i32) but NO native FP4 WMMA; llama.cpp already uses WMMA f16/bf16 for MMQ, int types stay DP4A. (3) microbench: WMMA-I8 **185-188 TMAC/s vs DP4A 31.9 T = 5.85-6.3x**. (4) MMVQ decode can't use WMMA (ncols=1, BW-bound per W20); only MMQ prefill could benefit. Next candidate (W22): env-gated MMQ-i8-WMMA for MXFP4 (A-fragment layout already int8; B raw q8_1) |
| W22 | Matrix audit CORRECTION: MMQ prefill already uses WMMA-I8 | [MMQ prefill WMMA-I8](W22_MATRIX_PREFILL_AUDIT.md): source + trace show `mma.cuh:1200` (wmma_i32_16x16x16_iu8_w32_gfx12) is wired into MMQ via `mmq_select_vec_dot(use_dp4a=false)` -> prefill MMQ on RDNA4 ALREADY runs INT8 WMMA (not DP4A). W21's "int types stay DP4A" corrected. MMQ is only ~7.8% of prefill (W17) so no prefill port to win; matrix hardware is already utilized for prefill. Decode remains the only unlucky path (ncols=1, no WMMA possible; BW-bound W20). Next real levers: native MXFP4 quality, MTP/acceptance, weight-stream geometry, GDN |
| W23 | WMMA-I8 prefill MEASURED +18% MXFP4 (but only if MMQ is used) | [MXFP4 MMQ WMMA-I8](W23_MXFP4_MMQ_WMMA_I8_PREFILL.md): current L1/L2 (batch 8192/ubatch 1024) use hipBLAS for prefill (MMQ cut at ne11<=128 for MXFP4 -> ZERO mul_mat_q calls in trace). Forcing MMQ (GGML_CUDA_FORCE_MMQ_RUNTIME=1, WMMA-I8): L1 prefill 2276.99 vs 1894.25 = **+20.2%**, L2 2122.14 vs 1797.15 = **+18.1%**; decode unchanged. Q4_K same force = -3.5% (G08 confirmed: hipBLAS right for Q4_K). +20% is MXFP4-specific (single E8M0 scale, int8 A, no dequant). NEXT: route MXFP4 large ubatch to MMQ (env default 128, propose default 4096 after 3-run A-B-A) |
| W24 | MXFP4 decode geometry ACCEPTED (+3.0-5.0%); MTP draft precision NEGATIVE; native BF16->MXFP4 quality gate NOT passed (+5.29% PPL) | [MXFP4 decode nwarps + MTP draft](W24_MXFP4_DECODE_NWARPS_MTP_DRAFT.md): MTP/acceptance lever CLOSED-NEGATIVE - up-quantizing the draft head-only or the whole final block (`blk.64.*=Q6_K`) leaves acceptance ~63% and slows decode (BW-bound stack); the acceptance gap is NOT draft-precision. Weight-stream geometry lever POSITIVE: RDNA4 ncols=1 `calc_nwarps` whitelist omitted MXFP4 (it ran 1 warp; Q4_K/Q6_K use 8). Adding MXFP4 -> nwarps=8: L1 decode 29.762 vs 28.342 avg control = **+5.0%**; L2 26.252 vs 25.478 = **+3.0%**; MTP L2 49.195 vs 48.317 = **+1.8%**. Prefill neutral. NVFP4 deliberately not whitelisted (unmeasured). W24b: native BF16->MXFP4 (HF Qwen/Qwen3.8-27B, upstream converter, no imatrix possible: `quantize_mxfp4` ignores quant_weights in fork AND upstream) PPL = **7.1391** vs Q4_K 6.7802 = **+5.29%** (vs requant 7.1937 = -0.76% only) -> quality gate NOT passed; keep Q4_K_M as quality baseline, MXFP4 = speed-only |
| W25 | MXFP4/NVFP4 quality-for-bytes OPTIMAL (+3.1%); MTP acceptance profile; MMQ gap | [MXFP4/NVFP4 quality + acceptance profile](W25_MXFP4_QUALITY_ACCEPTANCE_DEEP.md): (1) best quality-per-byte = **MXFP4-hybrid-attnQ6** (attn/output/token_embd -> Q6_K, rest MXFP4): PPL **6.9925 ±0.05375 (+3.13% vs Q4_K)**, 4.85 bpw; bit-identical 2 runs; L2 decode 24.26, prefill 1770. **NVFP4-native** (4.50 bpw): PPL **6.9840 ±0.05201 (+3.01%)** but prefill 489 t/s (-4x). Both far better than pure MXFP4 (+5.29%). (2) MTP acceptance **exact profile** `#acc rate/pos = (0.800, 0.600, 0.432)`, mean 2.83; n2 41.574, n4 44.155, p_min .5 44.246 -> n3 optimum 49.195; acceptance ~66% is draft-state/calibration ceiling, NOT a tunable knob. (3) small_k for MXFP4 measured REJECTED (25.835 vs 26.252 = -1.6%, reverted). MMQ routing gap open (closed in W26) |
| W26 | MXFP4 prefill -> MMQ/WMMA-I8 by default ACCEPTED (+17-21% prefill, decode neutral) | [MXFP4 MMQ routing 3-run A-B-A](W26_MXFP4_MMQ_ROUTING_ABA.md): `ggml_rdna4_mxfp4_mmq_max_ne11()` default 4096 (env override) + explicit MXFP4 case in RDNA4 `should_use_mmq`; NVFP4/Q4_K unchanged. 3-run A-B-A (batch 8192/ubatch 1024): L1 prefill **2253.06 vs 1864.56 avg = +20.8%**, L2 **2073.57 vs 1766.20 = +17.4%**; decode 29.375 vs 29.379 / 25.889 vs 25.894 = **0.0%**; aggregate L1 +9.6%, L2 +10.8%. Adoption confirm no-env: L2 2076.46/25.92. Q4_K control skipped (model removed by user; code path unchanged, W20 1813/22.30 stands), later rerun on UD-Q4_K_M (W26b, no regression). Combined W24+W26: L1 prefill +18.9%, L2 aggregate +5.1%. MMQ on RDNA4 already WMMA-I8 (W22), MXFP4 A-fragment already MMA layout - correctness fine |
| W28 | Linux L3 (98K) format sweep; MTP acceptance drops; r1 records | [W28 L3 98K sweep](W28_L3_98K_SWEEP.md): `ctx=98304`, synthetic 64,287/256, batch 8192/ubatch 1024, f8_e4m3, ROCm1,ROCm0. None: Q4_K_M(UD) 1539.60/20.80/4.735; **MXFP4-requant(UD) 1765.49/22.60/5.362 (+14.7% prefill, +13.3% aggregate)**; dense MXFP4 1769.05/22.90/5.387; hybrid 1718.59/21.63/5.199; NVFP4 473.85/21.75/1.736 (unusable). MTP n3: Q4 1329.83/33.27/4.568 (64.5%); MXUD 1502.55/33.68/5.081 (**52.4%** vs 66% at L2) - MTP aggregate below spec-none at 98K. Background Minecraft rerun `*-r0-mc` archived (prompt TPS -8%..-38% vs clean `r1`). W24/W26 wins hold at 98K |
| W30 | MTP acceptance long prompts: KV f16 tail 16 layers REJECTED (prefill cost) | [W30 MTP acceptance long prompt](W30_MTP_ACCEPTANCE_LONG_PROMPT.md): target = acceptance. `LLAMA_VK_MTP_KV_LAST_F16=16` (auto was 12) -> MXFP4 L3 acceptance **52.3% -> 67.9%** (171/252, 5 identical runs, deterministic), decode 33.8 -> 38.4 (+13.6%), **but prefill 1505 -> 1415 (-6%)** -> REJECTED (prompt eval is the primary objective; aggregate 5.09 -> 4.92). KV32/64 identical; KV12 flaky; KV8 57.3% - boundary 16 layers at 98K. Window expansion (2048..32768), host handoff, DEFER=0, full-context prefill all REJECTED (flaky one-offs 87-90% or no gain). L2 keeps auto 8 (KV16 gives 57.9% vs 66.0% - wrong); Q4 L3 KV16/KV32 still flaky (54-78%) - open |
| W14 | C3 audit done | [GATED_DELTA_NET decode cost audit](W14_GATED_DELTA_NET_DECODE_COST_AUDIT.md): S_v=128, H_v=48 (dt_rank), 49 GDN layers (interval 4), state 144 MiB f32 = 3 MiB/layer; decode grid (48,1,32)=1536 blocks x 128 threads, 6 MiB read-modify-write per layer per token, ~3.9 MFLOP - launch/latency bound, NOT bandwidth-bound. The W12 `13.5%` share is sync-inflated; modeled real cost ~5-8%. No GDN prototype before a device-trace run (`GGML_TRACE_GDN_TIMING`); MMVQ weight-stream remains first |
| W15 | C2 closed-rejected | [MTP draft-batch weight-stream audit](W15_MTP_DRAFT_BATCH_WEIGHT_STREAM_AUDIT.md): the MTP draft context runs ONE block + NextN head (not the full model) and the target verification is already batched (`sampled` + all drafts in one decode); draft steps are per-token and auto-regressively dependent, so within-sequence draft batching is impossible and the weight-stream premise is falsified. No code change |

Track status: ACTIVE 2026-08-14 (resumed after the Qwen3.8 rebaseline and
f8-KV fix). W13 source audit is done; measurement starts when the GPUs are
free again - the user reserves the GPUs for now, so no bench/GPU launches
until the next signal. Resume pointer = W13 "Measurement plan"; W14 (C3 GDN
audit) completed 2026-09-07 with MMVQ weight-stream still first.

## Phase-2 candidate shelf (exhausted 2026-08-14, see PHASE2_PLAN)

All phase-2 candidates were tested and rejected (H80 closed-rejected
2026-09-07 after the toolchain block lifted and the ROCm 10 A/B measured a
-28.2% prefill / -22.8% decode regression; H79 neutral; SR-requant worse
NMSE; H77 premise falsified + regression). The
post-W12 shelf lives in W12 "Direction set": decode MUL_MAT/MMVQ
weight-stream candidates first, GDN audit second, FA shelf leftovers demoted
(the untried vectorized fp8 tile loads remain documented there).

## Fences

- Backend policy unchanged: CPU/Vulkan/ROCm only; never restore CUDA.
- Driver-safety rules from AGENTS.md apply to every GPU run.
- No production change without focused correctness, exact-route proof,
  same-binary A-B-A, and 98K confirmation.
- The census and graph-trace tools from D100-D102 remain available and
  default-off.

## Lane

- `Qwen3.8-27B-Q4_K_M.gguf` (primary since 2026-08-14 rebaseline), q8_0/q8_0 production KV; f8_e4m3/f8_e4m3 for
  the opt-in native lanes;
- `ctx=49152,b=8192,ub=1024`, one slot, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`;
- `triage_diff`, seed 42, 128 tokens, `spec=none` (MTP comparisons only
  against an adjacent `spec=none` baseline);
- cold/no-reuse/no-prime/no-warmup, `-fit off`;
- 98K confirmation: `ctx=98304`, same recipe.

## Artifacts

- `isa/` — extracted code objects and per-kernel disassembly.
- `W###_*.md` — per-work-item verification notes.
- `RESULTS.md` — accepted/rejected candidates and their evidence.
