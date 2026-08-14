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
| W09 | done | [KV vs Infinity Cache](W09_KV_VS_INFINITY_CACHE.md): KV = 128 KiB/token (64 layers, 4 KV heads x 256, fp8) -> 3 GiB per GPU at 49K vs 64 MB L2; KV stream is use-once per token; H80 = streaming cache-policy hints |
| W08 | done | occupancy closed in W03: wave32, 156 VGPR -> 6 waves/SIMD = 24/CU; 2 CTAs per CU already, LDS-bound, 3 CTAs need <= 21,845 B |
| W10 | done | [MMVQ gemm audit](W10_MMVQ_GEMM_SHAPES.md): decode = MMVQ M<=4/K=5120/N<=17408; prefill = MMQ stream-k; hipBLAS off the hot path; follow-ups: Q3_K batch cap 1, small-K toggle consolidation |
| W11 | done | [backend debt audit](W11_BACKEND_DEBT_AUDIT.md): dead diagnostics (Vulkan FA P2-P5/NATIVE_DECODE/HALF_CMP, census), live fallbacks (do not remove), removal order for phase 3 |
| W12 | done | [decode-token census](W12_DECODE_TOKEN_CENSUS.md): MUL_MAT >= 50% of a 49K decode token (weight stream IS the bottleneck), FA ~10-20%, GDN ~13.5%; next candidates = MMVQ/MMQ weight-stream, then GDN; FA-level micro-opts demoted |

Track status: PAUSED 2026-08-14 after W12 (user switch to Qwen 3.8 27B
support). Resume pointer = W12 "Direction set" section.

## Phase-2 candidate shelf (exhausted 2026-08-14, see PHASE2_PLAN)

All phase-2 candidates were tested and rejected/blocked (H80 toolchain-blocked,
H79 neutral, SR-requant worse NMSE, H77 premise falsified + regression). The
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

- `Qwen3.6-27B-Q4_K_M.gguf`, q8_0/q8_0 production KV; f8_e4m3/f8_e4m3 for
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
