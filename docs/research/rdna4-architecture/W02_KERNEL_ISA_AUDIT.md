# W02: ISA audit of the 49K decode kernels (static structure)

Date: 2026-08-14

Source: `ggml-hip.gfx1201.tu23.disasm.txt` (llvm-objdump -d --mcpu=gfx1201 on
the existing build-rocm binary; no new build). Counts are **static** per
kernel body (one instance of each loop, not dynamic execution).

## Instruction histogram

Counts are static per kernel body. The production decode kernel is the **8-warp**
fullnative instantiation (`n8`); the 4-warp (`n4`) instantiation serves the
KQ-only / V-only / non-native D256 routes and is shown for contrast.

| Category | fullnative n8 (production decode) | fullnative n4 (KQ/V-only route) | f16 n4 |
| --- | --- | --- | --- |
| total instructions | 4322 | 7616 | 4990 |
| `s_wait_alu` | 528 (12.2%) | 788 (10.3%) | 683 (13.7%) |
| `s_wait_loadcnt` | 163 | 352 | 212 |
| `s_wait_dscnt` | 52 | 87 | 100 |
| `s_wait_kmcnt` | 12 | 15 | 14 |
| `s_wait_loadcnt_dscnt` | 5 | 5 | 5 |
| all `s_wait_*` | 760 (17.6%) | 1247 (16.4%) | 1014 (20.3%) |
| other ALU (address/mask/math) | 3019 (70%) | 5338 (70%) | 3087 (62%) |
| `v_wmma_f32_16x16x16_fp8_fp8` | 64 | 128 | - |
| `v_wmma_f16_16x16x16_f16` | - | - | 64 |
| `global_load_*` | 320 | 640 | 560 |
| of which `global_load_u8` | 256 | 512 | - |
| `global_store_*` | 18 | 36 | 36 |
| `ds_*` | 103 | 187 | 192 |
| `s_barrier_signal` / `s_barrier_wait` | 8 / 8 | 8 / 8 | 8 / 8 |
| full `s_barrier` | 0 | 0 | 0 |
| VOPD (`v_dual_*`) | 0 | 0 | 0 |

## Interpretations

1. **The kernel is ALU-dense with waits every ~4 instructions.** 70% of the
   static instruction stream is plain VALU (address math, masks, row sums,
   rescale, converts) and 17.6% is `s_wait_*` (12.2% `s_wait_alu`). Per-lane
   dependent chains (fp8 dequant, addressing, requant, final store) leave the
   compiler nothing independent to fill; it must explicitly wait out the ALU
   latency. This is consistent with the D096 diagnosis (scalar fp8 dequant =
   branchy ALU soup) and with D102's P*V = 55% of kernel time: the P*V phase
   is latency-bound scalar work, not WMMA-throughput-bound.

2. **No async copy, no vectorized tile loads.** K/V fp8 tiles arrive via
   per-lane scalar `global_load_u8` (256 static in the production kernel, 512
   in the 4-warp one), Q via `global_load_u16`,
   addressing via `global_load_b32/b64`. No `s_load_dwordx16` staging of
   tiles, no `global_load_b128` + swizzle. Every loaded element passes
   through a VGPR with its own address computation. The WMMA A/B fragment
   layouts (W01) explain why: fp8 fragments need lane-by-byte placement, and
   the code chose direct scalar loads over vectorized loads + redistribution
   (which would need `ds_swizzle_b32` / `v_permlane` passes).

3. **Barriers are already split.** 8 `s_barrier_signal` + 8
   `s_barrier_wait`, no full `s_barrier` — the CUDA-`__syncthreads()`
   semantics were compiled to the cheap RDNA form.

4. **Reductions use `ds_bpermute_b32`** (20 static in the production kernel;
   5-step 32-lane reduction chains) plus a small LDS staging area — softmax
   rowmax/rowsum stay in-register/permute and do not round-trip the whole P
   tile through LDS.

5. **P requant is 2 `v_cvt_pk_fp8_f32`** (packed two-lane conversion, with
   the D098 scale-128 pre-scale). gfx12 also has `V_CVT_SR_FP8_F32`
   (stochastic rounding), an untried quality lever for the E4M3-P NMSE
   problem (phase-2 candidate).

6. **f16 kernel**: d16 loads (`global_load_d16_hi_b16`) are also scalar
   16-bit loads; its 64 WMMA f16 per chunk is expected (fp8 WMMA has half
   the K-panel of f16 per instruction, so the native kernel needs 64 fp8
   WMMA for the same tile).

7. **VOPD is used in the production kernel** (45 `v_dual_*` pairs, all in
   the softmax/merge/requant arithmetic). The kernel is wave32 (VOPD is
   illegal in wave64, W01). The WMMA chains and their operand loads are not
   pairable, so the MMA phases get no dual-issue benefit.

## Candidate directions opened by this audit (phase 2, all need gates)

- Vectorize K/V fp8 tile loads (`global_load_b128` / wider) and redistribute
  into WMMA fragments via `ds_swizzle_b32` — trades VMEM instruction count
  and address VGPRs against LDS traffic and permutes.
- Stage uniform tile metadata on SGPRs (address bases, strides) — already
  partially done; extend so the per-lane address VALU work shrinks.
- Break dependent ALU chains in the requant/store merge (the 55% phase):
  batch converts, use `V_CVT_SR_FP8_F32` only if NMSE improves.
- Evaluate a wave64 variant: reverts VOPD to unavailable and halves the
  per-SIMD VGPR units (1024 -> 512), i.e. strictly worse for this kernel;
  drop from the candidate list unless a wave64-only feature appears.
- Occupancy: 248 VGPR forces 1 CTA/CU; the n8 instantiation (156 VGPR) shows
  the same math could fit 2 CTAs at 8 warps if the fp32-accumulator/working
  set is cut — W04 will size the accumulator and working-set budgets.
