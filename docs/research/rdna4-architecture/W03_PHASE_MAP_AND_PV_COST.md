# W03: chunk-loop phase map and PV-vs-KQ cost model

Date: 2026-08-14

Source: static ISA of the production decode kernel
`flash_attn_ext_f16<256,16,8,128,float,0,0,0,1,1,0>` (tu23 disasm) mapped onto
the D102 `clock64()` phase boundaries in `fattn-wmma-f16.cu`. No GPU runs.

## Kernel geometry (verified from source + ISA)

- 256 threads = 8 warps x **wave32**; `FATTN_KQ_STRIDE = 256` rows per chunk
  (inferred from `kqs_padded = 264` and the D098 29,568 B shared budget,
  both consistent with the ISA).
- Shared: KQ_or_V 16,896 B (fp32 KQ tile / fp32 VKQ parts union) + P_f8
  4,224 B + VKQ 8,448 B = 29,568 B.
- Accumulators: `KQ_c[1]` + `VKQ_c[2]` fragments = 8 + 16 = **24 VGPR per
  lane** (wave32 WMMA C/D = 8 VGPR per fragment, W01); Q_b = 32 VGPR; the
  other ~100 VGPR are staging, addresses and row state.

## Phase composition (D102 census shares at 49K decode)

| Phase | Census share | Contents (per 256-row chunk) |
| --- | --- | --- |
| KQ | 28.3% | 2 iterations x 16 serialized fp8 WMMA (one 16-deep D-chain each), scalar u8 K loads issued before each chain, 2 fp32 `store_matrix_sync` to LDS |
| softmax | 14.3% | fp32 KQ reload from LDS, mask add, rowmax via `ds_bpermute` + `v_max`, exp, rowsum reduce, **packed P requant (2 `v_cvt_pk_fp8_f32` + P_f8 store to LDS)**, rowsum rescale |
| PV | 55.6% | P_f8 B-fragment loads **from LDS** (16 `ds_load_u16` per chain), scalar u8 V loads interleaved with WMMA, 32 fp8 WMMA over 2 accumulator chains, fp32 `store_matrix_sync` to LDS |
| merge | 2.0% | fp32 VKQ parts reload from LDS, /128 + max-scale rescale, f32->f16 convert, VKQ accumulate |

## Why PV costs 2x KQ for the same 32 WMMA

The census measured 55.6% (PV) vs 28.3% (KQ) for equal WMMA counts. Verified
structural differences:

1. **B-operand source (corrected 2026-08-14).** The production source
   preloads all P_f8 B-fragments (KQ_b) into registers BEFORE the V loop
   (fattn-wmma-f16.cu PV section); the disasm confirms the 16 `ds_load_u16`
   are batched, not interleaved (median distance ds_load->next wmma = 226
   instructions in TU23). So PV's B IS in registers, like KQ's Q_b. The
   2x gap does NOT come from LDS B-fragment latency (H79 premise as written
   was wrong; the phase-2 candidate restructures V_a prefetch instead).
2. **V_a loads inside the chain**: V_a A-fragments arrive as scalar u8 global
   loads interleaved with the WMMA stream - the only per-chain operand fed
   from global memory directly into the mma chain.
3. **Two accumulator chains in PV** (VKQ_c[2]) vs one in KQ — more hazard
   bookkeeping, but also the only place where independent WMMA work exists.
4. **PV tail**: the fp32 `store_matrix_sync` of both accumulator chains into
   LDS before the barrier; KQ's tail is the same shape but half the size.

The LDS B-fragment path is the dominant structural suspect for the 2x gap;
it is exactly the P_f8 staging that D098 introduced so the fp8 V-MMA can
consume the same cache format that the KV cache stores. The merge roundtrip
(fp32 VKQ parts) is NOT the cost: it measures 2.0%.

## Register budget (static, from source + ISA metadata)

| Item | VGPR/lane |
| --- | --- |
| KQ_c (1 x 16x16 fp32, wave32 = 8 VGPR/frag) | 8 |
| VKQ_c (2 x 16x16 fp32) | 16 |
| Q_b B-fragments (16 x 16x16 fp8, wave32 = 2 VGPR/frag) | ~32 |
| K/V/P staging frags + addresses + row state + masks | ~100 |
| total | 156 |

Accumulators are 24 of 156 VGPR; the working set dominates. VGPR is NOT the
occupancy limit: at 156 VGPR the wave32 model gives 6 waves/SIMD = 24/CU,
so even 3 CTAs (24 waves) fit the register file. **LDS is the binding
constraint**: 3 x 29,568 = 88,704 > 64 KiB, so the kernel runs 2 CTAs per
CU, and reaching 3 CTAs needs LDS <= 21,845 B per CTA (a -26% LDS cut).

## Candidate directions (phase-2 shelf, updated)

- Prefetch P_f8 B-fragments one chain ahead (registers already hold the next
  fragment set; the LDS latency is the target) - the PV phase's LDS-latency
  serialization is the single largest modeled win.
- Keep P in registers across the softmax->PV handoff instead of the LDS
  roundtrip, at the cost of registers (no occupancy penalty: VGPR is not
  the limit; LDS is) - compatible with the 3-CTA LDS cut.
- `V_CVT_SR_FP8_F32` stochastic requant: quality gate first (D098 NMSE).
- 3 CTAs per CU: cut LDS from 29,568 B to <= 21,845 B (shrink/remove the
  P_f8 tile and the VKQ f16 accumulator tile, or split the KQ_or_V union
  reuse).
