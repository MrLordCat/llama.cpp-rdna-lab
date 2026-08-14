# W04: softmax and P-requant instruction stream (W5/W6 detail)

Date: 2026-08-14

Source: disasm of the production kernel (softmax region, offsets 2230-2360 of
the n8 body) + `fattn-wmma-f16.cu` + `fp8.cuh`. No GPU runs.

## Verified instruction stream (rowmax -> exp -> rowsum -> requant)

| Step | ISA | Notes |
| --- | --- | --- |
| mask add + offset | `v_dual_fmac_f32 v, s8, vx` pairs (VOPD) | slopeh x mask fused add, dual-issued; FATTN_KQ_MAX_OFFSET added via `v_dual_add_f32` with literal |
| rowmax (in-register) | `v_max3_num_f32` + 5-step `ds_bpermute_b32` chain | max3 = 2 max ops in one instruction (RDNA3+); 5 bpermute steps = 32-lane log reduction, wave32 confirmed |
| exp | `v_mul_f32` by 0x3fb8aa3b (log2e) then `v_exp_f32` | hardware exp (pseudo-scalar transcendental op); two FTZ thresholds: 0xc2aeac50 (-87.34) and 0xc1a00000 (-20.0), both via `v_cmp_*` + `v_cndmask_b32` |
| rowsum | `v_dual_fmac_f32` pairs + 5-step `ds_bpermute_b32` chain | dual-issued accumulate |
| P prescale | `v_mul_f32` by 0x43000000 (**128.0**, the D098 p_f8_scale) | applied to exp result before packing |
| P requant | **`v_cvt_pk_fp8_f32`** (2 per warp slice) | packs 2 lanes -> 2 E4M3 bytes; source: `ggml_cuda_fp32x2_to_f8_e4m3_p` -> `__hip_cvt_float2_to_fp8x2(f, __HIP_SATFINITE, __HIP_E4M3)` in fp8.cuh:47 |
| byte split + store | `v_lshrrev_b32 8` + `ds_store_b8` | low/high byte of the packed pair goes to P_f8[k] / P_f8[k+warp_size] in LDS |
| scheduling | `s_delay_alu instid0/instid1` hints everywhere | compiler-inserted VALU latency scheduling |

## Findings

1. The softmax phase is already fully optimized at the instruction level:
   VOPD dual-issue, `v_max3_num_f32`, hardware `v_exp_f32`, minimal bpermute
   steps, packed hardware fp8 conversion. There is no obvious per-instruction
   waste left in this phase.
2. The requant depth is 2 `v_cvt_pk_fp8_f32` per warp slice — the packed
   native conversion, not the scalar software encoder (`v_cvt_f32_fp8` +
   bit ops). W6 answer: **cvt_pk is already in use**; the only untried
   requant instruction is `V_CVT_SR_FP8_F32` (stochastic rounding, W01).
3. The softmax phase cost (14.3% per D102) is therefore dominated by the
   fp32 KQ tile LDS reload + the two barrier pairs around it, not by the
   arithmetic itself.
4. P_f8 store: 2 `ds_store_b8` per lane-pair slice; the B-fragment layout
   (wave32 B: lane={row[3],col[3:0]}, byte=row[1:0], W01) is what makes the
   PV phase read 16 `ds_load_u16` per chain afterwards (W03).
