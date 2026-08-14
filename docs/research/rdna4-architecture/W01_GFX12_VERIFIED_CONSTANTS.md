# W01: gfx1201 hardware constants, verified against official docs

Date: 2026-08-14

Sources:

- AMD "RDNA4" Instruction Set Architecture, PDF v1.4, 707 pages, downloaded
  from `docs.amd.com` khub
  (`https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture`),
  cached at `/tmp/rdna4-isa.txt` (page markers `@@PAGE N@@`).
- LLVM AMDGPU Usage (`https://llvm.org/docs/AMDGPUUsage.html`), fetched
  2026-08-14.

## Verified facts

| Fact | Value | Source |
| --- | --- | --- |
| gfx1201 identity | `amdgpu12.01`, dGPU, `wavefrontsize64`, cumode, architected flat scratch, packed work-item IDs; products: RX 9070 / RX 9070 XT / RX 9070 GRE | LLVM processor table |
| wave sizes | both wave32 and wave64 supported; shader compiled for one size | ISA 2.1 |
| wave size in this build | **wave32** kernels (empirical, 2026-08-14): 45 VOPD `v_dual_*` inside the production kernel body, 5-step 32-lane `ds_bpermute` reductions, `vcc_lo`/`exec_lo` only, no `vcc_hi`/`exec_hi`; no `-mwavefrontsize` flag in build files, so ROCm 7.1 hip-clang compiles gfx1201 as wave32 | tu23 disasm + build.ninja |
| VOPD (dual-VALU) | legal only for wave32; must not be used by wave64 | ISA 7.8 |
| VOPD usage | present in the production kernel: 45 `v_dual_*` (softmax/merge arithmetic is dual-issued; WMMA and its operand loads are not pairable) | tu23 disasm |
| wave32 issue | each instruction issues once per wave (the wave64 double-issue cost does not apply to these kernels) | ISA 2.1 |
| SGPRs per wave | 106 normal + VCC pair (106/107) + 16 TTMP | ISA 3.3.1.1 |
| VGPR max per wave | 256 VGPR; allocation in blocks of 16 (wave32) or 8 (wave64) = 512 DWORD units | ISA 3.3.2.1 |
| dynamic VGPR | `S_ALLOC_VGPR` grow/shrink; **wave32 only**; max 8 blocks; block 16 (max 128 VGPR) or 32 (max 256 VGPR) chip-wide config | ISA 3.3.3 |
| LDS | 128 KiB per WGP, 64 banks of DWORD RAM per WGP (2×32 banks, one set per CU-half); each bank 512×32 two-port (1R/1W per clock); all banks can store/load simultaneously; **max 64 KiB per work-group** | ISA 12.1 |
| work-group | all waves of a WG resident on one WGP, any of the 4 SIMD32; WGP supports up to 32 WGs; max 1024 work-items per WG | ISA 2.3 |
| CU mode | LDS split into two 64 KiB halves (one per SIMD32 pair); WG confined to one CU | ISA 2.3 |
| WGP mode | LDS contiguous 128 KiB; waves of a WG may spread over both CUs | ISA 2.3 |
| WMMA A layout (16x16, 8-bit, wave32) | row striped across VGPRs within a lane: lane={col[2],row[3:0]}, vgpr={col[3],col[1]}, byte=col[0] | ISA 7.12.2 |
| WMMA B layout (16x16, 8-bit, wave32) | row striped across lanes within one VGPR: lane={row[3],col[3:0]}, vgpr=row[2], byte=row[1:0] | ISA 7.12.2 |
| WMMA C/D layout (16x16, fp32, wave32) | lane={row[3],col[3:0]}, vgpr=row[2:0] -> **8 VGPR per lane** | ISA 7.12.2 |
| WMMA hazard | WMMA -> WMMA sharing/overlapping the D matrix requires >= 1 V_NOP or independent VALU between them for correctness | ISA 7.12.1 |
| fp8 converts | `V_CVT_PK_FP8_F32` (f32->2x fp8), `V_CVT_PK_F32_FP8` (2x fp8->f32), `V_CVT_F32_FP8`, `V_CVT_SR_FP8_F32` (**stochastic-rounding** f32->fp8) | ISA opcode lists |
| cross-lane ops | DPP16 (groups of 16), DS_SWIZZLE_B32 (groups of 32, rotate/broadcast/swap), DS_PERMUTE / DS_BPERMUTE_B32 (all 64 lanes in wave64) | ISA 7.9 |
| fp8 WMMA | `V_WMMA_F32_16X16X16_FP8_FP8` exists; A/B must come from VGPRs; C from VGPR or inline constant; no OPSEL/ABS/NEG | ISA 7.7 |
| occupancy model | LLVM `AMDGPUBaseInfo.cpp`: gfx12 wave32 total VGPR units per SIMD = 1024, alloc granule 16, max 16 waves per EU (SIMD); waves = 1024 / alignTo(vgpr, 16) | llvm-project source, fetched 2026-08-14 |
| LDS banks | 128 KiB per WGP, 64 DWORD banks (2x32 per CU pair); **each bank = 512x32 two-port RAM, 1R/1W per clock**; all banks can service a load or store simultaneously | ISA 12.1 |
| WMMA tile loads | ISA 11.6 gives the recommended vectorized load per fragment layout: row-major 16x16/16x32 8-bit -> `GLOBAL_LOAD_B128`; column-major 16x32 -> `GLOBAL_LOAD_TR_B128` (transpose load); 8-bit rows -> `GLOBAL_LOAD_B64/B32` | ISA 11.6 |
| WMMA throughput | **not published**: the ISA defines encodings/hazards but no FLOPs/clk; LLVM's scheduling model has no dedicated WMMA latency class (generic `Write32Bit`, VOP3PInstructions.td:1591) | ISA + llvm-project, checked 2026-08-14 |

## Not yet verified (needs another source; do not use in claims)

- Physical VGPR file size per CU in dwords (the ISA gives per-wave caps and
  LLVM gives the 1024-per-SIMD occupancy unit; the raw byte size is not
  stated by either).
- WMMA / DOT **throughput** rates: no official per-clock numbers exist for
  gfx1201 (checked ISA + LLVM scheduling model on 2026-08-14).
- LDS latency in cycles (bandwidth structure is known from ISA 12.1; the
  access latency value is not stated).
- VOPD pairing restrictions beyond the ISA 7.8 list.
- Infinity Cache (L2) size from an ISA-level source (product pages say
  64 MB for RX 9070 XT; the cache-op encodings are in the ISA, sizes are
  not).

## Immediate consequences for the audit

- The production decode kernel runs **wave32** (8 warps x 32 = 256 threads),
  so each instruction issues once per wave and **VOPD is available and used**
  (45 dual-issued pairs in the kernel; the WMMA chains and their loads are
  not pairable).
- fp32 WMMA accumulators cost 8 VGPR/lane per 16x16 fragment in wave32; the
  production kernel's accumulator arrays are 24 VGPR/lane (KQ_c 1 frag +
  VKQ_c 2 frags), so the remaining ~132 VGPR are working set - see W03.
- The WMMA D-matrix hazard forces serialization gaps in accumulation chains
  (matches the observed wait pattern; see W02/W03).
- `V_CVT_SR_FP8_F32` (stochastic rounding to E4M3) is a candidate for the P
  requant path that lost NMSE precision at scale 128 (D098) - phase-2
  candidate, needs an NMSE gate before any speed work.
- Occupancy (LLVM model): 156 VGPR -> 6 waves per SIMD, 24 per CU; LDS is
  the binding constraint at 2 CTAs per CU (already achieved); 3 CTAs need
  LDS <= 21,845 B per CTA.
