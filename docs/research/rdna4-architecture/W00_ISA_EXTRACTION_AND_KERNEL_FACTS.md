# W00: gfx1201 ISA extraction and hot-kernel facts

Date: 2026-08-14

Status: artifact-ready. Facts below are read from the existing `build-rocm`
binary (`ggml-hip.dll`, commit `bffa79f33` base + D102 census) and do not
require any new build.

## Extraction recipe (Windows HIP, RDC)

```bash
export PATH="/c/Program Files/AMD/ROCm/7.1/bin:$PATH"
llvm-objcopy -O binary --only-section=.hip_fat build-rocm/bin/ggml-hip.dll /tmp/hip.fat
# split at every 0x7fELF magic (129 embedded code objects, one per TU)
# for each blob: llvm-objdump -d --mcpu=gfx1201 <blob.co>
```

The `.hip_fat` PE section concatenates 129 relocatable ELF code objects, each
prefixed by a 512-byte stub plus a `__CLANG_OFFLOAD_BUNDLE__` header. TU 23
holds the `fattn-wmma-f16.cu` kernels. Artifacts live in `isa/` and are
untracked (102 MB total); regenerate with the recipe above.

## Hot-kernel roster (49K decode path)

Production full-native decode kernel: the dispatch uses the `f8_native_only`
specialization, so the running kernel is the **8-warp wave32** instantiation
`flash_attn_ext_f16<256, 16, 8, 128, float, false, false, false, true, true, false>`
(256 threads = 8 warps x 32 lanes):

| Metric | Value |
| --- | --- |
| VGPR | 156 |
| SGPR | 46 |
| AGPR | 0 |
| private/stack | 0 |
| threads | 256 (8 warps x wave32) |
| shared | 29,568 B = KQ_or_V 16,896 + P_f8 4,224 + VKQ 8,448 |

Related instantiations in the same TU:

- census variant of the production kernel `<256,16,8,128,float,0,0,0,1,1,1>`:
  157 VGPR / 56 SGPR - this is exactly what D102 measured;
- the 4-warp instantiations `<256,16,4,64,...>` (248 VGPR full-native, 246
  softcap, 212 f16) serve the KQ-only / V-only / non-native D256 routes, not
  the full-native production path;
- softcap full-native `<256,16,8,128,float,1,0,0,1,1,0>`: 154 VGPR / 48 SGPR
  (the 154-156 VGPR figures from D098 belong to this 8-warp family).

## Occupancy fact (LLVM model, 2026-08-14)

The kernels are compiled as **wave32** (evidence: 45 VOPD `v_dual_*` inside
the production kernel, 5-step 32-lane `ds_bpermute` reductions, `vcc_lo` /
`exec_lo` only; the build has no `-mwavefrontsize` flag, ROCm hip-clang
builds RDNA targets as wave32). LLVM `AMDGPUBaseInfo.cpp` for gfx12 wave32:
total VGPR units per SIMD = 1024, allocation granule = 16, max 16 waves per
SIMD. At 156 VGPR (rounded to 160): **6 waves per SIMD, 24 per CU**. One
CTA = 8 waves, so VGPR admits 3 CTAs (24 waves); LDS admits 2 CTAs
(2 x 29,568 = 59,136 <= 64 KiB), so **LDS is the binding constraint at
2 CTAs per CU - already achieved**. Three CTAs need LDS <= 21,845 B per CTA.

Hardware constants from the official RDNA4 ISA (both wave sizes supported,
VOPD wave32-only, 256 VGPR max per wave, 128 KiB LDS per WGP) are in W01.

## Next work items

- W1: verify the hardware constants from the AMD gfx12 ISA/RDNA4 docs. **done
  2026-08-14, see W01.**
- W2: audit the K/V tile loader instructions (no `cp.async`; what the
  compiler emits instead) in the full-native D256 body. **done 2026-08-14,
  see W02.**
- W3: barrier count and kind (`s_barrier` vs split barriers) per chunk loop.
  **done in W02: 8 signal + 8 wait, zero full barriers.**
- W4: WMMA fragment layout and the store/merge instruction stream. **layout
  verified in W01; store/merge stream audit still open.**
- W5: softmax reduction instruction stream (shared roundtrip vs permute).
  **partial in W02 (`ds_bpermute_b32` x40); rowmax/rowsum stream still open.**
- W6: fp8 conversion instructions in the requant path (`v_cvt_pk_*` present?).
  **partial in W02: 4x `v_cvt_pk_fp8_f32`; stochastic-rounding alternative
  flagged as a phase-2 candidate.**
