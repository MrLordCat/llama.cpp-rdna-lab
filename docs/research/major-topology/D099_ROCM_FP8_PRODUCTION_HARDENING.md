# D099: ROCm FP8 production hardening

Date: 2026-08-13

Status: complete. D098 closed the production Qwen3.6 D=256 gfx1201 F8/F8
route. D099 broadens correctness ownership without weakening the D098 fast
path or describing non-gfx12 hardware as native FP8.

## Objective

Make `f8_e4m3` KV a safe ROCm feature across the supported FlashAttention
shape surface:

- keep native gfx12 FP8 WMMA for eligible RDNA4 phases;
- use an explicit device-resident F8-to-f16 fallback everywhere else on HIP;
- support one-sided F8 mixed K/V types when the other side has a valid f16
  converter;
- make exact K/V aliases safe under both native and rollback modes;
- extend deterministic backend coverage for head sizes, masks, sinks, ALiBi,
  logit softcap, mixed types and aliases;
- preserve the validated D098 49K/98K Qwen route and rollback controls.

Native means that the corresponding FP8 operand is consumed by gfx12 FP8
WMMA. Older AMD targets use the portable conversion fallback and must never be
reported as native.

## Locked baseline

- model: `models/Qwen3.6-27B-Q4_K_M.gguf`;
- backend: Windows ROCm/HIP 7.1, Ninja, `gfx1201`;
- production order: `-dev ROCm1,ROCm0 -sm layer -ts 1,1`;
- safe lane: `ctx=49152,b=8192,ub=1024`, one slot, cold/no-reuse/no-warmup;
- FP8 candidate: `f8_e4m3/f8_e4m3`, FlashAttention on;
- D098 spec-none anchor: `1769.37/22.97/9.05`
  prompt/decode/aggregate TPS;
- D098 MTP n2 anchors: 49K `1751.58/41.98/6.29`, 98K
  `1482.86/35.82/2.99`.

No GPU discovery may overlap a server or benchmark. A 131K run is a residency
stress gate and is not required before focused correctness and compile gates.

## Ownership matrix

### RDNA4 native phases

Native KQ is eligible when K is F8, the standard K/V head dimensions match,
KV length is aligned to `FATTN_KQ_STRIDE`, and D is one of
`64,80,96,112,128,256`.

Native P*V is independently eligible when V is F8 and D is one of
`64,128,256`. The production four-wave WMMA topology has `VKQ_ratio == 1`
only for those dimensions. D80/D96/D112 keep native KQ but convert V to f16;
their ratios are `4/2/4`, and the native merge intentionally rejects multiple
accumulator groups. D112 also has a four-wave full-native resource model of
`38,784 B` LDS, above the local 32 KiB compute fence.

D256 full-native retains the spill-free eight-wave D098 body. D64/D128 use the
existing four-wave full-native body; D80/D96/D112 use its native-KQ/f16-V
variant. Lower-wave native-V variants remain research-only until separately
benchmarked and are not production defaults.

### Portable HIP fallback

Any supported FlashAttention head shape with at least one F8 K/V operand uses
the existing HIP byte-compatible converter and a f16 tile/WMMA body when a
native phase is unavailable. This is the only allowed path on pre-RDNA4 AMD
architectures or builds without rocWMMA.

`GGML_ROCM_FATTN_F8_NATIVE_KQ=0` remains the complete native rollback.
`GGML_ROCM_FATTN_F8_NATIVE_V=0` keeps native KQ where safe. The explicit
reference switch forces the portable path. Unsupported conversions remain
fail-closed.

### Mixed types and aliases

One-sided F8 is supported when the other K/V type has an existing HIP
to-f16 converter. The F8 side may stay native on eligible RDNA4 shapes; the
other side is passed directly when f16 or converted once into backend scratch.

An exact F8 K/V alias is native only when both phases are active. If native V
is disabled, native KQ must also stand down for the alias so the common
fallback converts the shared source once and reuses the f16 view.

## Gate ladder

1. Add policy helpers that separate native-KQ, native-V and portable fallback.
2. Extend dispatch/allocation without changing non-F8 or non-HIP behavior.
3. Add backend cases for:
   - D64/80/96/112/128/256 native coverage;
   - D40/72 and unaligned-KV portable fallback;
   - F8/F16, F16/F8, F8/Q8 and Q8/F8;
   - exact F8/F8 K/V alias under native and rollback;
   - mask/unmasked, sinks, ALiBi and D128 logit softcap;
   - retained D256 Qwen prefill/decode.
4. Build `test-backend-ops` and `llama-server` for gfx1201.
5. Compile-only configure/build for a pre-RDNA4 target when supported by the
   local ROCm SDK; do not make a runtime claim without hardware.
6. Run focused backend correctness on ROCm0, then an adjacent 49K D098
   spec-none regression bracket. Run 98K/131K only if source/resource or
   placement behavior changed materially.
7. Update the D099 note, `RESULTS_LOG.md`, support docs and canonical history.

## Stop rules

- Do not instantiate a native shape above the 32 KiB LDS fence.
- Do not silently route raw F8 bytes into an f16 kernel.
- Do not let mixed/alias handling allocate a zero scratch pointer.
- Do not regress the D256 D098 route, resource record, output correctness,
  placement or adjacent performance outside normal order noise.
- Negative native expansions revert to portable fallback; fallback coverage
  may remain when correctness-clean.
- No forceful server termination during model load, prompt evaluation or
  decode.

## Validation and result

- gfx1201: `test-backend-ops` and `llama-server` build with ROCm clang/Ninja.
- gfx1100: compile-only configure and full `test-backend-ops` plus
  `llama-server` build pass. Native gfx12 kernels are device-guarded; the
  portable F8-to-f16 route remains compiled for the older target. This is not
  a runtime performance claim.
- ROCm0 focused correctness: `19/19` pass. This includes the retained D098
  D256 reference/KQ/full-native prefill+decode cases and 13 D099 cases for
  alternate dimensions, fallback, mixed types, features and exact aliases.
- D098/D099 route-selection cases are explicitly ROCm-only in the shared
  backend harness. The same focused filter on Vulkan0 reports `0/0` applicable
  cases and exits cleanly instead of executing HIP env contracts on Vulkan.
- The compiler rejected the initial D80/D96/D112 native-V expansion because
  its merge requires `VKQ_ratio == 1`. Production ownership was therefore
  narrowed to D64/D128/D256 full-native; D80/D96/D112 are native-KQ with
  portable V. No lower-wave experimental route is enabled by default.
- Same-binary production 49K bracket (`b8192/ub1024`, one `triage_diff`,
  dual `ROCm1,ROCm0`, spec-none):

| run | KV | aggregate TPS | prompt TPS | decode TPS |
| --- | --- | ---: | ---: | ---: |
| `d099-rocm49k-q8-open-r1` | q8_0 | 8.73 | 1741.80 | 21.95 |
| `d099-rocm49k-f8-r2` | f8_e4m3 | 9.08 | 1809.62 | 22.96 |
| `d099-rocm49k-q8-close-r3` | q8_0 | 8.72 | 1739.22 | 21.93 |

Against the exact q8 center (`8.7217/1740.51/21.94`), FP8 retains
`+4.12%/+3.97%/+4.65%` aggregate/prompt/decode. D256 therefore has no
hardening regression. A new 98K/131K run was not required: D099 changes no
D256 kernel resource body or cache placement, and the adjacent 49K gate is
clean.
