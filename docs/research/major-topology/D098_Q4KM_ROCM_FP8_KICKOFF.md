# D098: Q4_K_M native ROCm FP8 KV

Date: 2026-08-11

Status: active on `master`. Plan locked; G1 device conversion and G2 reference
FlashAttention are correctness-complete. G3 native FP8 rocWMMA is next. No
ROCm FP8 performance claim exists yet.

## Objective

Bring the fork's `f8_e4m3` KV cache to the Windows ROCm/HIP backend for the
two RX 9070 XT (`gfx1201`) system. The first milestone is a correct,
fail-closed device-resident FP8 cache. The final route must consume the stored
FP8 bytes with native RDNA4 FP8 WMMA in FlashAttention, preserve MTP quality,
and beat an adjacent q8_0 control before any preset or README promotion.

This is a backend implementation, not a transfer of the Vulkan P5 speed claim.
The Vulkan SPIR-V path remains unchanged.

## Current evidence

The backend-neutral foundation already exists:

- `GGML_TYPE_F8_E4M3` has a stable one-byte type and CPU traits;
- CLI cache types and the context-scoped last8/last12 f16 MTP policy exist;
- finite encoded values use the OCP E4M3 bit layout, with exponent 15 reserved
  and the project-wide finite clamp fixed at `[-240, 240]`.

The HIP backend does not yet implement the type:

- `ggml/src/ggml-cuda/cpy.cu` has no F32-to-F8 or F8-to-F32 graph copy;
- `fattn-common.cuh` has no F8 K/V loader;
- `fattn.cu` rejects F8 before dispatch;
- no F8 vector or rocWMMA FlashAttention template is instantiated.

ROCm 7.1 on this machine supplies the required native pieces. `gfx1201` uses
OCP `__hip_fp8_e4m3`, and the vendored rocWMMA provides the gfx12
`fp8 x fp8 -> fp32` 16x16x16 builtin. The stored finite bytes are directly
loadable by that type. The GPU writer must nevertheless keep the repository's
deterministic rounding and `240` clamp instead of emitting OCP exponent-15
finite values up to 448.

External implementations validate the architecture but are not copied
wholesale: current vLLM recognizes RDNA4 OCP E4M3 KV; AITER tests unified
attention with E4M3 on gfx1201; CK `develop` describes gfx1201 FP8 FMHA
pipelines. Their layouts and LDS limits are references for G4, not a dependency
of the first implementation.

## Locked lanes

All GPU runs are sequential and use the production device order.

### 49K gate

- model: `models/Qwen3.6-27B-Q4_K_M.gguf`;
- binary: ROCm/HIP Ninja build for `gfx1201`;
- topology: `-dev ROCm1,ROCm0 -sm layer -ts 1,1`;
- `ctx=49152,b=8192,ub=1024`, FlashAttention on, one slot;
- cold/no-reuse/no-prime/no-warmup;
- q8_0/q8_0 spec-none control, then f8_e4m3/f8_e4m3 candidate.

### 98K and MTP confirmation

- preserve the D091 one-copy scheduler and device order;
- run MTP only after spec-none correctness passes;
- use n2, q8 last8 control, FP8 last8 at 49K and last12 at 98K;
- record accepted/generated draft tokens, prompt/decode rates, KV MiB,
  placement diagnostics and output sanity.

Never set `LLAMA_OUTPUT_DEVICE=ROCm1`. Do not run discovery or memory probes
while a server or benchmark is active.

## Gate ladder

### G0: plan and fail-closed boundary

Only HIP builds may advertise the new conversion path. Native CUDA behavior,
Vulkan shaders, storage IDs, GUI defaults, device splitting and presets remain
unchanged. Until attention is implemented, ROCm must still reject an F8
FlashAttention graph rather than silently fall back.

### G1: byte-compatible device conversion

Implement F32 -> F8 and F8 -> F32 graph copies in the shared CUDA-compatible
HIP layer, guarded by `GGML_USE_HIP`.

Requirements:

- exact bytes versus the CPU reference for zero/sign, subnormal boundaries,
  exponent transitions, tie cases, NaN and saturation;
- no exponent-15 finite output; maximum finite magnitude stays 240;
- decode the stored byte through native OCP E4M3 on gfx1201;
- add focused backend copy coverage without enabling native CUDA F8.

Result: passed on ROCm0 (`gfx1201`). F32 -> F8, F8 -> F32 and the explicit
32-value boundary vector all pass (`3/3`). The boundary vector covers signed
zero, subnormal ties and limits, normal transition, mantissa carry, saturation
and infinities. The HIP decoder uses the native OCP `__hip_fp8_e4m3` type for
all repository-valid finite bytes; encoding remains byte-identical to the
CPU/Vulkan finite-240 contract.

This gate also found a stale CPU encoder defect inherited from the early
Vulkan prototype: values in `[2^-9, 2^-6)` could enter the normal branch and
wrap an unsigned exponent, and mantissa carry was not folded into the exponent.
The CPU reference now uses the same bit algorithm as the accepted Vulkan/HIP
writer, so the comparator is independent and deterministic.

### G2: reference F8 FlashAttention

Add a default-off correct route that reads F8 K/V and converts each tile to
f16 for the already validated rocWMMA f16 attention body. Cover both vector
decode and batched prefill. This gate isolates storage and layout correctness;
it is not the final performance route.

Pass criteria: focused `FLASH_ATTN_EXT` backend comparisons for D=256 and the
active Qwen GQA shapes, no graph split/CPU fallback, finite coherent output and
documented agreement with f16.

Result: passed on ROCm0 behind `GGML_ROCM_FATTN_F8_REFERENCE=1`. The exact
Qwen3.6 D=256/GQA=6 shapes pass against an independent CPU E4M3 reference:
masked prefill `KV=512,N=16` and decode `KV=512,N=1` are both `OK` (`2/2`).
Without the environment gate both shapes remain unsupported, preserving the
fail-closed default. G2 preconverts F8 K/V to f16 and exercises the established
rocWMMA f16 body; it proves storage/layout correctness but is not a native-FP8
speed result.

### G3: native RDNA4 FP8 FlashAttention

Instantiate a gfx12-only rocWMMA body using `rocwmma::float8_t` for K/V tiles
and fp32 accumulators. Preserve the softmax and output contracts. Use CK/AITER
only as resource/layout references. Require compiled ISA to contain the gfx12
FP8 WMMA instruction and record VGPR, SGPR, LDS, scratch and occupancy.

The candidate must stay within the local 32 KiB compute-shared-memory limit;
the known RDNA4 AITER overflow is a design fence against blindly increasing
pipeline stages.

### G4: server correctness and adjacent q8/F8 performance

After G1-G3, run a short 49K spec-none smoke, then q8 -> F8 -> q8 on one clean
binary and idle session. Promote only a repeatable prompt/wall win outside
order noise with correct output. Raw F8 saves only 5.88% versus q8_0
(`1` versus `34/32` bytes per value); memory saving alone is not a speed claim.

### G5: MTP and 98K confirmation

F8 is accepted only if the same prompt/seed gives MTP acceptance no worse than
the adjacent q8 center and decode does not regress. The 98K run must preserve
the corrected D091 placement corridor. A crash, driver drop, corruption,
silent fallback or acceptance regression rejects the candidate.

## Initial source ownership

The first prototype owns only:

- HIP E4M3 conversion helpers and CPY dispatch;
- ROCm backend support checks for those exact conversions;
- focused copy tests;
- later, one default-off F8 attention reference route and its native gfx12
  successor.

No model-format, Vulkan, GUI, preset, peer-copy or public benchmark-table edit
belongs in G1-G3.

## Rollback and stop rules

Each gate is independently removable. Keep a default-off implementation only
when it is correctness-complete and directly enables the next gate. Stop and
document if bytes diverge from CPU/Vulkan, attention falls back, output is
corrupt, resources exceed the RDNA4 limit, adjacent q8 wins outside noise, or
MTP acceptance falls.
