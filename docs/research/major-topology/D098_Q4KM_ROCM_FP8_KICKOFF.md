# D098: Q4_K_M native ROCm FP8 KV

Date: 2026-08-11

Status: complete on `master`. G1-G3 conversion/attention correctness, G4
spec-none speed and G5 MTP/98K quality-placement gates all pass. The spill-free
eight-wave full-native body beats the bracketed q8_0 center at 49K by `+4.9%`
prompt, `+5.2%` decode and `+5.0%` aggregate; 49K/98K MTP acceptance is no
worse than q8. Native KQ+V is therefore the ROCm RDNA4 F8 default for the
guarded D=256 shape, with explicit environment rollback.

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

At kickoff the HIP backend did not implement the type:

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
- `ctx=49152,b=512,ub=512`, FlashAttention on, one slot for the spec-none
  kernel gate; MTP confirmation uses production `b=8192,ub=1024`;
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

### G3a: native FP8 KQ WMMA (landed)

First native gfx12 step: the Q*K^T phase runs `fp8 x fp8 -> fp32` on the
`v_wmma_f32_16x16x16_fp8_fp8` instruction. Q is converted to E4M3 in the
kernel (scale applied after the MMA), K is consumed directly from the raw F8
KV cache, and the P*V phase stays on the proven f16 reference leg. Owned only
by the production Qwen shape (D=256, GQA=6, K/V F8, no K/V alias).

- Backend default on guarded RDNA4 F8/F8 D=256 shapes. Roll back the native KQ
  phase with `GGML_ROCM_FATTN_F8_NATIVE_KQ=0`.
- ISA proof: a minimal gfx1201 rocWMMA `float8_t` TU compiled with the exact
  vendored rocWMMA 7.1 emits `v_wmma_f32_16x16x16_fp8_fp8`; probe kernel
  registers `vgpr 47 / sgpr 49` (scratch spills 2/25).
- Correctness: masked prefill `KV=512,N=16` and decode `KV=512,N=1` both pass
  against the independent CPU E4M3 reference (`2/2`, NMSE threshold).
- Regression: the `GGML_ROCM_FATTN_F8_REFERENCE=1` route still passes `2/2`.
- Compile note: the first revision reused the f16 B-fragment type for the
  post-softmax P matrix, which made rocWMMA reject the mixed f16 x fp8 MMA;
  P now uses a dedicated f16 B-fragment while only the Q*K^T phase consumes
  `float8_t`.

### G3b: native FP8 V phase (landed)

Extend the gfx12 body to the P*V phase: V is consumed raw from the F8 KV
cache, P is re-quantized to E4M3 and the MMA runs `fp8 x fp8 -> fp32` with
fp32 VKQ accumulators. Softmax values fall below the E4M3 subnormal floor
(`2^-9`), so P is pre-scaled by 128 into the normal range and the VKQ merge
compensates; this keeps the P error at the format's `2^-4` instead of the
subnormal rounding.

- Backend default when native KQ owns the guarded shape. Roll back only the
  native V phase with `GGML_ROCM_FATTN_F8_NATIVE_V=0` (KQ-only bisect).
- The E4M3 P re-quantization is a new precision contract: the focused
  `FLASH_ATTN_EXT` comparisons run with a documented `1e-3` NMSE allowance
  (2x the strict `5e-4` of the f16-leg paths). Measured error is `~7e-4`.
- Correctness: masked prefill `KV=512,N=16` and decode `KV=512,N=1` pass
  `2/2`; reference and native-KQ routes still pass `2/2` each with the
  strict allowance (single `test-backend-ops` run now covers all six F8
  cases, each test selects its own env gate before `supports_op`).
- Shared memory stays at `29568 B` (`KQ_or_V 16896` + `P_f8 4224` +
  `VKQ f16 8448`) under the 32 KiB RDNA4 fence; native kernels are
  instantiated only for the D=256/16-column shape so non-Qwen forms never
  pull the FP8 body.
- Regression: G1 CPY `3/3`, G2 reference `2/2`, G3a native KQ `2/2`; unsupported
  architectures, dimensions, types and K/V aliases still fail closed. The
  selected RDNA4 D=256 F8/F8 shape now chooses full-native without an env gate.

### G4: speed gates (spec-none gate passed)

Server smoke and A/B on the 12K lane (24576 chars, ctx 16384, spec=none,
no MTP, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, seed 42, no background game,
adjacent runs in one session, quick tasks):

| run | KV/FA path | prompt ptps | decode tps | vs f16 |
|-----|------------|-------------|------------|--------|
| r5  | f16 (baseline)            | 1770.3 | 24.20 |  0.0% |
| r1  | q8_0                       | 1760.8 | 23.30 | -0.5% |
| r4  | f8_e4m3, native KQ only   | 1764.3 | 23.35 | -0.3% |
| r3  | q8_0 (control)            | 1742.9 | 23.63 | -1.5% |
| r2  | f8_e4m3, native KQ+V      | 1629.5 | 22.04 | -8.0% |
| r6  | f8_e4m3, native KQ+V (rep)| 1620.0 | 22.00 | -8.5% |

Second session (V-phase optimization bisect, same lane and flags):

| run | KV/FA path | prompt ptps | decode tps | vs f16 |
|-----|------------|-------------|------------|--------|
| r8  | f16 (fresh control)      | 1728.2 | 24.48 |  0.0% |
| r9  | f8 KQ+V, P-in-softmax + f16 VKQ store | 1529.0 | 20.94 | -11.5% |
| r10 | f8 KQ+V, P-in-softmax, fp32 VKQ store (kept) | 1635.5 | 22.11 | -5.4% |

- Converting the fp32 VKQ accumulators to f16 before the LDS store costs
  ~6.5% (r9 vs r10): the extra registers (f16 copies of the accumulators)
  and conversion instructions hurt more than the halved LDS store traffic
  helps. On this kernel registers are the scarce resource, not LDS.
- Folding the P->E4M3 write into the softmax loop (dropping the separate
  re-quantization pass and its barrier) is neutral-to-slightly-positive
  (r10 1635.5 vs r2/r6 1629.5/1620.0) and is kept.
- Remaining full-native gap to f16 on this lane: ~5.5% prefill, ~6-9%
  decode. The bottleneck is the fp32 accumulation itself (register
  pressure) plus the fp8 MMA operand conversion, not LDS bandwidth.

Third session (2026-08-13, packed native P conversion, same 12K lane):

| run | KV/FA path | prompt ptps | decode tps |
|-----|------------|-------------|------------|
| s6r1 | f16 (fresh control) | 1785.66 | 24.89 |
| s6r2 | f8, native KQ only | 1831.48 | 23.80 |
| s6r3 | f8, native KQ+V, packed P conversion | 1725.87 | 23.00 |

- G4 now converts two pre-scaled P values with one gfx12
  `__hip_cvt_float2_to_fp8x2` operation instead of invoking the scalar
  software E4M3 encoder once per value. The values are finite and in
  `[0,128]`, so this OCP E4M3 operation cannot enter the exponent-15 range
  where the persistent KV-cache format differs. Persistent cache writes
  continue to use the byte-exact portable encoder.
- The focused F8 FA matrix still passes reference, native-KQ and native-V
  prefill/decode (`6/6`) on ROCm0. `test-backend-ops` and `llama-server`
  build successfully for `gfx1201`.
- Against the previous full-native r10, the packed candidate improves prompt
  `1635.5 -> 1725.87` (`+5.5%`) and decode `22.11 -> 23.00` (`+4.0%`) raw.
  The fresh KQ-only comparison leaves a smaller V-phase cost of `3.4%`
  decode, but full FP8 remains `7.6%` below f16 decode. Keep the optimization
  inside the existing default-off native-V gate; G4 is improved, not closed.

Spill/occupancy follow-up (2026-08-13, same 12K lane):

| run | full-FP8 body | prompt ptps | decode tps |
|-----|---------------|-------------|------------|
| s7r1 | native KQ only | 1817.05 | 23.62 |
| s7r2 | spill-free 4-wave KQ+V | 1818.58 | 24.28 |
| s8r1 | spill-free 4-wave KQ+V control | 1822.23 | 24.22 |
| s8r2 | spill-free 8-wave KQ+V | 1843.68 | 24.71 |

- The packed-P kernel originally compiled at `256 VGPR`, `196` VGPR spills
  and `788 B/thread` private scratch. The root cause was compiler expansion of
  the packed P conversion and native-V VKQ merge, not the `29568 B` LDS tile.
- Serializing those loops removes all spills: the four-wave body uses
  `246-248 VGPR`, `52-54 SGPR`, zero scratch; the eight-wave native-only body
  reduces that to `154-156 VGPR`, `48-50 SGPR`, zero scratch, with unchanged
  LDS. Focused reference/KQ/V prefill and decode remain `6/6` correct.
- Eight waves improve the adjacent four-wave 12K result by `+1.2%` prompt and
  `+2.0%` decode. The eight-wave specialization is therefore the selected
  body whenever the existing native-V gate is active; native KQ-only and all
  non-FP8 paths keep their established four-wave instantiations.

### 49K lane A/B (spec-none, one fresh session, adjacent runs)

30K-token prefill at KV=49152, seed 42, no MTP, `-dev ROCm1,ROCm0`:

| run | KV/FA path | prompt ptps | decode tps |
|-----|------------|-------------|------------|
| r1  | f16 (baseline)       | 1722.5 | 23.05 |
| r2  | f8_e4m3, native KQ+V | 1398.8 | 17.14 |
| r3  | f8_e4m3, native KQ   | 1757.5 | 19.14 |
| r4  | f16 (control)        | 1724.9 | 22.93 |

- The 49K hypothesis (f8 wins at long context because K/V reads halve) is
  REJECTED. Full-native loses 19% prefill and 26% decode.
- Prefill: native KQ is at/above parity (1757.5 vs 1722.5, +2%, within
  noise - first sign of the bandwidth effect but not a measurable win);
  the entire prefill loss is the native V phase, which scales badly with
  KV length (-5% at 8K, -19% at 49K).
- Decode: both phases regress at KV=49152 (KQ-only -17%, full -26%; at
  KV=8192 they were ~0% and -10%). The single-row decode loop is
  latency-bound, so the extra per-iteration FP8 conversion/accumulation
  overhead is exposed.
- Bottom line for the ROCm wmma path: FP8 KV on gfx1201 does not beat f16
  at any lane; only the prefill KQ phase is cost-neutral. FP8 KV remains a
  memory-saving option (half the cache), not a speed option, on this
  backend. No route is promoted; all FP8 routes stay opt-in.

Packed-P follow-up (2026-08-13, one `triage_diff` task per adjacent run):

| run | KV/FA path | prompt ptps | decode tps |
|-----|------------|-------------|------------|
| s6r4 | f16 (fresh control) | 1694.16 | 23.34 |
| s6r5 | f8, native KQ only | 1739.83 | 19.84 |
| s6r6 | f8, native KQ+V, packed P conversion | 1468.24 | 18.55 |

- The packed candidate improves the old full-native result from
  `1398.8/17.14` to `1468.24/18.55` (`+5.0%` prompt, `+8.2%` decode raw;
  about `+4.4%` decode after normalizing by the adjacent KQ-only control).
- It does not reverse the 49K decision: full-native remains `6.5%` below
  KQ-only decode and `20.5%` below f16 decode. The V leg and long-KV decode
  loop remain the open G4 bottlenecks; no default route is promoted.

Spill-free and eight-wave follow-up (2026-08-13, `b512/ub512`, one
`triage_diff` task per run):

| run | KV/FA path | prompt ptps | decode tps |
|-----|------------|-------------|------------|
| s7r3 | f8, native KQ only | 1720.72 | 19.64 |
| s7r4 | f8, spill-free 4-wave KQ+V | 1673.28 | 21.97 |
| s7r5 | f8, spill-free 4-wave KQ+V repeat | 1699.19 | 21.98 |
| s7r6 | f16 control | 1688.74 | 23.33 |
| s8r3 | f8, 4-wave KQ+V control | 1706.46 | 21.96 |
| s8r4 | f8, 8-wave KQ+V | 1760.53 | 22.84 |
| s8r5 | f8, 8-wave KQ+V repeat | 1761.85 | 22.92 |
| s8r6 | f16 control | 1691.43 | 23.36 |

- Eliminating the spills raises the earlier packed full-native result from
  `1468.24/18.55` to the four-wave center `1686.24/21.98` (`+14.8%` prompt,
  `+18.5%` decode raw). Eight waves add another `+3.2%` prompt and `+4.2%`
  decode versus the adjacent four-wave control.
- The eight-wave center is `1761.19/22.88`: prompt is above the adjacent f16
  control while decode is only about `2.1%` below it. This supersedes the
  historical conclusion above that full-native FP8 cannot approach f16.

Final same-binary G4 bracket after promoting eight waves inside the native-V
gate (no separate wave-count environment variable):

| run | KV/FA path | aggregate TPS | prompt ptps | decode tps |
|-----|------------|--------------:|------------:|-----------:|
| s9r1 | q8_0 open | 8.62 | 1686.34 | 21.84 |
| s9r2 | f8_e4m3, native KQ+V | 9.05 | 1769.37 | 22.97 |
| s9r3 | q8_0 close | 8.62 | 1687.16 | 21.81 |

- Versus the q8 center (`1686.75/21.83`, aggregate `8.62`), full FP8 gains
  `+4.9%` prompt, `+5.2%` decode and `+5.0%` aggregate, outside the current
  order-swing noise. G4 therefore passes for spec-none.
- These measurements authorized backend-default promotion after G5; setting
  `GGML_ROCM_FATTN_F8_NATIVE_KQ=0` restores the pre-native rollback path, while
  `GGML_ROCM_FATTN_F8_NATIVE_V=0` isolates the KQ-only route.

- Smoke (8K ctx, f8 KV + full native, spec=none) completed both quick tasks
  with valid output; `GGML_TRACE_FATTN_PATH` confirmed `native_kq=1
  native_v=1` on the D=256 path and K/V cache `f8_e4m3`.
- Blocking fix landed: `SET_ROWS` on ROCm did not accept `F8_E4M3`, so the
  server aborted while writing K rows (`cache_k_l3`). Added a f32->E4M3
  SET_ROWS path and the `supports_op` admission (HIP-only).
- Gate conclusion: G4 spec-none performance is closed positive. The next
  decision is G5 quality and long-context behavior, not another nearby V-loop
  micro-shape.

### G3: native RDNA4 FP8 FlashAttention (full)

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

Result: passed on 2026-08-13 with MTP n2, seed 42, one `triage_diff` task,
production `b8192/ub1024` and `-dev ROCm1,ROCm0 -sm layer -ts 1,1`.

49K bracket (30,187 prompt tokens; q8/FP8 use the context-scoped last8 f16
policy):

| run | KV | aggregate TPS | prompt ptps | decode tps | acceptance |
|-----|----|--------------:|------------:|-----------:|-----------:|
| s10r1 | q8 open | 5.93 | 1665.16 | 37.71 | 77/99 = 77.78% |
| s10r2 | full FP8 | 6.29 | 1751.58 | 41.98 | 79/95 = 83.16% |
| s10r3 | q8 close | 5.97 | 1678.90 | 38.21 | 77/99 = 77.78% |

Versus q8 center, FP8 gains `+4.8%` prompt, `+10.6%` decode and `+5.7%`
aggregate while acceptance improves `+5.38 pp`.

98K D091 placement bracket (57,893 prompt tokens):

| run | KV | aggregate TPS | prompt ptps | decode tps | acceptance |
|-----|----|--------------:|------------:|-----------:|-----------:|
| s11r1 | q8 open | 2.91 | 1445.14 | 33.43 | 78/98 = 79.59% |
| s11r2 | full FP8 | 2.99 | 1482.86 | 35.82 | 78/96 = 81.25% |
| s11r3 | q8 close | 2.90 | 1441.93 | 33.57 | 78/98 = 79.59% |

The 98K FP8 policy correctly selected last12 f16 (`5376 MiB` KV) versus q8
last8 (`4704 MiB`). Against q8 center, FP8 gains `+2.7%` prompt, `+6.9%`
decode and `+2.9%` aggregate while acceptance improves `+1.66 pp`; placement
remains stable with no crash, fallback or driver event.

Final default smoke used no native-FP8 env variables. Route trace proved
`native_kq=1 native_v=1` for both `Q rows=1024` prefill and `Q rows=1` decode;
the short dual-ROCm server run completed at `1688.85/25.99` prompt/decode.

## Initial source ownership

The first prototype owns only:

- HIP E4M3 conversion helpers and CPY dispatch;
- ROCm backend support checks for those exact conversions;
- focused copy tests;
- one explicit F8 attention reference route and its guarded native gfx12
  successor.

No model-format, Vulkan, GUI, preset, peer-copy or public benchmark-table edit
belongs in G1-G3.

## Rollback and stop rules

Each gate is independently removable. Set
`GGML_ROCM_FATTN_F8_NATIVE_KQ=0` to disable the native FP8 body completely, or
`GGML_ROCM_FATTN_F8_NATIVE_V=0` to retain native KQ with the f16 V leg. The
explicit G2 reference additionally requires `GGML_ROCM_FATTN_F8_REFERENCE=1`.
Stop and document if bytes diverge from CPU/Vulkan, attention falls back,
output is corrupt, resources exceed the RDNA4 limit, adjacent q8 wins outside
noise, or MTP acceptance falls.
