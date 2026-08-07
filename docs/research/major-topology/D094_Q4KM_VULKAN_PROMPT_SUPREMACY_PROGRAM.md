# D094: Q4_K_M long-prompt prompt-eval supremacy — Vulkan vs ROCm program

Date: 2026-08-04
Status: open design program — cycle 1 measured; one launch-level win accepted
(`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, +3.31% on 131k), supremacy gap narrowed
but not closed; source work required for the remaining gap
Model: `Qwen3.6-27B-Q4_K_M.gguf` (primary production model, D089)
Hardware: dual RX 9070 XT 16 GB, Windows 11, AMD proprietary driver
`32.0.31035.1003`, HIP SDK 7.1

## Cycle 1 results (2026-08-04, this session)

Lane: 131k, 57,463-token prompt (`rcc220000`), `b8192/ub1024`, `q8_0/q8_0`
KV, FA on, `spec=none`, cold/no-reuse, `-dev Vulkan1,Vulkan0 -sm layer
-ts 1,1`, fresh wn32 binary. Adjacent control r3: `1201.54` prompt tok/s
(50.88 s wall).

| Probe | pp tok/s | Delta | Verdict |
| --- | ---: | ---: | --- |
| control r1/r3 | 1201.26 / 1201.54 | — | stable |
| `-ts 5,6` | 1106.39 | −7.9% | reject |
| `-ts 27,37` | 1061.33 | −11.6% | reject |
| `GGML_VK_QK_LOW_TILE_SPLIT_K=3` | 1195.22 | −0.5% | reject |
| `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` | 1201.06 | ±0.0% | neutral |
| `q5_1/q5_1` KV | 1106.48 | −7.9% | reject |
| `q4_0/q4_0` KV | 1195.01 | −0.5% | reject |
| `--no-mmap` | 1189.88 | −1.0% | reject |
| `ub=2048` | 1186.93 | −1.2% | reject |
| `GGML_VK_FA_FORCE_SPLIT_K=2` (source knob) | 1194.24 | −0.6% | reject |
| `GGML_VK_FA_F32ACC=0` (f16acc, source knob) | 1194.34 | −0.6% | reject |
| `GGML_VK_FA_ROW_SPLIT=2` (source knob) | 1121.58 | −6.6% | reject |
| `GGML_VK_FA_ROW_SPLIT=8` | crash 0xC0000005 | — | reject, capped ≤4 |
| `GGML_VK_FA_BLOCK_COLS=128` + ROW_SPLIT=2 | timeout, scalar fallback (Br16/Bc32) | — | reject (matches D076) |
| **`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`** | **1241.31 (r3: 49.61/49.13/49.33 s)** | **+3.31%** | **accepted** |
| gfxq + `-ts 5,6` | 1181.66 | −4.8% vs gfxq | reject |
| gfxq + `GGML_VK_FA_F32ACC=0` | 1206.36 | −2.8% vs gfxq | reject |
| gfxq + `GGML_VK_MAX_NODES_PER_SUBMIT=1024` | 1232.16 | −0.7% vs gfxq | reject |
| gfxq + `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | 1201.15 | −3.2% vs gfxq | reject (graph optimize required) |
| gfxq + `GGML_VK_DISABLE_ASYNC=1` | 583.80 | −51% | reject (async required) |
| gfxq + `GGML_VK_PREFER_HOST_MEMORY=1` | failed run | — | reject |

12k lane (6.7k-token prompt, same config): device order matters. Fresh
adjacent rows: `Vulkan0,Vulkan1` = `1534-1563` (matches the old
validate-20260804 row), **`Vulkan1,Vulkan0` = `1669.79` (+6.8-8.8%)** —
the earlier 12k row was measured with the non-canonical device order.
gfxq on 12k is negative (`1646.00`, −1.4%) — keep gfxq off at 12k.

Verdict on the launch-level candidate queue: T101 (ts sweep) closed negative,
T103 (low-tile) negative, T104 (async/host/fusion) closed mostly negative,
T105 (q5_1 KV) negative on this lane (FA is not KV-byte-bound here), T102
(stale 98k/49k rows) still pending re-measure with wn32 + gfxq. The only
accepted item is graphics-queue (`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`), which is
WDDM-queue selection — no math change, response token counts and lengths
matched control.

## Measured bottleneck structure (cycle 1 + cycle 2 clean perf, 131k p60k)

Clean perf-verified shares (GGML_VK_PERF_LOGGER=1 run, gfxq config; note
the logger itself serializes dispatches and halves throughput, but relative
shares are representative):

- Quant MUL_MAT: `47.0%` (34,821 calls). q4_K `17408x1024x5120` (gate/up)
  at `~65 TFLOPS`; all forms land `55-65 TFLOPS` ≈ Q4_K dequant ceiling.
  Q4_K block is already nibble-packed (no q3quad-style win); int-dot/q8_1
  path is scalar-only in shader-gen and E099-rejected on Q3.
- FLASH_ATTN_EXT: `39.4%` (1,936 calls; 19 TFLOPS effective). Independent of
  KV type (q4_0 −2.3% FA time at 12k only; q5_1 negative), split_k (neutral),
  f16acc (neutral), row_split (2 worse, 8 crashes), Br/Bc (D081/D076 closed;
  Bc128+row_split2 falls back to scalar). Bottleneck is the fixed iteration
  structure of the coopmat1 shader (6 q-heads re-read the same KV);
  kernel tuning is exhausted.
- SSM/GDN tail: `~13%` total (GDN 2.1% + CONCAT 1.9% + GLU 1.8% + ADD 1.4%
  + CONT 0.8% + RMS fused 1.4% + SILU 0.5% + GET_ROWS 0.4% + SSM_CONV 0.4%
  + MUL 0.4% + CPY 0.2%). CONCAT measures `5.4 us` per call (5,808 calls),
  NOT the `435 us` seen in the cycle-1 trace run — that number was an
  artifact of the trace-instrumented bc128 probe and is retracted.
- No dual-GPU imbalance: ubatch timing shows a consistent per-ubatch cost
  with no per-device skew.

Remaining gap after cycle 1: 131k `1241.31` vs ROCm `1411.98` = `1.1375x`.
User target for cycle 2 is `>= 1.1x` local (from `1201.54` that is
`>= 1321.7` tok/s; gfxq contributes `+3.31%`, ~`+6.5%` still needed).

## Source-work queue for cycle 2 (ranked by expected ceiling)

| ID | Candidate | Mechanism | Ceiling |
| --- | --- | --- | --- |
| T201 | FA cross-head KV reuse | one KV pass shared by the 6 q-heads of a kv-head (currently KV is re-read per q-head) | 1.3-1.6x on FA if iteration-bound (39.4% -> ~20-25% => +8-12% wall) |
| T202 | FA warp-specialization / async K/V staging | split load/dequant/compute across subgroups to hide the fixed per-iteration cost | 1.2-1.5x on FA |
| T203 | K-compression for q8 KV (T5a) | fewer KV bytes per key with cheap dequant | blocked by cycle-1 evidence: FA time is KV-byte-independent on this lane |
| T204 | Q4_K int8 coopmat path | dot4 s8 path for q4_K weights | E099-fenced on Q3; needs fresh Q4 point proof first |
| T205 | fused GDN + CONCAT reduction | shave the SSM tail | CLOSED cycle 2: CONCAT measured `5.4 us/call` = 1.9% share; a fast concat shader (dim-0 contiguous, row-block) was built, verified (32/32 backend tests + model smoke) and A/B-measured neutral (±0.2%); kept as opt-in `GGML_VK_CONCAT_FAST=1`, off by default |

T201/T202 (FA body work) were gated this cycle — see "Cycle-2 FA gates"
below; both are closed at the coopmat1-tuning level. T205 is closed; micro-op
fusion of the SSM tail is not observable in wall time on this lane.

## Cycle-2 FA gates (T201/T202 status, 2026-08-04)

All FA levers were re-tested or closed analytically this cycle. Summary:

- VKGC resources for the q8_0 prefill pipeline (Br16/Bc64, row_split=4,
  128 threads): `102 VGPR / 69 SGPR / 26112 B LDS / 0 scratch` →
  2 workgroups/CU (LDS-limited at the 64 KiB/CU budget).
- FA time is strictly linear in KV: `1.4-1.5 us per 1k KV` per call
  (call = full ubatch; 16 attention layers of 65 total layers); fixed cost
  ~0. Effective throughput `~16 TFLOPS` on two GPUs (~8.5% of the 96 TFLOPS
  fp16 ceiling) - far below the `55-65 TFLOPS` of the MUL_MAT lane.
- Occupancy gate: `GGML_VK_FA_BLOCK_COLS=32` (3 wg/CU, ~19.5 KiB LDS,
  mask-opt disabled for the probe): 12k `7.359s` vs control `7.34s` ->
  neutral. Occupancy is NOT the limiter.
- Latency gate: dual QK^T accumulator over d-parity (2 independent coopmat
  chains, +8-16 VGPR): 12k `7.684s` -> `-4.5%`, rejected and reverted.
- T201 (cross-head KV reuse) closed analytically: coopMatLoad reads are per
  (subgroup, d) and cannot be shared through LDS; the 6-q-head N-accumulator
  variant blows VGPR and repeats the D081 (-0.98%) outcome.
- Conclusion: coopmat1 FA structure is exhausted on this lane (D076/D077/
  D081/E142/E145/E148 + cycle-2). Any future FA win needs a new dataflow
  (register-resident shared K-tile across q-heads, or non-coopmat FA), which
  is a dedicated multi-day project, not a tuning knob.

## Cycle-3 FA structure gates (2026-08-04, fundamental paths)

User approved deep work. 9070 XT cooperative-matrix shapes were enumerated
via a GGML_VULKAN_DEBUG build: **only 16x16x16** (f16/f32, f16/f16, int8,
fp8, bf16) - no K=32 or N=32/64 forms, so wider coopmat ops are impossible.

Structural shader experiments (all correct per 58/58 FA backend tests, all
reverted; 12k wall vs `7.34s` control):

- P^T-tile reuse in the V phase (P tiles are identical across all 16
  hsv_tiles; 64 loads/workgroup -> 4, register-resident PMat0..3, 4 PV
  accumulators, 4-slot pvsh): `8.0385s` (-9%). LDS grew to 32256 B and the
  PV chains became 16-deep; rejected.
- Same with single PV accumulator and single-slot pvsh (isolated P-reload
  cost): `7.8206s` (-6.5%). P reloads are NOT a bottleneck.
- 64-dim K staging per barrier pair in QK^T (4x fewer barriers: 32 -> 8 per
  workgroup iteration, kvsh 64x18): `7.8853s` (-7.5%). Barriers are NOT the
  bottleneck either.

Combined with cycle-2 (occupancy, latency, LDS) this closes every axis that
distinguishes FA from the 55-65 TFLOPS matmul lane: the gap comes from
N-accumulators + independent warps + 256 threads in mul_mm vs 1 accumulator +
128 threads + softmax barriers in FA. Approximating the matmul structure in
FA blows VGPR/LDS (D081; Br64 Qf/pvsh analysis) and is analytically closed.
A scalar MMQ-style FA (like mul_mmq, which is the actual 65 TFLOPS path) is
the only remaining route and is a multi-week project.

## Cycle-4: real FA cost split on 131k + f16-KV lane (2026-08-04)

The 12k A/B harness turned out to be host/dispatch-bound for FA (empty j-loop
on 12k: only -8% wall; all cycle-2/3 shader gates were masked by this).
Skip-diagnostics on the 131k lane (coopmat1, q8_0 KV, control `50.8s`):

- empty j-loop: `32.4s` (-36% wall) -> j-loop body = ~92% of FA time
- skip dequant-staging: `44.4s` (-12.7%) -> dequant = ~32% of FA
- skip muladds (QK^T + V): `46.2s` (-9%) -> WMMA = ~23% of FA
- skip coopMatLoads: `49.6s` (-2.4%) -> LDS reads = ~6% of FA
- rest (mask/softmax/barriers/stat-writes): ~39% of FA

Dequant-staging is the single largest FA cost. Eliminating it entirely with
an **f16 KV cache** (no staging, coopMatLoad straight from global):

- 131k f16-KV: `41.67 / 39.96 / 40.07s` (runs=3, avg `40.57s` -> ~1416 tok/s)
  vs q8_0 `50.8s` (1241 tok/s): **+14% wall; 1321.7 goal beaten**.
- 58/58 FLASH_ATTN backend tests pass (f16 path).
- Cost: separate lane - f16 KV uses ~945MB vs ~500MB at 131k (fits 2x16GB).
- Scalar/MMQ FA path (flash_attn.comp with MMQ int8 dots) was also A/B'd:
  stock `12.72s`; +LDS staging `8.29s`; +Br16 `7.49s` on 12k (coopmat parity),
  but on 131k `66.6s` (-28% vs coopmat) - rejected; coopmat1 stays default.
  Diagnostics knobs (env-only, no default change) left in vk_pipeline.inc:
  `GGML_VK_FA_FORCE_SCALAR`, `GGML_VK_FA_SCALAR_{BC,DSPLIT,RS,BR,SUBGROUP,STAGING,NOMASK}`.

Open: q8_0 lane still pays the dequant tax; a shader-side dequant
optimization (or pre-dequantized K/V copy kept in sync) could close part of
that gap without changing the KV cache type.

## Cycle-5: int8-coopmat QK^T (2026-08-04) - REJECTED

Goal: make FA consume q8_0 KV without dequantization by using the hardware
s8 x s8 -> s32 cooperative matrix form (present in the 9070 XT shape list).
Implementation under `COOPMAT_INT8` in flash_attn_cm1.comp:

- K staged as raw int8 (copy, no dequantize4); Q quantized to int8 with a
  per-row global amax (`subgroupClusteredMax` returns 0 on this driver -
  worked around via `subgroupMax` accumulated across the two row iterations);
  S kept in an int32 coopmat accumulator; dequant = 1 mul per S element.
- Accuracy: 338/338 q8_0 FA backend tests pass (tolerance 0.0005).
- 131k clean-lane A/B: control `53.9s` / `56.5s` vs int8 `65.1 / 67.2 / 70.4s`
  -> int8 is ~22-25% SLOWER despite removing the dequant tax.

Cause (most likely): the int32 accumulator is 16x16 int32 = 64 VGPR/lane vs
32 VGPR for f16, dropping occupancy from 2 to 1 workgroup per CU; the removed
K-dequant (~32% of FA on q8_0) cannot compensate. int8 WMMA is also not
faster than f16 WMMA on RDNA4.

Conclusion: q8_0 KV has no faster FA path on RDNA4 (scalar/MMQ route and
int8-coopmat both rejected). The f16-KV lane (+14%, 40.6s) remains the only
FA acceleration. All cycle-5 code reverted; post-revert control `56.5s`.

## Cycle-6: q8_0 pre-dequant staging for FA (2026-08-05) - ADOPTED

Insight from ROCm: the HIP path never showed a q8_0 prompt-eval penalty because
ggml-cuda's FA dequantizes K/V into an f16 extra buffer OUTSIDE the attention
kernel (fattn-wmma-f16.cu `K_to_f16`/`V_to_f16`), so the kernel runs the pure
f16 path. The old Vulkan cm1 staged q8_0 with in-kernel dequantize4 per KV
chunk (~32% of FA time on q8_0, measured in cycle-4/5).

Port: in `ggml_vk_flash_attn`, when K/V are q8_0 and N > 1 (prefill), two
`dequant_q8_0` dispatches (existing f16-output shader) convert K/V into
preallocated f16 buffers (`prealloc_fa_k16/v16`, with need_sync fencing, same
pattern as split_k). Effective types (k_type_eff = F16) drive tuning and the
pipeline lookup (`pipeline_flash_attn_f32_f16[k_type_eff]` + FaTypeK=16), so
the kernel runs `flash_attn_f32_f16_aligned_f32accf16` - zero dequant in the
hot loop. Decode (N == 1) keeps q8_0 (smaller KV reads, faster decode).

Results (131k, sequential, same session):
- control q8_0: 54.1s
- q8_0 + preconvert: 47.8s (-11.7%)
- f16 lane: 52.5s  -> q8+preconvert is the fastest lane overall

Accuracy: 3/3 FA backends, 0 FAIL. Memory: transient f16 staging only
(~66 MiB at 131k), KV cache stays q8_0 (2176 MiB vs 4352 MiB for f16).
Toggle: GGML_VK_FA_NO_PRECONVERT=1.

## Cycle-7: MTP on q8_0 KV - acceptance ceiling (2026-08-05)

GUI autotune (ctx=49k, spec=draft-mtp n=2, exact GUI env) prefers f16 KV over
q8_0+preconvert: 5.41 vs 4.57 aggregate tps. Full lane matrix (same session):

| 49k lane | prompt pps | decode dps | draft acceptance |
|---|---|---|---|
| mtp q8_0+preconv | 1296 | 29.4 | 0.397 (56/141) |
| mtp f16 | 1500 | 39.7 | 0.735 (75/102) |
| none q8_0+preconv | 1412 | 26.4 | - |
| none f16 | 1655 | 26.0 | - |
| mtp q8_0 rot-off | 1310 | 26.8 | 0.342 (51/149) |

MTP decode speedup: q8 +11% vs f16 +53%. Root cause: draft acceptance is
capped by q8 KV precision - target logits are computed on an 8-bit cache, so
correct draft tokens disagree with the noisy argmax and get rejected. The
local Hadamard rotation (`attn_rot_k`, src/llama-kv-cache.cpp) already adds
+16% acceptance (0.342 -> 0.397) but cannot close the gap. The FA path is not
the bottleneck (preconvert runs fine, decode N==1 keeps q8 reads).

Default policy: spec=none -> q8_0 + preconvert (fastest lane, 2176 MiB KV);
spec=mtp -> f16 KV (autotune already selects it). Future MTP-q8 options:
q6_k K-cache (more precise 8-bit), or keep f16 KV for MTP profiles.

## Goal

Determine whether the dual-Vulkan backend can match or exceed the dual-ROCm
backend on large-prompt prompt evaluation for the primary Q4_K_M model, and
rank the cheapest code/launch routes that close the measured gap. Any accepted
change must not regress the D089 ROCm production lane, must keep q8 KV as the
quality baseline, and must preserve decode.

## Measured state (same-lane pairs, b8192/ub1024, q8_0/q8_0 KV, FA on,
spec=none, cold/no-reuse)

Fresh binaries (post-D093 wn32 / post-D091 ROCm order):

| Lane | ROCm prompt tok/s | Vulkan prompt tok/s | V/R |
| --- | ---: | ---: | ---: |
| 12k (rcc24576) | 1802.41 (e355 r3) | 1563.57 (validate-20260804) | 0.87 |
| 131k, 59,213-token prompt (rcc220000) | 1411.98 (resume-20260803, `-ts 27,37`) | 1171.94 (validate-20260804, `-ts 1,1`) | 0.83 |

Stale Vulkan rows (pre-D093 broken bn256 default, do not use for the gap):

| Lane | ROCm prompt tok/s | Vulkan prompt tok/s | Note |
| --- | ---: | ---: | ---: |
| 49k (rcc96000) | 1778.59 (e353) | 1432.13 (e332) | Vulkan binary was pre-wn32 |
| 98k (rcc294912) | 1483.41 (d091) | 1171.17 (e332) / 1164.29 (e338) | Vulkan binary was pre-wn32 |
| 131k (rcc294912) | 447.59 (e332, broken split) | 1051.67 (e332, pre-wn32) | historic Vulkan win; ROCm row was an artifact of un-balanced split, not reproduced since (resume = 1412) |

Current gap on the two fresh lanes: Vulkan trails by `1.15x` (12k) and
`1.20x` (131k p60k). The 98k/49k Vulkan rows must be re-measured with the
accepted wn32 default before any supremacy conclusion: the pre-wn32 rows are
not comparable, and 12k improved `381 -> 1563` after the fix.

Historic Q3 precedent (`p002-vulkan130k-big-c152k`): Vulkan `626.06` vs ROCm
`363.81` prompt tok/s on a 57k-token Q3 prompt. Do not transfer this to Q4:
the ROCm row predates the D091 layer-balance/output-placement fixes and the
resume ROCm row at the same scale is now `1411.98`.

## Ceiling model

- 131k target to tie ROCm: `1171.94 -> 1411.98` = `1.2047x` local.
- 98k target to tie ROCm 1483: unknown until fresh Vulkan 98k is measured.
- 12k target to tie ROCm 1802: `1.1528x`; short lane is not the user-facing
  big-prompt lane and should not drive source work.
- H65 already framed the Q4 long-prompt `1500 tok/s` target (`1.2682x` over
  `1182.76` measured on XL) and stopped E281 at the residency gate; D094 is
  the successor program scoped to prompt-eval parity/supremacy on the D089
  model, with launch-level candidates first.

## Dual-GPU reality check (why "two GPUs" is not 2x here)

- Layer split (active on both backends) is a pipeline, not data-parallel:
  D084/D085 closed tensor split (127 host-mediated all-reduce boundaries per
  ubatch, ~3.4 ms each; BF16 native reduce `1032-1043` still below layer
  `1826`; Q8-compressed model `1250-1450` below layer). Windows exposes two
  singleton device groups; no peer primitive.
- Data-parallel (each GPU computes half of the same ubatch rows) requires a
  second model copy: Q4_K_M fits statically across 2x16 GiB only as one copy;
  a second instance does not fit (H65 residency gate).
- The realistic dual-GPU levers are therefore: pipeline stage balance
  (`-ts`), async/transfer-queue overlap, and output/KV placement — not a new
  parallel topology. T14 (hybrid PP/tensor) stays design-only until an
  upstream transport appears.

## Candidate queue (ranked, cheapest first) — cycle 1 status

| ID | Candidate | Type | Status (cycle 1) |
| --- | --- | --- | --- |
| T101 | Vulkan `-ts` sweep on 131k p60k | launch only | CLOSED negative: `5,6` −7.9%, `27,37` −11.6% (vs Q3 D080 precedent); `1,1` stays best, also under gfxq |
| T102 | Fresh Vulkan 98k and 49k controls with wn32 default | launch only | PENDING: re-measure with gfxq accepted; rows exist pre-gfxq only |
| T103 | `GGML_VK_QK_LOW_TILE_SPLIT_K=3` probe | env only | CLOSED negative (−0.5%) |
| T104 | Async/transfer-queue/host-memory probes | env only | CLOSED: transfer queue neutral; host-memory failed run; noasync with gfxq −51% |
| T105 | q5_1/q5_1 KV opt-in | launch + opt-in | CLOSED negative on 131k (−7.9%): FA on this lane is not KV-byte-bound; q8 baseline unchanged |
| T106 | Route/residency evidence pack | diagnostic | DONE: shares 40/46/14 (FA/matmul/rest); ubatch timing linear in KV |
| T107 | Source work | source | OPENED as cycle-2 queue T201-T205 (see above) |

Closed/not-reopened: tensor split (D084/D085), all-KV on one device
(H64: `815 tok/s`), output-placement-only fixes (D006), ubatch sweeps
(D003 cliff), `GGML_VK_DISABLE_F16` and broad f16 pivots (E259/E264),
Q3-only helpers (quad dequant, FFN down split-k), FA tile geometry
(D076/D081 + cycle-1 row_split/Bc128/f16acc/split-k probes), int-dot
q8_1 for quant matmul (E099).

## Legacy-path inventory (code audit, 2026-08-04)

- `GGML_VK_AMD_LARGE_MATMUL_VARIANT`: wn32 accepted (D093); `bn256` legacy
  default regressed to ~400 tok/s on the 131k lane; `GGML_VK_DISABLE_AMD_WN32_DEFAULT`
  and legacy `..._BN256_DEFAULT` alias remain as A/B switches.
- Generic (large-matmul disabled) route: 915.39 tok/s at 12k — slower than
  wn32 1094.53; not a legacy win.
- `q4_low_tile_candidate` in `ggml_vk_guess_split_k` (vk_dispatch.inc:80-84):
  the Q4_K shape branch exists but the AMD proprietary default activates only
  the Q3 branch — Q4 is a dormant code path, reachable via
  `GGML_VK_QK_LOW_TILE_SPLIT_K` (T103).
- MMVQ: prompt (`n=1024`) does not use MMVQ (`n > 1` routes to `mul_mat_q_f16`);
  decode uses it (`k=5120 >= 2048`, AMD default true). `GGML_VK_FORCE_MMVQ` /
  `GGML_VK_DISABLE_MMVQ` are decode diagnostics only.
- Q8_1/int-dot `quantize_y` prompt path exists for F32 src1 (E099 rejected on
  Q3); not re-tested on Q4 — low priority, include in T104 batch only if free.
- Upstream port check: `110d9caf3..b06fbc968` contains only 6 ggml-vulkan
  commits (MoE topk fusion, POOL_1D, Intel FWHT guard, quantized CONCAT, conv
  layout checks, IQ4_NL FA) — none touch Q4 matmul/FA q8/split-k/dual-device.
  Nothing to port. Upstream is behind this fork on the relevant Vulkan paths.

## Gate plan (sequential, one hardware owner) — cycle 1 status

1. Gate A: fresh paired controls — Vulkan 98k and 49k with wn32 default
   (`-dev Vulkan1,Vulkan0 -sm layer -ts 1,1`, output Vulkan1), adjacent ROCm
   control on the same session; r1 each, then r3 for the promising one.
   *(cycle 1: 131k control measured twice (1201.26/1201.54); 98k/49k still
   pending with gfxq.)*
2. Gate B: `-ts` sweep on 131k p60k — **done, negative**: `5,6` −7.9%,
   `27,37` −11.6%; `1,1` stays.
3. Gate C: env probes T103/T104 — **done, negative/neutral** (lowtile −0.5%,
   asyncTQ ±0, q5_1 −7.9%, q4_0 −0.5%, no-mmap −1%, ub2048 −1.2%,
   host-memory failed, noasync −51%).
4. Gate D: evidence pack — **done**: FA 40%, quant MUL_MAT ~46%, SSM tail
   ~6%; trace and perf artifacts under `build_logs/agent-workload/d094-*`.
5. Gate E: q5_1 KV opt-in probe — **done, negative** on this lane.
6. Decision (cycle 1): launch-level stack cannot reach `1.15x`; supremacy
   gap is `1.1375x` with the accepted graphics-queue win. Proceed to
   source-work queue (T205 first, then T201/T202) — see above.

## Stop conditions

- Any probe that regresses decode or the ROCm lane is reverted immediately.
- Do not change q8 KV default for speed; q5_1 stays opt-in behind quality
  smoke.
- Do not reopen tensor split, all-KV placement, or ubatch sweeps.
- Keep every measured row label-prefixed (`d094-...`) with full lane
  contract in `BENCH_RUNS.csv`.

## Validation trail

- This note contains no new measurements; all rows cite existing CSV labels.
- First hardware artifact must be Gate A (paired controls) with diagnostics
  preserved; then the note is updated with the trace pack before source work.
