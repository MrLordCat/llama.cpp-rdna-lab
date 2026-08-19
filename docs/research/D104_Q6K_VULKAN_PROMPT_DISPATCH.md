# D104 — Q6_K Vulkan prefill dispatch (RDNA4)

Date: 2026-08-16. Branch: research/q4-k16-quant. Owner: coordinator.

## Goal

Close the Q6_K Vulkan prefill gap vs Q4_K_M on the RX 9070 XT dual-GPU rig
(measured 2026-08-16, 12K lane, spec=none, quick tasks):

| config | prompt t/s | decode t/s |
|---|---|---|
| Q4_K_M (direct coopmat, no env) | 1493.4 | 29.5 |
| Q6_K direct coopmat m (no env)   | 930.2  | 24.0 |
| Q6_K direct coopmat l/wn32 + output=Vulkan1 (GUI env) | 985.8 | 24.1 |
| Q6_K GGML_VK_DISABLE_COOPMAT=1 (int8 dot) | 336.9 | 23.6 |
| Q6_K l variant sweep: wn48 252.5, bn64 246.0, block128-wn32 982.6 | | |

Decode gap (-18%) tracks the +23% weight bytes (bandwidth-bound) and is out of
scope. Target: prefill parity-to-Q4_K_M-per-byte, i.e. >= ~1220 t/s on the 12K
lane (Q4_K_M 1493 x 21.05/17.11 size ratio puts the pure-bandwidth floor at
~1214).

## Root cause (code evidence)

Q6_K prefill uses the coopmat (WMMA f16) `mul_mm` path. Its A-side dequant,
`mul_mm_funcs.glsl` `DATA_A_Q6_K` (~line 308):

- processes **2 values per idx** (generator default `load_vec_quant=2` for
  q6_k; Q6_K block is 210 B, not 4-byte divisible, so only 2-byte loads);
- per idx: `ql` (2 B) + `qh` (2 B) loads, shift/mask/or/`unpack8`, `-32`, scale
  mul -> one `f16vec2` store; `data_a[ib].d` is re-loaded on every idx
  (128x per block), `scales[is]` repeats every 16 idx;
- the `d`/`scales` loads + all ALU are repeated **per batch row**: the coopmat
  workgroup map is `batch_idx = gl_WorkGroupID.z` (1 WG per batch element,
  `mul_mm.comp` ~line 157), so the whole weight matrix is re-dequantized for
  every prefill row (m=512 -> 512x).

Q4_K needs fewer ops per value (4 values per 4-byte load) and was tuned
(D093-D095 wn32 l-shape), Q6_K got neither. Int8 (DP4a) alternative is 3x
worse; wn32 l is the best available shape (wn48/bn64 collapse 4x).

## Plan

- R1 (probe, no shader changes): env-gate `qx_needs_dequant` for Q6_K in
  `ggml_vk_mul_mat_q_f16` (vk_dispatch.inc) -> the existing generic route
  dequant_q6_k -> f16 staging (once per op, amortized over 512 rows) -> f16
  coopmat matmul. A/B on the 12K lane. This is the E139 structure; E139
  rejected it for Q3_K (Q3_K has a quad-dequant amortization Q6_K lacks), so
  Q6_K is a better candidate.
- R2 (if R1 insufficient): micro-opt the `DATA_A_Q6_K` branch (hoist
  d/scales, 4 values per iteration within 2-byte loads, cheaper bit math).
- R3 (if R1 wins): production route gating (type+shape policy, opt-in env or
  default after 49K confirmation) + revert guard.

## Lane contract (unchanged)

12K: agent_workload_bench.py quick tasks, runs 1, real-context 24576 chars,
-n 99, -dev Vulkan1,Vulkan0 -sm layer -ts 1,1 -fit off, --cache-ram 0,
--ctx-checkpoints 0, --seed 42, no-warmup. Canonical batch/ubatch from
2026-08-19: b8192/ub1024 (the R3 49K runs below are the one exception,
b512/ub128). GUI env baseline for Q6_K runs:
GGML_VK_FORCE_AMD_LARGE_MATMUL=1, GGML_VK_AMD_LARGE_MATMUL_VARIANT=wn32,
LLAMA_OUTPUT_DEVICE=Vulkan1 (985.8/24.1 reference).

Artifacts: build_logs/agent-workload/d104-*.

## Status

- [x] R1 predequant probe — **rejected**: 201.7/23.2 t/s vs direct 985.8/24.1
  (f16 staging re-reads 2.67x bytes per row vs 6-bit direct; same outcome as
  E139 for Q3_K). Probe gate `GGML_VK_PREDEQUANT_Q6K` left in vk_dispatch.inc
  for reuse; do not enable by default.
- [x] R2-v1: Q6_K coopmat dequant 4 values/idx (LOAD_VEC_A 2->4) — correct
  (11/11 focused Q6_K MUL_MAT cases pass on Vulkan0) but **rejected**:
  924.4/24.0 vs 985.8/24.1 (prompt -6.2%). Reverted. Root cause hypothesis:
  register pressure of 2x f16vec2 per idx in the l/wn32 warptile.
- [x] Route audit: the executed Q6_K matmul pipeline is **cm1 coopmat**
  (pipeline-create trace: matmul_q6_k_f32_f16acc_s spv_size=25116 = cm1
  binary; plain 13928 never created). The int8 `mul_mmq` shader is only
  generated for the `_q8_1` (quantize_y) path, never for prefill; the
  336.9 t/s `GGML_VK_DISABLE_COOPMAT` number is FA-regression-confounded,
  not an int8-matmul measurement.
- [x] R2-v2 (int8 mul_mmq route) — **not pursued**: the only non-q8_1 int8
  shader generation exists for the quantize_y path; routing prefill there
  would need new pipeline + warptile work, and the DP4a probe was already
  3x worse than WMMA. Closed as moot.
- [x] R2-v3 l-variant sweep — **exhausted, all rejected**: wm128-wn32
  244.0/23.6, block512 252.1/24.1 (both 4x collapse), joining wn48 252.5,
  bn64 246.0. The 256x128/wn32 l-tile is a sharp optimum; every alternative
  shape collapses 4x. bm64 skipped (pattern unambiguous).
- [x] R3 production gating + 49K gate — **closed 2026-08-19, gate NOT
  passed** (details in the R3 section below).

## R3: 49K gate (2026-08-19, Qwen3.6-27B-Q6_K, q8_0 KV, spec=none,
dual Vulkan, b512/ub128 lane exception, interleaved controls)

| config | prompt t/s | decode t/s |
|---|---:|---:|
| stock | 824.73 / 790.05 / 832.16 | 22.84 / 22.70 / 22.87 |
| wn32 + OUTPUT_DEVICE=Vulkan1 | 831.12 / 838.12 | 22.25 / 22.11 |
| wn32 only | 831.43 | 22.91 |
| OUTPUT_DEVICE=Vulkan1 only | 833.13 | 22.21 |

- The 12K prompt win (+6%) does not survive to 49K: in the warmed-up
  final triple (wn32-only 831.43 / outdev-only 833.13 / stock 832.16) all
  configs tie within 0.2%. The larger deltas in the first two pairs are
  drift (stock-r2 790.05 is an outlier below both its neighbours).
- `LLAMA_OUTPUT_DEVICE=Vulkan1` costs a stable -2.5..-3% decode
  (22.21-22.25 vs 22.70-22.91) — the cross-device output move hurts the
  per-token chain on Q6_K at 49K, more than the -0.5% measured for
  Q4_K_M in D105 §5.
- wn32 alone is decode-neutral (22.91, best single run) and prompt-
  neutral at 49K.

Verdict: no production gating change. The GUI checkboxes stay manual; do
NOT add `LLAMA_OUTPUT_DEVICE=Vulkan1` to any Q6_K preset (measured decode
cost on long context). The D104 12K recommendation (env pair for
prefill-heavy 12K use) stays valid but is not promoted to a default.

Artifacts: `d104-r3-stock-r{1,2,3}`, `d104-r3-wn32-r{1,2}`,
`d104-r3-wn32only-r1`, `d104-r3-outdev-r1`.

## Verdict

Q6_K Vulkan prompt (coopmat-cm1, WMMA) is ALU/dequant-bound in the A-shmem
stage; the canonical config is the GUI env pair + wn32 (985.8/24.1). The
micro-opt that would fix it (4 values/idx) costs 6% from register pressure,
the predequant route costs 4.9x from staging bandwidth, int8 costs 3x, and
the tile shape is already optimal (4x cliffs around it). The remaining gap
vs Q4_K_M (1493 t/s) is the price of 6-bit weights + 2-byte loads; no
shader-only fix found in this round. Production recommendation after R3:
for prefill-heavy 12K Q6_K use, enable wn32 via the manual GUI checkbox,
KV q8_0; do NOT set LLAMA_OUTPUT_DEVICE=Vulkan1 (stable -2.5..-3% decode
at 49K), and no automatic preset gating.
