# W23: MMQ WMMA-I8 for MXFP4 prefill is +18-20% (opt-in env measured)

Date: 2026-09-09

## Question

Where is WMMA-I8 used today, and what would L1/L2 numbers be with/without it?

## Findings

1. WMMA-I8 is the **only** MMQ path on RDNA4 (mma.cuh:1200 +
   mmq_select_vec_dot(use_dp4a=false)). MMQ is used ONLY for small ubatch:
   Q4_K ne11<=256, Q5_K<=256, Q2/Q3/Q6<=192, MXFP4/NVFP4/Q8_0... <=128
   (`ggml_cuda_should_use_mmq`, mmq.cu). Our bench2 contract uses
   batch=8192 / ubatch=1024 -> ne11=1024 -> **prefill runs dequant+hipBLAS**,
   and a runtime trace (GGML_TRACE_MMQ_PATH=1) shows ZERO `mul_mat_q_case`
   calls on L1. So in the current L1/L2 runs, WMMA-I8 has no prefill effect.

2. Forced-MMQ A/B (`GGML_CUDA_FORCE_MMQ_RUNTIME=1` -> all MUL_MAT_2D go
   through MMQ/WMMA-I8) on the MXFP4 model:

| run | prefill tok/s | decode tok/s |
| --- | --- | --- |
| L1 MXFP4 baseline (hipBLAS) | 1894.25 | 28.4916 |
| L1 MXFP4 force-MMQ (WMMA-I8) | **2276.99** | 28.1944 |
| L2 MXFP4 baseline (hipBLAS) x2 | 1799.47 / 1794.80 | 25.72 / 25.24 |
| L2 MXFP4 force-MMQ (WMMA-I8) | **2122.14** | 25.4422 |

- L1 prefill: **+20.2%**; L2 prefill: **+18.1%** (vs avg 1797.15).
- decode unchanged (noise). Aggregate L2 9.88 vs 9.29 (+6.4%).

3. Control on Q4_K (same force): L1 prefill 1847.68 vs 1915.98/1909.14
   baseline -> **-3.5%**. Confirms G08 (Q4_K MMQ is slower; hipBLAS is
   chosen for a reason). So the +18-20% is MXFP4-specific.

## Why MXFP4 wins while Q4_K loses

- MXFP4 MMQ: weights decode through a single 32-entry E8M0 scale and a
  look-up table, A-fragment is int8; the WMMA-I8 kernel does 4096 MAC/tile
  with 8 bytes/lane and no per-element dequant in the hot loop; hipBLAS
  needs a full dequant pass (int8->float) before GEMM, costing an extra
  weight read + F32 workspace.
- Q4_K MMQ: d + dmin + scales (dual-scale unpack per block) makes the MMQ
  load path heavier than hipBLAS's dequant; G08 measured hipBLAS faster at
  ne11 549/919/1024.

## Recommendation

For MXFP4/NVFP4 production prefill, route large batches to MMQ (WMMA-I8)
instead of hipBLAS. Suggested gate: extend mmq.cu RDNA4 switch with
`case GGML_TYPE_MXFP4: return ne11 <= ggml_rdna4_mxfp4_mmq_max_ne11();`
(env `GGML_MMQ_RDNA4_MXFP4_MAX_NE11`, default 128 to preserve behavior;
set to 4096 to adopt). Validate with 3-run A-B-A on L1+L2 before making it
default. Optional: a second check at ubatch 4096/8192 to confirm no
rollover.

## Artifacts

- logs: /tmp/mxfp4_l1_forcemmq.log, /tmp/mxfp4_l2_forcemmq.log,
  /tmp/q4k_l1_forcemmq.log, /tmp/mxfp4_l2_a2.log
- CSV: build_logs/bench/mxfp4-ab/*forcemmq*, *mxfp4-l2-a2*
