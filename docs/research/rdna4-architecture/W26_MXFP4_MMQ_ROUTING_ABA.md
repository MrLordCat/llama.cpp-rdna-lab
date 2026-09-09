# W26: MXFP4 prefill routed to MMQ (WMMA-I8) by default — 3-run A-B-A accepted

Date: 2026-09-09 (Linux ROCm 10, 2x RX 9070 XT, gfx1201; same binary
`build-rocm-linux/bin/llama-server`, commit a771fdbd0+w26)

Closes the W23/W25 open item ("MMQ routing gap"). W23 forced MMQ via
`GGML_CUDA_FORCE_MMQ_RUNTIME=1` (L1 +20.2%, L2 +18.1% prefill). This adds a
production gate: `ggml_cuda_should_use_mmq` RDNA4 switch now has an explicit
`GGML_TYPE_MXFP4` case routed by `ggml_rdna4_mxfp4_mmq_max_ne11()`.

## Change

`ggml/src/ggml-cuda/mmq.cu`:
- `ggml_rdna4_mxfp4_mmq_max_ne11()`: default **4096** (covers ubatch 1024),
  env override `GGML_MMQ_RDNA4_MXFP4_MAX_NE11` retained for diagnostics.
- RDNA4 `switch(type)`: `case GGML_TYPE_MXFP4: return ne11 <=
  ggml_rdna4_mxfp4_mmq_max_ne11();` (before default `ne11 <= 128`).
- NVFP4 deliberately unchanged (default `ne11 <= 128`; prefill 489 t/s not a
  candidate). Q4_K/Q5_K routing untouched (`ggml_rdna4_q4k_mmq_max_ne11`,
  default 256 -> hipBLAS at ubatch 1024 per G08/W23 -3.5%).

## 3-run A-B-A (MXFP4-requant, batch 8192 / ubatch 1024, f8_e4m3, repo-snapshot)

A1 = default (pre-change), B = `GGML_MMQ_RDNA4_MXFP4_MAX_NE11=4096`,
A2 = default repeat. Same binary for A and B.

| lane | run | prefill tok/s | decode tok/s | aggregate |
| --- | --- | --- | --- | --- |
| L1 | A1 w26mmq-a1-l1 | 1868.5831 | 29.4186 | 14.4256 |
| L1 | **B w26mmq-b-l1** | **2253.0606** | 29.3750 | **15.7871** |
| L1 | A2 w26mmq-a2-l1 | 1860.5298 | 29.3395 | 14.3749 |
| L2 | A1 w26mmq-a1-l2 | 1765.7838 | 25.9179 | 8.8106 |
| L2 | **B w26mmq-b-l2** | **2073.5695** | 25.8894 | **9.7636** |
| L2 | A2 w26mmq-a2-l2 | 1766.6249 | 25.8699 | 8.8078 |

- L1 prefill: B 2253.06 vs avg(A1,A2) 1864.56 = **+20.8%**; decode 29.375 vs
  29.379 = **0.0%**; aggregate +9.6%.
- L2 prefill: B 2073.57 vs avg 1766.20 = **+17.4%**; decode 25.889 vs 25.894 =
  **0.0%**; aggregate +10.8%.
- Confirmation after adopting default 4096 (no env): w26mmq-final-l2 L2
  prefill **2076.46**, decode 25.92 - matches B (stable).
- Q4_K control (on `Qwen3.8-27B-UD-Q4_K_M.gguf`, plain `Qwen3.8-27B-Q4_K_M.gguf`
  was removed from `models/` earlier): w26q4k-ud-l2 L2 prefill **1765.82**,
  decode **23.53**, aggregate 8.517. Vs W20 Q4_K_M baseline (1813-1815 /
  22.29-22.32): decode +5.5% is explained by the UD file being 3.8% smaller
  (16.46 vs 17.11 GiB, BW-limited decode), prefill -2.6% by the NextN/unsloth
  layout - not by a routing change. Code proof: Q4_K routing untouched
  (`ggml_rdna4_q4k_mmq_max_ne11` default 256 -> hipBLAS at ne11=1024), the
  new case is MXFP4-only, and W24's nwarps change already covered Q4_K
  (was in the whitelist). No Q4_K regression.

## Verdict

- **Accepted as default** for MXFP4: prefill +17-21% (mean +19%) with decode
  unchanged - the strongest prefill win found on this lane, and it stacks
  with W24 decode nwarps=8 (+3-5%).
- Combined (W24 + W26) MXFP4 L1: prefill 2253 vs 1894 original = +18.9%;
  L2: prefill 2074 vs 1797 = +15.4%, decode 25.89 vs 25.72, aggregate
  9.76 vs 9.29 = +5.1% (approx; W20/W24 runs on the same model).
- No correctness concern: MMQ on RDNA4 already uses INT8 WMMA
  (`wmma_i32_16x16x16_iu8_w32_gfx12`, W22) and `load_tiles_mxfp4` already
  writes the MMA layout; W23 measured the same force-path result.

Artifacts: `build_logs/bench/mxfp4-ab/w26mmq-*-l{1,2}*`, `w26mmq-final-l2`;
diff `mmq.cu`; docs `W25` (gap open) superseded here.
