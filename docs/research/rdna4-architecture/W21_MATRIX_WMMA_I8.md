# W21: RDNA4 matrix hardware - WMMA-I8 measured 5.85-6.3x DP4A (quality check + matrix audit)

Date: 2026-09-09

## Part 1 - MXFP4 quality check (perplexity, 512 chunks / corpus 924 KB)

- Corpus: concatenated repo docs (`/tmp/qc_corpus.txt`, ~240k tokens).
- Contract: `llama-perplexity -c 512 -b 512 -ub 512 --chunks 512 -ngl 999
  -dev ROCm1,ROCm0 -sm layer -ts 1,1 --flash-attn on --no-warmup --no-mmap`.

| model | PPL | delta |
| --- | --- | --- |
| Qwen3.8-27B-Q4_K_M | **6.7802 +/- 0.05045** | - |
| Qwen3.8-27B-MXFP4-requant (from Q4_K) | **7.1937 +/- 0.05471** | **+0.4135 (+6.1%)** |

- The MXFP4 file is a REQUIREMENT test (requant from Q4_K, no imatrix, 4.25
  BPW). The PPL penalty is expected and is NOT the best achievable MXFP4:
  a native BF16->MXFP4 with imatrix should be closer to Q4_K quality. The
  +6.1% here is the requant worst-case; production decision needs the native
  path or acceptance of the trade (decode +15%, MTP +22% at +0.41 PPL).

## Part 2 - Matrix hardware audit (RDNA4 / gfx1201)

Findings:
1. **gfx1201 has INT8 WMMA**: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`
   in rocWMMA 2.2.1 (`wmma_impl.hpp:1476`), m16n16k16, int8 x int8 -> i32,
   signed; each lane holds 8 bytes A, 8 bytes B, 8 i32 acc. Also 16x16x8
   legacy (`VRegI32x1`) and gfx1250 16x16x32/64 variants.
2. **No native FP4 WMMA on RDNA4** (rocWMMA has only f8_fnuz/bf8 via MFMA;
   llama.cpp `mma_block_scaled_fp4` is BLACKWELL_MMA_AVAILABLE only -
   `kind::mxf4` NVIDIA instruction).
3. llama.cpp already has gfx1201 WMMA for **f16/bf16** MMQ
   (`mma.cuh:936/1125/1160`), and **int-quantized MMQ also uses WMMA-I8**
   on RDNA4 (`mma.cuh:1200` + `mmq_select_vec_dot`); the earlier
   "int types stay DP4A" statement is corrected by
   [W22](W22_MATRIX_PREFILL_AUDIT.md).
4. **MMVQ decode cannot use WMMA** (ncols=1; WMMA needs >=16 columns), so
   matrix hardware cannot speed single-token decode - the decode limit stays
   the weight-stream bandwidth (W20 ~25.7 t/s MXFP4).
5. **MMQ prefill ALREADY runs on WMMA-I8** (verified in source + trace);
   therefore the matrix-hardware port does not exist as a new opportunity.
   The prefill limiter is NOT the MMQ compute (W17: MMQ 7.8% of prefill
   time; MXFP4 prefill neutral - W20).

## Part 3 - WMMA-I8 microbenchmark (gfx1201)

- `scripts/research/w21_bench_wmma_i8.hip` (8 independent accumulators,
  128 blocks x 256 threads, N=20M).
- Build: `hipcc -O3 --offload-arch=gfx1201 -o /tmp/w21/bench_wmma_i8 ...`;
  run with `LD_LIBRARY_PATH=/home/chris/rocm/lib`.

| | DP4A | WMMA-I8 | ratio |
| --- | --- | --- | --- |
| rep 0 | 29 810 GMAC/s | 187 766 GMAC/s | 6.30x |
| rep 1 | 31 871 GMAC/s | 186 524 GMAC/s | 5.85x |
| rep 2 | 31 773 GMAC/s | 185 848 GMAC/s | 5.85x |

- **WMMA-I8 = 5.85-6.3x the raw MAC rate of DP4A** on gfx1201. If a MMQ path
  can replace the DP4A inner loop with m16n16k16 i8 WMMA, the prefill compute
  floor drops ~6x; real gain is limited by load/convert overhead and memory
  streaming (expected 1.5-3x on prefill, not 6x).

## Decision / next

- **Quality**: MXFP4-requant = +0.41 PPL; usable for speed runs, needs native
  BF16->MXFP4 (+imatrix) for a quality claim.
- **Matrix**: decode-MMVQ WMMA is impossible (ncols=1); prefill MMQ already
  uses WMMA-I8 (see W22). There is no new MMQ-i8 port to build.
- **Actual next levers** (bandwidth/geometry or speculative):
  - native BF16->MXFP4 (+imatrix) quality gate;
  - MTP state and acceptance tuning;
  - decode weight-stream geometry (W-follow-up) - or GDN trace if time.
- Microbench result (186 T MAC/s) is now a confirmation that WMMA-I8 is
  already loaded, not a roadmap item.
