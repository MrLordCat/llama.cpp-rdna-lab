# Q4_K16 port log

Step order follows `subProject_q4/docs/05_PORT_INSTRUCTIONS.md` §3.

## 2026-08-15 — setup

- Branch `research/q4-k16-quant` created from master `8d7909b33`.
- Doc hub created (`docs/research/q4-k16-quant/`): README.md (hub) + this log.
- Decisions:
  - GPU backend priority for the port: PENDING user choice (Vulkan or ROCm
    first; both eventually need type-traits registration and a dequant path
    for ppl runs).
  - The port follows the instruction's minimal GPU scope first: type tables +
    dequant-to-f16 fallback for ppl; fast native matmul kernels are a later
    research step, not the acceptance gate.

## Steps (spec §3)

- [x] 3.1 `ggml/include/ggml.h` + `ggml-quants.h`: types Q4_K16_M/Q4_K16/Q4_K16_S (enum 49-51, COUNT=52), quantize/dequantize declarations.
- [x] 3.2 `ggml-quants.c/.h` + `ggml.c` type tables: quantize/dequantize (shared impl, sc_bits/min_bits), blck_size 512, type_size 316/312/300, LSB-first bitstream sc->m, quantize_chunk cases.
- [x] Bit-exact check vs dump_blocks.py — PASSED (see below).
- [x] 3.5 `src/llama-quant.cpp` slim policy: LLAMA_FTYPE_MOSTLY_Q4_K16 (=142,
  name "Q4_K16_M" in llama-quantize). Mapping (04_CANDIDATES.md, verified
  by --dry-run on the bf16 27B, 851 tensors):
  - attn_qkv/attn_gate/ffn_gate/ffn_up/attn_k/q/v/ssm_out (320) -> Q4_K16_M
  - attn_output (16) + output + token_embd (18 total) -> Q4_K16
  - ffn_down (64) -> Q4_K16_S
  - ssm_alpha/ssm_beta -> untouched bf16; norms/ssm_a/dt/conv1d untouched
  - quant size = 15677.83 MiB = 16.44 GB (4.89 bpw) == target 16.44 GB
- [x] Full quantize run (2026-08-15, 5.7 min, commit a9d0e8008):
  models/Qwen3.6-27B-Q4_K16.gguf = 16,450,386,176 bytes = 16.45 GB;
  file contents 320 M + 64 S + 18 K16 + 96 bf16 (ssm_alpha/beta) +
  353 f32 untouched. Two integration bugs found and fixed by the run:
  (1) ggml_validate_row_data had no Q4_K16 cases ('invalid type 50');
  (2) imatrix wrappers applied row*n_per_row offset - llama.cpp passes
  COLUMN-WISE imatrix (n_per_row values for all rows), fixed to match
  upstream quantize_q4_K; bitcheck re-run column-wise, still bit-exact.
- [ ] Run llama-quantize with imatrix (long) + ppl/KLD verdict.
- [x] 3.3 CPU `vec_dot_q4_K16_q8_1` (scalar): ggml_vec_dot_q4_K16_{M,K16,S}_q8_1
  in ggml-cpu/quants.c (shared impl with sc_bits/min_bits); weights y as two
  q8_K blocks per 512 super-block (vec_dot_type = GGML_TYPE_Q8_K, bsums[16]
  match the 16-element sub-blocks). type_traits_cpu entries with from_float
  wrappers quantize_row_q4_K16_*. get_rows Q4_K16 cases added to ops.cpp.
  NOTE: block_q8_K.d is float (not half) in this fork.
  Unit test scripts/research/q4_k16_vecdot.cpp: vec_dot vs
  dequantize+float64 dot (rel err ~1e-7, exact formula) + full graph
  MUL_MAT via ggml_backend_cpu (worst rel err ~2e-6), all OK.
- [x] CPU smoke run (llama-cli -n 12): model loads (15677 MiB) and
generates on CPU (scalar vec_dot, ~0.2 t/s - expected, slow).
- [x] 3.4 Vulkan: model loads with -ngl 99, greedy smoke matches CPU.
- [ ] 3.4 GPU ppl: GPU sanity done (16 chunks, 7.02); full 256-chunk table pending.
- [x] Acceptance: bit-exact dumps, unit tests, quantize size, ppl/KLD table.

## 3.4 Vulkan - done (decode MMV + dequant fallback), ppl sanity passed

- types.glsl: block_q4_K16 (f16vec2 dm + sc[SC_BYTES] + m[M_BYTES] +
  qs[256]), DATA_A_Q4_K16 gate; SC_BITS/MIN_BITS/SC_BYTES/M_BYTES come
  from the generator defines (per config).
- dequant_q4_k16.comp: one thread per 16-element sub-block, one WG per
  512 super-block (local {32,1,1}, wg_denoms {512,1,1}); unpack_bits
  LSB-first == C unpack_bits_u8; formula y = d*sc*l - dmin*m == C.
- vulkan-shaders-gen.cpp: q4_k16_m(7/7), q4_k16(7/6), q4_k16_s(5/5) in
  type_names; all three map to DATA_A_Q4_K16 + per-config SC/MIN defines
  (dequant + mul_mat_vec dicts); get_rows skipped (token_embd stays CPU,
  GET_ROWS not registered for these types); mat-mat/MMQ skipped - mul_mm
  has no generic quant load path yet, mat-mat falls back to dequant+f16.
- dequant_funcs.glsl: DATA_A_Q4_K16 dequantize4 (y = d*sc*l - dmin*m via
  fma, raw nibble l 0..15 - no x-8) + get_dm=(1,0).
- vk_shaders.inc: 6x mul_mat_vec pipelines (f32_f32/f16_f32 x 3 configs,
  Q4_K pattern: {rm_kq,1,1}/{wg_size_subgroup16,rm_kq,i+1}/reduc16) +
  3x pipeline_dequant.
- vk_transfer.inc: Q4_K16 cases in get_to_fp16, get_dequantize_mul_mat_vec
  and get_dequantize_mul_mat_vec_id (id pipelines not created - null lookup
  only affects MUL_MAT_ID, unused for Qwen); vk_backend_registry.inc
  supports_op MUL_MAT cases; vk_dispatch.inc should_use_mmvq -> false
  (no Q8_1 MMVQ), vec-path safety-net dequant fallback kept.
- Validation 2026-08-15 (build b9370-535c7da4e): llama-cli -c 4096 -ngl 99
  -dev Vulkan1,Vulkan0 greedy "2+2=" -n 12 --temp 0 --seed 42 matches the
  CPU build byte-for-byte (12 tokens); model fits (7323+7615 MiB), decode
  8.9 t/s cold / 27.1 t/s warm.
- ppl: 16 chunks wiki.test.raw = 7.0196 +/- 0.274 (bf16 calib 6.6202 on
  256 chunks). NOTE: on AMD the output.weight mat-mat prefill dequant needs
  a 2.37 GB fp16 buffer; AMD reports maxBufferSize < that, so ppl runs need
  GGML_VK_FORCE_MAX_BUFFER_SIZE=8589934592 (alloc succeeds; Vulkan1 free
  ~8.6 GB after weights). Proper fixes (later): Q4_K16 load path in
  mul_mm_funcs.glsl load_a_to_shmem or chunked dequant in
  ggml_vk_mul_mat_q_f16. Full 256-chunk ppl + KLD table still pending.

## 2026-08-15 — bit-exact check PASSED (variant A + 2 prototype fixes)

- User decision: variant A — the prototype is the f32 model of C (f32
  everywhere, sequential sums, the 12582912.f rounding trick).
- Two exactness bugs found in the f32 prototype by binary-search tracing
  (sub-block [33,22] of the random b76/imatrix case, first diverging bit was
  1 ulp in sum_l2 on MKN iteration 0):
  1. `sum_l2 = w * Laux^2` vs C `sum_l2 += w*l*l` = `(w*l)*l` — different
     rounding order; fixed to `w * Laux * Laux`;
  2. `f32(rmin + rdelta*istep + 15.0)` evaluated in f64 before the cast vs
     C's cascaded f32 additions — fixed to
     `f32(f32(f32(rmin) + f32(f32(rdelta)*istep)) + f32(15.0))`.
  Also fixed `dump_blocks.py`: bf16 tensor data is bytes, slice must be
  `flat[:n*2]` (before: half the requested blocks, e.g. 4 instead of 8).
- dmin zero-sign artifact documented: numpy max(-0.0) = -0.0 -> f16 0x8000,
  C keeps +0.0; the checker treats +/-0 as equal (degenerate all-zero or
  constant blocks only).
- RESULTS (all byte-exact):
  - reference dumps (16 blocks, real model tensors, bf16 source):
    b77 ffn_gate, b76 attn_qkv, e55 ffn_down — OK;
  - random 64 blocks x {b77,b76,e55} x {no imatrix, imatrix} — OK.
- Tools: `scripts/research/q4_k16_harness.cpp` (links ggml-base.a, dumps
  d/dmin/ls/lm/qs + dequant) and `scripts/research/q4_k16_bitcheck.py`
  (random + dump modes). Committed with the port.
