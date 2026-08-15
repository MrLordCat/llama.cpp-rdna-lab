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
- [ ] 3.3 CPU `vec_dot_q4_K16_q8_1` (scalar first, SIMD optional).
- [ ] 3.4 GPU minimal (backend Vulkan first): type traits, dequant_row for ppl, model loads with -ngl all, ppl matches CPU.
- [ ] Acceptance: bit-exact dumps, unit tests, quantize size, ppl/KLD table.

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
