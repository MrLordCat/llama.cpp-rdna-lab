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
- [ ] Bit-exact check vs dump_blocks.py — IN PROGRESS (see below).
- [ ] 3.3 CPU `vec_dot_q4_K16_q8_1` (scalar first, SIMD optional).
- [ ] 3.4 GPU minimal (backend Vulkan first): type traits, dequant_row for ppl, model loads with -ngl all, ppl matches CPU.
- [ ] 3.5 `src/llama-quant.cpp` slim policy (16.44 GB target).
- [ ] Acceptance: bit-exact dumps, unit tests, quantize size, ppl/KLD table.

## 2026-08-15 — §3.1+§3.2 done, bit-check WIP

- Blocks added in `ggml-common.h` (316/312/300 B with static_asserts),
  traits in `ggml.c`, quantize/dequantize in `ggml-quants.c`
  (make_qkx2_quants(16, 15, rmin=-1, rdelta=0.1, nstep=20, mad=false);
  imatrix: w = qw*sqrt(2*sum(x^2)/512 + x^2); else w = sqrt(sum(x^2)/16)+|x|;
  final levels recomputed from packed d/sc/dmin/m with denom==1 fallback).
- Tools: `scripts/research/q4_k16_harness.cpp` (links ggml-base.a) +
  `scripts/research/q4_k16_bitcheck.py` (random data, 3 configs, with/without
  imatrix; compares d/dmin/ls/lm/qs byte-wise + dequant).
- Fixes found by the check: imatrix row stride (`quant_weights + row*n_per_row`),
  harness argv off-by-one (qw ignored), dmin +/-0.0 sign artifact.
- REMAINING BLOCKER (needs user decision): the prototype computes
  make_qkx2 in float64 and np.rint; the C port (like upstream llama.cpp)
  is float32 with the 12582912.f half-even trick. The search is extremely
  sensitive: on identical data the f64 and f32 paths can pick DIFFERENT
  least-squares branches (e.g. scale 0.07956 vs 0.08709, min 0.4988 vs
  0.5325 - both nearly equal error), so byte-exact equality is impossible
  in principle. Measured residue: ~1-12 ls/2048, ~4-50 qs/16384 per config.
  Options: (A) make the prototype float32 (one-line style change in
  quants.py, then C == prototype bit-for-bit; subproject metrics may shift
  slightly and should be re-run); (B) implement the C path in double to
  mirror the f64 prototype (out of upstream style, slower quantize);
  (C) relax the criterion to value-equality. Decision pending user.
