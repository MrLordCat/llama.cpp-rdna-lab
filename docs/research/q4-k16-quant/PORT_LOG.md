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

- [ ] 3.1 `ggml/include/ggml.h` + `ggml-quants.h`: types Q4_K16_M/Q4_K16/Q4_K16_S, COUNT, quantize/dequantize declarations.
- [ ] 3.2 `ggml-quants.c/.h` + `ggml.c` type tables: quantize/dequantize (shared impl, sc_bits/min_bits), blck_size 512, type_size 316/312/300, quantize_chunk/dequantize cases.
- [ ] 3.3 CPU `vec_dot_q4_K16_q8_1` (scalar first, SIMD optional).
- [ ] 3.4 GPU minimal (backend TBD): type traits, dequant_row for ppl, model loads with -ngl all, ppl matches CPU.
- [ ] 3.5 `src/llama-quant.cpp` slim policy (16.44 GB target).
- [ ] Acceptance: bit-exact dumps, unit tests, quantize size, ppl/KLD table.
