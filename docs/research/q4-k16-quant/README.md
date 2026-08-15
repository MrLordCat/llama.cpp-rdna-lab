# Q4_K16 quant port hub

## Status

- Branch: `research/q4-k16-quant` (created 2026-08-15 from master
  `8d7909b33`, after the W13 C1b ROCm promotion).
- Owner: user + agent, shared.
- Docs here are kept SEPARATE from the rdna4-architecture track
  (`docs/research/rdna4-architecture/`) so the two experiments never mix.

## Goal

New 4-bit quant **Q4_K16** whose quality (SNR/wSNR) beats Q6_K at the size
of Q4_K_M. Full instructions and the Python source of truth live in the
subproject (separate git repo `subProject_q4`, gitignored in this fork):

- `subProject_q4/docs/05_PORT_INSTRUCTIONS.md` — the C++ port instruction
  (read first; this hub only mirrors status and decisions, not the spec).
- `subProject_q4/prototype/quants.py` — `quantize_q4_k16` /
  `dequantize_q4_k16` (source of truth for byte-level comparison).
- `subProject_q4/docs/04_CANDIDATES.md` — candidate results.

## Block layout (summary, mirror of spec §1)

Superblock 512 = 32 subblocks x 16. Fields: fp16 `d`, fp16 `dmin`,
uint8 `sc[32]`, uint8 `m[32]`, nibbles `qs[512]` (256 B, low nibble = even
index). Decode: `x = (d*sc[i]/NQS)*L - (dmin*m[i]/NQM)`, uniform levels 0..15.

| Type | sc_bits | min_bits | bytes/block | bpw | Use |
| --- | --- | --- | ---: | ---: | --- |
| Q4_K16_M | 7 | 7 | 316 | 4.9375 | attn_qkv/gate, ffn_gate/up, attn_k/q/v, ssm_out |
| Q4_K16 | 7 | 6 | 312 | 4.875 | attn_output |
| Q4_K16_S | 5 | 5 | 300 | 4.6875 | ffn_down |

## Acceptance criteria (summary of spec §5)

1. Byte-identical C++ quantize vs `prototype/dump_blocks.py` (all 4 configs,
   >= 16 blocks, 2+ tensors, with and without imatrix).
2. Unit tests: quantize->dequantize roundtrip, SNR > 24 dB.
3. `llama-quantize` with the slim policy -> 16.44 GB ± 0.1.
4. ppl/KLD on `wiki.test.raw`: target ppl <= ppl(Q6_K) at size <= Q4_K_M.
   If the true ppl comes out WORSE than Q6_K despite a correct bit-exact
   port — record it as a legitimate research result.

## Port progress

See `PORT_LOG.md` for the step-by-step journal (spec §3 order).

## Fences

- Work only on `research/q4-k16-quant`; do not mix with master/rdna4 work.
- Never touch `subProject_q4/**` from this fork's git (separate repo).
- No GPU bench runs without the user signal; follow the driver-safety rules
  from AGENTS.md for any backend work.
- Backends stay CPU/Vulkan/ROCm only (SUPPORTED_BACKENDS.md).
