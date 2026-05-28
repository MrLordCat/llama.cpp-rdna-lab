# D054 - P003 Ck-1/Ck-2 fast symbol+escape atlas gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only checkpoint (no prototype)

## Purpose

Close the first analytical checkpoints from
`P003_Q4_C2_THEORY_BACKLOG.md`:

1. Ck-1: symbol entropy atlas,
2. Ck-2: active-symbol/escape atlas.

This run is intentionally a fast exploratory pass (sampled), not the final
full-model atlas.

## Method

Command:

```bash
python scripts/research/q4_c2_symbol_atlas.py \
  --model models/Qwen3.6-27B-Q4_K_S.gguf \
  --label q4c2-symbol-atlas-qwen36-27b-q4ks-fast-r1 \
  --chunk-blocks 16384 \
  --max-tensors 24 \
  --max-blocks-per-tensor 131072
```

Sampling scope used in this gate:

- largest Q4 tensors only: `24`
- per-tensor block cap: `131072`
- analyzed Q4 payload bytes: `402,653,184` (`0.375 GiB`)

## Measured outputs

From
`build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_symbol_atlas.json`:

- global payload entropy: `3.867718 bpw`
- entropy ratio vs fixed 4-bit payload: `0.966929`
- active-symbol fixed-bit bound from `K_active`: `4.000000 bits`
- global active-symbol histogram (K=16 bin): `3,133,359` blocks
- global active-symbol histogram (K=15 bin): `12,333` blocks
- global active-symbol histogram (K<=14 bins): `36` blocks total

## Interpretation vs Ck-1/Ck-2

Ck-1 (entropy atlas):

- Non-uniformity exists (`3.8677 < 4.0`), but margin is modest
  (`~0.1323 bpw` theoretical headroom before side overhead).
- Fast pass does not support an aggressive entropy-only expectation yet.

Ck-2 (active-symbol/escape atlas):

- Practical sparse-alphabet assumption is weak on this sampled slice.
- Almost all sampled blocks are full-alphabet (`K=16`), with tiny tails at
  `K=15` and almost no `K<=14` events.
- This is unfavorable for SPRE-style escape savings on dominant tensors unless
  additional structure exists beyond simple active-symbol count.

## Gate decision

- Keep H45 (EBNS) open, but with stricter overhead budgeting because entropy
  headroom appears narrow.
- Downgrade near-term confidence for H46 (SPRE) on dominant tensors based on
  sampled `K_active` evidence.
- Do not unlock prototypes yet. Continue theory-only pipeline with:
  - full-scope Ck-1/Ck-2 run (no sampling caps),
  - Ck-3 tuple atlas,
  - Ck-4 decode complexity budget.

## Caveats

This is a fast exploratory gate and must not be treated as final global proof.
The full-corpus unsampled atlas remains required before any candidate ranking
or prototype authorization.

## Artifacts

- `scripts/research/q4_c2_symbol_atlas.py`
- `build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_symbol_atlas.json`
- `build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_symbol_atlas.md`
- `docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md`
