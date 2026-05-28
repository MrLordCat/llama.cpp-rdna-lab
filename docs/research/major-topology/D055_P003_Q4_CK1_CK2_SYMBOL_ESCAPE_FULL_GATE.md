# D055 - P003 Ck-1/Ck-2 full symbol+escape atlas gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only checkpoint (full unsampled)

## Purpose

Complete Ck-1 and Ck-2 from
`docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md` using a full
unsampled atlas run.

This supersedes D054 fast sampling for checkpoint closure.

## Method

Command:

```bash
python scripts/research/q4_c2_symbol_atlas.py \
  --model models/Qwen3.6-27B-Q4_K_S.gguf \
  --label q4c2-symbol-atlas-qwen36-27b-q4ks-full-r1 \
  --chunk-blocks 16384
```

Run mode:

- no tensor cap (`max_tensors=0`),
- no per-tensor block cap (`max_blocks_per_tensor=0`),
- full Q4 tensor coverage.

## Measured outputs

From
`build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-full-r1.q4_c2_symbol_atlas.json`:

- total model: `15.004 GiB`
- Q4-covered bytes: `12.355 GiB` (`82.35%` of model)
- payload bytes analyzed: `10.983 GiB`
- Q4 tensors analyzed: `348`
- global payload entropy: `3.864885 bpw`
- entropy ratio vs fixed 4-bit payload: `0.966221`
- expected fixed bits from active-symbol count: `3.999999`
- global active-symbol histogram:
  - `K=16`: `91,735,445` blocks
  - `K=15`: `392,783` blocks
  - `K<=14`: `1,052` blocks total

## Interpretation vs Ck-1/Ck-2

Ck-1 (symbol entropy atlas):

- Full-corpus entropy remains below `4.0`, confirming non-uniform payload
  statistics.
- Practical theoretical headroom is still narrow: about `4.0 - 3.864885 =
  0.135115 bpw` before headers/side-costs.

Ck-2 (active-symbol/escape atlas):

- Full-corpus active-symbol distribution is strongly concentrated at full
  alphabet (`K=16`).
- Sparse-alphabet tails exist but are small in global weight, so simple
  escape-driven gains on dominant tensors are limited.

## Gate decision

1. Ck-1 status: closed (full atlas complete).
2. Ck-2 status: closed (full active-symbol atlas complete).
3. H45 (EBNS): remains viable, but must pass strict overhead and decode-complexity
   budgets due narrow headroom.
4. H46 (SPRE): downgraded for dominant-tensor impact; can survive only as a
   mixed-policy auxiliary where tails are demonstrably favorable.
5. Prototype unlock remains blocked until Ck-3/Ck-4/Ck-5 are completed.

## Next theory steps

- Ck-3: tuple repetition/dictionary coverage atlas.
- Ck-4: decode complexity budget model.
- Ck-5: mixed-policy optimizer and ranked shortlist.

## Related artifacts

- `docs/research/major-topology/D054_P003_Q4_CK1_CK2_SYMBOL_ESCAPE_FAST_GATE.md`
- `docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md`
- `scripts/research/q4_c2_symbol_atlas.py`
- `build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-full-r1.q4_c2_symbol_atlas.json`
- `build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-full-r1.q4_c2_symbol_atlas.md`
