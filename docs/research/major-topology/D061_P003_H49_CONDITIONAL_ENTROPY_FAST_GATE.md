# D061 - P003 H49 conditional entropy fast gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only fast exploratory gate

## Purpose

Start H49 analytical queue from D060 by measuring first-order conditional
entropy signal on representative Q4 payload scope.

## Method

New script:

- `scripts/research/q4_c2_conditional_entropy_atlas.py`

Run:

```bash
python scripts/research/q4_c2_conditional_entropy_atlas.py \
  --model models/Qwen3.6-27B-Q4_K_S.gguf \
  --label q4c2-condent-atlas-qwen36-27b-q4ks-fast-r1 \
  --chunk-blocks 16384 \
  --max-tensors 24 \
  --max-blocks-per-tensor 131072
```

## Results

From
`build_logs/agent-workload/q4c2-condent-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_conditional_entropy_atlas.json`:

- sampled Q4 bytes: `1.740 GiB`
- sampled payload bytes: `0.375 GiB`
- global unigram entropy: `3.867718 bpw`
- global first-order conditional entropy: `3.867718 bpw`
- global delta (`H1 - Hcond`): `0.000000 bpw`

Tensor-level deltas are near-zero (`~0.000004-0.000006 bpw`) on top sampled
payload tensors.

## Interpretation

1. First-order adjacent-symbol context on raw nibble stream does not provide a
   meaningful compression headroom lift on this fast scope.
2. This is a negative signal for the naive H49 formulation that assumes strong
   first-order locality.
3. H49 is not closed yet, but continuation requires a materially different
   context definition (for example, higher-order/page-structured contexts or
   alternative symbol ordering) before full unsampled run.

## Gate decision

1. H49 fast gate outcome: negative on current first-order stream formulation.
2. Do not run full unsampled with the same formulation.
3. Rework H49 context model first, then re-enter fast gate.

## Related artifacts

- `docs/research/major-topology/D060_P003_Q4_C2_REOPEN_ADMISSION_GATE.md`
- `scripts/research/q4_c2_conditional_entropy_atlas.py`
- `build_logs/agent-workload/q4c2-condent-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_conditional_entropy_atlas.json`
- `build_logs/agent-workload/q4c2-condent-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_conditional_entropy_atlas.md`
