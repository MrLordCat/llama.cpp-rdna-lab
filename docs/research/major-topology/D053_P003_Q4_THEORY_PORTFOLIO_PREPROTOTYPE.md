# D053 - P003 theory portfolio (pre-prototype only)

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only gate (no prototype)

## Purpose

Shift P003 to a pure theory/research phase before any converter/runtime
implementation.

User requirement for this phase:

1. no implementation prototypes yet,
2. focus on new compression concepts,
3. preserve quality and performance constraints,
4. only move to prototypes after a working theory is established.

## Fixed numeric constraints from D050-D052

For `models/Qwen3.6-27B-Q4_K_S.gguf` and target `13.0 GiB`:

- current total: `15.004 GiB`
- required reduction: `2.004 GiB`
- metadata-only ceiling: `1.373 GiB`
- mandatory residual after full metadata path: `0.631 GiB`

Therefore C2 (payload-side) is mandatory. Required effective payload corridor:

- best case (full C1 metadata): about `3.7701 bpw`
- weaker C1 (`meta_save_frac=0.60`): about `3.5701 bpw`

So theory must justify a practical payload corridor around `3.57-3.77 bpw`
without quality/perf collapse.

## New C2 Theory Portfolio (no code)

### T-C2-1: Entropy-bounded nibble stream (EBNS)

Concept:

- Keep exact 4-bit symbols as source alphabet.
- Pack symbols by superblock using entropy coding with strict decode budget.
- Use fixed-size decode pages to keep GPU access deterministic.

Why it can work:

- Q4 symbol distributions are typically non-uniform by tensor/layer.
- If average entropy is below `4.0`, effective payload can drop below `4.0 bpw`.

Main risks:

- decode control-flow divergence,
- random-access overhead,
- quality is safe only if coding is strictly lossless for symbols.

Pre-prototype evidence required:

1. entropy histograms per tensor and per superblock,
2. expected bpw under bounded page headers,
3. decode-step complexity model (ops/byte, branches, table loads).

### T-C2-2: Superblock palette remapping with escapes (SPRE)

Concept:

- For each superblock, remap 16-symbol nibble alphabet to a smaller active
  local alphabet when usage is sparse.
- Encode indices with variable local bitwidth and explicit escape path for rare
  symbols.

Why it can work:

- many blocks do not use all 16 symbols uniformly.
- local alphabet reduction can lower average payload bits without changing value
  semantics.

Main risks:

- control complexity of local dictionaries,
- extra lookup latency,
- pathologically flat distributions reduce benefit.

Pre-prototype evidence required:

1. active-symbol-count distribution across tensors,
2. projected bpw with dictionary/escape overhead,
3. worst-case fallback ratio.

### T-C2-3: Pattern dictionary coding for nibble tuples (PDNT)

Concept:

- treat short nibble runs/tuples as macro-symbols,
- use a bounded dictionary per tensor family,
- keep fallback literal mode for uncovered tuples.

Why it can work:

- transformer weights often show repeated local nibble motifs after quantization.

Main risks:

- dictionary miss penalty,
- larger side metadata,
- memory locality risk in decode path.

Pre-prototype evidence required:

1. tuple-frequency Zipf curves,
2. top-K coverage vs dictionary size,
3. net bpw after dictionary overhead.

### T-C2-4: Layer-adaptive mixed C2 policy (LAMC2)

Concept:

- do not enforce one payload method globally,
- select best C2 mode per tensor (or per tensor class),
- keep strict fallback to legacy payload where confidence is low.

Why it can work:

- tensor statistics are heterogeneous; one-size-fits-all is likely suboptimal.

Main risks:

- policy complexity,
- fragmentation of runtime routes,
- configuration drift.

Pre-prototype evidence required:

1. per-tensor candidate ranking,
2. expected global bpw from mixed policy,
3. fallback share estimate.

## Quality and performance hard guardrails (theory phase)

Before any prototype is allowed, each theory candidate must pass all four
paper-gates:

1. Compression gate:
   - predicted net payload in target corridor (`3.57-3.77 bpw`) after realistic
     headers/fallback overhead.
2. Correctness gate:
   - coding transform must be symbol-lossless or have explicitly bounded error
     with recovery plan (preferred: fully lossless).
3. Runtime budget gate:
   - estimated decode overhead must stay below a bounded ceiling relative to
     legacy Q4 decode path.
4. Integration gate:
   - clear fail-closed fallback path to legacy Q4 for unsupported tensors.

## Pre-prototype research deliverables

No implementation prototypes yet. Required next deliverables are analytical:

1. entropy atlas for Q4 symbols by tensor family,
2. active-symbol atlas and escape-rate forecast,
3. tuple/dictionary coverage atlas,
4. comparative cost model (bpw benefit vs decode complexity),
5. ranked shortlist of top two C2 candidates for later prototype authorization.

## Decision

Freeze implementation work for P003 C2 until the above theory package is
complete. Continue only with analytical studies and documented design gates.

## Related artifacts

- `docs/research/major-topology/D050_P003_Q4_TARGET13_FEASIBILITY_GATE.md`
- `docs/research/major-topology/D051_P003_Q4_PHASE1_TENSOR_PLAN_GATE.md`
- `docs/research/major-topology/D052_P003_Q4_C2_REQUIREMENTS_TABLE.md`
- `docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md`
