# D057 - P003 Ck-3 full tuple atlas gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only checkpoint (full unsampled)

## Purpose

Close Ck-3 from
`docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md` using a full
unsampled tuple atlas run.

This supersedes D056-fast sampling for checkpoint closure.

## Method

Command:

```bash
python scripts/research/q4_c2_tuple_atlas.py \
  --model models/Qwen3.6-27B-Q4_K_S.gguf \
  --label q4c2-tuple-atlas-qwen36-27b-q4ks-full-r1 \
  --chunk-blocks 32768 \
  --progress-interval-sec 3
```

Run mode:

- no tensor cap (`max_tensors=0`),
- no per-tensor block cap (`max_blocks_per_tensor=0`),
- full Q4 tensor coverage (`348` tensors),
- live progress confirmed from GGUF load through tensor/chunk processing.

## Measured outputs

From
`build_logs/agent-workload/q4c2-tuple-atlas-qwen36-27b-q4ks-full-r1.q4_c2_tuple_atlas.json`:

- total model: `15.004 GiB`
- Q4-covered bytes: `12.355 GiB` (`82.35%` of model)
- payload bytes analyzed: `10.983 GiB`
- payload symbols: `23,585,095,680`
- tuple atlas global best modeled bpw (`fixed-code + escape-literal proxy`):
  - `L=2`: `4.500000 bpw` at `K=256`, `coverage=1.000000`
  - `L=3`: `5.514632 bpw` at `K=1024`, `coverage=0.538009`
  - `L=4`: `5.242693 bpw` at `K=16`, `coverage=0.001827`

Runtime trace quality note:

- full run completed in `592.7s` with periodic live progress messages,
  confirming no hidden stall.

## Interpretation vs Ck-3 target

C2 corridor from D052 remains about `3.57-3.77 bpw`.

Under the current Ck-3 tuple coding proxy:

- none of tested tuple lengths (`L=2,3,4`) approach the required corridor,
- best global modeled point remains at `4.5 bpw` (`L=2`),
- longer tuples require larger indexes/escape cost and remain far above target.

Therefore the current H47 tuple-dictionary formulation is not competitive as a
primary C2 route.

## Gate decision

1. Ck-3 status: closed (full atlas complete).
2. H47 status: rejected for current fixed-code+escape formulation as a primary
   route toward target13 corridor.
3. H47 may only re-open if a materially different tuple coding model is proposed
   with a new overhead and complexity proof.
4. Prototype unlock remains blocked until Ck-4 and Ck-5 are completed.

## Next theory steps

- Ck-4: decode complexity budget model.
- Ck-5: mixed-policy optimizer and ranked shortlist.

## Related artifacts

- `docs/research/major-topology/D056_P003_Q4_CK3_TUPLE_ATLAS_FAST_GATE.md`
- `docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md`
- `scripts/research/q4_c2_tuple_atlas.py`
- `build_logs/agent-workload/q4c2-tuple-atlas-qwen36-27b-q4ks-full-r1.q4_c2_tuple_atlas.json`
- `build_logs/agent-workload/q4c2-tuple-atlas-qwen36-27b-q4ks-full-r1.q4_c2_tuple_atlas.md`
