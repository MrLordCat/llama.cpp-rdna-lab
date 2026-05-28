# D056 - P003 Ck-3 fast tuple atlas gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only checkpoint (fast exploratory)

## Purpose

Start Ck-3 (tuple repetition atlas) with a bounded fast pass before full
unsampled closure.

## Method

Command:

```bash
python scripts/research/q4_c2_tuple_atlas.py \
  --model models/Qwen3.6-27B-Q4_K_S.gguf \
  --label q4c2-tuple-atlas-qwen36-27b-q4ks-fast-r1 \
  --chunk-blocks 8192 \
  --max-tensors 24 \
  --max-blocks-per-tensor 131072
```

Scope:

- tuple lengths: `L=2,3,4`
- dictionary sizes: `K=16,64,256,1024`
- sampled Q4 tensors: `24`
- analyzed payload: `0.375 GiB` (`805,306,368` symbols)

## Measured outputs

From
`build_logs/agent-workload/q4c2-tuple-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_tuple_atlas.json`:

- best modeled bpw (`fixed-code + escape-literal` proxy):
  - `L=2`: `4.500003 bpw` at `K=256` (`coverage=1.000000`)
  - `L=3`: `5.528656 bpw` at `K=1024` (`coverage=0.534506`)
  - `L=4`: `5.242850 bpw` at `K=16` (`coverage=0.001788`)

## Interpretation

- Under the current conservative tuple coding proxy, sampled tuple-dictionary
  routes do not approach the required C2 corridor (`3.57-3.77 bpw`).
- `L=2` collapses to full 2-symbol alphabet (`K=256`) and behaves like a wider
  index coding, not a compression win.
- `L=3/L=4` show insufficient effective coverage for the tested fixed-code
  dictionary settings.

## Gate decision

- Keep H47 open only as a conditional candidate pending full unsampled Ck-3 and
  possible model refinements.
- No prototype unlock from D056-fast.
- Immediate next step is full unsampled Ck-3 to confirm whether this negative
  signal persists at corpus scale.

## Artifacts

- `scripts/research/q4_c2_tuple_atlas.py`
- `build_logs/agent-workload/q4c2-tuple-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_tuple_atlas.json`
- `build_logs/agent-workload/q4c2-tuple-atlas-qwen36-27b-q4ks-fast-r1.q4_c2_tuple_atlas.md`
- `docs/research/major-topology/P003_Q4_C2_THEORY_BACKLOG.md`
