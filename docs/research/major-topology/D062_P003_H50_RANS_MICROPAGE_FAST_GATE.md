# D062 - P003 H50 rANS micropage fast gate

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: theory-only fast exploratory gate

## Purpose

Execute the first H50 fast gate from D060 backlog using a bounded-rANS
micropage overhead model anchored to full-corpus entropy from D055.

## Method

New script:

- `scripts/research/q4_c2_rans_micropage_gate.py`

Run:

```bash
python scripts/research/q4_c2_rans_micropage_gate.py \
  --source-symbol-atlas-json build_logs/agent-workload/q4c2-symbol-atlas-qwen36-27b-q4ks-full-r1.q4_c2_symbol_atlas.json \
  --label q4c2-rans-micropage-qwen36-27b-q4ks-fast-r1
```

Model assumptions:

- source entropy floor uses D055 full `H1=3.864885 bpw`,
- micropage symbols scanned in `{512,1024,2048,4096,8192}`,
- per-page headers scanned in `{64,96,128}` bits,
- coder inefficiency scanned in `{0.010,0.020,0.035}` bpw.

## Results

From
`build_logs/agent-workload/q4c2-rans-micropage-qwen36-27b-q4ks-fast-r1.q4_c2_rans_micropage_gate.json`:

- feasible config count under corridor `<=3.7701 bpw`: `0`
- best modeled point: `3.882697 bpw`
  (`page_symbols=8192`, `header_bits=64`, `coder_overhead=0.010`)
- best-point margin vs corridor max: `-0.112597 bpw`

## Interpretation

1. Naive order-0/bounded-rANS micropage formulation cannot reach the D052
   corridor even under optimistic overhead settings.
2. The dominant blocker is the D055 source entropy floor itself (`3.864885`),
   already above corridor max before practical overhead terms.
3. H50 continuation requires a materially stronger entropy model (for example,
   conditional/context restructuring) rather than simple micropage/header tuning.

## Gate decision

1. H50 fast gate outcome: negative for current order-0 micropage formulation.
2. Do not run full unsampled with this formulation.
3. Park H50 until redesigned with a context model that can move source entropy
   floor below corridor while staying within complexity budget.

## Related artifacts

- `docs/research/major-topology/D060_P003_Q4_C2_REOPEN_ADMISSION_GATE.md`
- `scripts/research/q4_c2_rans_micropage_gate.py`
- `build_logs/agent-workload/q4c2-rans-micropage-qwen36-27b-q4ks-fast-r1.q4_c2_rans_micropage_gate.json`
- `build_logs/agent-workload/q4c2-rans-micropage-qwen36-27b-q4ks-fast-r1.q4_c2_rans_micropage_gate.md`
