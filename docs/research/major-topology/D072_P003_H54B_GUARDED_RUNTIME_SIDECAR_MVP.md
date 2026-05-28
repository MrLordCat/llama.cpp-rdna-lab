# D072 - P003 H54-B guarded runtime sidecar MVP

Date: 2026-05-28  
Owner: Copilot/perf workspace  
Status: runtime-sidecar prototype (guarded, fail-closed)

## Scope

First runtime-oriented prototype step after D071: build a fail-closed sidecar
artifact that can be used by a future runtime gate without changing defaults.

## What Was Implemented

New script:

- `scripts/research/q4_metacomp_guarded_runtime_sidecar.py`

Design:

- Input guarded manifest from D071.
- Process only `selected=true` rows.
- For each selected Q4 tensor, build value-aware 16-level codebook on capped
  sample (`max-elements-per-tensor`).
- Preserve explicit fallback entries for all non-selected or invalid rows.
- Emit runtime gate contract as opt-in env pair and explicit rollback.

Fail-closed contract:

- Missing tensor -> fallback.
- Non-Q4 tensor -> fallback.
- Dequant/build failure -> fallback.
- Non-selected rows -> fallback.

## Run

- model: `models/Qwen3.6-27B-Q4_K_S.gguf`
- guarded manifest: `q4metacomp-guarded-prototype-qwen36-27b-q4ks-r1`
- label: `q4metacomp-guarded-runtime-sidecar-qwen36-27b-q4ks-r3`
- sample cap: `131072` elements per tensor
- selected cap: `64` tensors (MVP bounded pass)

## Results

- selected rows: `64`
- fallback rows: `0`
- selected entropy (orig sample): `3.867974 bpw`
- selected entropy (new sample): `3.279216 bpw`
- selected entropy delta: `-0.588758 bpw`

Runtime gate metadata included in sidecar:

- enable: `LLAMA_Q4_METACOMP_ENABLE=1`
- sidecar path: `LLAMA_Q4_METACOMP_SIDECAR=<path>`
- rollback: `unset LLAMA_Q4_METACOMP_ENABLE LLAMA_Q4_METACOMP_SIDECAR`

## Decision

Keep as guarded runtime-sidecar MVP artifact.

- No default runtime behavior changed.
- Next step is runtime reader/integration under the same fail-closed contract,
  followed by controlled A/B and quality checks.

## Artifacts

- `scripts/research/q4_metacomp_guarded_runtime_sidecar.py`
- `build_logs/agent-workload/q4metacomp-guarded-runtime-sidecar-qwen36-27b-q4ks-r3.q4_metacomp_guarded_runtime_sidecar.json`
- `build_logs/agent-workload/q4metacomp-guarded-runtime-sidecar-qwen36-27b-q4ks-r3.q4_metacomp_guarded_runtime_sidecar.md`
