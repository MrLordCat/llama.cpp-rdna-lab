# E185 - DFlash Speed Gate (ROCm Q3 4k)

## Metadata

- Experiment ID: E185
- Date: 2026-05-22
- Owner: Copilot
- Branch/Commit: local working tree (no commit yet)
- Target lane: DFlash speed sanity gate (no promotion)

## Hypothesis

DFlash should provide a strong speedup versus `spec=none` on ROCm for the active Qwen3.6-27B Q3 lane.

## Method

Matched A/B runs via `scripts/agent_workload_bench.py` on `build-rocm-vec/bin/llama-server.exe`.

Shared shape:

- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- ctx/batch/ubatch: `4096 / 2048 / 1024`
- KV: `q4_0 / q4_0`
- flash-attn: on
- tasks: `quick` + `triage_diff`
- max tokens: `64`
- runs: `1`

Tested pairs:

1. Cold (`--no-reuse`) baseline `spec=none` vs DFlash.
2. Reuse-enabled baseline `spec=none` vs DFlash.

Also tested two draft choices:

- external draft: `models/Qwen3.5-9B-Q6_K.gguf`
- self-draft: `models/Qwen3.6-27B-Q3_K_S.gguf`

## Results

### Cold pair (no-reuse)

- baseline: `e185-rocm4k-short-spec-none-r1` -> `25.93 TPS`
- DFlash + external draft: `e185-rocm4k-short-dflash-r1` -> `2.96 TPS` (`0.114x`, `-88.6%`)
- DFlash + self-draft: `e185-rocm4k-short-dflash-selfdraft-r1` -> `1.69 TPS` (`0.065x`, `-93.5%`)

### Reuse-enabled pair

- baseline: `e185-rocm4k-reuse-spec-none-r1` -> `25.09 TPS`
- DFlash + self-draft: `e185-rocm4k-reuse-dflash-selfdraft-r1` -> `1.71 TPS` (`0.068x`, `-93.2%`)

## Diagnostics Notes

For external draft (`Qwen3.5-9B`), server diagnostics include repeated decode failures:

- `decode: failed to initialize batch`
- `llama_decode: failed to decode, ret = -1`

This indicates the current DFlash path is not yet in a performant/stable configuration for this lane.

## Decision

- Outcome: Reject for speed claims.
- Keep only as diagnostic evidence.

## Next Technical Steps

1. Fix DFlash decode/batch initialization failures first (stability gate).
2. Enforce stricter draft compatibility checks for DFlash (not only presence of draft model).
3. Re-benchmark with lane-matched A/B only after stability gate passes.
