# E266: ROCm MTP Depth-8 Decode Profile

Date: 2026-07-07

## Goal

Re-check the Unsloth Qwen3.6 MTP expectation on the local ROCm/RDNA4 path after
the MTP hot-path fixes. Earlier `n_max=1..2` measurements were below baseline
despite healthy acceptance, which made the result look like a ROCm backend
failure. This experiment tests whether the missing gain is actually a
ROCm-specific draft-depth choice.

## Kept Contract

- Model: `models/Qwen3.6-27B-Q3_K_S_mtp.gguf`
- Backend: ROCm, default two-GPU layer split, `-ngl 999`
- Context: `ctx=4096`
- Shape: `batch=512`, `ubatch=128`
- KV: `q4_0/q4_0`, FlashAttention on
- Workload: `quick:triage_diff`
- Output: `max_tokens=256`
- Sampling: `temperature=0.0`, `top_p=0.9`
- Cold policy: `--no-reuse --no-v2-prime-pass`, thinking enabled
- Server: `build-rocm-vec/bin/llama-server.exe`

Code state includes:

- MTP hook-prefill skips the vocab LM head when no logits are needed.
- MTP hidden rows are copied in one bulk transfer before per-sequence stashing.
- MTP draft sampling uses GPU argmax where available.

## Results

Short prompt, r3 confirmation:

| Mode | Label | Aggregate TPS | Decode tok/s | Prompt tok/s | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | `mtp-temp0-postbuild-none-confirm3` | `25.6412` | `26.29` | `744.72` | - |
| MTP `n_max=8` | `mtp-temp0-postbuild-n8-confirm3` | `42.1258` | `44.68` | `515.81` | `54.33%` |

Delta: `1.6428x` aggregate completion TPS and `1.699x` decode tok/s versus
the same ROCm two-GPU baseline.

Depth scan, r1:

| MTP depth | Aggregate TPS | Decode tok/s | Acceptance |
| ---: | ---: | ---: | ---: |
| `4` | `24.9690` | `26.00` | `76.19%` |
| `6` | `21.5639` | `22.34` | `62.42%` |
| `8` | `41.5827` | `44.55` | `54.33%` |
| `10` | `37.8592` | `40.41` | `44.61%` |
| `12` | `37.2331` | `39.73` | `38.59%` |

Small repo-context check (`ctx=4096`, safe-capped to `1577` prompt tokens):

| Mode | Label | Aggregate TPS | Decode tok/s | Prompt tok/s | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | `mtp-repoctx-temp0-none-r1` | `22.3927` | `26.36` | `943.14` | - |
| MTP `n_max=8` | `mtp-repoctx-temp0-n8-r1` | `29.3322` | `41.54` | `620.31` | `50.37%` |

## Decision

Keep the MTP code hot-path fixes and treat `--spec-draft-n-max 8` as the current
measured ROCm two-GPU generation-heavy profile for this local Qwen3.6 MTP GGUF.
Do not conclude that the CUDA-guide `n_max=2` is optimal on RDNA4/ROCm; the
local best is hardware/backend dependent.

Do not promote this as a 130k/60k-prompt headline yet. MTP still slows prompt
eval because hook-prefill must advance the MTP context, so long-prompt claims
need a separate same-lane A/B. That follow-up is E267, where the large-prompt
gate is negative until the MTP prefill overhead is fixed.

## Artifacts

- `build_logs/agent-workload/mtp-temp0-postbuild-none-confirm3.diagnostics.md`
- `build_logs/agent-workload/mtp-temp0-postbuild-none-confirm3.server.log`
- `build_logs/agent-workload/mtp-temp0-postbuild-n8-confirm3.diagnostics.md`
- `build_logs/agent-workload/mtp-temp0-postbuild-n8-confirm3.server.log`
- `build_logs/agent-workload/mtp-repoctx-temp0-none-r1.diagnostics.md`
- `build_logs/agent-workload/mtp-repoctx-temp0-n8-r1.diagnostics.md`
