# E267: ROCm MTP Big-Prompt Gate

Date: 2026-07-08

## Goal

Check whether the positive E266 ROCm MTP `n_max=8` result survives the practical
large-prompt lane. E266 confirmed a generation-heavy speedup, but did not claim
the 130k-context / 60k-prompt-token workflow because MTP hook-prefill still has
to advance the MTP context.

## Kept Contract

- Model: `models/Qwen3.6-27B-Q3_K_S_mtp.gguf`
- Backend: ROCm, default two-GPU layer split, `-ngl 999`
- Server: `build-rocm-vec/bin/llama-server.exe`
- Context: `ctx=131072`
- Shape: `batch=512`, `ubatch=128`
- KV: `q4_0/q4_0`, FlashAttention on
- Workload: `quick:triage_diff`
- Prompt source: `--real-context-mode repo-snapshot --real-context-chars 152000`
- Prompt tokens: `56371`
- Output: `max_tokens=256`
- Sampling: `temperature=0.0`, `top_p=0.9`
- Cold policy: `--no-reuse --no-v2-prime-pass`, thinking enabled
- Background policy: fail if another server is already running

## Results

| Mode | Label | Aggregate TPS | Wall s | Prompt tok/s | Prompt ms | Decode tok/s | Decode ms | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `mtp-bigprompt-none-r1` | `2.1774` | `117.5701` | `537.93` | `104793.21` | `20.19` | `12682.62` | - |
| MTP `n_max=8` | `mtp-bigprompt-n8-r1` | `1.5997` | `160.0303` | `386.54` | `145835.45` | `18.15` | `14102.51` | `54.62%` |

Delta versus same-lane baseline:

- Aggregate completion TPS: `-26.5%`
- Prompt eval throughput: `-28.1%`
- Decode throughput: `-10.1%`
- Prompt eval time: `+41.04 s`
- Wall time: `+42.46 s`

MTP statistics from `mtp-bigprompt-n8-r1`:

- `#gen tokens = 379`
- `#acc tokens = 207`
- local acceptance: `54.62%`
- `#calls(get,decode) = 375 384`
- MTP internal decode time: `1555.522 ms`
- MTP sample time: `0.858 ms`

## Interpretation

Acceptance is not the blocker. The large-prompt run has acceptance in the same
range as the successful E266 short/generation-heavy lane, but the wall clock is
dominated by prompt processing. The MTP run adds about `41 s` to prompt eval on
the `56371`-token prompt, which is larger than any possible decode-side win for a
`256`-token answer. At `ctx=131072`, decode is also about `10%` slower with MTP
enabled, so this lane needs more than draft-depth tuning.

The likely next code target is the MTP prefill/hook path: either make
`handle_mtp_for_ubatch` substantially cheaper on ROCm, or prove and implement a
safe lazy/after-prefill MTP initialization path that avoids replaying the whole
large prompt through the MTP context. Any lazy path must first prove that the MTP
state semantics remain correct for Qwen3.6.

## Decision

Reject MTP `n_max=8` as the cold large-prompt default for the current code.
Keep the E266 MTP hot-path fixes and the `n_max=8` generation-heavy profile, but
do not claim a 130k/60k-prompt speedup until the MTP prefill overhead is fixed.

## Artifacts

- `build_logs/agent-workload/mtp-bigprompt-none-r1.csv`
- `build_logs/agent-workload/mtp-bigprompt-none-r1.diagnostics.md`
- `build_logs/agent-workload/mtp-bigprompt-none-r1.server.log`
- `build_logs/agent-workload/mtp-bigprompt-n8-r1.csv`
- `build_logs/agent-workload/mtp-bigprompt-n8-r1.diagnostics.md`
- `build_logs/agent-workload/mtp-bigprompt-n8-r1.server.log`
