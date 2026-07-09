# E269 ROCm Dual Split Baseline Drop

## Metadata

- Experiment ID: E269
- Date: 2026-07-09
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`, baseline after `3f11d6190`
- Target lane: ROCm dual-GPU baseline without MTP, `Qwen3.6-27B-Q3_K_S_mtp.gguf`

## Hypothesis

- Statement: The baseline decode drop under two-GPU split is caused by split scheduling / cross-device transfer overhead, not by Qwen/MTP model correctness.
- Mechanism: `-sm layer` keeps the model resident on both GPUs, but a single request still runs layer blocks serially and crosses devices through scheduler copies. On Windows ROCm, peer copies are disabled by default and cross-device transfers are host-staged.
- Why now: MTP is now positive on practical dual-GPU (`+62.2%` wall), but the dual baseline itself is below ROCm1-only (`25.06` vs `29.93` decode tok/s). Fixing or reducing that baseline drop raises both baseline and MTP absolute speed.

## Math / Theory

- Assumptions:
  - `-sm none` is single-GPU and is only a diagnostic lane.
  - `-sm layer` is the practical long-context residency lane.
  - Windows ROCm peer-copy opt-in may be unsafe on this RDNA4 setup, so it is a diagnostic-only candidate until proven stable.
- Expected speedup corridor:
  - Safe split-mode/ratio tuning target: recover at least half of the observed `~16%` decode loss.
  - Risky peer-copy path target: prove whether host-staged cross-device copies are the dominant loss before any default change.
- Failure conditions:
  - Any candidate with errors, driver instability, incorrect output symptoms, or lower decode TPS than current layer split is rejected.

## Implementation Plan

1. Minimal code surface to change: start with no-code split-mode and tensor-split A/B; inspect scheduler copy path before editing.
2. Guard rails: keep `GGML_ROCM_ENABLE_PEER_COPY` off by default; if tested, label as diagnostic-only.
3. Rollback path: no-code candidates need no rollback; code changes must be env-gated until r3/stability confidence.

## Benchmark Plan

- Baseline command: `--spec-type none`, `-dev ROCm1,ROCm0 -sm layer -ts 1,1`
- Candidate commands: `-sm row`, `-sm tensor`, and layer `-ts` ratios such as `2,1` / `3,1`
- Number of runs: `r1` scouts first; promote only promising candidates to `r3`
- Artifacts path: `build_logs/agent-workload/rocm-dual-split-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- server log split count and residency lines

## Current Baseline

| Mode | Device/split | Label | Aggregate TPS | Decode tok/s | Prompt tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| single diagnostic | `-dev ROCm1 -sm none` | `rocm1-mtp-polish-mt256-none-r1` | `28.9357` | `29.93` | `598.78` |
| practical dual layer | `-dev ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-layer-mtp-polish-mt256-none-r1` | `24.3710` | `25.06` | `618.77` |

## Result

- Outcome: keep partial fix
- Delta: practical dual layer baseline improved from `24.3710` aggregate / `25.06` decode tok/s to `25.6137` aggregate / `26.26` decode tok/s r3 (`+5.10%` aggregate, `+4.80%` decode). Best r1 placement/code scout reached `25.4359` aggregate / `26.17` decode tok/s with `-dev ROCm0,ROCm1 -sm layer -ts 1,3 -mg 1`.
- Confidence: medium-high for the host-staged buffer-copy patch; r3 stable with `0` errors and task TPS stdev `0.1707`.
- Recommendation: keep the safe buffer-level host-stage copy path. Do not enable `GGML_ROCM_ENABLE_PEER_COPY=1` by default: direct peer-copy produced an immediate 1-token/empty completion in the diagnostic run.

## Measured Results

Baseline and keep candidate use:

`Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm, `ctx=8192`, `b512/ub128`, `q4_0/q4_0`, FlashAttention on, `max_tokens=256`, `temperature=0.0`, `quick:triage_diff`, no reuse/no prime, thinking enabled, `--spec-type none`.

| Variant | Device/split | Label | Runs | Aggregate TPS | Decode tok/s | Prompt tok/s | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | `-dev ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-layer-mtp-polish-mt256-none-r1` | 1 | `24.3710` | `25.06` | `618.77` | baseline |
| host-stage buffer copy | `-dev ROCm1,ROCm0 -sm layer -ts 1,1` | `rocm-dual-split-bufferhostcopy-mt256-none-r3` | 3 | `25.6137` | `26.26` | `749.89` | keep |
| best placement scout after patch | `-dev ROCm0,ROCm1 -sm layer -ts 1,3 -mg 1` | `rocm-dual-split-bufferhostcopy-dev01-mg1-ts1_3-mt256-none-r1` | 1 | `25.4359` | `26.17` | `606.56` | optional profile |

No-code scouts before the code patch:

| Variant | Label | Aggregate TPS | Decode tok/s | Decision |
| --- | --- | ---: | ---: | --- |
| row split | `rocm-dual-split-row-mt256-none-r1` | `0.0000` | - | reject: no valid decode / timeout |
| tensor split | `rocm-dual-split-tensor-mt256-none-r1` | `0.0000` | - | reject: `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` |
| tensor split, fit off | `rocm-dual-split-tensor-fitoff-mt256-none-r1` | `0.0000` | - | reject: tensor split + q4 KV unsupported/crashed |
| layer `-ts 2,1` | `rocm-dual-split-layer-ts2_1-mt256-none-r1` | `19.3897` | `19.90` | reject |
| layer `-ts 1,2` | `rocm-dual-split-layer-ts1_2-mt256-none-r1` | `21.7593` | `22.48` | reject |
| layer `-dev ROCm1,ROCm0 -mg 1` | `rocm-dual-split-layer-mg1-mt256-none-r1` | `24.9866` | `25.69` | useful placement scout |
| layer `-dev ROCm0,ROCm1 -mg 1` | `rocm-dual-split-dev01-mg1-mt256-none-r1` | `25.1185` | `25.81` | useful placement scout |
| layer `-dev ROCm0,ROCm1 -ts 1,3 -mg 1` | `rocm-dual-split-dev01-mg1-ts1_3-mt256-none-r1` | `25.1269` | `25.85` | useful placement scout |

Rejected code/diagnostic probes:

| Probe | Label | Result | Decision |
| --- | --- | --- | --- |
| backend async host-stage copy | `rocm-dual-split-hoststage-async-mt256-none-r1` | `19.7278` aggregate / `20.29` decode tok/s | reject and revert; moving the transfer into `cpy_tensor_async` made scheduler sync behavior worse |
| direct ROCm peer copy opt-in | `rocm-dual-split-peercopy-dev01-mg1-mt64-none-r1` | only `1` predicted token, empty response | reject; correctness failure, keep direct peer-copy disabled by default |

## Notes

- Surprises:
  - Direct peer-copy did not just fail performance; it corrupted the generation path enough to stop immediately. This validates the conservative Windows ROCm default.
  - The upstream `3fc4e1052` scheduler change is not the main answer for this lane because its headline CPU->CUDA async copy path is explicitly disabled for HIP/MUSA, while our hot path is cross-device hidden-state copy.
- Follow-up action:
  - Investigate why HIP direct peer-copy produces immediate EOS/empty output before considering any peer-copy default.
  - Consider a pinned host-stage buffer or lower-sync host staging, but only behind an A/B gate; the current keep patch recovers a small safe slice, not the whole single-GPU gap.
  - Re-test the kept patch on the real `ctx=131072` large-prompt lane before making a 130k headline claim.
