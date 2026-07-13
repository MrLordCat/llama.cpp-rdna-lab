# E281 Q4 1500 tok/s Long-Prompt Plan

## Metadata

- Experiment ID: E281
- Date: 2026-07-12
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`, dirty research tree preserved
- Target lane: dual RX 9070 XT, Qwen3.6-27B MTP-enabled Q4, `ctx=131072`, cold-first long prompt

## Goal And Lane Contract

- Primary target: at least `1500 prompt tok/s` on a real repo-snapshot prompt of roughly `55k-60k` tokens.
- Metric: server-reported `prompt_eval_tps`, not aggregate TPS and not a short `llama-bench` projection.
- Context: `ctx=131072`, `b=8192`, `ub=1024`, `q8_0/q8_0` KV, FlashAttention on.
- Session policy: `--cache-ram 0 --ctx-checkpoints 0`, no reuse, no v2 prime pass, thinking enabled.
- Spec policy: establish and optimize prefill with `--spec-type none`; validate MTP only after the prefill route is selected.
- Placement: dual layer split with GPU1 first and output on GPU1; do not force all KV to one device.
- Power state, prompt token count, residency lines, backend, and build ID are mandatory parts of every result.
- Confirmation: use r1 scouts; claim the target only after same-lane r3 confirmation with zero errors.

## Models

| Model | Size | MTP structure | Quant mix | Role |
| --- | ---: | --- | --- | --- |
| `Qwen3.6-27B-Q4_K_M.gguf` | `17.11 GB` | 866 tensors, 15 `blk.64.*`, 4 NextN | Q4_K 294, Q5_K 48, Q6_K 67 | speed/quality reference |
| `Qwen3.6-27B-UD-Q4_K_XL.gguf` | `17.91 GB` | 866 tensors, 15 `blk.64.*`, 4 NextN | Q4_K 225, Q5_K 70, Q6_K 66, Q8_0 49 | preferred quality candidate |
| `Qwen3.6-27B-Q4_K_S.gguf` | `16.12 GB` | MTP-enabled local reference | mostly Q4_K/Q5_K | historical control only |

XL selection rule: choose XL as the main model when its prompt throughput is at least 95% of Q4_K_M and it stays fully GPU-resident on the target lane. Otherwise optimize M first and retain XL as the quality lane.

## Hypothesis

- Statement: the new hardware and current device/output placement should put Q4 above the old `935 tok/s` result, but reaching 1500 requires identifying whether the remaining wall is quantized matmul, long-KV attention/GDN, or split topology.
- Mechanism: Q4_K_M and XL exercise different kernel mixes. Same-lane traces can distinguish a type-specific route problem from common layer-split or attention overhead.
- Why now: the old Q4 measurement used a different motherboard, model, request size, and runtime state, so it cannot serve as the optimization baseline.

## Required Improvement Gate

For measured baseline `B`, the required local speedup is `1500 / B`:

| Baseline B | Required total speedup |
| ---: | ---: |
| 1000 | 1.500x |
| 1100 | 1.364x |
| 1200 | 1.250x |
| 1300 | 1.154x |
| 1400 | 1.071x |

Do not prototype a kernel whose traced wall share and plausible local gain cannot close at least one third of the measured gap.

## Execution Plan

### Phase 1: Clean Baseline Matrix

1. Run Q4_K_M and XL on Vulkan with the exact lane contract and `spec=none`.
2. Repeat the same pair on ROCm; change only backend/device names.
3. Record prompt tokens and model/KV/compute residency. Reject any run with RAM compute spill or mismatched token count.
4. Select the best backend/model pair. Use the 95% rule for XL.

Measured r1 matrix, all with exactly 56,456 prompt tokens:

| Backend | Model | Prompt tok/s | Decode tok/s | Delta vs Vulkan XL |
| --- | --- | ---: | ---: | ---: |
| Vulkan | Q4_K_M MTP GGUF, spec none | 1163.99 | 20.87 | -1.59% |
| Vulkan | UD-Q4_K_XL MTP GGUF, spec none | **1182.76** | 20.38 | baseline |
| ROCm | Q4_K_M MTP GGUF, spec none | 1053.28 | 16.60 | -10.95% |
| ROCm | UD-Q4_K_XL MTP GGUF, spec none | 1024.23 | 16.75 | -13.40% |

Decision: continue with Vulkan XL. It is faster than M while using the higher-quality quant mix. The measured target gap is `1500 / 1182.76 = 1.2682x`.

The user observed Shared GPU Memory when active MTP draft decoding was enabled. Q4 optimization therefore keeps `spec=none`; MTP decode is not part of the main Q4 profile. In the `spec=none` Vulkan XL run all 66/66 layers were GPU-offloaded and the startup memory breakdown still reported 4,256 MiB free on Vulkan1 and 5,577 MiB free on Vulkan0. The 682 MiB host model buffer is ordinary CPU-resident model data, not layer spill.

Vulkan pipeline-parallel negative control:

- `LLAMA_MTP_PIPELINE_PARALLEL=1`, still `spec=none`: prompt `1179.94 tok/s`, decode `13.58 tok/s`.
- Control: prompt `1182.76`, decode `20.38`.
- Decision: reject forced pipeline parallelism; it does not explain the prefill gap and materially hurts decode.

### Phase 2: Route Trace

1. Capture one unsynchronized route/count trace and one narrowly synchronized timing trace.
2. Split wall time into Q4_K, Q5_K, Q6_K/Q8_0, FlashAttention, GDN/recurrent ops, scheduler copies, and other work.
3. Compare M versus XL traces. A common slowdown points to runtime/FA/split; a divergent slowdown points to quant-type kernels/selectors.
4. Reconstruct the target with the measured wall shares before editing code.

### Phase 3A: Kernel/Runtime Route

Use this branch when quantized matmul or a shared runtime component dominates.

1. Verify the selected ROCm MMQ/dequant route for every hot Q4/Q5 shape; E070 only proved `ne11<=1024` on the older lane.
2. Verify Vulkan compressed Q4/Q5 coopmat/MMQ selection and resource use for the same shapes.
3. Prefer a direct compressed-dot/body improvement over broad dequant-to-F16 staging.
4. Keep every prototype env-gated until point, short-lane, and long-lane gates pass.

### Phase 3B: Tensor Split Rehabilitation

Treat `-sm tensor` as a separate topology project, not a flag sweep.

1. First trace the existing F16-KV Meta path on a small deterministic prefill. Current reference is only `185.94 prompt tok/s` versus `952.47` for layer split.
2. Identify synchronization, gather/reduction, graph duplication, and backend dispatch costs before adding quantized KV.
3. Require at least 80% of layer-split prompt speed on the small lane before implementing q8 KV support.
4. Add q8 KV only after compute scaling is healthy, then test the long lane.
5. Promote tensor split only if it beats the best layer-split Q4 prefill by at least 5% with identical output and no driver instability.

This ordering prevents spending time on KV plumbing while the core Meta compute path remains several times too slow.

### Phase 4: MTP Validation

Deferred for Q4 on this 32 GiB dual-GPU system. Active MTP enters Shared GPU Memory in the user's live observation, so it is not a candidate for the main Q4 profile. Reopen only if a later residency change leaves sufficient dedicated-VRAM headroom; any reopened run must use at least 128 generated tokens and report prompt throughput, decode speed, local acceptance, coverage, effective acceptance, and aggregate TPS separately.

## Benchmark Plan

- Baseline: current best dual layer route, `spec=none`, cold-first, one real prompt.
- Candidate: one changed dimension per A/B.
- Runs: r1 scout, r3 only for a promising result or final target confirmation.
- Artifacts: `build_logs/agent-workload/e281-*` through the canonical benchmark script.
- Driver safety: no `hipMemGetInfo`, no Vulkan staging script, no executable version/help probe, and soft server shutdown only.

## Stop Gates

- Stop a model lane if it spills active compute to RAM under the fixed contract.
- Stop a route if the point result is below control or its modeled ceiling cannot materially close the gap.
- Stop tensor-split work before q8 KV if small-lane F16 compute stays below 80% of layer split.
- Do not claim 1500 from short prompts, cache reuse, priming, reduced token count, or a different power state.

## Result

- Outcome: stopped before trace by the active-compute residency gate
- Delta: current best is Vulkan XL at `1182.76 prompt tok/s`; target requires `1.2682x`
- Confidence: high for the r1 model/backend selection because token count and lane parameters were identical; high for the stop decision because the user observed Shared GPU Memory rising during the actual large-prompt compute phase
- Recommendation: do not continue Q4 kernel or tensor-split work on this fixed `ctx=131072`, q8 KV, dual-16-GiB contract. Startup accounting retained 4.3/5.6 GiB nominal free, but it does not capture the transient long-prompt compute peak seen by WDDM. Resume only with additional dedicated VRAM or a separately approved lane that changes model/context/KV residency; do not hide the problem by treating Shared GPU Memory as acceptable.

## Notes

- MTP training in the source architecture does not guarantee that a GGUF includes the draft head. Both newly downloaded files were checked locally and contain the full MTP block.
- The prompt target is independent of speculative decode. MTP is validated after prefill so its overhead cannot obscure the Q4 optimization result.
- No synchronized Vulkan trace or source prototype was started after the residency failure. The branch is stopped cleanly at the baseline gate.
