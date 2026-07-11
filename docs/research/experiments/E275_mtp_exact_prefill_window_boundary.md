# E275 MTP Exact Prefill Window Boundary

## Metadata

- Experiment ID: E275
- Date: 2026-07-11
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`
- Target lane: dual RX 9070 XT, `Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm and Vulkan, short trace first and 49152-token confirmation second

## Hypothesis

- Statement: the 512-token MTP prefill window is applied at physical-batch granularity, so the whole final prompt batch pays target NextN extraction and draft-context catch-up.
- Mechanism: `server_context::update_slots()` toggles `common_speculative_set_process_enabled()` after `batch_view.n_tokens` is chosen. The 2026-07-11 49152-token runs had a 6206-token final batch, and all 6206 rows were processed with MTP although the configured window was 512.
- Expected result: split the prompt batch at the exact window boundary, leaving the prefix on the normal target graph and enabling NextN only for the final 512 prompt tokens. Prompt eval should move toward the same-GGUF `spec=none` lane on both backends.
- Decode is a separate gate: Vulkan draft/verify timing and pipeline-parallel behavior must be measured independently after the prompt bug is removed.

## Analytical Gate

- `projected`: reducing MTP-enabled tail rows from 6206 to 512 removes 91.75% of the avoidable tail work.
- `measured pre-fix`: Vulkan MTP prompt `1207.88 tok/s` versus recent same-shape Vulkan non-MTP around `1356-1399 tok/s`; ROCm MTP prompt `1214.05 tok/s` versus non-MTP/ngram around `1199-1272 tok/s`, with run-to-run and spec differences requiring a fresh same-command A/B.
- Gate decision: proceed with trace-first implementation because the server log proves the incorrect row count directly.

## Implementation

1. `tools/server/server-context.cpp`: stop filling a prompt batch exactly at `task_tokens - LLAMA_SPEC_PREFILL_WINDOW`; the next server iteration contains only the MTP tail.
2. `src/llama-context.cpp`: add `ctx=target|mtp` to `LLAMA_UBATCH_TIMING` rows.
3. `src/llama-context.cpp`: add diagnostic `LLAMA_MTP_PIPELINE_PARALLEL=0|1` control. ROCm defaults remain unchanged until the A/B proves a better route; Vulkan remains disabled by default unless explicitly forced.
4. `ggml/src/ggml-cuda/mmvq.cu`: keep the tuned Q3_K MMVQ path for `N=1` and send RDNA4 Q3_K `N>=2` to MMQ by default. `GGML_MMVQ_RDNA4_Q3K_MAX_BATCH=8` restores the legacy multi-column MMVQ route for diagnostics.

## Trace Contract

- Keep thinking enabled, cold-first, no reuse/no prime.
- Use `LLAMA_UBATCH_TIMING=1` and sync timing only for diagnostic runs.
- Compare identical backend/model/context/batch/ubatch/KV/device split.
- Record target PP rows, target TG verify rows, draft-context rows, graph reuse, scheduler switch, compute, and sync time.
- Do not use `hipMemGetInfo`, executable `--version`/`--help` probes, or the Vulkan DLL staging script during the benchmark sequence.

## Results

### Exact prefill boundary

The trace now shows two prompt batches at the boundary: `6776 + 512` on the 12k lane and `38376 + 512` on the 49k lane. Only the final 512 rows enable target NextN extraction and draft-context catch-up.

49k cold-first lane, `38888` prompt tokens, `max_tokens=64`:

| Backend | Mode | Aggregate TPS | Prompt tok/s | Decode tok/s | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| ROCm | none | `1.90` | `1275.42` | `20.88` | - |
| ROCm | MTP n2 | `1.87` | `1232.19` | `24.84` | `74.00%` |
| ROCm | MTP n4 | `1.88` | `1233.79` | `26.75` | `57.33%` |
| Vulkan | none | `2.19` | `1448.29` | `28.80` | - |
| Vulkan | MTP n2 | `2.16` | `1407.52` | `34.06` | `76.00%` |

The remaining long-prompt cost is `-3.3%` ROCm and `-2.8%` Vulkan prompt throughput, down from the pre-fix double-digit regression. MTP does not accelerate target prefill itself, so a 39k-prompt/64-output cold request remains roughly wall-neutral even with faster decode.

### ROCm verify route

Synchronized ubatch trace (`n_max=2`):

| Route | Target N=3 avg | MTP overhead per round | Decode tok/s |
| --- | ---: | ---: | ---: |
| legacy RDNA4 Q3_K multi-column MMVQ | `94.44 ms` | about `8.0 ms` | `22.08` |
| pipeline parallelism disabled | `92.64 ms` | about `8.0 ms` | `22.39` |
| Q3_K N>=2 routed to MMQ | `65.12 ms` | about `8.2 ms` | `30.89` |
| Vulkan comparator | `37.90 ms` | about `8.5 ms` | `47.19` |

This localizes the ROCm bottleneck to the generic multi-column Q3_K MMVQ body, not acceptance, MTP draft-head cost, or pipeline parallelism. The tuned RDNA4 `N=1` route uses its special small-k/fused path; `N=2..8` did not and scaled almost linearly. Routing only those multi-column shapes to MMQ cuts target N=3 verify time by `31.0%`.

Clean ROCm 12k/256-token depth sweep after making the MMQ cutoff default:

| Mode | Aggregate TPS | Prompt tok/s | Decode tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| none | `17.16` | `1521.13` | `25.23` | - |
| MTP n1 | `17.16` | `1381.20` | `26.44` | - |
| MTP n2 | `19.92` | `1368.12` | `33.85` | `79.19%` |
| MTP n3 | `20.02` | `1304.35` | `35.41` | `65.76%` |
| MTP n4 | `20.57` | `1381.16` | `35.58` | `56.91%` |
| MTP n5 | `20.46` | `1382.79` | `35.17` | - |
| MTP n6 | `19.95` | `1375.52` | `33.80` | - |
| MTP n8 | `18.68` | `1376.42` | `30.38` | `35.08%` |

Best measured n4 delta: `+19.9%` aggregate and `+41.0%` decode. N2 remains the safer cross-backend default; ROCm-specific tuning should use n4 on this model.

Rejected controls:

- disabling ROCm MTP pipeline parallelism: only `22.08 -> 22.39 tok/s`;
- two-warp RDNA4 Q3_K multi-column MMVQ: `23.41 tok/s`, rejected and reverted;
- deep n8 after the new route: acceptance falls to `35.08%` and decode to `30.38 tok/s`.

## Decision

- Keep the exact 512-token prompt-tail boundary on both backends.
- Keep `ctx=target|mtp` ubatch trace labels and the diagnostic pipeline-parallel control.
- Keep RDNA4 Q3_K MMVQ only for `N=1`; use MMQ for `N>=2` by default. `GGML_MMVQ_RDNA4_Q3K_MAX_BATCH=8` is the negative-control restore knob.
- Recommend MTP n4 for ROCm generation-heavy work and n2 for Vulkan/cross-backend safety.
- The next route toward `1.6x` ROCm decode is a fused RDNA4 Q3_K multi-column verify kernel (or equivalent MMQ fusion), targeting the remaining `65 ms` target verify body. More scheduler toggles or deeper drafts are not the next step.
