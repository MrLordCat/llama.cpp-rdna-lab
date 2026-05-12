# E007: H08 Current UBatch 480/490 Boundary Recon

## Metadata

- Experiment ID: E007
- Date: 2026-05-12
- Owner: Copilot
- Branch/Commit: 96decb6af + local docs/script changes
- Target lane: current autotune-best neighborhood, ctx=32768, batch=2560, ubatch=480/490, q4_0/q4_0, ngram-mod, repo-snapshot first request, no reuse, no v2 prime

## Hypothesis

- Statement: The current ubatch cliff is near 480/490, not the old 824/832 boundary.
- Mechanism: Larger physical `n_ubatch` changes prompt prefill graph/kernel shape enough to cause severe slowdown or timeout/failure.
- Why now: Current autotune history shows best at ubatch=480, slight drop at 485, and errors at 490+.

## Recon Plan

1. Reproduce the current best at ubatch=480 with tracing.
2. Reproduce the failure at ubatch=490 with the same command shape.
3. Test whether opt-in shape context cap can make an ubatch=490 request behave like the safe ubatch=480 physical context.

## Commands

Common benchmark shape:

```bash
python scripts/agent_workload_bench.py --history-version v2 --build-id bld-20260508113803-2d592989 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks v2-mini --runs 1 --ctx-size 32768 --batch-size 2560 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --real-context-chars 21872 --no-reuse --allow-ctx-above-16k --no-v2-prime-pass --background-server-policy fail --request-timeout 120 --task-fail-timeout 0 --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"
```

Variants:

- `e007-h08-ub480-trace`: `--ubatch-size 480`, with `LLAMA_TRACE_DELTA_NET_CONTRACT=1 LLAMA_UBATCH_TRACE=1`.
- `e007-h08-ub490-trace`: `--ubatch-size 490`, with the same tracing.
- `e007-h08-ub490-shape480-nocap-trace`: `--ubatch-size 490`, with `LLAMA_UBATCH_SPLIT_POLICY=shape-score LLAMA_UBATCH_SHAPE_PREFERRED=480`, no context cap.
- `e007-h08-ub490-cap480-trace`: `--ubatch-size 490`, with `LLAMA_UBATCH_SPLIT_POLICY=shape-score LLAMA_UBATCH_SHAPE_PREFERRED=480 LLAMA_UBATCH_SHAPE_CONTEXT_CAP=1`.
- `e007-h08-ub512-cap480`: `--ubatch-size 512`, with the same cap env as above.

## Result

| Label | UBatch arg | Physical n_ubatch | Split/context policy | TPS | Prompt eval TPS | Decode eval TPS | Result |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `e007-h08-ub480-trace` | 480 | 480 | default | 9.8079 | 946.88 | 25.68 | safe corridor |
| `e007-h08-ub490-trace` | 490 | 490 | default | 4.0122 | 282.77 | 25.69 | cliff reproduced |
| `e007-h08-ub490-shape480-nocap-trace` | 490 | 490 | shape-score preferred=480 | 4.0649 | 288.52 | 25.12 | still bad |
| `e007-h08-ub490-cap480-trace` | 490 | 480 | shape-score preferred=480 + context cap | 9.7788 | 943.96 | 25.64 | recovered |
| `e007-h08-ub512-cap480` | 512 | 480 | shape-score preferred=480 + context cap | 9.8052 | 947.86 | 25.69 | recovered |

Key log evidence:

- `e007-h08-ub490-cap480-trace.server.log`: `LLAMA_UBATCH_SHAPE_CONTEXT_CAP enabled -> n_ubatch 490 -> 480`.
- `e007-h08-ub512-cap480.server.log`: `LLAMA_UBATCH_SHAPE_CONTEXT_CAP enabled -> n_ubatch 512 -> 480`.
- `e007-h08-ub490-shape480-nocap-trace` and `e007-h08-ub490-cap480-trace` have the same traced GDN contract shape summary, but only the cap run recovers throughput.

Interpretation:

1. The regression is prefill-only: decode stays around 25.6 tok/s while prompt eval falls from ~947 tok/s to ~283 tok/s.
2. Shape-score split planning alone does not fix the cliff when physical `n_ubatch` remains 490.
3. Capping physical `n_ubatch` to 480 restores performance for both `-ub 490` and `-ub 512`.
4. Therefore the immediate fix path is not a GDN chunk-size override; it is policy hardening around physical context `n_ubatch` for this model/lane.

Decision:

- Superseded by the follow-up PP reserve root-cause patch below.
- Do not promote `LLAMA_UBATCH_SHAPE_CONTEXT_CAP=1` or `LLAMA_UBATCH_SHAPE_PREFERRED=480` as the final policy; they were useful discriminators but still avoided the slow layout instead of fixing why it appeared.
- Keep this section as historical evidence that the cliff was reserve/layout-related rather than a GDN chunk-size issue.

## Follow-up Root-Cause Patch (2026-05-12)

After deeper tracing, the cliff was isolated to **scheduler reserve layout**, not to runtime split shape or kernel route selection:

- `ub490` no-cap with shape-score preferred `480` was still slow despite matching GDN/MMQ/FATTN trace shapes.
- `process_ubatch` timing showed device sync time exploding while host build/alloc/input times stayed small.
- GDN per-launch timing (`GGML_TRACE_GDN_TIMING=1`) was ~3.5x slower in no-cap runs, with identical chunk shapes.
- This points to reserve-time allocation/layout pressure from reserving PP graph at `n_tokens=min(n_ctx,n_ubatch)=490`.

Superseded intermediate fix:

- A shape-score reserve-size patch made `ub490` fast, but `ub512` still needed preferred `480`. This proved the direction but was not a generic fix.
- The shape-score reserve-size patch was removed from `llama_context`; final code does not cap reserve `n_tokens` via planner output.

Final implemented fix:

- `llama_context::sched_reserve()` now reserves PP graph outputs from actual decode output count instead of always reserving full `n_tokens` outputs.
- `decode()` supplies the actual `n_outputs_all`; all-output decode keeps full reserve via sentinel.
- `encode()` keeps full reserve to preserve encoder/all-output behavior.

Validation (no context cap):

| Label | UBatch arg | Context cap | TPS | Prompt eval TPS | Result |
| --- | ---: | --- | ---: | ---: | --- |
| `e007-fix-ub490-nocap-r1` | 490 | off | 0.13 (max-tokens=1 probe) | ~942-955 class | fast prefill restored |
| `e007-fix-ub490-nocap-full` | 490 | off | 9.81 | 943.68 | cliff removed |
| `e007-fix-ub512-nocap-full` | 512 | off | 9.80 | 944.60 | above-cliff request also stable |
| `e007-fix-ub480-baseline-full` | 480 | off | 9.78 | 941.96 | baseline unchanged |

Validation update (auto shape-score, no preferred):

| Label | UBatch arg | Preferred | TPS | Prompt eval TPS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `e007-autofix2-ub490-nopref-r1` | 490 | 0/unset | 0.13 (max-tokens=1 probe) | fast class | reserve auto-picked 480 and stayed fast |
| `e007-autofix2-ub490-nopref-full` | 490 | 0/unset | 9.80 | 945+ class | fast |
| `e007-autofix2-ub512-nopref-r1` | 512 | 0/unset | 0.04 (max-tokens=1 probe) | slow class | planner kept 512, cliff returned |
| `e007-autofix2-ub512-nopref-full` | 512 | 0/unset | 3.98 | 281.25 | cliff returned |
| `e007-autofix2-ub512-pref480-full` | 512 | 480 | 9.78 | 945.27 | fast again |

Final validation update (no context cap, no shape-score/preferred, no diagnostic PP env):

| Label | UBatch arg | Auto reserve log | Wall | Prompt eval TPS | Result |
| --- | ---: | --- | ---: | ---: | --- |
| `e010-ub490-final-ppout` | 490 | `PP reserve outputs 490 -> 1` | 7.41s | 966.26 | fast |
| `e010-ub512-clean-ppout` | 512 | `PP reserve outputs 512 -> 1` | 7.32s | 979.33 | fast |

Conclusion update:

1. The direct `ub490+` cliff is removable without forcing physical context cap or lowering requested `-ub`.
2. Root cause is oversized PP output reservation creating a bad scheduler/compute-buffer layout for normal server decode.
3. The final fix is output-aware PP graph reservation; shape-score/preferred remains a diagnostic lane, not the production answer for this issue.

## Ceiling Break Update (2026-05-12, large-context autotune)

After the root-cause fix landed, GUI autotune on the same `ctx=32768` lane produced a new practical ceiling above the old `~10 TPS` corridor.

Measured autotune summary:

| Label | ctx | batch | ubatch | KV | spec | Aggregate TPS | Errors |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-073021-cfg01` | 32768 | 2560 | 480 | q4_0 | ngram-mod | 9.9751 | 0 |
| `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-114709-cfg01` | 32768 | 5120 | 1024 | q4_0 | ngram-mod | 10.8176 | 0 |

Delta versus prior stable best on this lane:

- absolute: `+0.8425 TPS`
- relative: `+8.45%`

Server-log evidence for the new run (`...114709-cfg01.server.log`):

- `sched_reserve: PP reserve outputs 1024 -> 1`
- `prompt eval time = 6374.97 ms / 7125 tokens (1117.65 tok/s)`
- `eval time = 4685.94 ms / 120 tokens (25.61 tok/s)`
- `total time = 11060.91 ms / 7245 tokens`

Interpretation:

1. The scheduler no longer reserves full PP outputs for decode at large `ubatch`, so `ubatch=1024` remains stable instead of falling into a reserve/layout cliff.
2. In this run, speculative drafts were still zero (`#gen drafts = 0`), so the gain is primarily from the prefill/runtime path unlocked by the reserve fix and larger stable batch/ubatch settings.
3. This is a real lane-level ceiling break for the active 32k autotune lane, not a prime-pass artifact.

Operational outcome:

- New practical large-context target for this lane is now `~10.8 TPS`+.
- Future candidates should compare against `gui-autotune-Qwen3.6-27B-Q3_K_S-20260512-114709-cfg01` as the current large-context reference on `ctx=32768`.

## Notes

- Do not use `--v2-prime-pass` for these runs.
- Compare against the current autotune/history best, not a new sweep baseline.
- A background `llama-server` from `build-rocm-upstream-stock` was stopped before the measured runs to avoid shared GPU load.
