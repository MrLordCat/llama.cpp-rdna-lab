# E053 Post-C01 Selection Gate

## Metadata

- Experiment ID: E053
- Date: 2026-05-18
- Owner: Copilot
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, `ubatch=2048`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: after closing C01, the next useful TPS branch should be selected from current `ubatch=2048` trace evidence instead of old `ubatch=192` C01 assumptions.
- Mechanism: the current bench is prompt-heavy and dominated by large `cublas_backend` `MUL_MAT` calls, especially Q3_K dequant plus GEMM staging, with GATED_DELTA_NET as the second measured prompt hotspot. A fresh diagnostic gate can confirm whether Q3_K dequant/layout still has the largest actionable wall ceiling.
- Why now: E045 recentered the active lane to `ubatch=2048`, E046-E052 closed the broad route toggles and C01 branch, and `POST_C01_ACCELERATION_SCAN_2026-05-18.md` recommends a no-code E053 gate before further implementation.

## Math / Theory

- Assumptions:
  - Current r3 baseline: `prefill-current-ub2048-base-r3 = 11.6534 TPS`, prompt eval `1197.5567 tok/s`.
  - E045 prompt trace: `MUL_MAT 64.81%`, `GATED_DELTA_NET 14.50%`, `FLASH_ATTN_EXT 4.78%`.
  - E049 Q3_K split timing: Q3_K traced calls `src0 32.29%`, `src1 6.74%`, `GEMM 60.97%`; one dequant-heavy Q3_K shape had `src0 78.23%`.
  - E051 estimated Q3_K dequant effective wall share at about `16.69%`.
- Expected speedup corridor:
  - A `10%` local improvement in the prior `16.69%` prompt-trace proxy projects about `+1.5%` prompt-path gain, below the keep threshold unless implementation risk is tiny.
  - Full aggregate TPS must discount decode time. If current prompt+decode share leaves Q3_K dequant near `10%` full-wall share, a dequant-only candidate needs about `25%` local improvement to clear `+2%` aggregate.
  - GDN needs a large specialized-kernel win to justify implementation because its full-wall share is lower than its prompt-trace share.
- Failure conditions:
  - Split timing sync overhead distorts absolute TPS; traces are diagnostic-only.
  - Combining cuBLAS split timing and GDN sync timing in one run over-synchronizes the lane and can obscure per-hotspot attribution.
  - If Q3_K dequant effective wall share falls below `15%`, do not start another dequant kernel probe without a stronger local model.

## Implementation Plan

1. No runtime code changes.
2. Run analytical checks and a trace-off control to anchor the session.
3. Run cuBLAS split timing separately from GDN timing.
4. Parse artifacts for Q3_K dequant share, largest dequant-heavy shapes, and GDN timing distribution.
5. Run one kernel-full trace if the QKV/RoPE-adjacent share still needs a fresh current-lane check.
6. Choose one next code candidate only if modeled wall ceiling is at least `2%`.

## Benchmark Plan

- Analytical gate:
  - `python scripts/research/formula_sanity_checks.py`
  - `python scripts/research/speedup_model.py --baseline-tps 11.6534 --prefill-share 0.1669 --flash-prefill-speedup 1.15 --draft-len 1 --accept-rate 0 --spec-overhead 0 --decode-kernel-speedup 1.0`
  - `python scripts/research/required_acceptance.py --target-wall 1.02 --draft-len 1 --prefill-share 0.1669 --prefill-speedup 1.15 --decode-kernel-speedup 1.0 --spec-overhead 0.0`
- Control command:
  - `python scripts/agent_workload_bench.py --label prefill-e053-control-r1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids triage_diff,review_bug --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --no-v2-prime-pass --no-disable-thinking --max-tokens 120`
- cuBLAS split trace:
  - same command with label `prefill-e053-cublas-split-r1` and env `GGML_TRACE_CUBLAS_SPLIT_TIMING=1 GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`.
- GDN trace:
  - same command with label `prefill-e053-gdn-trace-r1` and env `GGML_TRACE_GDN_PATH=1 GGML_TRACE_GDN_TIMING=1 GGML_TRACE_GDN_TIMING_SYNC_HIP=1`.
- Kernel-full trace:
  - same command with label `prefill-e053-kernel-full-r1` and `--trace-preset kernel-full`.
- Number of runs:
  - `--runs 1`; diagnostic gate only.
- Artifacts path:
  - `build_logs/agent-workload/prefill-e053-control-r1.*`
  - `build_logs/agent-workload/prefill-e053-cublas-split-r1.*`
  - `build_logs/agent-workload/prefill-e053-gdn-trace-r1.*`
  - `build_logs/agent-workload/prefill-e053-kernel-full-r1.*`

## Metrics

- aggregate completion TPS for trace-off control
- prompt eval TPS for trace-off control
- Q3_K `src0` dequant effective wall share
- largest Q3_K dequant-heavy shapes
- GDN timing histogram and hot contract
- QKV/RoPE-adjacent prompt share
- next-candidate modeled wall ceiling

## Result

- Outcome: diagnostic success; proceed to P1 only with a stricter local-gain gate.
- Control:
  - `prefill-e053-control-r1 = 11.7681 TPS`, errors `0`.
  - Prompt eval mean `1207.06 tok/s`; decode eval mean `29.86 tok/s`.
  - Prompt eval mean `6145.59 ms`, decode eval mean `4018.44 ms`, so prompt is about `60.46%` of prompt+decode time in this run.
- cuBLAS split trace:
  - `prefill-e053-cublas-split-r1 = 11.1456 TPS`; sync trace only, not a speed claim.
  - Large split rows: `4456`, total `13052.68 ms`.
  - All traced calls: `src0 27.13%`, `src1 6.15%`, `GEMM 66.72%`.
  - Q3_K traced calls: rows `2792`, total `10369.29 ms`, `src0 3386.36 ms` (`32.66%`), `src1 710.16 ms` (`6.85%`), `GEMM 6272.75 ms` (`60.49%`).
  - Q3_K without the first one-time GEMM outlier: `src0 33.87%`, `src1 7.10%`, `GEMM 59.03%`.
  - Largest dequant-heavy shape remains Q3_K `row_diff=6144, ne00=5120, ne10=5120, ncols=2048`: rows `288`, total `1841.00 ms`, `src0 1440.13 ms` (`78.23%`), `src1 62.46 ms`, `GEMM 338.42 ms`.
  - Tail shapes `ncols=1278` and `ncols=1259` for the same `6144x5120` projection also stay dequant-heavy at about `76-77% src0`.
- GDN trace:
  - `prefill-e053-gdn-trace-r1 = 11.2312 TPS`; sync trace only, not a speed claim.
  - Hot contract confirmed: `KDA=0`, `keep_intermediates=0`, `n_seqs=1`, `S_v=128`, prompt `n_tokens=2048`, `chunked_prefill=1`, `chunk_size=128`, `fast_exp=0`.
  - Separate GDN timing rows are dominated by sync instrumentation: prompt-2048 rows `4608`, total `9893.97 ms`, p50 `0.364 ms`, p95 `20.057 ms`, max `53.729 ms`. Use this for contract/histogram only, not wall-share math.
- Kernel-full trace:
  - `prefill-e053-kernel-full-r1 = 10.4167 TPS`; diagnostic overhead only.
  - Large prompt node total: `13851.911 ms`.
  - `MUL_MAT`: `8934.030 ms` (`64.50%`).
  - Q3_K `MUL_MAT`: `7508.598 ms` (`54.21%`).
  - `GATED_DELTA_NET`: `2044.496 ms` (`14.76%`).
  - `FLASH_ATTN_EXT`: `666.847 ms` (`4.81%`).
  - `ROPE`: `95.128 ms` (`0.69%`).
  - H06 QKV/RoPE-adjacent slice: `2560.164 ms` (`18.48%` of large prompt node total).
- Modeled ceiling:
  - Same proxy as E049/E051: Q3_K `src0` dequant remains about `16.7%` of prompt trace (`32.66%` of Q3_K split time, or about `25.9%` of all split time combined with the `64.5%` MUL_MAT prompt share).
  - Discounting current decode time, Q3_K dequant-only full-wall share is about `10.1%`.
  - A `20%` local Q3_K dequant/layout improvement projects `1.0172x`, `11.9700 TPS`.
  - A `25%` local Q3_K dequant/layout improvement projects `1.0207x`, `12.0112 TPS`.
- Delta: no candidate delta; diagnostic-only. Control r1 is `+0.98%` vs the E045 r3 baseline (`11.6534 -> 11.7681`), within normal r1/session variance.
- Confidence: high for hotspot ordering and branch selection; low for absolute trace TPS because all timing traces synchronize streams.
- Recommendation: P1 remains the next code branch, but only for a Q3_K dequant/layout design with a plausible `>=25%` local win or a combined dequant+GEMM effect. P2 GDN specialized prefill kernel is second. P3 H06 has a real `18.5%` prompt slice, but graph-level fusion was already rejected and the remaining kernel-level target is broader/riskier than P1.

## Notes

- Surprises: the `6144x5120@ncols2048` Q3_K projection repeated the exact `78.23% src0` dequant-heavy signature from E049, so this is stable rather than trace noise.
- Follow-up action: start E054 as a Q3_K dequant/layout design gate. Do not repeat MMQ route, compute16, hipBLASLt, Stream-K, GDN chunk-size, or 128-thread dequant probes.