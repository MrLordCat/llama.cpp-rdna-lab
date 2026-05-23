# E191 ROCm Post-Pairdot Bottleneck Recapture

## Metadata

- Experiment ID: E191
- Date: 2026-05-23
- Owner: Codex
- Branch/Commit: master after `3080b223c`
- Hypothesis ID: H39 ROCm decode parity with Vulkan
- Target lane: L1 ROCm H39, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: before the next large ROCm Q3_K route change, the clean default lane needs a fresh post-E190 bottleneck recapture because E190 proved that a local fused MMVQ bucket win can regress real server wall time.
- Mechanism: collect a short real-context server trace with CUDA-node timing, MMVQ timing/resources, and clean default ROCm code, then rank top wall buckets and pre-sync shifts.
- Why now: E190 rejected y-reuse-only pair-dot despite `615.144 -> 552.412 ms` local fused bucket improvement. The next patch must target the actual current route bottleneck, not the previous mental model.

## Math / Theory

- Assumptions: use the E190 paired control as freshest same-build clean reference: aggregate `12.9580 TPS`, prompt eval mean `5774.7067 ms`, decode eval mean `4070.82 ms`, prefill wall share about `0.5865`.
- Kernel-only wall ceiling from this split:
  - `1.02x` decode-only local speedup projects `1.0082x` wall, `13.0639 TPS`;
  - `1.05x` decode-only local speedup projects `1.0201x` wall, `13.2183 TPS`;
  - `1.10x` decode-only local speedup projects `1.0391x` wall, `13.4641 TPS`;
  - `1.27x` decode-only local speedup projects `1.0964x` wall, `14.2069 TPS`.
- Analytical gate commands already run:
  - `python scripts/research/formula_sanity_checks.py` passed.
  - `python scripts/research/required_acceptance.py --target-wall 1.02,1.05 --draft-len 4 --prefill-share 0.5865 --decode-kernel-speedup 1.00 --spec-overhead 1.0` is a speculative sanity reference only; E191 is non-spec.
  - `speedup_model.py` was not used as the primary projection because its default speculative model is not a pure kernel-only route model.
- Failure conditions:
  - trace is not real-context (`task_prompt_tokens` should be around `7413`, not the invalid E190 `159`);
  - background server or override environment contaminates the run;
  - top buckets do not match route evidence, requiring a second clean trace before code work.

## Implementation Plan

1. Minimal code surface to change: none.
2. Guard rails:
   - clean code only;
   - no `GGML_MMVQ_Q3K_FUSED_PAIRDOT`;
   - no `HSA_OVERRIDE_GFX_VERSION`;
   - no background `llama-server`.
3. Rollback path: not applicable, diagnostic-only.

## Benchmark Plan

- Baseline command: clean default L1 `agent_workload_bench.py` real-context trace with `--runs 1`.
- Candidate command: none.
- Number of runs: 1 diagnostic trace; r3 only after a code candidate exists.
- Artifacts path: `build_logs/agent-workload/e191-rocm-postpairdot-*`.

## Metrics

- aggregate completion TPS
- prefill/decode split
- CUDA_NODE top buckets
- MMVQ top buckets and resource state
- pre-sync buckets
- error rate

## Result

- Outcome: diagnostic keep.
- Delta: no candidate. Trace-only run measured `6.8100 TPS`, prompt eval `1045.3 tok/s`, decode eval `29.28 tok/s`, prompt tokens `7489`, errors `0`.
- Confidence: high for route structure, low for wall-speed headline because sync/resource tracing intentionally slows the server.
- Recommendation: split the next work by target metric:
  - practical real-context wall: H35 large Q3_K `cublas_backend` prefill route is now the next bottleneck;
  - pure decode parity: keep H39 MMVQ route open, but do not use this prompt-heavy trace as a decode-only proof.

## Measured Data

Artifact: `build_logs/agent-workload/e191-rocm-postpairdot-trace-r1.server.log`.

Server summary:

| Metric | Value |
| --- | ---: |
| aggregate completion TPS | `6.8100` |
| prompt eval TPS | `1045.3` |
| decode eval TPS | `29.28` |
| prompt eval ms | `7164.45` |
| decode eval ms | `2185.5` |
| task prompt tokens | `7489` |
| errors | `0` |

Top parsed CUDA-node groups:

| Group | Sum | Count | Avg |
| --- | ---: | ---: | ---: |
| `CUDA_NODE op=MUL_MAT kind=forward` | `4944.803 ms` | `3495` | `1.4148 ms` |
| `CUDA_NODE op=GATED_DELTA_NET` | `1016.668 ms` | `336` | `3.0258 ms` |
| `CUDA_NODE op=FLASH_ATTN_EXT` | `339.244 ms` | `112` | `3.0290 ms` |
| `CUDA_NODE op=RMS_NORM` | `234.050 ms` | `1457` | `0.1606 ms` |
| `CUDA_NODE op=GLU` | `192.400 ms` | `315` | `0.6108 ms` |
| `CUDA_NODE op=ADD` | `192.124 ms` | `1066` | `0.1802 ms` |
| `MMVQ` | `148.009 ms` | `1077` | `0.1374 ms` |

`MUL_MAT forward` route split from `cold_steady_trace_split.py`:

| Route / Type | Sum | Share | Count | Avg |
| --- | ---: | ---: | ---: | ---: |
| `cublas_backend|q3_K` | `3891.530 ms` | `78.70%` | `1396` | `2.788 ms` |
| `cublas_backend|f32` | `565.603 ms` | `11.44%` | `784` | `0.721 ms` |
| `cublas_backend|q4_K` | `264.766 ms` | `5.35%` | `192` | `1.379 ms` |
| `mul_mat_vec_q_direct|q3_K` | `155.475 ms` | `3.14%` | `639` | `0.243 ms` |
| `mul_mat_vec_f_direct|f32` | `41.260 ms` | `0.83%` | `336` | `0.123 ms` |

MMVQ resource/timing split confirms the pair-dot target is no longer the practical wall center for this real-context lane:

| MMVQ bucket | Sum | Count | Avg | Resource |
| --- | ---: | ---: | ---: | --- |
| `q3_K ncols_dst=2 small_k=0 fusion=0` | `89.847 ms` | `349` | `0.2574 ms` | `regs=70`, `occ=100%`, `block=(32,1,1)` |
| `q3_K ncols_dst=1 small_k=1 fusion=1` | `27.931 ms` | `290` | `0.0963 ms` | `regs=84`, `occ=87.5%`, `block=(32,2,1)` |
| `q3_K ncols_dst=1 small_k=1 fusion=0` | `14.615 ms` | `290` | `0.0504 ms` | `regs=88`, `occ=87.5%`, `block=(32,2,1)` |

## Interpretation

E190 improved a local fused MMVQ bucket but did not improve wall TPS. E191 explains why that can happen on the real-context lane: with `7489` prompt tokens and only `64` generated tokens, the parsed work is dominated by large prompt/prefill `cublas_backend q3_K`, not by decode MMVQ.

This does not close H39 decode parity. It says the route-chain should not blindly continue MMVQ micro-edits when the user-visible repo-snapshot wall lane is blocked by large-N Q3_K cuBLAS staging/GEMM. E192 therefore refreshes H35 split timing before any new large route prototype.

## Notes

- This experiment is intentionally a map-refresh, not an optimization claim.
- The invalid E190 no-real-context r1 had only `159` prompt tokens; E191 fixes that by validating prompt tokens around `7489`.
