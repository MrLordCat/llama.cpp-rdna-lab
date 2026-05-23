# E196 ROCm Decode Route Recapture

## Metadata

- Experiment ID: E196
- Date: 2026-05-23
- Owner: Codex
- Branch/Commit: master after E195 rollback
- Target lane: H39 ROCm decode-heavy route, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: before another ROCm Q3_K kernel rewrite, the current clean decode-heavy route must be recaptured after E190/E195 because prior local wins did not translate to wall TPS.
- Mechanism: if the fresh route still shows Q3_K fused/direct MMVQ as the dominant decode center, the next useful code path must be a topology-level Q3_K route redesign; if the route shifted toward FA/norm/runtime, the next bottleneck should change.
- Why now: E195 lowered VGPR for one fused Q3_K specialization but regressed real decode TPS, proving resource counters alone are insufficient.

## Math / Theory

- Assumptions:
  - use the fresh clean decode-heavy baseline as the practical control;
  - use E116 Vulkan q4/f16 around `40-41 tok/s` as the parity reference;
  - do not mix this decode-heavy lane with the E191/E192 repo-snapshot prompt-heavy L1 lane.
- Projected:
  - `+5%` decode kernel speedup from E195 control `31.9110 TPS` projects `33.5066 TPS`;
  - `+10%` projects `35.1021 TPS`;
  - `+27%` projects `40.5270 TPS`, roughly Vulkan-q4 parity.
- Failure conditions:
  - route trace does not show Q3_K MMVQ as the dominant decode center;
  - wall timing is dominated by prefill/repo-snapshot rather than decode;
  - trace-only improvements are used as speed claims.

## Implementation Plan

1. Minimal code surface to change: none; this is a measurement and route-selection gate.
2. Guard rails: clear override env, require no background `llama-server`, keep `--spec-type none`, use real `llama-server` benchmark rather than synthetic microbench.
3. Rollback path: no source changes.

## Benchmark Plan

- Baseline command: clean ROCm decode-heavy r3, `max_tokens=512`, `quick/triage_diff,review_bug`.
- Route command: sync/resource ROCm trace, `max_tokens=16`, graph disabled for per-kernel timing attribution.
- Number of runs: r3 for clean baseline, r1 for diagnostic trace.
- Artifacts path: `build_logs/agent-workload/e196-*`.

## Metrics

- aggregate completion TPS (wall)
- decode eval TPS and ms
- prompt eval TPS and ms
- Q3_K fused/direct MMVQ share
- FA/norm/runtime share

## Result

- Outcome: keep diagnostic, no source change.
- Delta: current clean ROCm r3 is `31.9233 TPS` aggregate / `32.3833 tok/s` decode; current clean Vulkan r3 is `40.8007 TPS` aggregate / `41.795 tok/s` decode. Vulkan is `+27.81%` aggregate on this decode-heavy contract, with bootstrap delta CI `[+8.6818,+9.0336] TPS`.
- Confidence: high for the backend gap and route center. Both clean runs are r3 over the same two real server tasks, and the fresh ROCm/Vulkan route delta matches the older E149 shape split.
- Recommendation: continue H39, but only with a topology-level Q3_K route-body candidate. Do not spend more time on generic fusion, graph launch, static SWIGLU branch removal, lower-VGPR-only variants, direct-only `nwarps` changes, pair-dot/preload helpers, or transient q8 layout as standalone patches.

### Clean Runtime

| Backend | Aggregate TPS | Decode eval | Prompt eval | Errors |
| --- | ---: | ---: | ---: | ---: |
| ROCm | `31.9233` | `32.3833 tok/s` | `737.7967 tok/s` | `0` |
| Vulkan | `40.8007` | `41.795 tok/s` | `559.0317 tok/s` | `0` |

Decision stats:

- aggregate delta: `+8.8773 TPS` / `+27.81%` for Vulkan;
- per-task normal CI: ROCm `[31.8166,32.0310]`, Vulkan `[40.6352,40.9677]`;
- bootstrap delta CI: `[+8.6818,+9.0336] TPS`;
- verdict: positive backend gap.

### Fresh ROCm Route Split

`e196-rocm-decode-synctrace-r1` is diagnostic only because graph-disabled sync timing slows decode to `6.01 tok/s`.

Parsed Q3_K route delta against fresh Vulkan perf-log:

| ROCm Q3_K bucket | Calls | Total ms | Share |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_q_fused q3_K->f32` | `4292` | `1212.51` | `56.95%` |
| `mul_mat_vec_q_direct q3_K->f32` | `4350` | `667.16` | `31.33%` |
| `mul_mat_q_direct q3_K->f32` | `349` | `249.50` | `11.72%` |

Fresh Vulkan perf-log shape split:

| Vulkan Q3_K bucket | Calls | Total ms | Share |
| --- | ---: | ---: | ---: |
| `MUL_MAT_VEC q3_K` | `8190` | `420.55` | `72.32%` |
| `MUL_MAT_ADD_VEC q3_K` | `2370` | `160.96` | `27.68%` |

Top aligned shapes remain:

| Shape | ROCm share | Vulkan share |
| --- | ---: | ---: |
| `q3_K m=17408 n=1 k=5120` | `33.14%` | `48.39%` |
| `q3_K m=5120 n=1 k=17408` | `20.88%` | `25.48%` |
| `q3_K m=10240 n=1 k=5120` | `12.06%` | `11.00%` |
| `q3_K m=6144 n=1 k=5120` | `9.83%` | `7.30%` |

MMVQ resource/timing summary for ROCm Q3_K:

| ncols_dst | small_k | fusion | ncols_x | Count | Sum ms | Avg ms | Regs | Occ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `1` | `1` | `5120` | `1923` | `676.110` | `0.3516` | `84` | `87.50%` |
| `1` | `1` | `0` | `5120` | `4320` | `554.893` | `0.1284` | `88` | `87.50%` |
| `1` | `1` | `1` | `17408` | `1923` | `415.253` | `0.2159` | `84` | `87.50%` |
| `2` | `0` | `0` | `5120` | `270` | `75.714` | `0.2804` | `70` | `100.00%` |

Coarse steady Q3 path components:

| Component | Sum ms | Share |
| --- | ---: | ---: |
| `dequant_load_vec_q3` | `751.849` | `49.38%` |
| `compute_core_q3` | `500.224` | `32.86%` |
| `fallback_cublas` | `270.363` | `17.76%` |

Interpretation: the residual gap is still Q3_K route-body dominated. The likely transferable Vulkan advantage is its Q3_K q8_1 matvec topology (`MUL_MAT_VEC`/`MUL_MAT_ADD_VEC`), not broad op fusion or pipeline overhead.

## Notes

- Surprises: prompt eval is faster on ROCm even in this short prompt lane, but the long generation makes decode dominate and Vulkan wins strongly. The fresh Vulkan perf-log with tracing is much slower (`30.63 tok/s`) and must not be used as a clean speed baseline.
- Follow-up action: inspect whether a ROCm Q3_K wave/subgroup topology closer to Vulkan's q8_1 matvec can be prototyped without the already rejected failure modes: lower grid width, VDR2 register cliff, pair-dot live-state growth, or static fusion branch removal.

## Artifacts

- `build_logs/agent-workload/e196-rocm-decode-clean-r3.diagnostics.md`
- `build_logs/agent-workload/e196-vulkan-decode-clean-r3.diagnostics.md`
- `build_logs/agent-workload/e196-rocm-decode-synctrace-r1.server.log`
- `build_logs/agent-workload/e196-vulkan-decode-perflog-r1.server.log`
- `build_logs/agent-workload/e196-current-rocm-vulkan-decode-route-delta-q3k.md`
- `build_logs/agent-workload/e196-rocm-decode-mulmat-steady-split.md`
- `build_logs/agent-workload/e196-rocm-q3-path-components.md`
- `build_logs/agent-workload/e196-rocm-vulkan-clean-decode-r3-stats.md`
