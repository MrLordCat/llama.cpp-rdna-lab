# E176 ROCm F32 M-RoPE MMVF Gate

## Metadata

- Experiment ID: E176
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `7e6239407`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: The small f32 M-RoPE GEMMs with `ncols_dst=16/24` may be faster through the lightweight MMVF route than hipBLAS.
- Mechanism: E169 shows repeated f32 `cublas_backend` shapes in decode: `256x256 @ 24` and `64x64 @ 16`. They are small enough that hipBLAS overhead can dominate, while the existing MMVF kernel already handles nearby `ncols_dst<=8` cases.
- Why now: Q3_K and RMS retunes are hitting either register cliffs or low-ceiling failures. This is a route-selection check for a distinct f32 secondary bucket.

## Math / Theory

- Assumptions: E169 traced the two target shapes at `61.955 ms` out of `2940.555 ms`, about `2.11%` of decode-node time.
- Expected speedup corridor: a `+0.5%` wall gain needs about `1.31x` local speedup; a `+1%` wall gain needs about `1.88x` local speedup. `+2%` would require an unrealistic `14.14x` local speedup.
- Failure conditions: reject if same-build r1 aggregate TPS regresses, even if the target f32 shapes improve locally; this route is too small to carry a default by itself.

Analytic gate:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\required_local_speedup.py --share 0.0211 --goals 1.005,1.01,1.02,1.05
```

Gate output:

- formula sanity checks passed;
- required local speedup: `1.3085x` for `+0.5%`, `1.8841x` for `+1%`, `14.1406x` for `+2%`; `+5%` infeasible.

## Implementation Plan

1. Minimal code surface to change: temporary env-gated `GGML_CUDA_MMVF_F32_16_24=1` in `ggml/src/ggml-cuda/mmvf.cu`, with template cases for `ncols_dst=16` and `24`.
2. Guard rails: no default change; compare no-env control and candidate on the same dirty build.
3. Rollback path: revert MMVF cases and selector env if wall TPS does not improve.

## Benchmark Plan

- Baseline command: active H39 quick `triage_diff`, `--runs 1`, `--max-tokens 128`, no reuse, `spec=none`.
- Candidate command: same command with `GGML_CUDA_MMVF_F32_16_24=1`.
- Trace command: graph-disabled sync trace with route logging and `--max-tokens 16`.
- Artifacts path: `build_logs/agent-workload/e175-rocm-mmvf-f32-16-24-*`.

## Metrics

- aggregate completion TPS;
- decode eval tok/s;
- error rate;
- target f32 shape route and timing.

## Result

- Outcome: local-positive but wall-negative
- Delta: same-build no-env control `30.0378 TPS` / `32.44 tok/s`; candidate `29.0836 TPS` / `31.47 tok/s`.
- Confidence: enough to reject as default. The local target shapes improved, but their share is too small and wall TPS regressed.
- Recommendation: revert the code. Keep the finding in the route map: small f32 M-RoPE GEMMs can be locally accelerated, but this is not a standalone H39 speed route.

Runtime:

| Variant | Aggregate TPS | Decode eval | Delta vs control |
| --- | ---: | ---: | ---: |
| no-env control | `30.0378` | `32.44 tok/s` | baseline |
| MMVF f32 `16/24` | `29.0836` | `31.47 tok/s` | `-3.18%` aggregate |

Trace target buckets:

| Shape | E169/default route | E176 MMVF route | Interpretation |
| --- | ---: | ---: | --- |
| f32 `256,256 x 256,24` | `cublas_backend`, `240` calls, `33.717 ms`, `0.1405 ms` avg | `mul_mat_vec_f_direct`, `240` calls, `25.651 ms`, `0.1069 ms` avg | local `1.31x` faster |
| f32 `64,64 x 64,16` | `cublas_backend`, `240` calls, `28.238 ms`, `0.1177 ms` avg | `mul_mat_vec_f_direct`, `240` calls, `19.917 ms`, `0.0830 ms` avg | local `1.42x` faster |
| f32 `64,64 x 64,96` | `cublas_backend`, `240` calls, `27.851 ms`, `0.1160 ms` avg | unchanged route, `30.962 ms`, `0.1290 ms` avg | neighboring noise/regression offsets part of target win |

## Notes

- Surprises: The local route hypothesis was correct, but the wall result was still negative. This is a good example of why low-share targets need a hard wall gate.
- Follow-up action: Do not promote MMVF `16/24` alone. It could be reconsidered only as a small component inside a larger f32/M-RoPE route stack with measured wall headroom.

## Artifacts

- `build_logs/agent-workload/e175-rocm-mmvf-f32-16-24-control-r1.diagnostics.md`
- `build_logs/agent-workload/e175-rocm-mmvf-f32-16-24-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e175-rocm-mmvf-f32-16-24-trace-r1.server.log`
