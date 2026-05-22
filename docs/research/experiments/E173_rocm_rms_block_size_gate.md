# E173 ROCm RMS Norm Block Size Gate

## Metadata

- Experiment ID: E173
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `040723d44`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: RDNA4 decode RMS_NORM might run faster with a smaller large-row block size (`256` or `512` threads) than the current `1024`-thread path.
- Mechanism: E169 shows `RMS_NORM` fused is `8.96%` of traced decode-node time, and the dominant RMS shape is `5120` columns. Smaller blocks may reduce occupancy pressure and scheduling overhead for one-token rows if the extra per-thread loop work stays cheap.
- Why now: E166-E170 showed Q3_K MMVQ local-state changes are hitting register/occupancy cliffs. RMS is a separate secondary route with a cheap topology probe.

## Math / Theory

- Assumptions: E169 traced RMS share is `0.0896`; RMS plus ROPE/SET_ROWS share is `0.1167`, but this experiment only changes RMS.
- Expected speedup corridor: RMS alone needs about `1.124x` local speedup for `+1%` wall and `1.280x` local speedup for `+2%` wall. A `+5%` wall gain would require an unrealistic `2.134x` local RMS speedup.
- Failure conditions: reject if r1 aggregate TPS is below the same-session control or if the `5120,1,1,1` RMS trace bucket slows.

Analytic gate:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\required_local_speedup.py --share 0.0896 --goals 1.01,1.02,1.05,1.10
python scripts\research\required_local_speedup.py --share 0.1167 --goals 1.01,1.02,1.05,1.10
```

Gate output:

- formula sanity checks passed;
- RMS-only required local speedup: `1.1242x` for `+1%`, `1.2801x` for `+2%`, `2.1343x` for `+5%`; `+10%` is infeasible from RMS alone;
- RMS+ROPE+SET_ROWS corridor would need `1.2020x` local for `+2%`, but this probe did not change ROPE/SET_ROWS.

## Implementation Plan

1. Minimal code surface to change: temporary env-gated `GGML_CUDA_RMS_NORM_BLOCK_SIZE=256|512|1024` in `ggml/src/ggml-cuda/norm.cu`.
2. Guard rails: no default change; same-session r1 control before candidates.
3. Rollback path: revert the env-gated code if neither `256` nor `512` beats the control.

## Benchmark Plan

- Baseline command: active H39 quick `triage_diff`, `--runs 1`, `--max-tokens 128`, no reuse, `spec=none`.
- Candidate command: same command with `GGML_CUDA_RMS_NORM_BLOCK_SIZE=256`, then `512`.
- Resource/timing command: short graph-disabled sync trace for the regressing `256` candidate.
- Artifacts path: `build_logs/agent-workload/e171-rocm-rms-*`.

## Metrics

- aggregate completion TPS;
- decode eval tok/s;
- error rate;
- RMS_NORM `5120,1,1,1` bucket timing in sync trace.

## Result

- Outcome: regression / no keep
- Delta: same-session r1 control `28.8451 TPS` / `31.19 tok/s`; block256 `28.1982 TPS` / `30.42 tok/s`; block512 `28.7389 TPS` / `31.04 tok/s`.
- Confidence: enough to reject. The wall signal is negative/neutral, and trace confirms the main RMS bucket slows with `256`.
- Recommendation: revert the temporary env-gated RMS block-size code. Do not continue simple RMS block-size retuning on H39 unless a future trace changes the RMS shape mix.

Candidate r1:

| Variant | Aggregate TPS | Decode eval | Delta vs control |
| --- | ---: | ---: | ---: |
| control/default `1024` | `28.8451` | `31.19 tok/s` | baseline |
| block `256` | `28.1982` | `30.42 tok/s` | `-2.24%` aggregate |
| block `512` | `28.7389` | `31.04 tok/s` | `-0.37%` aggregate |

Trace check:

| RMS bucket | E169/default | E173 block256 | Interpretation |
| --- | ---: | ---: | --- |
| fused `5120,1,1,1` | `1937` calls, `150.685 ms`, `0.0778 ms` avg | `1937` calls, `177.150 ms`, `0.0915 ms` avg | `+17.6%` local regression |
| fused `5120,159,1,1` | `127` calls, `11.686 ms`, `0.0920 ms` avg | `127` calls, `11.944 ms`, `0.0940 ms` avg | slight regression |

## Notes

- Surprises: The plausible occupancy/smaller-block idea loses because the extra per-thread loop work and reduction balance dominate the one-row decode RMS shape.
- Follow-up action: keep RMS as a secondary map entry, not an active retuning target. The next H39 branch should return to a structural Q3_K route or another route with a larger measured ceiling.

## Artifacts

- `build_logs/agent-workload/e171-rocm-rms-control-r1.diagnostics.md`
- `build_logs/agent-workload/e171-rocm-rms-block256-r1.diagnostics.md`
- `build_logs/agent-workload/e171-rocm-rms-block512-r1.diagnostics.md`
- `build_logs/agent-workload/e171-rocm-rms-block256-trace-r1.server.log`
