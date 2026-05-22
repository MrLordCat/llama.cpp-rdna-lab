# E170 ROCm Q3_K MMVQ VDR2 Gate

## Metadata

- Experiment ID: E170
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after `ead33088e`
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: RDNA4 Q3_K MMVQ may benefit from processing two adjacent Q3_K `iqs` fragments per thread (`VDR=2`) instead of the current `VDR=1`.
- Mechanism: E169 shows Q3_K fused + direct remains about `39.84%` of traced decode time. Vulkan's active q8_1 Q3_K decode route uses a larger K fragment per invocation (`K_PER_ITER=16`) through integer-dot shaders, while ROCm Q3_K MMVQ still schedules one Q3_K fragment per thread. A VDR2 probe halves the number of thread groups per Q3_K block and may reduce loop/scheduling overhead while preserving the E151 `nwarps=2` / `rows_per_block=2` topology.
- Why now: E162-E168 rejected local helper/register tweaks. The next candidate needs a broader route change that can affect both fused FFN and direct Q3_K buckets.

## Math / Theory

- Assumptions: E169 traced Q3_K fused + direct share is `0.3984` of decode-node time.
- Expected speedup corridor: a `5%` wall gain from Q3_K alone requires about `1.136x` local speedup. A `10%` wall gain requires about `1.296x` local speedup.
- Failure conditions: If VDR2 raises registers, reduces latency hiding, or increases per-thread work enough to slow the dominant fused buckets, reject even if occupancy looks better.

Analytic gate:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\required_local_speedup.py --share 0.3984 --goals 1.02,1.05,1.10,1.27
```

Gate output:

- formula sanity checks passed;
- required local speedup: `1.0518x` for `+2%`, `1.1358x` for `+5%`, `1.2956x` for `+10%`, `2.1442x` for full `1.27x` parity.

## Implementation Plan

1. Minimal code surface to change: `ggml/src/ggml-cuda/vecdotq.cuh` Q3_K MMVQ helper and `ggml/src/ggml-cuda/mmvq.cu` VDR selection.
2. Guard rails: RDNA4/Q3_K-only experiment; no persistent default unless r1 and resource trace are both credible.
3. Rollback path: revert the two code hunks and keep this note as rejected if speed or resource buckets regress.

## Benchmark Plan

- Baseline: E151 best r3 `30.3145 TPS` / `32.2467 tok/s`; E169 route gate after E151.
- Candidate command: active H39 quick `triage_diff`, `--runs 1`, `--max-tokens 128`.
- Resource command: active H39 quick `triage_diff`, `--max-tokens 16`, graph disabled, `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`, `GGML_TRACE_MMVQ_RESOURCES=1`.
- Artifacts path: `build_logs/agent-workload/e170-*`.

## Metrics

- aggregate completion TPS;
- decode eval tok/s;
- error rate;
- MMVQ fused/direct Q3_K bucket timings;
- regs/occupancy/shared-memory changes.

## Result

- Outcome: regression
- Delta: r1 aggregate `30.3145 -> 26.8812 TPS` versus E151 best r3 (`-11.33%`); decode eval `32.2467 -> 28.90 tok/s` (`-10.38%`).
- Confidence: high enough to reject without r3 promotion. The regression is large and the resource trace explains it.
- Recommendation: reject and revert the VDR2 code. Do not retry larger Q3_K per-thread VDR unless a future design avoids the fused-kernel register cliff.

Candidate r1:

| Metric | E151 kept baseline | E170 VDR2 |
| --- | ---: | ---: |
| Aggregate TPS | `30.3145` | `26.8812` |
| Decode eval | `32.2467 tok/s` | `28.90 tok/s` |
| Errors | `0` live sanity in E151 | `0` |

Resource trace, decode-only Q3_K buckets:

| Bucket | E163/E151 route | E170 VDR2 | Interpretation |
| --- | ---: | ---: | --- |
| fused `ncols_x=5120`, `grid.x=8704` | `0.355 ms`, `84 regs`, `87.5% occ` | `0.409 ms`, `141 regs`, `56.25% occ` | dominant bucket hits a register cliff |
| fused `ncols_x=17408`, `grid.x=2560` | `0.219 ms`, `84 regs`, `87.5% occ` | `0.226 ms`, `141 regs`, `56.25% occ` | smaller but still slower |
| direct `ncols_x=5120`, `grid.x=5120` | `0.156 ms`, `88 regs`, `87.5% occ` | `0.177 ms`, `90 regs`, `100% occ` | occupancy alone is not the bottleneck |
| direct `ncols_x=5120`, `grid.x=3072` | `0.124 ms`, `88 regs`, `87.5% occ` | `0.137 ms`, `90 regs`, `100% occ` | larger per-thread work loses despite higher nominal occupancy |

## Notes

- Surprises: The direct Q3_K kernels reached `100%` reported occupancy but still slowed, matching E166/E167: occupancy improvements are not sufficient when per-thread work/register pressure harms latency hiding.
- Follow-up action: Keep E151 topology. The next promising Q3_K route must reduce duplicated work without doubling local state in the fused FFN kernel. Specifically avoid a design that keeps both gate/up VDR2 fragments live in one thread.

## Artifacts

- `build_logs/agent-workload/e170-rocm-decode-q4-q3-vdr2-r1.diagnostics.md`
- `build_logs/agent-workload/e170-rocm-decode-q4-q3-vdr2-r1.server.log`
- `build_logs/agent-workload/e170-rocm-decode-q4-q3-vdr2-resources-r1.diagnostics.md`
- `build_logs/agent-workload/e170-rocm-decode-q4-q3-vdr2-resources-r1.server.log`
