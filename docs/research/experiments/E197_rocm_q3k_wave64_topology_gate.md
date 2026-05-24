# E197 ROCm Q3_K Wave64 Topology Gate

## Metadata

- Experiment ID: E197
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E196 route recapture
- Target lane: H39 ROCm decode-heavy route, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: ROCm Q3_K one-token MMVQ may be losing part of the Vulkan gap because the hot path uses two wave32-style warps plus shared-memory cross-warp reduction, while Vulkan q8_1 matvec runs a larger subgroup-style reduction topology.
- Mechanism: for `Q3_K`, `ncols_dst=1`, `small_k=1`, keep the E151 row batching/grid width (`rows_per_block=2`) but launch a candidate kernel as `block=(64,1,1)` and reduce with width `64`. This removes the explicit `threadIdx.y` cross-warp shared reduction while keeping the number of rows per block unchanged.
- Why now: E196 shows H39 is still Q3_K route-body dominated (`56.95%` fused + `31.33%` direct in parsed Q3_K), while E167/E179 prove row/grid-width loss is negative and E170 proves larger per-thread fragments hit a register cliff.

## Math / Theory

- Assumptions:
  - Q3_K MMVQ parsed Q3_K share from E196 is `0.8828` if fused+direct are considered together;
  - fused-only share is `0.5695`;
  - this candidate must not change row grid width or VDR.
- Projected:
  - for all Q3_K MMVQ, a `+2%` wall gain needs `1.0227x` local, `+5%` needs `1.0570x`, `+10%` needs `1.1148x`, parity `1.278x` needs `1.3270x`;
  - for fused-only, `+5%` wall needs `1.0912x` local and parity would need `1.6180x`, so direct must not regress.
- Failure conditions:
  - build fails;
  - output sanity fails or repeated-symbol corruption appears;
  - resource trace shows incorrect route activation or dominant Q3_K buckets regress;
  - clean r1 loses wall/decode speed.

## Implementation Plan

1. Minimal code surface to change: `ggml/src/ggml-cuda/mmvq.cu` only.
2. Add env gate: `GGML_MMVQ_Q3K_WAVE64=1`.
3. Activate only for `type == GGML_TYPE_Q3_K`, `ncols_dst == 1`, and `small_k == true`.
4. Keep default code path unchanged.
5. Rollback path: revert the `mmvq.cu` patch and rebuild `build-rocm-vec`.

## Benchmark Plan

- Build: `cmake --build build-rocm-vec --config Release -j 2`.
- Resource gate: `e197-rocm-q3k-wave64-res-r1`, graph disabled, MMVQ timing/resource trace, `max_tokens=16`.
- Runtime gate: `e197-rocm-q3k-wave64-clean-r1`, clean decode-heavy, `max_tokens=512`, compare to E196 clean ROCm r3 `31.9233`.
- Sanity: inspect output/diagnostics for correctness; if candidate looks positive, run paired r3.
- Artifacts path: `build_logs/agent-workload/e197-*`.

## Metrics

- aggregate completion TPS
- decode eval TPS
- error rate
- Q3_K fused/direct MMVQ timing
- regs/occupancy/shared memory
- output sanity

## Result

- Outcome: rejected and reverted. The temporary `mmvq.cu` patch was removed and
  `build-rocm-vec` was rebuilt back to baseline code.
- Resource/trace gate:
  - route activation worked: dominant `Q3_K`, `ncols_dst=1`, `small_k=1`
    buckets reported `block=(64,1,1)`;
  - fused `ncols_x=5120`: E196 `676.110 ms` -> E197 `681.567 ms`
    (`+0.81%` slower);
  - direct `ncols_x=5120`: E196 `554.893 ms` -> E197 `557.519 ms`
    (`+0.47%` slower);
  - fused `ncols_x=17408`: E196 `415.253 ms` -> E197 `418.187 ms`
    (`+0.71%` slower);
  - regs/occupancy stayed effectively unchanged for the hot buckets
    (`84/88 regs`, `87.50%` occupancy), so the topology did not buy a
    measurable resource win.
- Clean runtime gate:
  - E196 clean ROCm r3 reference: `31.9233 TPS`, decode `32.3833 tok/s`;
  - E197 wave64 clean r1: `31.6788 TPS`, decode `32.365 tok/s`, `0` errors;
  - wall delta vs reference: `-0.2445 TPS` (`-0.77%`), not a positive signal.
- Confidence: high enough to reject this branch because both the local
  resource/timing gate and the clean real-server gate were non-positive.
- Recommendation: do not pursue `Q3_K` MMVQ wave64/row-warp variants as a
  near-term ROCm decode parity route. The likely mistake in the premise is that
  removing the explicit cross-warp reduction is not the limiter; the current
  `(32,2,1)` topology probably benefits from the existing two-wave/K-split
  scheduling and latency hiding. Future candidates must reduce real Q3_K work or
  move the fused/direct route closer to Vulkan's matvec family without losing
  grid width, adding VDR/register pressure, or relying on occupancy-only wins.

## Notes

- This is intentionally not an occupancy-only experiment. The expected benefit is removing a reduction topology difference while preserving row batching/grid width.
- If HIP wave64 shuffle semantics are wrong for RDNA4 in this build, reject immediately as correctness/build risk.
- The build gate passed, so the rejection is runtime/resource based rather than
  compiler support based.

## Artifacts

- `build_logs/agent-workload/e197-wave64-build.log`
- `build_logs/agent-workload/e197-rocm-q3k-wave64-res-r1.server.log`
- `build_logs/agent-workload/e197-rocm-q3k-wave64-res-r1.diagnostics.md`
- `build_logs/agent-workload/e197-wave64-vs-e196-trace-compare.md`
- `build_logs/agent-workload/e197-wave64-q3k-resource-summary.md`
- `build_logs/agent-workload/e197-rocm-q3k-wave64-clean-r1.diagnostics.md`
- `build_logs/agent-workload/e197-wave64-revert-build.log`
