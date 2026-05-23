# E195 ROCm Q3_K Static Swiglu No-Bias Gate

## Metadata

- Experiment ID: E195
- Date: 2026-05-23
- Owner: Codex
- Branch/Commit: local `master` after `0898d5246`
- Hypothesis ID: H39
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the dominant fused Q3_K decode branch can improve if Qwen's common `gate + up -> SWIGLU` no-bias case uses a static specialized kernel instead of the generic runtime fusion branch.
- Mechanism: E152 shows residual Q3_K decode is still led by `mul_mat_vec_q_fused q3_K->f32` (`579.66 ms`, `64.62%` of parsed Q3_K MMVQ time), with top FFN shapes `m=17408,n=1,k=5120` and `m=5120,n=1,k=17408`. The current kernel compiles a generic fused template but keeps runtime checks for gate, x bias, gate bias, and GLU op. A static no-bias SWIGLU variant may remove branch/register pressure without changing launch geometry or q8 layout.
- Why now: E178/E179/E180/E190/E194 reject nearby selector, layout, y-reuse, and primitive-swap routes. This is a slightly larger branch-policy change inside the measured fused Q3_K path, still env-gated and easy to revert.

## Math / Theory

- Assumptions:
  - E151 clean post-win r3: `30.3145 TPS`, decode `32.2467 tok/s`, prompt eval `233.86 ms`, decode eval `3970.1533 ms`.
  - Baseline prefill share for the short-decode gate is about `0.0556`, so a whole-decode `1.01x`, `1.03x`, `1.05x` improvement projects `30.6006`, `31.1719`, `31.7420 TPS`.
  - This candidate only affects the fused Q3_K subroute, not the whole decode path. Therefore it needs a clear local resource/timing win before any r3 runtime gate.
- Analytical gate commands:
  - `python scripts/research/formula_sanity_checks.py` passed.
  - `python scripts/research/speedup_model.py --baseline-tps 30.3145 --prefill-share 0.0556 --draft-len 1 --accept-rate 0 --spec-overhead 0 --flash-prefill-speedup 1.0 --decode-kernel-speedup 1.01`
  - same for `1.03` and `1.05`.
- Expected speedup corridor:
  - Continue to r1/r3 only if the resource trace does not raise registers/occupancy pressure and the dominant fused Q3_K bucket improves.
  - Reject if local improvement is below noise or if wall r1 moves prompt/decode in opposite directions, following E190.
- Failure conditions:
  - specialization does not trigger on Qwen graph,
  - build failure,
  - register increase or occupancy drop,
  - local fused bucket tie/regression,
  - r1/r3 wall regression,
  - output sanity issue.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/mmvq.cu`: add env-gated Q3_K `ncols_dst=1` static SWIGLU/no-bias launch specialization inside `mul_mat_vec_q_switch_fusion`.
2. Guard rails:
   - default path unchanged unless `GGML_MMVQ_Q3K_STATIC_SWIGLU_NOBIAS=1`;
   - trigger only for `type == GGML_TYPE_Q3_K`, `ncols_dst == 1`, `fusion.gate != nullptr`, no x/gate bias, and `glu_op == GGML_GLU_OP_SWIGLU`;
   - resource/timing gate before r3.
3. Rollback path: revert `mmvq.cu` if build/resource/runtime gate fails.

## Benchmark Plan

- Build gate: `cmake --build build-rocm-vec --config Release -j`.
- Resource gate: short sync trace with `GGML_MMVQ_Q3K_STATIC_SWIGLU_NOBIAS=1`, `GGML_TRACE_MMVQ_RESOURCES=1`, `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`, `max_tokens=16`.
- Runtime gate: H39 clean short-decode `triage_diff,review_bug`, `max_tokens=512`, first r1 then r3 only if r1/resource are promising.
- Artifacts path: `build_logs/agent-workload/e195-rocm-q3k-static-swiglu-*`.

## Metrics

- aggregate completion TPS
- decode eval tok/s and ms
- prompt eval tok/s and ms
- Q3_K fused/direct MMVQ bucket timings
- regs / occupancy / LDS
- trigger count / output sanity
- errors

## Result

- Outcome: reject and revert.
- Delta: clean r1 wall `31.9110 -> 30.6142 TPS` (`-4.06%`), decode eval `32.45 -> 31.11 tok/s` (`-4.13%`), prompt eval essentially tied (`633.685 -> 633.295 tok/s`).
- Confidence: medium-high. This is only r1, but the resource/timing gate did not show a local fused-Q3_K improvement and the clean wall gate regressed clearly enough to stop before r3.
- Recommendation: do not keep static SWIGLU/no-bias Q3_K specialization. Reducing VGPR in the `ncols_x=5120` fused bucket is not sufficient; the active cost is not simply runtime branch/register pressure inside the generic fusion body.

## Measured Data

Artifacts:

- `build_logs/agent-workload/e195-rocm-q3k-static-swiglu-control-res-r1.server.log`
- `build_logs/agent-workload/e195-rocm-q3k-static-swiglu-cand-res-r1.server.log`
- `build_logs/agent-workload/e195-static-swiglu-resource-compare.md`
- `build_logs/agent-workload/e195-rocm-q3k-static-swiglu-control-clean-r1.diagnostics.md`
- `build_logs/agent-workload/e195-rocm-q3k-static-swiglu-cand-clean-r1.diagnostics.md`

Build gates:

| Step | Result |
| --- | --- |
| first build | failed because `GGML_ASSERT` cannot be called from a HIP `__global__` kernel |
| fixed build | passed |
| revert build | passed; `build-rocm-vec` restored to clean source after rejection |

Resource/timing gate, sync trace:

| Bucket | Control | Candidate | Interpretation |
| --- | ---: | ---: | --- |
| `MMVQ type=11/q3_K ncols_dst=1 small_k=1 fusion=1` | `580.240 ms` | `580.369 ms` | tie/slightly worse |
| `ncols_x=5120`, fused Q3_K | `342.370 ms`, `regs=84`, `occ=87.5%` | `343.617 ms`, `regs=46`, `occ=100%` | resource improved, time did not |
| `ncols_x=17408`, fused Q3_K | `209.982 ms`, `regs=84` | `209.303 ms`, `regs=84` | specialization did not lower regs here |
| all CUDA nodes | `3374.574 ms` | `3359.858 ms` | trace-level all-node noise, not a fused-Q3 win |

Clean wall gate:

| Metric | Control r1 | Candidate r1 | Delta |
| --- | ---: | ---: | ---: |
| aggregate completion TPS | `31.9110` | `30.6142` | `-4.06%` |
| decode eval TPS | `32.45` | `31.11` | `-4.13%` |
| prompt eval TPS | `633.685` | `633.295` | `-0.06%` |
| prompt eval ms | `248.875` | `248.255` | `-0.620 ms` |
| decode eval ms | `15777.29` | `16457.985` | `+680.695 ms` |
| errors | `0` | `0` | `0` |

The candidate was stopped at r1. No r3 was run.

Rollback:

- The temporary `mmvq.cu` code was reverted.
- `cmake --build build-rocm-vec --config Release -j 2` passed after rollback.

## Notes

- Surprises: the specialization did exactly one attractive thing on paper (`regs=84 -> 46`, occupancy `87.5% -> 100%` for the `ncols_x=5120` fused bucket), but that bucket's measured time did not improve and clean decode regressed. This is the same workflow lesson as E190 in a sharper form: resource counters alone are not enough when the route is likely limited by dequant/dot instruction mix, memory traffic, scheduling cadence, or graph-level interactions.
- Follow-up action: do not pursue more branch-removal/static-fusion micro-specializations without instruction-level evidence. The next H39 route needs to change actual Q3_K dot/dequant topology or scheduling, not just reduce generic fusion control flow.
