# E190 ROCm Q3_K Fused Pair-Dot Probe

## Metadata

- Experiment ID: E190
- Date: 2026-05-23
- Owner: Codex
- Branch/Commit: master after `1f8403355`
- Target lane: L1 ROCm H39, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: a streaming Q3_K pair-dot helper can reduce duplicated `q8_1` activation reads in fused FFN MMVQ without repeating E165's register/occupancy cliff.
- Mechanism: when the fused path computes x and gate against the same `q8_1` block, compute both dot products inside one helper while loading each `q8_1` lane once.
- Why now: E189 shows isolated decode micro wins below `+2%` are too small for the current wall mix, but a clean fused route-body improvement still has enough ceiling if it avoids the E165 register jump.

## Math / Theory

- Assumptions: use E187 L1 r3 as baseline (`12.4256 TPS`, prompt/decode split `6069.3967/4197.9733 ms`).
- Expected speedup corridor: `+0%..+3%` wall if the fused Q3_K buckets improve without prefill regression.
- Failure conditions:
  - candidate resource trace materially increases registers or lowers occupancy in dominant fused buckets;
  - local fused buckets improve but wall shifts into prompt/prefill and r3 does not confirm;
  - output/server errors appear.

## Implementation Plan

1. Minimal code surface to change:
   - `ggml/src/ggml-cuda/vecdotq.cuh`: add Q3_K pair-dot helper.
   - `ggml/src/ggml-cuda/mmvq.cu`: add env-gated launch specialization under `GGML_MMVQ_Q3K_FUSED_PAIRDOT=1`.
2. Guard rails:
   - default path unchanged;
   - candidate only for `type == GGML_TYPE_Q3_K`, `ncols_dst == 1`, fused route with gate present;
   - resource/timing gate before r3.
3. Rollback path: revert code if build/resource/runtime gate fails.

## Benchmark Plan

- Baseline command: E187/E188 L1 baseline and route evidence.
- Candidate resource command: L1 short trace with `GGML_MMVQ_Q3K_FUSED_PAIRDOT=1`, `GGML_TRACE_MMVQ_RESOURCES=1`, `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`.
- Candidate runtime command: L1 quick `triage_diff`, first r1, then r3 only if resource/timing is not worse.
- Artifacts path: `build_logs/agent-workload/e190-rocm-q3k-pairdot-*`.

## Metrics

- aggregate completion TPS
- decode eval tok/s
- Q3_K fused/direct bucket timings
- regs / occupancy / LDS
- errors

## Result

- Outcome: reject and revert.
- Delta: paired real-context r3 control `12.9580 TPS` vs candidate `12.8560 TPS`, `-0.1020 TPS` (`0.9921x`, `-0.79%`).
- Confidence: medium-high. The short resource trace was locally positive, but the paired real-context r3 moved both prompt and decode in the wrong direction.
- Recommendation: do not keep the pair-dot route. Future Q3_K work needs a larger route/topology change, not just shared `q8_1` reuse inside the existing fused MMVQ loop.

## Measured Data

### Resource / Timing Gate

Dominant fused Q3_K bucket, `ncols_x=5120`, short sync trace:

| Variant | Sum | Count | Avg | Median | Regs | Occupancy | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | `615.144 ms` | `962` | `0.6394 ms` | `0.3570 ms` | `84` | `87.5%` | `(32,2,1)` |
| pair-dot | `552.412 ms` | `962` | `0.5742 ms` | `0.3350 ms` | `95` | `100.0%` | `(32,2,1)` |

This passed the resource gate: unlike E165, there was no occupancy cliff, and the local fused bucket looked faster despite higher VGPR use.

### Runtime Gate

The first r1 probe `e190-rocm-q3k-pairdot-r1` is invalid for lane comparison because it missed `--real-context-mode repo-snapshot` and only used `159` prompt tokens. It is ignored for the decision.

Correct real-context probes:

| Variant | Runs | Aggregate TPS | Decode TPS Mean | Prompt TPS Mean | Prompt Eval Mean | Decode Eval Mean | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate r1 | 1 | `12.6039` | `31.50` | `1223.92` | `6056.77 ms` | `4063.98 ms` | 0 |
| control r3 | 3 | `12.9580` | `31.4433` | `1284.85` | `5774.7067 ms` | `4070.82 ms` | 0 |
| candidate r3 | 3 | `12.8560` | `31.3267` | `1271.2333` | `5838.6167 ms` | `4086.1667 ms` | 0 |

The paired r3 rejects the candidate:

- aggregate wall: `12.9580 -> 12.8560 TPS` (`-0.79%`);
- decode: `31.4433 -> 31.3267 tok/s` (`-0.37%`);
- prompt: `1284.85 -> 1271.2333 tok/s` (`-1.06%`);
- eval time shifted into both prompt and decode despite the local fused bucket improvement.

## Interpretation

The experiment was useful because it demonstrated a route-chain trap: a local fused MMVQ bucket can improve while the real server lane regresses. In this case, shared `q8_1` activation reuse was not the limiting cost of the current L1 route. The pair-dot helper increased instruction/register complexity and did not translate into lower end-to-end graph time.

Next Q3_K candidates should change a larger part of the route, for example launch topology, graph-level fusion policy, or a new specialized fused route for the actual top post-trace bucket. Do not repeat y-reuse-only pair/preload probes unless a fresh trace shows `q8_1` load pressure is the measured bottleneck.

## Follow-up: pair-dot disable A/B via host-side gate

Date: 2026-05-24

Goal:

- separate two hypotheses that were still entangled after the original rejection:
   - pair-dot is conceptually useful but the current always-on wiring is wrong;
   - pair-dot reuse is simply not the bottleneck, so both on/off stay near-tied.

Implementation:

- kept the default path unchanged;
- added host-side dispatch gate `GGML_MMVQ_Q3K_DISABLE_PAIRDOT` in `ggml/src/ggml-cuda/mmvq.cu` so the same fused Q3_K kernel family can be launched with pair-dot on or off without illegal device-side env access.

Lane and measurement contract:

- same L1 lane: `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on;
- paired trace run with `GGML_TRACE_MMVQ_RESOURCES=1`, `GGML_TRACE_MMVQ_TIMING=1`, `GGML_TRACE_MMVQ_TIMING_SYNC=1`.

Artifacts:

- pair-dot on:
   - `build_logs/agent-workload/e190-rocm12k-pairdot-on-trace-r1.server.log`
   - `build_logs/agent-workload/e190-rocm12k-pairdot-on-trace-r1.diagnostics.md`
- pair-dot off:
   - `build_logs/agent-workload/e190-rocm12k-pairdot-off-trace-r1.server.log`
   - `build_logs/agent-workload/e190-rocm12k-pairdot-off-trace-r1.diagnostics.md`

Measured result:

| Variant | Aggregate TPS | Decode TPS Mean | Prompt Eval Mean | Decode Eval Mean |
| --- | ---: | ---: | ---: | ---: |
| pair-dot on | `7.6684` | `30.44` | `6209.77 ms` | `2102.67 ms` |
| pair-dot off | `7.6747` | `30.62` | `6214.68 ms` | `2090.30 ms` |

Point/resource signal on the fused Q3_K route:

- pair-dot on: hot fused buckets use `95 regs`, `100%` occupancy;
- pair-dot off: the same fused buckets drop to `84 regs`, `87.5%` occupancy;
- despite the lower register footprint, representative fused points stay near-tied rather than clearly faster.

Interpretation update:

- lowering VGPR pressure from `95 -> 84` without a point-ms win is strong evidence that pair-dot live-state is not the controlling cost on this route;
- the missing gain is not hiding behind one more helper rewrite or branch cleanup inside the same MMVQ body;
- the next branch should be route-level and remove a larger class of work:
   - ephemeral same-step staging reuse across sibling Q3_K matmuls,
   - a shape-specialized non-persistent `Q3_K x F16` direct kernel for the hot Qwen buckets,
   - or a storage/layout contract change that lets ROCm use a more vector-friendly Q3_K path.

Updated decision:

- keep the host-side disable gate only as an experimental knob;
- do not treat pair-dot on/off selection as an optimization target by itself;
- move the research center to staging/layout/route-body redesign.

## Notes

- This probe intentionally does not touch launch geometry, rows-per-block, nwarps, q8 layout, or fusion policy.
- Code was reverted after the r3 rejection; default ROCm path remains unchanged.
- Build gate passed after fixing an intermediate `half * float` ambiguity in the candidate helper.
