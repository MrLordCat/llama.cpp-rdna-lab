# C01 - MUL_MAT forward

Quick return anchor:
- `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md`

Current contract note (2026-05-17):
- quick benchmark task pair is `triage_diff,review_bug`.
- older `review_bug,patch_sim` references in this file are historical experiment context only.

## Resume checkpoint after chatflow detour (2026-05-13)

Fresh resume run (lane contract from playbook):
- `build_logs/agent-workload/c01-resume-r1-resources.server.log`
- `build_logs/agent-workload/c01-resume-r1-resources.csv`
- aggregate TPS: `6.3221`

Mandatory pre-edit gate status on this run:
- shape presence gate (`qtype=11`, `ncols_max=192`): PASS (`count=26524`),
- cold/steady split (`MUL_MAT forward`, steady `<5 ms`): steady dominated by `mul_mat_q_direct|q3_K` (`78.20%` share),
- q3 coarse component split (steady):
	- `compute_core_q3`: `84.25%`,
	- `fallback_cublas`: `14.38%`,
	- `dequant_load_vec_q3`: `1.37%`.

Comparison checkpoints:
- vs global decode baseline (`decode-trace-current-ctx12288-ub192-r1`): not comparable for runtime verdict due different task mix.
- vs C01-compatible baseline (`e013-c01-two-tasks-trace-r1-resources`):
	- runtime: `6.6897 -> 6.3221 TPS` (`-5.5%`) on first resume run,
	- route compare: no major route flip in primary target buckets;
		`MUL_MAT forward` average timing ratio remained close (`0.992`).

Control rerun on the same lane:
- `build_logs/agent-workload/c01-resume-r2-control.server.log`
- aggregate TPS: `6.3309`
- vs `c01-resume-r1-resources`: `+0.14%` (inconclusive/noise-level)
- trace compare (`c01_r1 -> c01_r2`) shows near-identical hotspot timings (`avg ratio ~0.999` for `MUL_MAT forward`).

Interpretation:
- C01 route topology remained consistent after return.
- The runtime level around `~6.33 TPS` is now stable across two same-lane reruns; continue hypothesis work from this point.

## Post-resume quick probe A/B (stream-k knob, 2026-05-13)

Candidate:
- `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=144`

Lane/result:
- baseline: `c01-resume-r2-control` -> `6.3309 TPS`
- candidate: `c01-resume-r3-streamk144` -> `6.3354 TPS`
- runtime delta: `+0.07%` (borderline/inconclusive)

Hotspot-time signal (trace compare `c01_r2_control -> c01_r3_streamk144`):
- `CUDA_NODE op=MUL_MAT kind=forward`: `15448.053 -> 15428.358 ms` (`-19.695 ms`)
- `CUDA_NODE op=MUL_MAT`: `15603.131 -> 15580.533 ms` (`-22.598 ms`)
- no route topology change; effect is micro-level and needs stronger confirmation.

Verdict:
- keep as knob-only signal (no default change).
- requires a paired rerun and/or stronger candidate before promotion.

## Stream-k knob confirmation (runs=3, no trace overhead)

Same lane (`review_bug,patch_sim`, `ctx=12288`, `b=6144`, `ub=192`, `no-reuse`), but with `runs=3` for final confirmation:

- control: `c01-next-control-r3` -> `9.1787 TPS`
- candidate `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=144`: `c01-next-streamk144-r3` -> `9.0528 TPS`
- delta: `-1.37%` (statistical verdict: negative, bootstrap 95% CI excludes zero on negative side)

Interpretation:
- The previous `r1` micro-positive signal did not hold under `runs=3` confirmation.
- For effective acceleration, `stream-k NE11=144` should not be a primary direction (keep only as optional debug knob).

Current priority after this confirmation:
1. Continue C01 on `mul_mat_q_direct|q3_K` core path (steady dominant share) with route-local hypotheses.
2. Favor ideas that can reduce q3 core kernel time directly (not selector noise): shared/register pressure, memory-layout/loads, launch granularity for `ncols~139/140` cluster.
3. Use dual-metric gate (TPS + hotspot-time) and keep `runs=3` for any near-borderline candidate.

## Post-resume probe B: force MMQ runtime (2026-05-13)

Candidate:
- `GGML_CUDA_FORCE_MMQ_RUNTIME=1`

Runtime confirmation (`runs=3`, no trace overhead):
- baseline: `c01-next-control-r3` -> `9.1787 TPS`
- candidate: `c01-next-force-mmq-r3` -> `9.1127 TPS`
- delta: `-0.72%` (borderline/inconclusive runtime verdict)

Hotspot/route causality check (trace pair):
- compare: `c01-resume-r2-control.server.log` -> `c01-next-force-mmq-trace-r1.server.log`
- `CUDA_NODE op=MUL_MAT kind=forward`: `15448.053 -> 15532.074 ms` (`+84.021 ms`)
- MMQ q3 bucket (`type=11, ncols_max=192`): `9558.537 -> 9659.888 ms` (`+101.351 ms`)
- steady route split did not reduce `cublas_backend|f32` share (it slightly increased from `13.33%` to `13.43%`).

Verdict:
- reject as acceleration path for this lane.
- do not spend further cycles on forced-MMQ global routing; it increases target q3 hotspot time.

## Current cost snapshot

- Center: `CUDA_NODE op=MUL_MAT kind=forward`
- sum_ms: `1717.322`
- count: `8454`
- avg_ms: `0.203`
- Priority: `P1`

Source trace:
- `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`

## Full sub-trace (current baseline)

### Top node names by sum_ms

| Node name | sum_ms | count | avg_ms |
| --- | ---: | ---: | ---: |
| `node_200` | 124.818 | 21 | 5.944 |
| `node_34` | 32.351 | 21 | 1.541 |
| `node_13` | 31.140 | 21 | 1.483 |
| `result_output` | 24.280 | 21 | 1.156 |
| `linear_attn_out-0` | 9.421 | 21 | 0.449 |

### Top shapes (`ne`) by sum_ms

| ne | sum_ms | count | avg_ms |
| --- | ---: | ---: | ---: |
| `(256,48,1,1)` | 123.910 | 16 | 7.744 |
| `(17408,159,1,1)` | 87.390 | 126 | 0.694 |
| `(10240,1,1,1)` | 86.933 | 768 | 0.113 |
| `(17408,140,1,1)` | 85.563 | 126 | 0.679 |
| `(17408,139,1,1)` | 84.905 | 126 | 0.674 |
| `(5120,1,1,1)` | 80.279 | 784 | 0.102 |

## A/B signal already observed

- Control: `decode-trace-current-ctx12288-ub192-r1` -> `26.30 TPS`
- Candidate: `GGML_MMVQ_QWEN_FORCE_SMALL_K=1` (`decode-trace-current-ctx12288-ub192-force-smallk-r1`) -> `26.66 TPS`
- Trace compare: `build_logs/agent-workload/decode-trace-smallk-compare.md`
- Notable route delta: MMVQ moved from `small_k=0` to `small_k=1`, and total `CUDA_NODE` time dropped.

## Two-task r1 validation (requested lane)

- Task set: `review_bug, patch_sim` only
- Runs: `r1`
- Baseline (new default small_k):
	- `c01-two-tasks-r1-default-smallk` -> `28.02 TPS`
	- `c01-two-tasks-r1-default-smallk-rerun` -> `28.06 TPS`
- Control with small_k disabled:
	- `c01-two-tasks-r1-disable-smallk` -> `27.86 TPS`
- Net uplift on this lane: about `+0.16..+0.20 TPS` (`~+0.6..+0.7%`) in favor of default small_k.

Artifacts:
- `build_logs/agent-workload/c01-two-tasks-r1-default-smallk.jsonl`
- `build_logs/agent-workload/c01-two-tasks-r1-default-smallk-rerun.jsonl`
- `build_logs/agent-workload/c01-two-tasks-r1-disable-smallk.jsonl`

## Trace validation on two-task r1 lane

Serial `kernel-full` traces (no parallel overlap):

- default small_k: `c01-two-tasks-trace-r1-default-smallk-serial` -> `26.68 TPS`
- disable small_k: `c01-two-tasks-trace-r1-disable-smallk-serial` -> `26.46 TPS`

Key trace deltas (`disable -> default`):

- `CUDA_NODE`: `1793.128 -> 1767.849 ms` (`-25.279 ms`)
- `CUDA_NODE op=MUL_MAT`: `1113.771 -> 1099.978 ms` (`-13.793 ms`)
- `CUDA_NODE op=MUL_MAT kind=forward`: `956.045 -> 943.461 ms` (`-12.584 ms`)
- MMVQ route migration for qwen-hot types:
	- `small_k=0` buckets disappear,
	- `small_k=1` buckets appear with lower average kernel time.

Compare report:
- `build_logs/agent-workload/c01-two-tasks-trace-smallk-default-vs-disable.md`

## Route decision map (code-verified)

- File: `ggml/src/ggml-cuda/mmvq.cu`
- `small_k` auto-heuristic is evaluated inside `should_use_small_k(...)`.
- Generic RDNA path still disables auto-small_k by default.
- Qwen-hot RDNA4 path (`Q3_K/Q4_K/Q6_K`, `ncols_dst=1`) now defaults to `small_k=1`.
- Explicit env overrides remain supported:
	- `GGML_MMVQ_QWEN_FORCE_SMALL_K=1` forces enable.
	- `GGML_MMVQ_QWEN_DISABLE_SMALL_K=1` forces disable.

## C01 verdict

- Decision: `keep`
- Reason: stable same-lane decode uplift on requested two-task `r1` lane, with matching reduction in `MUL_MAT forward` trace cost.
- Risk: low. Override escape hatch is preserved via env flags if any model-specific regression appears.

## Deep route drilldown (where C01 still spends time)

Source:
- `build_logs/agent-workload/c01-two-tasks-trace-r1-default-smallk-serial.server.log`

### Route-level split inside `MUL_MAT forward` (`TOTAL=969.854 ms`)

| Route + type | sum_ms | share |
| --- | ---: | ---: |
| `mul_mat_q_direct | q3_K` | 386.811 | 39.88% |
| `mul_mat_vec_q_direct | q3_K` | 214.295 | 22.10% |
| `cublas_backend | f32` | 205.952 | 21.24% |
| `mul_mat_vec_f_direct | f32` | 67.749 | 6.99% |
| `mul_mat_vec_q_direct | q4_K` | 44.489 | 4.59% |
| `mul_mat_q_direct | q4_K` | 37.753 | 3.89% |
| `mul_mat_vec_q_direct | q6_K` | 12.805 | 1.32% |

### Important distinction: cold spike vs steady-state

- `node_200` (`cublas_backend|f32`) has a huge one-off spike:
	- `count=11`, `sum=111.333`, `median=0.152`, `max=109.980 ms`.
- Similar first-pass spikes exist for:
	- `node_34`: `max=21.953 ms`,
	- `node_13`: `max=20.835 ms`.
- If we isolate calls with `total_ms < 5` (steady-ish window), route shares become:
	- `mul_mat_q_direct|q3_K`: `47.30%`
	- `mul_mat_vec_q_direct|q3_K`: `24.08%`
	- `cublas_backend|f32`: `11.94%`

Conclusion: for sustained throughput, the main pressure is not cublas spike itself but `q3_K` direct routes (about `71%` in steady slice).

### Concrete hotspots by single call

Top expensive individual calls in this trace:

1. `node_200` `ne=(256,48,1,1)` -> `109.980 ms` (cold spike)
2. `node_34` `ne=(48,2,1,1)` -> `21.953 ms` (cold spike)
3. `node_13` `ne=(10240,2,1,1)` -> `20.835 ms` (cold spike)
4. `linear_attn_out-0` `ne=(5120,140,1,1)` -> `6.875 ms`
5. `node_13` `ne=(10240,140,1,1)` -> `6.749 ms`

### Updated C01 next target

- Priority route to optimize next: `mul_mat_q_direct|q3_K` (not MMVQ small_k path).
- Practical next A/B for C01:
	1. separate cold-first vs warmed decode claim (to avoid one-off cublas spike contamination),
	2. route-level A/B aimed at `q3_K` direct path (`ncols_dst~139/140` cluster),
	3. verify that any gain remains when `review_bug,patch_sim` only and `runs=1`.

## C01 deeper route focus (q3_K direct)

Baseline used:
- `build_logs/agent-workload/c01-two-tasks-trace-r1-streamk-default-serial.server.log`

`mul_mat_q_direct|q3_K` contribution:
- total in `MUL_MAT forward`: `386.397 ms`
- steady slice (`total_ms < 5`): `379.574 ms`

Top `ne` buckets inside `mul_mat_q_direct|q3_K`:

| ne | sum_ms | count |
| --- | ---: | ---: |
| `(17408,140,1,1)` | 84.034 | 126 |
| `(17408,139,1,1)` | 83.980 | 126 |
| `(5120,139,1,1)` | 53.885 | 79 |
| `(5120,140,1,1)` | 53.724 | 79 |
| `(10240,140,1,1)` | 27.644 | 48 |
| `(10240,139,1,1)` | 21.167 | 48 |

Interpretation: dominant sustained pressure is the `139/140` batch cluster in q3_K direct MMQ path (mostly FFN-related shapes), not MMVQ.

## C01 experiment E1: RDNA4 stream-k threshold for q3_K cluster

What was tested (two-task lane, `review_bug,patch_sim`, `r1`):
- default threshold behavior: `c01-two-tasks-r1-streamk-default` -> `28.07 TPS`
- candidate `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=128`: `c01-two-tasks-r1-streamk-ne11-128` -> `28.13 TPS`
- aggressive `...=96`: `c01-two-tasks-r1-streamk-ne11-96` -> `27.35 TPS` (unstable/worse)

Trace pair (`default` vs `128`) conclusion:
- `MUL_MAT forward` sum increased (`939.913 -> 945.236 ms`)
- `MUL_MAT` cold spikes (`node_200`, `node_34`, `node_13`) became larger
- despite tiny TPS noise-level gain in one non-trace run, route-cost signal is not consistently positive

Decision for E1:
- `reject` (do not keep stream-k threshold tweak in code)
- code reverted to baseline behavior.

## C01 experiment E2: force q3_K 139/140 out of MMQ

Idea:
- tighten RDNA4 q*_K MMQ gate from `ne11<=192` to `<=128` to route `ne11=139/140` through cublas path.

Two-task `r1` result (`review_bug,patch_sim`):
- default gate: `c01-two-tasks-r1-rdna4-qkmax-default` -> `28.27 TPS`
- candidate (`qk_max=128`): `c01-two-tasks-r1-rdna4-qkmax-128` -> `25.56 TPS`

Decision for E2:
- `reject` (strong regression, high variance)
- code reverted.

Interpretation:
- On this lane, keeping q3_K `139/140` inside MMQ is clearly better than spilling to cublas.

## C01 experiment E3: force larger MMQ x-tile on RDNA4

Idea:
- keep route unchanged, but force MMQ selector to skip `xbest=80` and start from larger tile (`xbest=96/112`) for q3_K `ncols_max=139/140`.

What was tested (`review_bug,patch_sim`, `r1`):
- baseline: `c01-two-tasks-r1-rdna4-minx-default` -> `27.91 TPS`
- candidate `GGML_MMQ_RDNA4_MIN_X=96`: `c01-two-tasks-r1-rdna4-minx-96` -> `27.95 TPS` (noise-level)
- candidate `GGML_MMQ_RDNA4_MIN_X=112`: `c01-two-tasks-r1-rdna4-minx-112-candidate` -> `27.93 TPS` (no gain)

MMQ timing cross-check (target cluster):
- default (`xbest=80`):
	- `ncols_max=140`: `162.176 ms` (`349` calls)
	- `ncols_max=139`: `156.866 ms` (`349` calls)
- forced (`xbest=96`):
	- `ncols_max=140`: `174.449 ms` (`349` calls)
	- `ncols_max=139`: `169.128 ms` (`349` calls)

Decision for E3:
- `reject` (target MMQ cluster became slower; TPS gain not stable)
- code reverted.

## C01 experiment E4: force exact MMQ x-tile (64/96)

Idea:
- bypass heuristic `xbest=80` and force exact MMQ x-tile to map performance curve on the target q3_K cluster.

Two-task `r1` TPS:
- default selector: `c01-two-tasks-r1-forcex-default` -> `28.05 TPS`
- force `x=64`: `c01-two-tasks-r1-forcex-64` -> `27.97 TPS`
- force `x=96`: `c01-two-tasks-r1-forcex-96` -> `28.03 TPS`

MMQ timing (type=11/q3_K, `ncols_max=139/140`, `349` calls each):
- default (`xbest=80`):
	- `140`: `162.176 ms`
	- `139`: `156.866 ms`
- force `x=64`:
	- `140`: `193.804 ms`
	- `139`: `188.365 ms`
- force `x=96`:
	- `140`: `176.930 ms`
	- `139`: `169.765 ms`

Decision for E4:
- `reject` (`xbest=80` is best among tested points for this lane)
- code reverted.

## C01 experiment E5: RDNA4 MMQ y-tile (128 -> 64)

Idea:
- reduce MMQ y-tile on RDNA4 from default `128` to `64` to test potential register/shared-memory relief.

Two-task `r1` TPS:
- default (`mmq_y=128`): `c01-two-tasks-r1-rdna4-y-default` -> `28.08 TPS`
- candidate (`mmq_y=64`): `c01-two-tasks-r1-rdna4-y-64` -> `28.06 TPS`

MMQ timing (type=11/q3_K, `ncols_max=139/140`, `349` calls each):
- default (`mmq_y=128`):
	- `140`: `162.176 ms`
	- `139`: `156.866 ms`
- candidate (`mmq_y=64`):
	- `140`: `239.652 ms`
	- `139`: `229.054 ms`

Decision for E5:
- `reject` (target MMQ cluster became substantially slower)
- code reverted.

## Hypotheses (to validate)

1. High-cost `MUL_MAT forward` slices are shape-sensitive (`(256,48,1,1)` and long skinny matrices).
2. Decode route has MMVQ-driven coupling into `MUL_MAT` wall cost (indirect speed gain already visible).
3. Some `MUL_MAT` nodes are memory-bound and benefit from reduced sync/launch pressure rather than arithmetic changes.

## Trace workflow for C01

1. Reproduce baseline kernel trace.
2. Extract `MUL_MAT forward` by node and by `ne` (already done once).
3. Run controlled A/B variants and compare traces with `scripts/research/compare_kernel_traces.py`.
4. Mark each variant as `keep/neutral/reject` by same-lane TPS + `sum_ms` deltas.

## Acceptance gate for C01

- Primary: stable decode TPS gain over baseline on same lane.
- Secondary: measurable `sum_ms` reduction in `MUL_MAT forward` without regressions in other major centers.
- Promotion: requires at least one confirmation rerun.

### Dual-metric verdict policy (TPS + hotspot-time)

For C01 we now record two outcomes per candidate:

1. runtime verdict:
- based on lane TPS.

2. hotspot verdict:
- based on expensive-place timing deltas in trace (first of all `CUDA_NODE op=MUL_MAT kind=forward`, plus target MMQ bucket timing).

Important:
- if TPS is neutral/noisy but hotspot-time in the target expensive place improves, this is still a positive research signal (`hotspot-positive`) and should be kept for iteration notes.
- merge/default decisions still require stable end-to-end gain.

## E6 analytic gate before new kernel edits

To avoid low-yield selector sweeps, C01 now uses a simple Amdahl pre-gate for the target q3_K direct cluster.

- Baseline share inside `MUL_MAT forward` (current C01 trace):
	- `f = 386.397 / 969.854 = 0.3984`
- Model:
	- `S = 1 / ((1 - f) + f / s_local)`
	- where `s_local` is local speedup of the target cluster.

Required local speedup corridor from this gate:

| target center speedup `S` | required local `s_local` |
| ---: | ---: |
| `1.01x` | `1.0255x` |
| `1.02x` | `1.0518x` |
| `1.03x` | `1.0789x` |
| `1.05x` | `1.1357x` |

Implication:
- candidates below about `+5%` local gain on the q3_K cluster are unlikely to produce a stable end-to-end uplift on this lane.

## C01 experiment E011: narrow q3_K 139/140 no-stream-k probe

Idea:
- Test one narrow mechanism-only candidate (no route changes): disable stream-k only for `RDNA4 + Q3_K + ne11 in {139,140}` via env flag.

Implementation status:
- Prototype was added as env-gated code path in `ggml/src/ggml-cuda/mmq.cu` and then reverted after verification.

Requested lane A/B (`review_bug,patch_sim`, `r1`):
- baseline: `e011-c01-two-tasks-r1-baseline` -> `9.34 TPS`
- candidate: `e011-c01-two-tasks-r1-candidate-nostreamk139140` -> `9.36 TPS`

Trace validation (`kernel-full` + MMQ timing):
- baseline: `e011-c01-two-tasks-trace-r1-baseline-mmqtiming` -> `6.64 TPS`
- candidate: `e011-c01-two-tasks-trace-r1-candidate-nostreamk139140-mmqtiming` -> `6.56 TPS`
- MMQ timing lines show `ncols_max=192` on this lane slice, so the `139/140`-scoped toggle was effectively not exercised.

Decision for E011:
- `reject`
- reason: candidate condition did not activate on measured shape bucket (`ncols_max=192`), and trace run regressed.
- code reverted.

Actionable lesson:
- Before coding shape-specific gates, first confirm that the target shape bucket is present in the exact lane variant being benchmarked.

## C01 experiment E012: RDNA4 stream-k threshold gate for observed `ncols_max=192`

Idea:
- keep default behavior unchanged,
- add an env-gated RDNA4 stream-k threshold override,
- test candidate at `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=192` on the same two-task lane.

Code:
- `ggml/src/ggml-cuda/mmq.cu`
- new env-gated helper: `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11` (default stays `256`).

Lane TPS (`review_bug,patch_sim`, `r1`):
- baseline: `e012-c01-two-tasks-r1-streamk-default` -> `14.40 TPS`
- candidate: `e012-c01-two-tasks-r1-streamk-min192` -> `14.42 TPS`

Trace hotspot deltas (`kernel-full + MMQ timing`, sync-applied):
- `CUDA_NODE`: `22430.960 -> 22379.713 ms` (`-51.247 ms`)
- `CUDA_NODE op=MUL_MAT`: `14575.007 -> 14541.438 ms` (`-33.569 ms`)
- `CUDA_NODE op=MUL_MAT kind=forward`: `14417.721 -> 14384.724 ms` (`-32.997 ms`)
- target MMQ bucket (`mul_mat_q_case type=11, ncols_max=192`):
	- `8944.730 -> 8936.004 ms` (`-8.726 ms`, same call count `25128`)

Decision for E012 (initial 192 probe):
- runtime verdict: `neutral-positive` (small TPS uplift, r1)
- hotspot verdict: `positive` (clear reduction in target expensive places)
- overall: `iterate/keep-as-knob` (default unchanged, env-gated candidate retained for broader sweep)

E012-R1 full-point sweep (all practical RDNA4 stream-k points):

Checked thresholds (`GGML_MMQ_RDNA4_STREAM_K_MIN_NE11`):
- `128, 144, 160, 176, 192, 208, 224, 240, 256, 320, 9999`

Sweep runtime ranking (agg TPS):
- best: `144` (`14.4325`)
- then: `160` (`14.4230`), `128` (`14.4029`), `176` (`14.4008`)
- baseline point `256`: `14.2724`

E012-R1 hotspot confirmation (`256` vs best `144`, both with `--task-hard-timeout 0`):
- `CUDA_NODE`: `22625.835 -> 22440.438 ms` (`-185.397 ms`)
- `CUDA_NODE op=MUL_MAT`: `14639.932 -> 14576.019 ms` (`-63.913 ms`)
- `CUDA_NODE op=MUL_MAT kind=forward`: `14480.357 -> 14420.098 ms` (`-60.259 ms`)
- target MMQ bucket (`mul_mat_q_case type=11, ncols_max=192`):
  - `8968.599 -> 8959.550 ms` (`-9.049 ms`, same call count `25128`)

Updated decision after full sweep:
- runtime verdict: `positive` for `skmin=144` vs `256`
- hotspot verdict: `positive`
- overall: `keep-as-knob` (best candidate now `skmin=144`), default still unchanged pending extra confirmation.

Artifacts:
- `build_logs/agent-workload/e012-c01-sweep-summary.md`
- `build_logs/agent-workload/e012-c01-two-tasks-r1-sweep-skmin-*.csv`
- `build_logs/agent-workload/e012-c01-two-tasks-trace-r1-sweep-skmin-256-mmqtiming-nohard.server.log`
- `build_logs/agent-workload/e012-c01-two-tasks-trace-r1-sweep-skmin-144-mmqtiming-nohard.server.log`
- `build_logs/agent-workload/e012-c01-trace-compare-skmin256-vs-144-nohard.md`

## E013 diagnostics toolkit (added)

### 1) Per-kernel MMQ resource telemetry

Code path:
- `ggml/src/ggml-cuda/mmq.cuh`

Enable:
- `GGML_TRACE_MMQ_TIMING=1`
- `GGML_TRACE_MMQ_TIMING_SYNC=1`
- `GGML_TRACE_MMQ_RESOURCES=1`

Now `mul_mat_q_case: timing` includes (when available):
- `regs`
- `nbytes_shared` + `shared_pct`
- `max_blocks_per_sm`
- `max_threads_per_sm`
- `occupancy_pct`
- `waves_per_sm`

This gives lane-local resource context for MMQ buckets without external profiler.

### 2) Shape presence gate before shape-scoped experiments

Script:
- `scripts/research/c01_shape_presence_gate.py`

Example:

```bash
python scripts/research/c01_shape_presence_gate.py \
	build_logs/agent-workload/<trace>.server.log \
	--qtype 11 --ncols 139,140 --min-count 1 --strict
```

Rule:
- if gate fails, reject shape-scoped probe before coding.

### 3) Cold-vs-steady split protocol

Script:
- `scripts/research/cold_steady_trace_split.py`

Example:

```bash
python scripts/research/cold_steady_trace_split.py \
	build_logs/agent-workload/<trace>.server.log \
	--op MUL_MAT --kind forward --steady-max-ms 5 --top 12
```

Outputs route shares separately for:
- `cold` (spikes)
- `steady` (sustained window)

### 4) Coarse q3 path components (proxy split)

Script:
- `scripts/research/c01_q3_path_components.py`

Example:

```bash
python scripts/research/c01_q3_path_components.py \
	build_logs/agent-workload/<trace>.server.log --kind forward --steady-max-ms 5
```

Notes:
- components are explicit proxies from route lines (`compute_core_q3`, `dequant_load_vec_q3`, `fallback_cublas`), not kernel-internal cycle counters.

### 5) Statistical decision layer for borderline deltas

Script:
- `scripts/research/decision_stats.py`

Example:

```bash
python scripts/research/decision_stats.py \
	--baseline build_logs/agent-workload/<base>.csv \
	--candidate build_logs/agent-workload/<cand>.csv \
	--bootstrap 3000 --borderline-pct 1.0
```

Outputs:
- aggregate delta
- per-task 95% CI
- bootstrap delta CI
- effect size (Cohen d)
- statistical verdict (`positive`/`negative`/`inconclusive`)

## C01 experiment E014: post-E013 MMQ selector/resource pressure

Fresh post-E013 resource baseline:
- label: `c01-poste013-r1-resources`
- artifact: `build_logs/agent-workload/c01-poste013-r1-resources.server.log`
- trace aggregate: `6.61 TPS`
- shape gate: `qtype=11 ncols=192` PASS (`192:26524`, `91:349`, `90:349`)
- steady `MUL_MAT forward`: `15616.091 ms`
- steady `mul_mat_q_direct|q3_K`: `12325.249 ms` (`78.93%`)
- q3 coarse split: `compute_core_q3=84.72%`, `fallback_cublas=14.07%`, `dequant_load_vec_q3=1.21%`

No-trace selector probes against fresh default (`9.41 TPS`):
- forced `mmq_x=64`: `8.90 TPS`
- forced `mmq_x=80`: `8.34 TPS`
- forced `mmq_x=112`: `8.76 TPS`
- forced `mmq_x=128`: `8.59 TPS`
- apparent `mmq_x=88/104`: looked similar to default, but trace showed the override did not activate (`mmq_x_best=96`, `mmq_x_forced=0`)

Additional probes:
- real `mmq_x=104` by temporary granularity-8 override: hard timeout at `30.01s`, reject.
- post-E013 stream-k retest: `skmin=192 -> 9.43 TPS`, `skmin=144 -> 9.41 TPS`; no repeatable gain over default.
- `mmq_y=64`: compile-time reject; `mmq_write_back_mma` static assert requires `nwarps * tile_C::I == mmq_y`.
- RDNA4 `launch_bounds(..., 1)`: runtime `9.48 TPS`, but target trace worsened:
  - `MMQ type=11 ncols_max=192`: `9949.928 -> 10005.326 ms`
  - decision: reject as non-causal/runtime-noise; code reverted.

Decision:
- `reject`
- reason: none of the simple selector/resource probes improved both wall runtime and target MMQ bucket.
- code state: all temporary probes reverted; `llama-server` rebuilt after rollback.

Next C01 direction:
- stop scalar selector sweeps for this bucket.
- inspect Q3_K MMQ compute/load internals (`load_tiles_q3_K`, scale/min unpack, accumulator/write-back pressure) with the same target-positive rule.

## C01 experiment E015: RDNA4 MMQ `mmq_y=64/nwarps=4`

Idea:
- Pair the previously failed `mmq_y=64` direction with `nwarps=4` on RDNA4.
- This preserves the MMA write-back invariant (`nwarps * tile_C::I == mmq_y`) while reducing the Q3_K MMQ shared-memory footprint.

Code:
- `ggml/src/ggml-cuda/mmq.cuh`
- RDNA4 host/device `mmq_y`: `128 -> 64`
- RDNA4 host/device `nwarps`: `8 -> 4`

Paired lane A/B (`review_bug,patch_sim`, `runs=3`):
- baseline: `c01-e015-control-postrevert-r3` -> `9.3974 TPS`
- candidate: `c01-e015-rdna4-y64w4-r3` -> `9.6080 TPS`
- delta: `+0.2107 TPS` (`+2.24%`)
- bootstrap CI: `[+0.1855,+0.2368]` TPS
- verdict: positive

Trace validation (`c01-poste013-r1-resources` -> `c01-e015-rdna4-y64w4-trace-r1`):
- trace TPS: `6.61 -> 6.69`
- `CUDA_NODE op=MUL_MAT kind=forward`: `15498.053 -> 14984.576 ms` (`-513.477 ms`)
- `MMQ`: `10887.326 -> 10381.647 ms` (`-505.679 ms`)
- target bucket `MMQ type=11 ncols_max=192`: `9949.928 -> 9551.391 ms` (`-398.537 ms`)
- Q3 resource line:
  - baseline: `mmq_y=128`, `shared_pct=88.09`, `occupancy_pct=12.50`, `waves_per_sm=8.00`
  - candidate: `mmq_y=64`, `shared_pct=54.49`, `occupancy_pct=6.25`, `waves_per_sm=4.00`

Decision:
- `keep`
- reason: paired r3 positive, bootstrap positive, and target hotspot positive.

Residual risk:
- This is RDNA4-wide MMQ policy, not Q3-only. The active C01 lane improved Q3_K and Q4_K MMQ buckets, but broader RDNA4 MMQ-heavy lanes should be watched.

## C01 experiment E016: post-y64/w4 force-x follow-up

Idea:
- Re-test force-x after E015 because the old force-x sweep used the previous `mmq_y=128/nwarps=8` geometry.

No-trace r1 results:
- E015 default reference: `mmq_x_best=96`, r3 `9.6080 TPS`
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=64`: `9.02 TPS`
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=80`: `8.20 TPS`
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=112`: `9.06 TPS`
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=128`: `8.77 TPS`

Decision:
- `reject`
- reason: all valid force-x points are below the E015 default selector.
- keep: default `mmq_x=96` on the active bucket.

## C01 experiment E017: Q3_K theory gate + k-pair8 probe

New analytic gate:
- script: `scripts/research/c01_mmq_q3_theory_gate.py`
- command:

```bash
python scripts/research/c01_mmq_q3_theory_gate.py build_logs/agent-workload/c01-e015-rdna4-y64w4-trace-r1.server.log --ncols 192
```

Gate output for active bucket:
- current: `mmq_x=96`, `mmq_y=64`, shared `35712` bytes, x tile count `2`
- Q3 x tile shared: `21504` bytes
- Q8 y tile shared: `13824` bytes
- misc shared: `384` bytes

Theory decisions:
- Q3 half-scale packing at `x96`: projected `33664` bytes, still above `32 KiB`, so it cannot unlock a second block/SM. Do not test first.
- Q3 half-scale packing + `x80`: projected `31360` bytes, but x tile count rises `2 -> 3`; too risky before cheaper probes.
- k-pair8: same shared and tile count, halves outer k-loop/dB loads. Proceeded to r1 test.

Runtime gate:
- candidate: `c01-e017-rdna4-q3-kpair8-r1`
- aggregate: `9.59 TPS`
- E015 reference: `9.6080 TPS`

Decision:
- `reject`
- reason: theory-positive but too small in practice; below E015 reference at r1.
- code reverted and `llama-server` rebuilt.

## C01 prompt/prefill focus recalibration trace (2026-05-14)

Reason:
- The focus question was revisited using the same C01 comparison bench rather than GUI autotune.
- The correct framing for this lane is prompt/prefill first: prompt eval dominates wall time, and the active `ncols=192` MMQ bucket is a prompt/prefill shape.

Command:

```bash
python scripts/agent_workload_bench.py --label focus-c01-current-hotspots-r1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full
```

Artifacts:
- `build_logs/agent-workload/focus-c01-current-hotspots-r1.server.log`
- `build_logs/agent-workload/focus-c01-current-hotspots-r1.csv`
- `build_logs/agent-workload/focus-c01-current-hotspots-r1-analysis.md`

Wall timing:
- aggregate completion TPS: `6.7990`
- prompt eval: `26826.35 ms / 14773 tokens = 550.69 tok/s`
- decode eval: `8399.61 ms / 240 tokens = 28.57 tok/s`
- prompt share of llama timing: `76.15%`

Fresh hotspot ranking:
- `CUDA_NODE` total: `22127.070 ms`
- `CUDA_NODE op=MUL_MAT kind=forward`: `14412.924 ms` (`65.14%`)
- `MUL_MAT src0=q3_K type=f32`: `11615.341 ms` (`52.49%`)
- `MMQ`: `9846.201 ms` (`44.50%`)
- target bucket `MMQ type=11 ncols_max=192`: `9048.863 ms` (`40.89%`)
- `GATED_DELTA_NET forward`: `1467.855 ms` (`6.63%`)
- `RMS_NORM fused`: `1063.827 ms` (`4.81%`)
- `ADD forward`: `824.194 ms` (`3.72%`)
- `FLASH_ATTN_EXT forward`: `615.449 ms` (`2.78%`)

Phase split:
- sync-only `CUDA_NODE` prompt phase: `17000.705 ms` (`76.83%`), count `83616`
- sync-only `CUDA_NODE` decode phase: `4760.101 ms` (`21.51%`), count `26848`
- outside/reserve: `366.264 ms` (`1.65%`), count `1842`

Prompt-phase target:
- `CUDA_NODE op=MUL_MAT kind=forward`: `11282.686 ms` (`66.37%` of prompt CUDA_NODE)
- `MUL_MAT src0=q3_K type=f32`: `9066.495 ms` (`53.33%`)
- `MMQ`: `7974.079 ms` (`46.90%`)
- `MMQ type=11 ncols_max=192`: `7490.845 ms` (`44.06%`)
- next center: `GATED_DELTA_NET forward = 1174.486 ms` (`6.91%`)

Mandatory gates:
- shape presence gate for qtype `11`, `ncols_max=192`: PASS (`26524` hits)
- cold/steady split: steady `mul_mat_q_direct|q3_K = 11408.481 ms` (`78.28%` of steady `MUL_MAT forward`)
- q3 component proxy: steady `compute_core_q3 = 11408.481 ms` (`83.91%`), `fallback_cublas = 2007.167 ms` (`14.76%`), `dequant_load_vec_q3 = 180.025 ms` (`1.32%`)

Decision:
- keep C01 as the primary focus, with prompt/prefill as the main optimization target.
- next work should target Q3_K MMQ compute/load internals on `type=11, ncols_max=192, mmq_x=96, mmq_y=64`.
- do not move the main focus to GUI autotune, ngram speculative, or MMVQ based on current C01 evidence.

## C01 experiment E018: Q3_K prefill scale preload

Idea:
- In the RDNA4 Q3_K WMMA path, `x_df[i*stride + k0/4]` depends on row and k-fragment but not on `j0`.
- Candidate cached these scale values in registers once per k-fragment before the `j0` loop, reducing repeated LDS loads in theory.

Analytic screen:
- Active geometry: `mmq_x=96`, `mmq_y=64`, `tile_C::J=16`, `tile_C::ne=8`, `ntx=1`.
- `j0` loop iterations: `96 / 16 = 6`.
- Per thread per k-fragment, x-scale LDS loads could drop from `48` to `8`, at the cost of about `8` extra float registers.

Measured screen:
- non-trace r1: `c01-e018-q3-scale-preload-r1` -> `9.6317 TPS`.
- trace r1 compare: `focus-c01-current-hotspots-r1` -> `c01-e018-q3-scale-preload-trace-r1`.
- `CUDA_NODE`: `22127.070 -> 22498.676 ms` (`+371.606 ms`).
- `CUDA_NODE op=MUL_MAT kind=forward`: `14412.924 -> 14562.136 ms` (`+149.212 ms`).
- `MMQ`: `9846.201 -> 9905.244 ms` (`+59.043 ms`).
- target bucket `MMQ type=11 ncols_max=192`: `9048.863 -> 9103.787 ms` (`+54.924 ms`).

Decision:
- `reject`
- reason: target hotspot regressed despite slight non-trace r1 TPS noise.
- code state: runtime code reverted and `llama-server` rebuilt.

Next C01 direction:
- avoid adding per-thread register arrays unless the projected limiting-term win is larger.
- prefer candidates that reduce shared footprint/tile count, remove work from `load_tiles_q3_K`, or alter memory layout without increasing inner-loop register pressure.

## C01 experiment E019: Q3_K load_tiles scale fusion

Idea:
- Fuse Q3_K scale unpack/store into the first `load_tiles_q3_K` quant-bit load pass for MMA/WMMA paths.
- Lanes `kqsx=0..3` already participate in the row load and can cover the four scale groups, so the separate scale pass looked removable.

Analytic screen:
- Active geometry: `mmq_x=96`, `mmq_y=64`, `nwarps=4`, `threads_per_row=16`.
- The removed pass was one extra traversal over `64` Q3_K rows with `4` scale lanes per row.
- Expected gain was intentionally small: `0.2-1.0%` if `load_tiles_q3_K` loop overhead was visible.

Measured screen:
- candidate: `c01-e019-q3-loadtiles-fuse-scales-r1`
- aggregate completion TPS: `8.2082`
- E015 reference: `9.6080 TPS`
- prompt eval TPS mean: `694.15`
- decode eval TPS mean: `30.465`

Decision:
- `reject`
- reason: gross regression in cheap screen, far outside the expected noise band.
- trace: skipped because the non-trace gate failed.
- code state: runtime code reverted and `llama-server` rebuilt.

Next C01 direction:
- do not fuse more scale/min unpack into the first Q3_K quant-load pass without a stronger resource argument.
- keep looking for changes that reduce shared footprint/tile count or improve scheduling with minimal added per-lane work.

## C01 experiment E020: Q3_K half-scale compact x96

Idea:
- Store precomputed Q3_K scales in shared memory as `half` and use a Q3-only compact MMA stride (`84 -> 72` ints).
- The goal was to reduce dynamic shared below `32 KiB` while keeping `mmq_x=96` and the same `2` x tiles for `ncols=192`.

Analytic screen:
- E015 resource baseline: shared `35712`, regs `160`, `max_blocks_per_sm=1`, waves `4.00`.
- Projected compact x96 shared: `32640`, below `32768`, with unchanged tile count.
- Theory gate artifact: `build_logs/agent-workload/c01-e020-q3-halfscale-compact-theory.md`.

Measured result:
- Runtime r3: `c01-e015-rdna4-y64w4-r3` -> `c01-e020-q3-halfscale-compact-r3`
- Aggregate TPS: `9.6080 -> 9.6017` (`-0.07%`)
- Decision stats: bootstrap 95% CI `[-0.0380, +0.0239]` TPS, verdict `inconclusive`.
- Valid target trace: `c01-e020-q3-halfscale-compact-trace-r1b`.
- Resource telemetry: shared `35712 -> 32640`, `max_blocks_per_sm=1 -> 2`, occupancy `6.25% -> 12.50%`, waves `4.00 -> 8.00`, regs `160 -> 158`.
- Target bucket vs E015 trace: `MMQ type=11 ncols_max=192` `9551.391 -> 9451.261 ms` (`-100.130 ms`).
- Total `MMQ`: `10381.647 -> 10300.173 ms` (`-81.474 ms`).

Decision:
- `research-positive / no default`
- reason: the MMQ target improved and the shared-memory theory was confirmed, but aggregate runtime did not beat the current best.
- code state: runtime code reverted and `llama-server` rebuilt.

Next C01 direction:
- compact Q3 shared layout alone is not enough.
- future variants should only revisit this if paired with a scheduling/pre-sync fix or a layout that avoids the non-MMQ slowdowns seen in trace.

## C01 experiment E021: dense Q3 MMQ staging

Idea:
- Temporarily enable the existing RDNA4 MMQ staging loop for dense `Q3_K`.
- The gate was env-only in the prototype (`GGML_RDNA4_DENSE_Q3_MMQ_STAGING=1`) and did not change default behavior.

Analytic screen:
- Current E015 split: Q3 x tile `21504` bytes, Q8 y tile `13824` bytes, misc `384` bytes.
- Staged projection: `2 * 21504 + 13824 + 384 + 4 = 57220` bytes.
- It fits `64 KiB`, preserves `mmq_x=96`, and keeps the `2` x-tile count for `ncols=192`.
- Risk: still above `32 KiB`, so no two-block occupancy; any gain must come from better load scheduling.

Measured screen:
- reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`.
- candidate: `c01-e021-dense-q3-staging-r1 = 8.6216 TPS`.
- delta: `-10.27%`.

Activation trace:
- artifact: `build_logs/agent-workload/c01-e021-dense-q3-staging-activation-r1.server.log`.
- confirmed `rdna4_staging_req=1` and `rdna4_staging_eff=1` on `type=11,ncols_max=192`.
- per-call `MMQ type=11 ncols_max=192` timing worsened about `25.9%` (`0.447 ms -> 0.563 ms`).
- `MUL_MAT ne=(17408,192,1,1)` average worsened `0.671 ms -> 0.842 ms`.

Decision:
- `reject`
- reason: the active Q3 bucket slowed decisively when staging was actually enabled.
- code state: runtime code reverted and `llama-server` rebuilt.
- keep the existing MoE-only staging gate unchanged.

## C01 experiment E023: RDNA4 F32 cuBLAS GemmEx route

Idea:
- The same C01 trace shows a secondary small-GEMM center: `MUL_MAT f32 ne=(48,192)` from Qwen SSM alpha/beta projections.
- It uses `cublas_backend`; a cheap RDNA4-only probe checked whether `cublasGemmEx` picks a better rocBLAS path than `cublasSgemm`.

Analytic screen:
- One SSM alpha/beta GEMM is about `2 * 48 * 192 * 5120 = 94.4M` FLOPs.
- Baseline target share was about `1.25 s` in the focus trace, so even a `10%` local win would only be roughly `0.6%` wall-time.
- Gate: keep only if aggregate TPS and the `MUL_MAT f32 ne=(48,192)` target both improve.

Measured screen:
- reference: `c01-e015-rdna4-y64w4-r3 = 9.6080 TPS`.
- candidate: `c01-e023-rdna4-f32-gemmex-r1 = 9.42 TPS`.
- target trace: baseline avg `0.1712 ms`, candidate avg `0.1850 ms` (`+8.1%` slower).

Decision:
- `reject`
- reason: both runtime and target timing regressed.
- code state: env-gated `GemmEx` branch reverted and `llama-server` rebuilt.

Next C01 direction:
- do not spend more time on the generic F32 cuBLAS route unless a later trace shows a larger F32 share or a concrete rocBLAS shape-specific knob.
- return to Q3_K MMQ/prefill or a different non-Q3 center only with a larger theoretical ceiling than E023.

## C01 experiment E027: force-x sub-32KiB probe

Fresh return trace:
- artifact: `build_logs/agent-workload/c01-return-20260516-r1-resources.server.log`
- trace-overhead TPS: `6.4253`
- shape gate: `type=11,ncols_max=192` PASS (`26524` hits)
- active geometry: `mmq_x=96`, `mmq_y=64`, shared `35712`, regs `160`, `max_blocks_per_sm=1`, waves `4.00`
- steady `MUL_MAT forward`: `mul_mat_q_direct|q3_K = 12171.789 ms` (`78.31%`)

Idea:
- Try a valid sub-32KiB force-x point to get more active blocks/SM without the E020 half-scale layout.
- `x72` looked attractive analytically because it would be near `32256` bytes, but RDNA4 WMMA requires x granularity `16`.
- Trace confirmed `x72` is rejected by selector validation and falls back to `mmq_x=96`.
- `x64` is valid and below `32 KiB`, but it uses `3` x tiles for `ncols=192` instead of `2`.

Measured screen:
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=72`: `9.53 TPS`, but invalid/no-op; trace still shows `mmq_x_best=96`.
- `GGML_MMQ_RDNA4_Q3_FORCE_MMQ_X=64`: `8.90 TPS` vs current baseline `9.4111 TPS` (`-5.43%`).

Decision:
- `reject`
- reason: the only valid below-32KiB force-x point tested (`x64`) is decisively slower, and `x72` is not a valid WMMA tile geometry.
- code state: no code changes.

Next C01 direction:
- simple force-x is closed unless tile geometry changes.
- future Q3_K work needs a real shared layout or scheduling change; otherwise scout the F32 batched/cuBLAS subcenter with a more specific idea than E023 `GemmEx`.

## C01 experiment E028: ngram-mod 24/48/64 confirmation

Context:
- E026 found `ngram-mod 24/48/64` promising but inconclusive on the current C01 lane.
- Before this confirmation, quantize/MMF code probes were tried and reverted:
  - quant MMQ block size `64/32/256` did not beat default `128`.
  - Q8_1 quant `__float2int_rn` did not beat `roundf`.
  - RDNA4 F32 MMF threshold `32/64` did not beat cuBLAS.

Measured confirmation:
- control: `c01-e028-clean-control-r3 = 9.4890 TPS`.
- candidate: `c01-e028-ngram244864-r6 = 10.3689 TPS`.
- delta: `+0.8799 TPS` (`+9.27%`).
- decision stats: bootstrap 95% CI `[+0.5192,+1.3106]` TPS, verdict `positive`.
- prompt eval: `855.5400 -> 851.3758 TPS` (`0.9951x`).
- decode eval: `30.1433 -> 45.1508 TPS` (`1.4979x`).
- spec stats: local acceptance `0.581422`, coverage `0.040580`, effective acceptance `0.023594`.

Decision:
- `keep as opt-in / no default`
- reason: the C01 speedup is real in this repeated-task sample, but it comes from sparse speculative coverage rather than a kernel/prefill fix.
- command:

```bash
python scripts/agent_workload_bench.py --label c01-e028-ngram244864-r6 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 6 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --server-extra "--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64"
```

Next C01 direction:
- if the goal is immediate practical speed, expose/document this as an opt-in repeated/steady preset.
- if the goal is default cold-first speed, continue kernel/runtime work; `ngram-mod` should not replace the C01 default while effective acceptance is workload-dependent.

## C01 experiment E029: cold-first recheck for ngram-mod 24/48/64

Context:
- After E028, the same ngram profile was still treated as repeated/steady opt-in.
- A strict cold-first gate was required because the first `r1` pair was inconclusive (`+0.10%`, CI crossing zero).

Measured cold gate:
- `r1` probe:
	- control: `c01-e029-cold-control-r1 = 9.4381 TPS`
	- candidate: `c01-e029-cold-ngram244864-r1 = 9.4476 TPS`
	- decision stats: `inconclusive`.
- powered `r3` pair:
	- control: `c01-e029-cold-control-r3 = 9.3031 TPS`
	- candidate: `c01-e029-cold-ngram244864-r3 = 10.0948 TPS`
	- delta: `+0.7918 TPS` (`+8.51%`).
	- bootstrap 95% CI: `[+0.2943,+1.3488]` TPS.
	- verdict: `positive`.
- extended `r6` pair:
	- control: `c01-e029-cold-control-r6 = 9.2468 TPS`
	- candidate: `c01-e029-cold-ngram244864-r6 = 10.2456 TPS`
	- delta: `+0.9988 TPS` (`+10.80%`).
	- bootstrap 95% CI: `[+0.6980,+1.3441]` TPS.
	- verdict: `positive`.
- phase clue:
	- prompt eval remains neutral/slightly lower (`839.27 -> 835.29 tok/s` mean),
	- decode eval improves strongly (`29.69 -> 42.97 tok/s` mean), with higher variance.
- speculative log stats (`spec_log_stats.py`):
	- `gen_drafts=4`, `acc_drafts=4`,
	- `gen_tokens=246`, `acc_tokens=218`,
	- `token_accept_ratio=0.8862`.

Decision:
- `keep as opt-in accelerated profile`
- reason: cold-first gate is now positive on powered A/B, but improvement remains speculative-driven and variance-sensitive.
- default policy: keep `spec=none` as conservative default; allow `ngram-mod 24/48/64` as documented opt-in accelerator for this lane.

Next C01 direction:
- for kernel-default claims, continue no-spec (`spec=none`) runtime/kernel work and compare only against cold clean baseline.
- for practical speed mode, maintain `ngram-mod 24/48/64` as an explicit opt-in profile and continue tracking coverage/acceptance stability.

## C01 experiment E030: cold/warm metric split

Context:
- Measurement policy changed to track true cold run #1 and warm/repeated rows separately.
- E029 `r6` headline was an all-runs aggregate, so it mixed the first cold request with later repeated requests.

Bench infrastructure update:
- `scripts/agent_workload_bench.py` now writes `run` to per-run CSV.
- `--stats-ignore-first-run` now prints both:
  - cold-only stats for `run == 1`,
  - warm-only stats for `run > 1`.

Fresh same-session split:
- clean `c01-e030-clean-split-r2`:
  - all: `9.4569 TPS`,
  - cold run #1: `9.47 TPS`,
  - warm excluding run #1: `9.45 TPS`.
- `ngram-mod 24/48/64` `c01-e030-ngram244864-split-r2`:
  - all: `10.0476 TPS`,
  - cold run #1: `9.46 TPS`,
  - warm excluding run #1: `10.72 TPS`.

Spec stats:
- local acceptance `0.868852`,
- coverage `0.011236`,
- effective acceptance `0.009762`.

Decision:
- `ngram-mod 24/48/64` remains a practical warm/session opt-in accelerator.
- It is not a cold-first default/kernel win on the corrected metric because cold run #1 is neutral.
- New C01 default claims must report `run == 1`; speculative/session claims must report `run > 1`.

Adjacent checks:
- server warmup enabled (`--no-no-warmup`) did not improve split metrics.
- `ubatch=224` and `ubatch=160` regressed, so the current `ubatch=192` remains the local lane setting.

Next C01 direction:
- continue no-spec Q3_K MMQ prefill work for cold-first speed.
- do not widen into GUI autotune-style parameter sweeps unless a fresh trace shows the bottleneck moved.

## C01 experiment E031: Q4_K force-x sub-32KiB probe

Context:
- Fresh E030 resource trace showed a small secondary `Q4_K` MMQ center:
  `mul_mat_q_direct|q4_K = 964.363 ms` (`6.17%` of steady `MUL_MAT forward`).
- Resource telemetry for `type=12,ncols_max=192`:
  `mmq_x=96`, `mmq_y=64`, shared `33664`, regs `200`, `max_blocks_per_sm=1`.

Analytic screen:
- `x80` should reduce shared memory below the 32 KiB boundary and may allow `2` blocks/SM.
- It also changes `ncols=192` from `2` x tiles to `3`, so the expected wall ceiling is small and risk is high.

Measured screen:
- same-build control: `c01-e031-q4force-control-r1 = 9.4522 TPS`.
- candidate: temporary env-gated `GGML_MMQ_RDNA4_Q4_FORCE_MMQ_X=80`,
  `c01-e031-q4force-x80-r1 = 9.4026 TPS`.
- decision stats: bootstrap 95% CI `[-0.0591,-0.0400]`, verdict `negative`.
- prompt eval regressed `853.885 -> 846.515 tok/s`; decode was unchanged.

Decision:
- `reject`
- reason: the x-tile count penalty dominates any possible residency benefit.
- code state: temporary runtime code reverted and `llama-server` rebuilt.

Next C01 direction:
- do not continue Q4 force-x sub-32KiB on this lane.
- return to Q3_K MMQ internals or scout a different center only if trace share and modelled ceiling are larger.

## C01 experiment E032: F32 MMF wide SSM probe

Context:
- E030 shape split showed a large secondary F32 cuBLAS SSM shape:
  `src0=(5120,48,1,1)`, `src1=(5120,192,1,1)`, `dst=(48,192,1,1)`,
  steady `1294.385 ms`.
- E023 had already rejected `cublasGemmEx`, so this probe tried a different route:
  no-id tiled `MMF` for `ncols_dst > 16`, guarded by `GGML_CUDA_RDNA4_F32_MMF_WIDE`.

Measured screen:
- trace control: `c01-e032-mmfwide-control-r1 = 6.3561 TPS`.
- trace candidate: `c01-e032-mmfwide-candidate-r1 = 6.4398 TPS`.
- decision stats were positive on trace wall time, but the route activation gate failed.

Activation/root-cause check:
- target SSM rows stayed on `cublas_backend`; only existing tiny `ncols=2` rows used `mul_mat_vec_f_direct`.
- RDNA4 `MMF` is not a cheap F32 target in this code path:
  `AMD_WMMA_AVAILABLE` `mul_mat_f` supports `half2`/`nv_bfloat162`, while F32 `should_use_mmf`
  depends on `amd_mfma_available`.
- SSM also has `src0_ne[1]=48`, which does not satisfy the current `MMF_ROWS_PER_BLOCK=32` divisibility gate.

Decision:
- `reject / no-activation`
- reason: the intended F32 SSM route cannot activate on current RDNA4 MMF without a real new F32 kernel design.
- code state: prototype reverted and `llama-server` rebuilt.

Next C01 direction:
- do not revisit F32 SSM through cheap `MMF` routing.
- continue Q3_K MMQ internals, or move only to centers with a concrete supported route and a larger modeled ceiling.

## C01 experiment E038: H06 Q/K rotation graph fusion screen

Context:
- After E037 gate, a minimal env-gated graph prototype was tested to fuse Q and K rotation into one path.

Prototype (temporary, reverted):
- env flag: `GGML_EXPERIMENTAL_QK_ROT_FUSION=1`
- path: concatenate `q_cur` and `k_cur` on head axis, apply one `ggml_mul_mat_aux`, split back with `ggml_view_4d`.

Measured screen (`runs=1`, same C01 lane):
- control: `c01-e038-h06-control-r1 = 11.2031 TPS`
- candidate: `c01-e038-h06-fused-r1 = 11.1688 TPS`
- delta: `-0.31%`
- prompt eval mean: `835.05 -> 832.29 tok/s`
- decode eval mean: `29.515 -> 29.44 tok/s`

Decision:
- `reject`
- reason: negative runtime screen on target lane.
- code state: reverted and rebuilt.

Next H06 direction:
- do not continue this graph-level concat/split variant.
- if H06 is revisited, target kernel-level QKV/RoPE integration rather than graph-level concat overhead.

## C01 experiment E036: E020 pre-sync companion + H06 gate snapshot

Context:
- E020 (`half-scale compact x96`) improved the target MMQ q3 bucket, but end-to-end runtime remained neutral.
- This follow-up compares E015 vs E020 trace timing fields inside the same bucket to explain where the gain was absorbed.

Trace pair and filter:
- baseline: `build_logs/agent-workload/c01-e015-rdna4-y64w4-trace-r1.server.log`
- candidate: `build_logs/agent-workload/c01-e020-q3-halfscale-compact-trace-r1b.server.log`
- rows: `mul_mat_q_case: timing type=11 ... ncols_max=192`

Aggregated bucket metrics (`count=26524` in both runs):
- E015: `pre_sync=1282.705`, `enqueue=135.698`, `sync=9446.242`, `total=9579.561` ms.
- E020: `pre_sync=1476.664`, `enqueue=153.422`, `sync=9326.433`, `total=9479.791` ms.
- delta (`E020-E015`):
	- `pre_sync: +193.959 ms`
	- `enqueue: +17.724 ms`
	- `sync: -119.809 ms`
	- `total: -99.770 ms`

Interpretation:
- The compact layout helped kernel-side sync, but a larger pre-sync tax consumed most of that win.
- This explains why E020 stayed runtime-neutral despite target hotspot improvement.

H06 gate snapshot (same-session trace-based ceiling):
- source: `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`
- total `GGML_TRACE_CUDA_NODE_TIMING`: `3303.800 ms`
- attention/QKV/RoPE-related node-name slice (`rope|rot|attn|q_|k_|v_|wq|wk|wv|query|key|value`):
	- `575.093 ms` (`17.41%` share)
- rough ceiling on current lane:
	- 10% gain inside this slice -> `~+1.74%` CUDA_NODE
	- 20% gain inside this slice -> `~+3.48%` CUDA_NODE

Decision:
- Keep E020 reverted as default.
- Promote H06 from backlog to active next implementation gate (largest remaining plausible multi-percent center).

## C01 experiment E052: H21 hipBLASLt Stream-K route gate

Context:
- A no-code environment gate was run after RDNA4 docs review to test whether hipBLASLt/Stream-K can improve the active C01 lane.
- Lane contract (unchanged): `tasks=triage_diff,review_bug`, `ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, `no-reuse`, thinking on, `runs=1`.

Measured screen (`build-rocm-vec`, same lane):
- control: `c01-h21-hipblaslt-control-r1 = 9.3660 TPS`
- `ROCBLAS_USE_HIPBLASLT=1`: `c01-h21-hipblaslt-on-r1 = 9.2753 TPS` (`-0.97%`)
- `ROCBLAS_USE_HIPBLASLT=1 TENSILE_SOLUTION_SELECTION_METHOD=2`:
	`c01-h21-hipblaslt-sel2-r1 = 9.2692 TPS` (`-1.03%`)
- `... + TENSILE_STREAMK_FIXED_GRID=64`:
	`c01-h21-hipblaslt-sel2-grid64-r1 = 9.2741 TPS` (`-0.98%`)
- `... + TENSILE_STREAMK_MAX_CUS=32`:
	`c01-h21-hipblaslt-sel2-maxcus32-r1 = 9.2515 TPS` (`-1.22%`)

Activation check:
- diagnostic run with `HIPBLASLT_LOG_LEVEL=4 HIPBLASLT_LOG_MASK=127`:
	`c01-h21-hipblaslt-sel2-log-r1 = 9.2672 TPS` (`-1.05%`).
- server log shows hipBLASLt initialization (`HIPBLASLT_TENSILE_LIBPATH ...`), but no route-level per-GEMM evidence appeared in this lane log.

Decision:
- `reject`
- reason: all tested hipBLASLt/Stream-K env combinations are consistently below fresh control on the target C01 lane.
- code state: no code changes.

Artifacts:
- `build_logs/agent-workload/c01-h21-hipblaslt-control-r1.*`
- `build_logs/agent-workload/c01-h21-hipblaslt-on-r1.*`
- `build_logs/agent-workload/c01-h21-hipblaslt-sel2-r1.*`
- `build_logs/agent-workload/c01-h21-hipblaslt-sel2-grid64-r1.*`
- `build_logs/agent-workload/c01-h21-hipblaslt-sel2-maxcus32-r1.*`
- `build_logs/agent-workload/c01-h21-hipblaslt-sel2-log-r1.*`
