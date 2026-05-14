# C01 Resume Playbook (Pause/Resume)

## Scope

This file is the fast handoff anchor when C01 work is paused and later resumed.

Primary center:
- `CUDA_NODE op=MUL_MAT kind=forward`

## Active Goals (next return)

1. Keep C01 decisions causal and reproducible (not trial-and-error).
2. Preserve dual-metric policy:
   - runtime (`aggregate TPS`),
   - hotspot-time (`MUL_MAT forward` and MMQ target bucket).
3. Use new diagnostics before any new kernel tweak:
   - shape presence gate,
   - cold-vs-steady split,
   - statistical verdict,
   - MMQ resource telemetry.

## Current Known State

1. Best-known stream-k knob point on current C01 lane:
   - `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=144` (keep-as-knob),
   - default remains unchanged until extra confirmation.
2. MMQ resource trace is now available with:
   - `GGML_TRACE_MMQ_RESOURCES=1`
3. Latest resume checkpoint artifacts (2026-05-13):
   - `build_logs/agent-workload/c01-resume-r1-resources.server.log`
   - `build_logs/agent-workload/c01-resume-r2-control.server.log`
4. Resume gate summary (latest run):
   - shape gate for q3/MMQ `ncols_max=192`: PASS (`count=26524`),
   - steady split: `mul_mat_q_direct|q3_K` dominates (`78.20%`),
   - q3 coarse split steady share: `compute_core_q3=84.25%`, `fallback_cublas=14.38%`.
5. Previous C01 resource trace remains the closest comparison point:
   - `build_logs/agent-workload/e013-c01-two-tasks-trace-r1-resources.server.log`
6. e013 -> latest resume route compare:
   - no route flip in main target buckets (`MUL_MAT forward` avg ratio `0.992`,
     MMQ q3 `ncols_max=192` avg ratio `1.008`),
   - runtime is lower by about `-5.4%` vs e013 (`6.6897 -> 6.3309 TPS`).
7. r1 -> r2 control rerun verdict:
   - runtime delta is borderline/inconclusive (`+0.14%`),
   - route composition and hotspot timings are effectively stable (`avg ratio ~0.999`).
8. E013 MMVQ Q3_K fast path kept:
   - `GGML_TYPE_Q3_K` on RDNA4 `ncols_dst=1` now uses `nwarps=2`,
   - paired non-trace result: `9.1629 -> 9.3847 TPS` (`+2.42%`),
   - trace hotspot improved: `MMQ type=11 ncols_max=192 -96.409 ms`.
9. E014 post-E013 selector/resource screen:
   - fresh trace artifact: `build_logs/agent-workload/c01-poste013-r1-resources.server.log`,
   - steady `mul_mat_q_direct|q3_K`: `12325.249 ms` (`78.93%`),
   - force-x, stream-k retest, `mmq_y=64`, and RDNA4 `launch_bounds(..., 1)` produced no target-positive keep candidate,
   - all temporary code probes were reverted and `llama-server` was rebuilt.
10. E015 RDNA4 MMQ y64/w4 kept:
   - code: RDNA4 MMQ uses `mmq_y=64` and `nwarps=4`,
   - paired r3: `9.3974 -> 9.6080 TPS` (`+2.24%`),
   - bootstrap CI: `[+0.1855,+0.2368]` TPS,
   - trace target: `MMQ type=11 ncols_max=192` improved by `-398.537 ms`.
11. E016 post-y64/w4 force-x follow-up:
   - force `mmq_x=64/80/112/128` all regressed against E015 default,
   - keep selected `mmq_x=96` for the active bucket.

## Lane Contract (resume baseline)

- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- tasks: `review_bug,patch_sim`
- ctx: `12288`
- batch: `6144`
- ubatch: `192`
- kv: `q4_0/q4_0`
- no-reuse: on
- thinking: on (default benchmark comparability)
- runs: `1` for iteration, `3` only for final borderline confirmation.

## Mandatory Pre-Edit Gate (on resume)

1. Run shape presence gate on the exact target trace:

```bash
python scripts/research/c01_shape_presence_gate.py <trace.server.log> --qtype 11 --ncols 192 --min-count 1 --strict
```

2. Run cold-vs-steady split to avoid spike contamination:

```bash
python scripts/research/cold_steady_trace_split.py <trace.server.log> --op MUL_MAT --kind forward --steady-max-ms 5 --top 12
```

3. Run statistical verdict for baseline vs candidate CSV:

```bash
python scripts/research/decision_stats.py --baseline <base.csv> --candidate <cand.csv> --bootstrap 3000 --borderline-pct 1.0
```

## First Command To Resume C01

```bash
GGML_TRACE_MMQ_RESOURCES=1 GGML_TRACE_MMQ_TIMING=1 GGML_TRACE_MMQ_TIMING_SYNC=1 python scripts/agent_workload_bench.py --label c01-resume-r1-resources --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug,patch_sim --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full
```

## Decision Rule (unchanged)

- Promote default only when both are stable:
  - runtime positive (or clearly above noise with CI support),
  - hotspot-positive in target expensive place.
- Keep env knob if signal is promising but not yet confirmed.

## Immediate Next Step (after latest resume gate)

1. `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=144` has now failed `runs=3` confirmation
   (`c01-next-control-r3` vs `c01-next-streamk144-r3`, `-1.37%`, negative verdict),
   so do not use it as acceleration direction.
2. `GGML_CUDA_FORCE_MMQ_RUNTIME=1` is also rejected for this lane:
   runtime `-0.72%` and hotspot-time regression on `MUL_MAT forward` / MMQ q3 bucket.
3. E013 closed the MMVQ Q3_K side center with a kept narrow policy change.
4. E015 kept the first direct C01 MMQ policy win after E013.
5. Continue C01 with a fresh post-E015 control. Next route-local work should inspect
   Q3_K MMQ compute/load internals beyond tile size:
   `load_tiles_q3_K`, scale/min unpack, accumulator/write-back pressure.
   Do not continue force-x sweeps unless a later change alters shared layout or selector math.
6. If a candidate is hotspot-positive but runtime-neutral, keep it as research-positive
   and confirm again with a paired control rerun.
7. If a candidate is runtime-positive, proceed to `runs=3` confirmation before any
   keep/default decision.
