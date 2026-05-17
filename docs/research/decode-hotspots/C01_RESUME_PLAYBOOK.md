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
12. E017 added a Q3_K MMQ theory gate:
   - script: `scripts/research/c01_mmq_q3_theory_gate.py`,
   - half-scale at `x96` rejected analytically because it stays above `32 KiB` shared,
   - k-pair8 tested after passing the gate but rejected at r1 (`9.59 TPS` vs E015 `9.6080`),
   - temporary code probe was reverted and `llama-server` rebuilt.
13. E018/E019 Q3_K load/scale probes were rejected:
   - scale preload regressed target `MMQ type=11 ncols_max=192` (`9048.863 -> 9103.787 ms`),
   - load_tiles scale fusion regressed runtime (`9.6080 -> 8.2082 TPS`),
   - both runtime probes were reverted and `llama-server` rebuilt.
14. E020 Q3_K compact half-scale x96 was research-positive but not kept:
   - shared `35712 -> 32640`, `max_blocks_per_sm=1 -> 2`, target `9551.391 -> 9451.261 ms`,
   - runtime r3 was neutral (`9.6080 -> 9.6017 TPS`), so code was reverted.
15. E021 dense Q3 MMQ staging was rejected:
   - activation confirmed, but runtime fell to `8.6216 TPS` and target average worsened about `25.9%`.
16. E022/E024 C05/GDN chunk probes were rejected:
   - `GGML_GDN_FAST_EXP=1`, `GGML_GDN_CHUNK_SIZE=192`, and `GGML_GDN_CHUNK_SIZE=128`
     all stayed below the E015 reference.
17. E023 RDNA4 F32 cuBLAS `GemmEx` route was rejected:
   - runtime `9.6080 -> 9.42 TPS`,
   - target `MUL_MAT f32 ne=(48,192)` avg `0.1712 -> 0.1850 ms`,
   - code reverted and `llama-server` rebuilt.
18. E025 current-environment retest:
   - E015 default retested at `9.4111 TPS` vs old `9.6080 TPS`,
   - no-code retests all stayed below the same-session baseline:
     `streamk144=9.3837`, `force-MMQ=9.3746`, `GDN fast_exp=9.3625`,
     `GDN chunk192=9.3485`, `GDN chunk128=9.3522`.
   - Use `9.4111 TPS` for same-session comparisons unless a fresh baseline recovers.
19. E026 FATTN/ngram scout:
   - FATTN is a low-ceiling target on current C01: `FLASH_ATTN_EXT forward` is only
     `~2.58%` of sync CUDA_NODE time, with WMMA F16 route active for `D=256/q_rows=192`.
   - `ngram-mod 24/48/64` measured `9.7225 TPS` vs `9.4111 TPS`, but the bootstrap
     verdict is inconclusive and effective acceptance is only `0.00675`.
   - `ngram-simple` and `ngram-mod n_match=12` are rejected/neutral for this lane.
   - Do not promote ngram to cold-first default; keep `24/48/64` only as opt-in
     repeated/steady-task candidate.
20. E027 force-x sub-32KiB probe:
   - fresh C01 return trace preserved the target: `type=11,ncols_max=192`, `mmq_x=96`,
     `mmq_y=64`, shared `35712`, waves `4.00`.
   - `x72` is invalid/no-op because RDNA4 WMMA granularity is `16`; trace fell back to `mmq_x=96`.
   - valid `x64` measured `8.90 TPS` vs current baseline `9.4111 TPS`.
   - Do not continue force-x sweeps unless tile geometry changes.
21. E028 C01 ngram-mod confirmation:
   - clean control r3: `c01-e028-clean-control-r3 = 9.4890 TPS`.
   - opt-in candidate r6: `c01-e028-ngram244864-r6 = 10.3689 TPS` with
     `--spec-type ngram-mod --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64`.
   - delta: `+9.27%`, bootstrap 95% CI `[+0.5192,+1.3106]` TPS, verdict `positive`.
   - decode eval improved `30.1433 -> 45.1508 TPS`; prompt eval was neutral/slightly lower.
   - spec stats: local acceptance `0.581422`, coverage `0.040580`, effective acceptance `0.023594`.
   - Keep as opt-in repeated/steady-task preset only; do not make it the cold-first default.
22. E029 cold-first ngram recheck:
    - initial `r1` pair was inconclusive (`9.4381 -> 9.4476 TPS`, `+0.10%`).
    - powered `r3` pair is positive: `9.3031 -> 10.0948 TPS` (`+8.51%`),
       bootstrap CI `[+0.2943,+1.3488]` TPS.
    - extended `r6` pair confirms the signal: `9.2468 -> 10.2456 TPS` (`+10.80%`),
       bootstrap CI `[+0.6980,+1.3441]` TPS.
    - decode eval improved (`29.685 -> 42.973` tok/s mean), prompt eval stayed neutral/slightly lower.
    - keep `ngram-mod 24/48/64` as an explicit opt-in accelerated profile; keep `spec=none`
       as the conservative default because speculative coverage/variance is workload-sensitive.
23. E030 cold/warm metric split:
    - `agent_workload_bench.py` now writes `run` to CSV and prints cold-only run #1 stats
      when `--stats-ignore-first-run` is used.
    - same-session split showed clean `9.4569` all / `9.47` cold / `9.45` warm TPS.
    - `ngram-mod 24/48/64` showed `10.0476` all / `9.46` cold / `10.72` warm TPS.
    - interpretation: ngram is a warm/session opt-in accelerator, not a cold-first default win.
    - E029 aggregate `r6` is still useful, but it should not be described as pure cold-first
      because aggregate-all mixes run #1 and later repeated runs.
24. E031 Q4_K force-x sub-32KiB probe:
    - secondary Q4_K bucket is present (`~6.17%` of steady `MUL_MAT forward`).
    - `type=12,ncols_max=192` resource state is `mmq_x=96`, shared `33664`, regs `200`,
      `max_blocks_per_sm=1`.
    - temporary `GGML_MMQ_RDNA4_Q4_FORCE_MMQ_X=80` measured `9.4026 TPS` vs same-build
      control `9.4522 TPS`; decision stats negative.
    - code reverted and server rebuilt; do not continue Q4 force-x unless tile geometry changes.

## Current Metric Contract

- `Cold-first baseline` for C01 (clean/spec=none): quote `run == 1`, not aggregate-all.
   - latest same-session split: `9.47 TPS` (`c01-e030-clean-split-r2`, run #1).
   - E029 aggregate clean reference remains `9.2468 TPS`, but it is an all-runs aggregate.
   - use cold run #1 for default/kernel/runtime claims.
- `Repeated/steady clean baseline` for C01: quote rows with `run > 1`.
   - latest same-session split: `9.45 TPS` (`c01-e030-clean-split-r2`, excluding run #1).
   - historical repeated/steady clean: `9.4890 TPS` (`c01-e028-clean-control-r3` all-runs aggregate).
   - use for session/speculative opt-in claims.
- `Repeated/steady opt-in reference`: `10.3689 TPS`
   - artifact: `c01-e028-ngram244864-r6`
   - keep as opt-in preset, not cold-first default.

- `Cold opt-in reference`: `10.2456 TPS`
   - artifact: `c01-e029-cold-ngram244864-r6`
   - all-runs aggregate; after E030, use split cold/warm rows before making cold claims.

Rule:
- do not compare a repeated/steady speculative candidate directly against the cold-first baseline;
- every new claim in this lane must say which row set it targets: `run == 1`, `run > 1`,
  or all-runs aggregate.
- E030 showed `ngram-mod 24/48/64` is cold-neutral but warm-positive.

## Lane Contract (resume baseline)

- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- tasks: `triage_diff,review_bug`
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
GGML_TRACE_MMQ_RESOURCES=1 GGML_TRACE_MMQ_TIMING=1 GGML_TRACE_MMQ_TIMING_SYNC=1 python scripts/agent_workload_bench.py --label c01-resume-r1-resources --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids triage_diff,review_bug --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --max-tokens 120 --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --task-fail-timeout 0 --trace-preset kernel-full
```

Note:
- historical references to `review_bug,patch_sim` in older experiment entries remain for reproducibility only.

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
5. Continue C01 from E015 as the kept code path. Historical best is `9.6080 TPS`,
   current cold-first split baseline is `9.47 TPS`, current repeated/steady clean baseline is
   `9.45-9.49 TPS`, and current opt-in ngram confirmation is `10.37-10.72 TPS` for repeated/steady tasks. Q3_K scale-load fusion,
   dense staging, GDN chunking (`128/192`), and F32 `GemmEx` are now closed negative branches.
   FATTN is also low-ceiling on the current C01 lane. `ngram-mod 24/48/64` is now a
   confirmed opt-in C01 plus after E028, but not a cold-first default. Force-x
   sub-32KiB is closed for Q3 after E027 and for Q4 after E031. F32 SSM cheap
   `MMF` routing is closed after E032 because the target stays on `cublas_backend`;
   current RDNA4 `MMF` has no cheap F32 WMMA route and the SSM shape misses the rows-per-block gate.
   Next route-local work should either:
   - find a Q3_K MMQ idea with a larger modeled ceiling than E018/E020, or
   - scout a different center only if its current trace share gives a plausible wall-time payoff.
   Run the Q3 theory gate before any new Q3 kernel probe.
6. If a candidate is hotspot-positive but runtime-neutral, keep it as research-positive
   and confirm again with a paired control rerun.
7. If a candidate is runtime-positive, proceed to `runs=3` confirmation before any
   keep/default decision.
