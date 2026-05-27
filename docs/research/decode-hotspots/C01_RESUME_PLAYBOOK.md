# C01 Resume Playbook (Pause/Resume)

## Scope

This file is the fast handoff anchor when C01 work is paused and later resumed.

Primary center:
- `CUDA_NODE op=MUL_MAT kind=forward`

## Current Pause Checkpoint (2026-05-27)

The active performance branch is paused by user request before starting a
separate public `llama-bench` comparison track. C01 itself remains historical;
the current paused branch is P002 130k dense Qwen3.6 Vulkan/ROCm work.

Active P002 lane:
- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- backend focus: Vulkan; ROCm is paused after D013-D027.
- contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`
- workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`
- cold-first: no reuse, no v2 prime pass, thinking on.

Current accepted P002 baseline:
- D012 `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3`
- `2.0013 TPS`, prompt `1053.1067 tok/s`, decode `42.7233 tok/s`
- opt-in env stack: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`,
   `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256`,
   `GGML_VK_QK_LOW_TILE_SPLIT_K=3`, `GGML_VK_Q3K_QUAD_DEQUANT=1`.

Latest decision before pause:
- D034 closed the fresh `ctx=131072` slow-pocket recheck as diagnostic only.
- Fresh full-server D012 controls can fall to `~0.37 TPS`, but direct
   `llama-bench pp4096` and full-server `ctx=65536` stayed fast.
- Best partial backend-host KV recovery was `1.9826 TPS`, below D012 and with
   decode regressed to `36.98 tok/s`; all D034 code prototypes were reverted.
- Do not use the `0.37 TPS` slow pocket as a speed baseline.

First command to resume P002 should be a clean D012 same-lane control before any
new code edit:

```bash
PATH="/c/Strawberry/c/bin:$PATH" GGML_VK_ALLOW_GRAPHICS_QUEUE=1 GGML_VK_AMD_LARGE_MATMUL_VARIANT=bn256 GGML_VK_QK_LOW_TILE_SPLIT_K=3 GGML_VK_Q3K_QUAD_DEQUANT=1 python scripts/agent_workload_bench.py --server-bin build-vulkan/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --label p002-resume-d012-control-r1 --ctx-size 131072 --batch-size 512 --ubatch-size 256 --gpu-layers 999 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --parallel 1 --max-tokens 16 --tasks quick --task-ids triage_diff --real-context-mode repo-snapshot --real-context-chars 24576 --no-disable-thinking --no-reuse --no-v2-prime-pass --allow-ctx-above-16k --runs 1 --background-server-policy fail --server-extra "--spec-type none --no-mmap" --cache-ram 0 --ctx-checkpoints 0 --write-diagnostics
```

After that, resume from `docs/research/major-topology/README.md`, especially
T3a. The next valid route must be a true Q3_K compute body/compressed-dot route
or a lifetime design that preserves decode while recovering residency.

## Closed Status (2026-05-18)

C01 is closed as the active default research branch for the current bench.

Repository-wide performance work is archived in `docs/research/archive/2026-05-fast-probe-cycle/PERFORMANCE_ARCHIVE_2026-05-18.md`. Read that file before using this playbook.

Use this playbook only if C01 is intentionally reopened. Reopen requires one of:
- a fresh current-bench trace showing a materially different `MUL_MAT forward` shape/route mix than the documented Q3_K `ncols_max=192` center,
- or a new Q3/MMQ design with modeled wall-time ceiling above `2%`, clear route activation evidence, and no overlap with the rejected E016-E027/E031/E032/E052 branches.

Default next work should come from the repo-wide docs/experiment scan, not from continuing C01 by inertia.

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

## Closed Branch Policy (2026-05-18)

1. `GGML_MMQ_RDNA4_STREAM_K_MIN_NE11=144` failed `runs=3` confirmation
   (`c01-next-control-r3` vs `c01-next-streamk144-r3`, `-1.37%`, negative verdict).
2. `GGML_CUDA_FORCE_MMQ_RUNTIME=1` is rejected for this lane:
   runtime `-0.72%` and hotspot-time regression on `MUL_MAT forward` / MMQ q3 bucket.
3. E013 closed the MMVQ Q3_K side center with a kept narrow policy change.
4. E015 kept the direct C01 MMQ policy win after E013.
5. E016-E027/E031/E032/E052 close the follow-up C01 queue: force-x, Q3/Q4 layout/load probes,
   dense staging, GDN chunking, F32 SSM route experiments, and hipBLASLt/Stream-K env gates.
6. `ngram-mod 24/48/64` remains a useful opt-in repeated/steady profile, but it is not a C01 kernel/default fix.
7. Do not start another C01 code probe unless the reopen conditions in `Closed Status` are met.
