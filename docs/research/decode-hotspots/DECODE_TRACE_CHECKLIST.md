# Prompt/Decode Speed - Trace Checklist (Current Focus)

## Baseline profile (current route)

- Lane: `tasks=quick`, `ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, `no-reuse`, `max_tokens=256`.
- Baseline run: `decode-trace-current-ctx12288-ub192-r1`.
- Aggregate TPS: `26.30`.
- Main artifacts:
  - `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.server.log`
  - `build_logs/agent-workload/decode-trace-current-ctx12288-ub192-r1.diagnostics.md`
  - `build_logs/agent-workload/decode-trace-smallk-compare.md`

## Main cost centers (by summed total_ms)

| Priority | Center | sum_ms | count | avg_ms | Status |
| --- | --- | ---: | ---: | ---: | --- |
| P1 | CUDA_NODE `MUL_MAT kind=forward` | 1717.322 | 8454 | 0.203 | closed as active branch (2026-05-18) |
| P2 | MMVQ `type=11/q3_K ncols_dst=1` | 339.110 | 4618 | 0.073 | done: E013 kept RDNA4 Q3_K `nwarps=2` |
| P3 | CUDA_NODE `MUL_MAT kind=fused` | 326.936 | 2298 | 0.142 | queued |
| P4 | CUDA_NODE `RMS_NORM kind=fused` | 209.981 | 4389 | 0.048 | queued |
| P5 | CUDA_NODE `GATED_DELTA_NET kind=forward` | 149.095 | 1008 | 0.148 | queued |

## Conditional checklist (workflow)

- [x] Freeze baseline route and collect full kernel trace.
- [x] Build hotspot ranking by `sum(total_ms)`.
- [x] Create one document per center.
- [x] Start deep trace from `MUL_MAT forward`.
- [x] Complete full sub-trace map for `MUL_MAT forward` (closed for current branch):
  - top node names,
  - top tensor shapes (`ne`),
  - route deltas vs control A/B.
- [x] Produce root-cause hypothesis set for `MUL_MAT forward` (memory bound, shape inefficiency, launch granularity, sync pressure).
- [x] Run micro A/B test for MMVQ Q3_K side center and keep reproducible gain (E013).
- [x] Run first micro A/B screen for top `MUL_MAT forward` selector/resource hypothesis (E014 negative).
- [x] Run first deeper Q3_K MMQ geometry A/B and keep reproducible gain (E015).
- [x] Scout FATTN and ngram after current-environment retest (E026).
- [x] Close simple force-x sub-32KiB probe (E027 negative/invalid).
- [x] Scout F32 SSM alternate MMF route (E032 no-activation; current RDNA4 MMF cannot cheaply target F32 SSM).
- [x] Continue Q3_K MMQ compute/load micro A/B and keep only reproducible gains (closed; no remaining low/medium-risk C01 candidate).
- [x] Move to next center only after `MUL_MAT forward` has a closed trace + hypothesis verdict.

## Next-step runbook

1. Treat C01 as closed for current-bench TPS work.
2. Reopen only if a fresh trace changes the C01 shape/route mix, or a new Q3/MMQ design passes a `>=2%` wall-ceiling preflight and is not a duplicate reject.
3. Select the next branch from the accumulated docs/experiments scan before editing code.

## Latest Resume Checkpoint (2026-05-13)

- Return command executed: `c01-resume-r1-resources` (lane contract preserved).
- Artifact:
  - `build_logs/agent-workload/c01-resume-r1-resources.server.log`
- Mandatory gates on latest trace:
  - shape gate (`qtype=11`, `ncols_max=192`): PASS (`count=26524`),
  - cold/steady split: steady still dominated by `mul_mat_q_direct|q3_K`,
  - q3 path coarse split (steady): `compute_core_q3=84.25%`, `fallback_cublas=14.38%`.
- Comparison notes:
  - comparing against global decode baseline (`decode-trace-current-ctx12288-ub192-r1`) is methodologically invalid due different task mix,
  - apples-to-apples check vs previous C01 resource run (`e013-c01-two-tasks-trace-r1-resources`) shows `-5.4%` runtime and no major route flip,
  - control rerun (`c01-resume-r2-control`) is stable vs `c01-resume-r1-resources` (`+0.14%`, inconclusive/noise-level).

Latest artifacts from return sequence:
- `build_logs/agent-workload/c01-resume-r1-resources.server.log`
- `build_logs/agent-workload/c01-resume-r2-control.server.log`
- `build_logs/agent-workload/c01-resume-r2-control.csv`

## E013 MMVQ Q3_K closure

- Candidate: RDNA4 `GGML_TYPE_Q3_K`, `ncols_dst=1`, `nwarps=2`.
- Paired non-trace control: `9.1629 TPS`.
- Candidate non-trace: `9.3847 TPS`, `+2.42%`.
- Bootstrap CI: `[+0.2019, +0.2442]` TPS.
- Decision: keep; Q4_K unchanged pending a Q4-heavy lane.

## Pause/Resume workflow

Before switching to another problem:

1. Refresh `C01_RESUME_PLAYBOOK.md` with current lane, best point, and first resume command.
2. Ensure latest artifacts are in `build_logs/agent-workload/` and referenced in C01 notes.
3. Mark open vs done items in this checklist.

## E026 FATTN/ngram scout

- FATTN trace on the current lane shows `FLASH_ATTN_EXT forward = 638.004 ms` out of `24758.198 ms` sync CUDA_NODE (`~2.58%`).
- Active FATTN route is WMMA F16 for `D=256`, `q_rows=192`, `selected_cols=16`; no obvious selector miss was found.
- Decision: do not spend current C01 code budget on FATTN unless the lane shifts to longer-context/FATTN-heavy work.
- ngram-mod `24/48/64` measured `9.7225 TPS` vs `9.4111 TPS`, but the verdict is inconclusive and the effective acceptance is only `0.00675`.
- ngram-simple generated zero drafts; `n_match=12` was neutral.
- Decision: ngram-mod remains opt-in for repeated/steady workloads, not a cold-first default.

## E027 force-x sub-32KiB scout

- Fresh return trace: `build_logs/agent-workload/c01-return-20260516-r1-resources.server.log`.
- Target remains `type=11,ncols_max=192,mmq_x=96,mmq_y=64`, shared `35712`, waves `4.00`.
- `x72` is invalid/no-op because RDNA4 WMMA granularity is `16`; trace stayed at `mmq_x=96`.
- `x64` is valid but regressed to `8.90 TPS` vs current baseline `9.4111 TPS`.
- Decision: simple force-x route is closed; further Q3_K work needs layout/scheduling changes, not selector values.

When returning to C01:

1. Read `C01_RESUME_PLAYBOOK.md` first.
2. Re-run one fresh resource trace and shape gate before any new code change.
3. Use decision stats for borderline runtime deltas.

Current return status:
- step 1: done,
- step 2: done (`c01-resume-r1-resources`),
- step 3: done (stats against e013 C01-compatible baseline).

## E014 C01 selector/resource screen

- Fresh post-E013 artifact: `build_logs/agent-workload/c01-poste013-r1-resources.server.log`
- Active steady target: `mul_mat_q_direct|q3_K = 12325.249 ms` (`78.93%` of steady `MUL_MAT forward`).
- Tested: force-x sweep, post-E013 stream-k retest, `mmq_y=64`, RDNA4 `launch_bounds(..., 1)`.
- Decision: no keep candidate; temporary code probes reverted and `llama-server` rebuilt.
- Next: inspect Q3_K MMQ compute/load internals, not more scalar selector sweeps.

## E015 RDNA4 MMQ y64/w4 keep

- Code: RDNA4 MMQ now uses `mmq_y=64` with `nwarps=4`.
- Paired r3: `c01-e015-control-postrevert-r3` -> `c01-e015-rdna4-y64w4-r3`.
- Runtime: `9.3974 -> 9.6080 TPS` (`+2.24%`), bootstrap CI `[+0.1855,+0.2368]`.
- Target trace: `MMQ type=11 ncols_max=192` improved `9949.928 -> 9551.391 ms`.
- Status: keep; continue C01 from post-E015 baseline.

## Focus Recalibration (2026-05-14)

- Fresh same-lane trace: `build_logs/agent-workload/focus-c01-current-hotspots-r1.server.log`.
- Lane preserved: `quick review_bug,patch_sim`, `ctx=12288`, `b=6144`, `ub=192`, `q4_0/q4_0`, `spec=none`, no-reuse, `max_tokens=120`.
- Wall timing is prompt-heavy: prompt eval `26826.35 ms`, decode eval `8399.61 ms`; prompt share `76.15%`.
- Phase split confirms this at trace level: sync-only `CUDA_NODE` prompt phase `17000.705 ms` (`76.83%`), decode phase `4760.101 ms` (`21.51%`), outside/reserve `366.264 ms`.
- Kernel ranking still points at C01:
  - `CUDA_NODE op=MUL_MAT kind=forward`: `14412.924 ms` (`65.14%` of CUDA_NODE),
  - `MUL_MAT src0=q3_K type=f32`: `11615.341 ms` (`52.49%`),
  - `MMQ type=11 ncols_max=192`: `9048.863 ms` (`40.89%`).
- Prompt-phase top target: `MMQ type=11 ncols_max=192 = 7490.845 ms` (`44.06%` of prompt CUDA_NODE).
- Shape gate remains PASS for Q3_K `ncols_max=192` (`26524` hits).
- Cold/steady split: steady `mul_mat_q_direct|q3_K` is `11408.481 ms` (`78.28%` of steady `MUL_MAT forward`).
- Decision at the time: keep C01 as the main focus, but frame it as prompt/prefill first. Next work should inspect Q3_K MMQ compute/load internals, not GUI autotune, ngram, or MMVQ.
- Superseded by 2026-05-18 closeout: C01 is closed as the active branch; next work should be selected from the docs/experiment scan.

## Current quick-bench contract (2026-05-17)

- Active quick task pair for C01 and nearby lanes: `triage_diff,review_bug`.
- Historical mentions of `review_bug,patch_sim` in this checklist are archived experiment context, not the current default.

## C01 diagnostics toolkit

- shape gate: `python scripts/research/c01_shape_presence_gate.py ...`
- cold/steady split: `python scripts/research/cold_steady_trace_split.py ...`
- q3 coarse component split: `python scripts/research/c01_q3_path_components.py ...`
- statistical verdict: `python scripts/research/decision_stats.py ...`
- trace compare: `python scripts/research/compare_kernel_traces.py ...`

## E018/E019 Q3_K load/scale probes

- E018 scale preload:
  - candidate `c01-e018-q3-scale-preload-r1` looked neutral-positive in non-trace TPS,
  - target trace regressed: `MMQ type=11 ncols_max=192` `9048.863 -> 9103.787 ms`,
  - decision: reject; code reverted.
- E019 load_tiles scale fusion:
  - candidate `c01-e019-q3-loadtiles-fuse-scales-r1`,
  - aggregate `8.2082 TPS` vs E015 reference `9.6080 TPS`,
  - decision: reject before trace; code reverted and server rebuilt.
- Next:
  - avoid extra register arrays or fused scale/min unpack in hot Q3_K lanes unless the resource model has a larger predicted win.
  - prefer C01 probes that change tile count/shared footprint or scheduling without adding work to the quant-load lanes.

## E020 Q3_K compact half-scale result

- Candidate: Q3-only compact MMA shared layout, `84 -> 72` int stride, precomputed scales stored as half.
- Valid artifact: `build_logs/agent-workload/c01-e020-q3-halfscale-compact-trace-r1b.server.log`.
- Resource result:
  - shared `35712 -> 32640`,
  - `max_blocks_per_sm=1 -> 2`,
  - occupancy `6.25% -> 12.50%`,
  - waves `4.00 -> 8.00`,
  - regs `160 -> 158`.
- Target result vs E015 trace:
  - `MMQ type=11 ncols_max=192`: `9551.391 -> 9451.261 ms`.
- Runtime result:
  - r3 `9.6080 -> 9.6017 TPS`,
  - bootstrap CI `[-0.0380,+0.0239]`, inconclusive.
- Decision:
  - no default; code reverted and server rebuilt.
  - useful clue: shared-residency unlock is real, but alone does not translate to aggregate lane speed.

## E021 dense Q3 MMQ staging

- Candidate: temporarily enable RDNA4 MMQ staging for dense `Q3_K`.
- Theory: staged shared footprint for the active x96/y64 bucket is `57220` bytes, so it fits `64 KiB` and preserves tile count.
- Runtime: `9.6080 -> 8.6216 TPS` (`-10.27%`).
- Activation confirmed: `rdna4_staging_req=1`, `rdna4_staging_eff=1`.
- Target signal: per-call `MMQ type=11 ncols_max=192` worsened about `25.9%`.
- Decision: reject; runtime code reverted and `llama-server` rebuilt.

## E022 C05 GDN chunk192 probe

- Current route: prompt GDN uses `n_tokens=192`, default internal chunks `96 + 96`, `fast_exp=0`.
- `GGML_GDN_FAST_EXP=1`: `9.59 TPS`, below current best.
- `GGML_GDN_CHUNK_SIZE=192`: `9.58 TPS`, below current best.
- Decision: reject; keep current GDN chunk policy.

## E023 C01 F32 cuBLAS GemmEx probe

- Target: small F32 SSM alpha/beta GEMMs, `MUL_MAT f32 ne=(48,192)`, currently routed through `cublas_backend`.
- Candidate: env-gated RDNA4 `cublasGemmEx(... CUDA_R_32F, CUBLAS_COMPUTE_32F ...)` instead of `cublasSgemm`.
- Runtime: `9.6080 -> 9.42 TPS`.
- Target timing: avg `0.1712 ms -> 0.1850 ms` (`+8.1%` slower).
- Decision: reject; code reverted and `llama-server` rebuilt.

## E024 C05 GDN chunk128 probe

- Candidate: `GGML_GDN_CHUNK_SIZE=128` for active `n_tokens=192`, changing GDN chunks from `96+96` to `128+64`.
- Runtime: `9.6080 -> 9.43 TPS`.
- Decision: reject; trace skipped and no code changes.

## E028 C01 ngram-mod confirmation

- Clean control: `c01-e028-clean-control-r3 = 9.4890 TPS`.
- Candidate: `c01-e028-ngram244864-r6 = 10.3689 TPS`.
- Delta: `+9.27%`, bootstrap CI `[+0.5192,+1.3106]`, verdict `positive`.
- Decode eval improved `30.1433 -> 45.1508 TPS`; prompt eval did not improve.
- Spec stats: local acceptance `0.581422`, coverage `0.040580`, effective acceptance `0.023594`.
- Decision: keep as opt-in repeated/steady preset only; not a cold-first default and not a C01 kernel fix.

## Current Metric Policy For C01

- `Cold-first baseline`: quote `run == 1`; latest same-session split is `9.47 TPS`
  (`c01-e030-clean-split-r2`).
- `Repeated/steady clean baseline`: quote `run > 1`; latest same-session split is
  `9.45 TPS` (`c01-e030-clean-split-r2`), with historical repeated clean `9.4890 TPS`
  (`c01-e028-clean-control-r3` all-runs aggregate).
- `Repeated/steady opt-in reference`: `10.3689 TPS` (`c01-e028-ngram244864-r6`)

Use:
- kernel/default claims must beat the cold-first baseline,
- speculative/session opt-in claims must beat the repeated/steady clean baseline,
- do not compare these two classes through a single mixed headline number,
- all-runs aggregate is allowed only when clearly labeled as mixed/session aggregate.

## E029 cold-first ngram recheck

- First `r1` pair was inconclusive (`9.4381 -> 9.4476 TPS`, `+0.10%`).
- Powered `r3` pair is positive: `9.3031 -> 10.0948 TPS` (`+8.51%`), bootstrap CI `[+0.2943,+1.3488]` TPS.
- Extended `r6` pair confirms it: `9.2468 -> 10.2456 TPS` (`+10.80%`), bootstrap CI `[+0.6980,+1.3441]` TPS.
- Improvement is decode-led (`29.685 -> 42.973 tok/s mean`) while prompt stays neutral/slightly lower.
- Decision: keep `ngram-mod 24/48/64` as explicit opt-in accelerated profile; keep clean `spec=none` as default for kernel/default claims.

## E030 cold/warm metric split

- Bench infrastructure now writes `run` to CSV and prints cold-only run #1 stats with
  `--stats-ignore-first-run`.
- Same-session split:
  - clean: `9.4569` all / `9.47` cold / `9.45` warm TPS.
  - `ngram-mod 24/48/64`: `10.0476` all / `9.46` cold / `10.72` warm TPS.
- Decision:
  - ngram remains a warm/session opt-in accelerator, not a cold-first default win.
  - E029 all-runs aggregate should not be described as pure cold-first.
- Adjacent no-code checks:
  - server warmup did not improve cold/warm,
  - `ubatch=224` regressed to `8.03 TPS`,
  - `ubatch=160` regressed to `8.88 TPS`.

## E031 Q4_K force-x sub-32KiB probe

- Secondary `Q4_K` MMQ bucket is measurable but small:
  `mul_mat_q_direct|q4_K = 964.363 ms` (`6.17%` of steady `MUL_MAT forward`).
- Resource state for `type=12,ncols_max=192`: `mmq_x=96`, shared `33664`, regs `200`,
  `max_blocks_per_sm=1`.
- Temporary `GGML_MMQ_RDNA4_Q4_FORCE_MMQ_X=80` regressed:
  `9.4522 -> 9.4026 TPS`, decision stats negative.
- Decision: reject; code reverted and server rebuilt.
- Next: do not continue Q4 force-x unless the active ncols/tile geometry changes.

## Center documents

- `docs/research/decode-hotspots/C01_mul_mat_forward.md`
- `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md`
- `docs/research/decode-hotspots/C02_mmvq_q3_ncols1.md`
- `docs/research/decode-hotspots/C03_mul_mat_fused.md`
- `docs/research/decode-hotspots/C04_rms_norm_fused.md`
- `docs/research/decode-hotspots/C05_gated_delta_net_forward.md`
