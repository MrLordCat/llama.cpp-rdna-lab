# R001 phase-2 plan (checklist, sequential)

Status: GPUs free (user signal 2026-08-14). Phase-1 constraint lifted: builds and
benchmarks are allowed again. Each item lists its verification; items are
closed strictly in order.

## Phase 0 - close the research tails (CPU-only, no GPU)

- [x] 0.1 Finalize occupancy (W08): fold the wave32/2-CTA/LDS-bound summary into
  W03 and mark W08 closed in the README (no separate doc needed).
- [x] 0.2 W10 audit (source-only): which gemm shapes the decode path actually
  dispatches (MMVQ small-batch shapes for Q4_K_M at ncols=1..4, MMQ prefill
  shapes), and where hipBLASLt/rocBLAS is or is not on the route.
  Deliverable: `W10_MMVQ_GEMM_SHAPES.md`.
- [x] 0.3 Internet research tails: WMMA throughput and LDS bandwidth for
  gfx1201 (GPUOpen/AMD/LLVM sources; mark unverifiable items explicitly).
  Update W01 "Not yet verified".
- [x] 0.4 Write the backend-debt audit done 2026-08-14:
  `W11_BACKEND_DEBT_AUDIT.md` (three categories: dead diagnostics, live
  fallbacks, real dead code; with the env-var zoo list).
- [x] 0.5 Commit the R001 docs on `research/rdna4-arch-exploit` — done,
  commit 23b12050d (after `git diff --check`).

## Phase 1 - re-establish baselines (GPU work starts here)

- [x] 1.1 Driver-safety precheck: no `llama-server`/bench processes, no game
  load on the GPUs (per AGENTS.md); idle GPU memory check — clean
  (Steam without games, VRAM ~2.5 GB baseline).
- [x] 1.2 Adjacent baseline: dual-ROCm 49K lane, `spec=none` — decode
  21.12 t/s raw / 1653.7 ptps, no error, clean teardown
  (`r001-p2-49k-f8-specnone-baseline-r1`).
- [x] 1.3 Adjacent MTP baseline: `--spec-type draft-mtp --spec-draft-n-max 2`
  — decode 38.55 t/s, ptps 1575.9, draft_n 96 / accepted 78 (81%)
  (`r001-p2-49k-f8-mtp2-baseline-r1`).

## Phase 2 - experiments, cheapest/safest first

Each experiment: focused correctness (test-backend-ops / FLASH_ATTN_EXT lane)
-> A-B-A on the locked lane -> 98K confirmation if it passes the >=3% decode
gate. Rejected candidates are documented and reverted.

- [x] 2.1 H80: cache-policy hints on the KV global loads (streaming/no-retain).
  BLOCKED by the toolchain: `__builtin_nontemporal_load` is silently dropped
  by ROCm 7.1 clang (byte-level encoding diff = zero), and llvm-mc (AOMP-18)
  accepts no glc/slc/nt/cache_policy/th modifier for global_load. Deferred
  until a toolchain with an expressible TH field (see HYPOTHESES H80).
- [x] 2.2 H79: PV V_a prefetch one chain ahead in `fattn-wmma-f16.cu`
  (original premise corrected in W03: B-fragments were already in registers).
  REJECTED: neutral (-0.5% decode, noise), reverted.
- [x] 2.3 `V_CVT_SR_FP8_F32` stochastic requant. REJECTED: NMSE worse than RNE
  (native_v=1 test cases ERR 0.0011-0.0013 vs ~0.0007; 12/19 vs 19/19).
- [x] 2.4 H77: LDS cut to <= 21,845 B/CTA for a 3rd CTA. REJECTED: (a) premise
  falsified - decode grid = (1, 8, 24) = 192 blocks = exactly 2 CTAs/CU, the
  3rd CTA would idle; (b) register-acc variant (LDS 21,120 B) pushed VGPR
  156 -> 181 and regressed decode -3.8% (21.49 vs 22.33 t/s).
- [x] 2.5 W10 follow-up: CLOSED. (a) The "Q3_K capped at batch 1 on RDNA4"
  claim in W10 was an audit error - the current RDNA4 table returns 4 and the
  dispatch uses it (mmvq.cu:233, runtime_compute.inc:1955); no cap fix needed.
  (b) QWEN small-K toggles folded into 3.1/3.4 (winner to default, loser removed).
- [x] 2.6 D104: SKIPPED - re-run only after an accepted change lands; all of
  2.1-2.4 were rejected/blocked, so there is nothing to re-census.

## Phase 3 - consolidation

- [x] 3.1 Close the loop per candidate in `docs/research/` + `RESULTS_LOG.md`:
  all phase-2 candidates rejected/blocked (H80 blocked, H79 neutral, SR-requant
  worse NMSE, H77 -3.8%); verdicts recorded in RESULTS_LOG and HYPOTHESES;
  rejected code reverted in source.
- [x] 3.2 MTP on ROCm: CONFIRMED on the debt-cleaned binary (2026-08-14,
  fresh boot after GPU1 re-enable, 49K dual-ROCm, f8 KV): draft-mtp n=2
  decode 39.98 t/s (acceptance 78/96) vs adjacent spec=none 22.72 t/s
  = 1.76x; the 38.9 t/s target is met and MTP remains the production
  decode configuration.
- [x] 3.3 Debt cleanup batch 1 (from W11): remove dead diagnostic branches
  (Vulkan FA F8_P2-P5 transforms, NATIVE_DECODE route, HALF_CMP splitter,
  ROCm census scaffolding) - committed f704ad8f2 (2026-08-14).
  Batch 2 (2026-08-14, debt close): loser MMVQ/MMQ toggle branches removed
  (QWEN small-K, PAIRDOT-disable, VK16 kernel family, FORCE_MMQ_X,
  MOE_MMQ_STAGING), PERF notes moved to docs/research/, suite gap documented;
  see W11 "batch 2 executed".
- [x] 3.4 Document the surviving env-var surface in one registry doc:
  `docs/research/ENV_VARS.md` (2026-08-14, post-cleanup surface incl. the
  removed-by-f704ad8f2 list).
- [x] 3.5 GitHub visibility: DROPPED by user decision (2026-08-14). Phase 2
  produced no performance change (all candidates rejected; decode stayed at
  the pre-existing MTP level), so a release has no new content; the
  turboquant project is closed. Revisit only if a future change warrants it.

## Standing validation rules (from AGENTS.md)

- Benchmark only with the bench harness (server-bin, no sleep polls, no
  hard kills); adjacent runs; equal lane settings; record game/background
  load if any.
- `python -m compileall -q gui scripts run.py` for Python changes;
  `cmake --build build-rocm -j 4 --target llama-server` for source changes;
  always finish with `git diff --check`.
- Never run `llama-server --version/--help` as a probe; stop servers
  gracefully.
