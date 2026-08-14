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
- [ ] 0.5 Commit the R001 docs on `research/rdna4-arch-exploit` (only after
  `git diff --check` is clean).

## Phase 1 - re-establish baselines (GPU work starts here)

- [ ] 1.1 Driver-safety precheck: no `llama-server`/bench processes, no game
  load on the GPUs (per AGENTS.md); then idle GPU memory check.
- [ ] 1.2 Adjacent baseline: dual-ROCm 49K lane, `spec=none`, D089 settings
  (`-dev ROCm1,ROCm0 -sm layer`, Qwen3.6-27B Q4_K_M, 147456 chars). Expect
  ~38.9 tok/s decode. Record cold-first and steady separately.
- [ ] 1.3 Adjacent MTP baseline: same lane, `--spec-type draft-mtp` with the
  MTP GGUF (acceptance + TPS). Only needed before any MTP-adjacent change.

## Phase 2 - experiments, cheapest/safest first

Each experiment: focused correctness (test-backend-ops / FLASH_ATTN_EXT lane)
-> A-B-A on the locked lane -> 98K confirmation if it passes the >=3% decode
gate. Rejected candidates are documented and reverted.

- [ ] 2.1 H80: cache-policy hints on the KV global loads (streaming/no-retain).
  Smallest, most reversible change; verify flags in the disasm before/after.
- [ ] 2.2 H79: prefetch P_f8 B-fragments one WMMA chain ahead in
  `fattn-wmma-f16.cu` (target: PV phase 55.6% -> closer to KQ's 28.3%).
- [ ] 2.3 `V_CVT_SR_FP8_F32` stochastic requant: NMSE gate FIRST (quality
  budget from D098: <= 1e-3 documented E4M3-P contract); speed only after the
  quality gate passes.
- [ ] 2.4 H77: LDS cut to <= 21,845 B/CTA for a 3rd CTA (shrink P_f8 / VKQ
  tiles, reuse the KQ_or_V union). Larger restructure; only after 2.1-2.3
  are settled (they touch the same buffers).
- [ ] 2.5 W10 follow-up: only if the audit (0.2) shows a real gemm-shape gap.
- [ ] 2.6 D104: whole-graph per-kernel census on the 49K decode token (deferred
  from D102; re-run only after any accepted change lands).

## Phase 3 - consolidation

- [ ] 3.1 Close the loop per candidate in `docs/research/` + `RESULTS_LOG.md`:
  accepted ones land in source; rejected ones are reverted and their env
  toggles removed (no dormant experiment flags left behind).
- [ ] 3.2 MTP on ROCm: confirm the stable 38.9 tok/s decode and make it the
  production track (if unchanged by phase 2).
- [ ] 3.3 Debt cleanup batch 1 (from W11): remove dead diagnostic branches
  (Vulkan FA F8_P2-P5 transforms, NATIVE_DECODE route, HALF_CMP splitter,
  ROCm census scaffolding) - only after phase-2 conclusions are firm.
- [ ] 3.4 Document the surviving env-var surface in one registry doc
  (`docs/research/ENV_VARS.md` or similar).
- [ ] 3.5 GitHub visibility (needs user decisions): release v0.1.0 with
  Windows binaries, enable issues, social preview image.

## Standing validation rules (from AGENTS.md)

- Benchmark only with the bench harness (server-bin, no sleep polls, no
  hard kills); adjacent runs; equal lane settings; record game/background
  load if any.
- `python -m compileall -q gui scripts run.py` for Python changes;
  `cmake --build build-rocm -j 4 --target llama-server` for source changes;
  always finish with `git diff --check`.
- Never run `llama-server --version/--help` as a probe; stop servers
  gracefully.
