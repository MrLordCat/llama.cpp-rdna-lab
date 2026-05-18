# E045 - Prefill ubatch 2048 Recenter

## Metadata

- Experiment ID: E045
- Date: 2026-05-17
- Owner: Codex
- Branch/Commit: local `master`
- Target lane: Qwen3.6-27B-Q3_K_S cold-first prefill lane, `ctx=12288`, `batch=6144`, KV `q4_0/q4_0`, `triage_diff,review_bug`, `spec=none`, no reuse, thinking on.

## Hypothesis

- Statement: the current prefill baseline should be recentered above the old C01 `ubatch=192/1024` points if RDNA4 can run the prompt in larger backend chunks without crossing the residency cliff.
- Mechanism: increasing `ubatch` moves large prompt matmuls from the old `Q3_K ncols=192` MMQ center to larger `cublas_backend`/dequant shapes. If the compute buffer still fits VRAM, fewer graph slices and larger GEMM-like shapes should improve prompt eval.
- Why now: after MTP work was closed, the user asked for autonomous prefill speed search and a fresh baseline. The machine had recently rebooted, so old baselines were not assumed current.

## Math / Theory

- Baseline `ubatch=1024` r3:
  - aggregate completion TPS: `11.4240`
  - prompt eval: `1146.9633 tok/s`
  - prompt eval mean: `6465.03 ms`
  - decode eval: `29.9683 tok/s`
- Candidate `ubatch=2048` r3:
  - aggregate completion TPS: `11.6534`
  - prompt eval: `1197.5567 tok/s`
  - prompt eval mean: `6192.1383 ms`
  - decode eval: `29.4850 tok/s`
- Expected wall gain from prompt-only improvement:
  - prompt time improved by about `272.9 ms` per task (`6465.0 -> 6192.1 ms`, `+4.41%` prompt eval TPS),
  - decode slowed by about `65.5 ms`,
  - net per-task time improvement is about `207 ms`, matching the observed aggregate gain of about `+2.01%`.
- Failure conditions:
  - `ubatch >= 3072` starts losing the advantage,
  - `ubatch=6144` hits a severe residency/allocator cliff (`1.19 TPS`),
  - broad forced MMQ at `ubatch=2048` regresses (`10.00 TPS`).

## Implementation Plan

1. Minimal code surface to change: none in this experiment.
2. Guard rails: keep this as a benchmark/profile finding until a GUI/profile default change is requested and separately tested.
3. Rollback path: use previous `ubatch=1024` baseline or old C01 `ubatch=192` lane for decode/MMQ-specific experiments.

## Benchmark Plan

- Baseline command:
  - `python scripts/agent_workload_bench.py --label prefill-current-ub1024-base-r3 ... --ctx-size 12288 --batch-size 6144 --ubatch-size 1024 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"`
- Candidate command:
  - same command with `--ubatch-size 2048`
- Number of runs:
  - r1 sweep, r3 confirmation for `ubatch=2048`.
- Artifacts path:
  - `build_logs/agent-workload/prefill-current-ub1024-base-r3.*`
  - `build_logs/agent-workload/prefill-current-ub2048-base-r3.*`
  - `build_logs/agent-workload/prefill-current-ub1024-trace-r1.server.log`
  - `build_logs/agent-workload/prefill-current-ub2048-trace-r1.server.log`

## Metrics

- aggregate completion TPS (wall)
- prompt eval TPS/ms
- decode eval TPS/ms
- route trace and operation share
- failure/regression checks for adjacent knobs

## Result

- Outcome: win for prefill profile/baseline; no code change.
- Delta:
  - `11.4240 -> 11.6534 TPS` (`+2.01%`) vs `ubatch=1024` r3.
  - prompt eval `1146.96 -> 1197.56 tok/s` (`+4.41%`).
  - decode eval `29.97 -> 29.49 tok/s` (`-1.61%`), so the win is prompt-heavy.
- Confidence:
  - confirmed with r3; individual task TPS stdev `0.1507`.
  - adjacent sweep supports a peak near `2048`: `1280=11.53`, `1536=11.64`, `2048=11.74` r1, `3072=11.67`, `6144=1.19`.
- Recommendation:
  - Use `ubatch=2048` as the current cold-first prefill search baseline on this lane.
  - Do not promote `ubatch=6144`.
  - Do not pursue broad forced MMQ for this large-prefill route.

## Trace Findings

- `ubatch=1024` prompt trace:
  - total prompt node time: `15491.838 ms`
  - `MUL_MAT`: `10118.950 ms` (`65.32%`)
  - `GATED_DELTA_NET`: `1954.640 ms` (`12.62%`)
  - `FLASH_ATTN_EXT`: `676.960 ms` (`4.37%`)
  - `MUL_MAT` by source type: `q3_K 84.27%`, `f32 9.79%`, `q4_K 5.90%`
- `ubatch=2048` prompt trace:
  - total prompt node time: `13969.454 ms`
  - `MUL_MAT`: `9053.320 ms` (`64.81%`)
  - `GATED_DELTA_NET`: `2024.980 ms` (`14.50%`)
  - `FLASH_ATTN_EXT`: `668.210 ms` (`4.78%`)
  - `MUL_MAT` by source type: `q3_K 84.32%`, `f32 9.80%`, `q4_K 5.83%`
- The `ubatch=2048` hot Q3_K route is large `cublas_backend`, not the old C01 `MMQ type=11 ncols_max=192` bucket.

## Negative Gates

- GDN knobs on `ubatch=1024`:
  - same-session r1 baseline: `11.37 TPS`
  - `GGML_GDN_CHUNK_SIZE=256`: `11.30 TPS`
  - `GGML_GDN_CHUNK_SIZE=512`: `11.33 TPS`
  - `GGML_GDN_CHUNK_SIZE=1024`: `11.30 TPS`
  - `GGML_GDN_FAST_EXP=1`: `11.32 TPS`
  - chunk `256` + fast-exp: `11.31 TPS`
- Batch sweep with `ubatch=2048`:
  - `batch=4096`: `11.60 TPS`
  - `batch=8192`: `11.55 TPS`
  - `batch=12288`: `11.55 TPS`
  - default `batch=6144` remains the best confirmed setting.
- Broad MMQ force:
  - `GGML_CUDA_FORCE_MMQ_RUNTIME=1`, `ubatch=2048`: `10.00 TPS`, rejected.

## Notes

- `ubatch=6144` was intentionally tested as a cliff probe and is not safe on the current 16 GB VRAM setup.
- The next code-level search should treat `ubatch=2048` as the profile baseline and inspect large-shape Q3_K dequant/cublas, GDN, or allocator/residency behavior rather than continuing old `ncols=192` MMQ-only assumptions.
