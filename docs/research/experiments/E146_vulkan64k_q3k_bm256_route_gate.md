# E146 Vulkan 64k Q3_K BM256 Route Gate

## Metadata

- Experiment ID: E146
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E145 (`d5d3c8360`)
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: the active large Q3_K coopmat route may improve with `BM=256` for hot 64k prompt shapes by reducing workgroup count and B/activation reloads without increasing `BN` or reducing `BK`.
- Mechanism: E143 proved larger `BN` is the wrong way to reduce repeated A-side work because it raises VGPR/LDS/scratch and loses occupancy. `BM256` is a different route: it keeps `BN=128,BK=32,WM/WN=64/64`, doubles M rows per workgroup, and halves M-block count. Static model says A-pair dequant proxy stays roughly flat, while B reload and barrier/workgroup proxy halve.
- Why now: after E143/E144/E145, the next Q3_K probe should target a different route dimension with a measurable traffic/workgroup mechanism, not another nearby A-dequant or FA split tweak.

## Math / Theory

- E134 Q3_K share proxy: `0.5228`; all-Q3_K alone needs about `1.357x` local speedup to close the lane.
- Static scout for `17408x1024,k=5120` plus `5120x1024,k=5120`:
  - base: `128x128x32`, `20480 B` Q3 LDS, workgroups `1088/320`, B reload `1760 MiB`, A pairs `461.37M`;
  - `bm256`: `256x128x32`, `31744 B` Q3 LDS, workgroups `544/160`, B reload `880 MiB`, A pairs `461.37M`.
- Static scout for reverse hot shape `5120x1024,k=17408`:
  - base: workgroups `320`, B reload `1360 MiB`, A pairs `356.52M`;
  - `bm256`: workgroups `160`, B reload `680 MiB`, A pairs `356.52M`.
- Failure conditions:
  - LDS is close to the 32 KiB limit and may reduce occupancy or fail support on driver details.
  - Larger M tile may increase register pressure or reduce scheduling flexibility enough to dominate B traffic reduction.
  - If prompt speed does not improve, this route does not justify full 64k server validation.

## Implementation Plan

1. Minimal code surface to change: temporary `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bm256` branch in Vulkan AMD large matmul warptile setup.
2. Guard rails: default route untouched unless the env variant is set.
3. Rollback path: revert the host-side env branch if the pp/resource gate fails.

## Benchmark Plan

- Baseline command: q4/q4 pp7488, FlashAttention on, `b8192/ub1024`, route trace and Q3_K pipeline stats.
- Candidate command: same with `GGML_VK_AMD_LARGE_MATMUL_VARIANT=bm256`.
- First gate: pipeline resource stats for `matmul_q3_k` and pp7488.
- Full 64k server run only if pp/resource gate is positive.

## Metrics

- prompt eval tok/s
- Q3_K route trace (`BM/BN/BK` through pipeline resource fingerprint and variant activation)
- driver pipeline stats (VGPR/SGPR/LDS/scratch)
- 64k real-server prompt eval if candidate survives

## Analytic Gate

Commands:

```powershell
python scripts\research\vulkan_warptile_static_scout.py --variants base,bm256,bm256-bn256 --shapes 17408x1024,5120x1024 --k-size 5120
python scripts\research\vulkan_warptile_static_scout.py --variants base,bm256,bm256-bn256 --shapes 5120x1024 --k-size 17408
python scripts\research\vulkan_q3k_prebuild_gate.py --candidate "Q3_K BM256 route reducing B reload/workgroups for hot Vulkan 64k shapes" --baseline-pp 978.88 --goal-total-speedup 1.1596 --target-share 0.5228 --local-gain-pct 5
```

- Static decision: build `bm256`; reject `bm256-bn256` statically because Q3 shader LDS is `45056 B`, above the device limit.
- Original prebuild gate decision: build-candidate.
- Workflow correction after measurement: this should have been blocked by the E098 `bm256` large-tile prior. `scripts/research/vulkan_q3k_prebuild_gate.py` now has explicit `bm256`/`bn256`/`bn192`/large-tile priors and lets rejected historical analogs block a build even when an optimistic `--local-gain-pct` is supplied. The corrected gate returns `skip-build-unless-new-topology` for the same `BM256` candidate.

## Result

- Outcome: regression; no code kept.
- Delta:
  - baseline `BM128/BN128/BK32`: `972.84 tok/s`, `113 VGPR / 45 SGPR / 20480 B LDS / 0 scratch`;
  - candidate `BM256/BN128/BK32`: `916.62 tok/s`, `94 VGPR / 45 SGPR / 31744 B LDS / 0 scratch`;
  - prompt delta: `-5.78%`.
- Confidence: high enough to reject this route before a full 64k server run. Route trace confirmed the same Q3_K pipeline name and the resource fingerprint changed as expected.
- Recommendation: reject and revert the env-gated `bm256` branch. Halving B reload/workgroup proxy is not enough when the shader consumes nearly the whole 32 KiB LDS budget. Future BM changes need a design that reduces LDS/residency pressure at the same time, or a different topology than current `mul_mm.comp`.
- Workflow lesson: static B/workgroup proxy must be paired with an LDS/occupancy blocker check and a historical-variant match. In future, a large-tile candidate that reuses current `mul_mm.comp` should be skipped unless it names a new topology that avoids the E098/E143/E146 resource failure mode.

## Artifacts

- `build_logs/agent-workload/e146-vulkan-q3k-bm256-static-scout.md`
- `build_logs/agent-workload/e146-vulkan-q3k-bm256-baseline-pp7488.log`
- `build_logs/agent-workload/e146-vulkan-q3k-bm256-candidate-pp7488.log`
- `scripts/research/vulkan_q3k_prebuild_gate.py`
