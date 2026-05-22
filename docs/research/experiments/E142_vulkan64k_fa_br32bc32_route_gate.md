# E142 Vulkan 64k FA Br32/Bc32 Route Gate

## Metadata

- Experiment ID: E142
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E141 (`c5e985857`)
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: the active coopmat1 q4 FlashAttention route may improve if one workgroup handles `Br=32` query rows while reducing `Bc` to `32` to stay within the RDNA4 32 KiB LDS budget.
- Mechanism: E129 tested `Bc=32` at the default `Br=16` and lost because it doubled KV loop chunks without increasing query-row reuse. `Br32/Bc32` is a different route: it keeps q4/q4 and single-dispatch coopmat1, but halves the number of query row blocks, so each long-KV traversal feeds twice as many Q rows. The `Bc=32` half-tile is a residency guard, not the primary mechanism.
- Why now: E138 rejected split/reduce, E141 rejected f16/q8 KV route pivots, and H38 still needs a single-dispatch FA long-KV candidate that stays in the q4/q4 coopmat1 path.

## Math / Theory

- E134 FA share: `0.4160`.
- Required local FA speedup to close the full 64k gap alone: about `1.494x`.
- Local FA speedups as stack components:
  - `1.10x` FA local projects about `1.393 TPS`;
  - `1.20x` FA local projects about `1.441 TPS`;
  - `1.35x` FA local projects about `1.503 TPS`.
- Static LDS estimate for `HSK=HSV=256`, f32acc:
  - default `Br16/Bc64`: about `26 KiB`, matching E132 (`26112 B`);
  - candidate `Br32/Bc32`: about `26-27 KiB`, expected to fit;
  - direct `Br32/Bc64`: about `40 KiB`, expected to fail support.

## Implementation Plan

1. Minimal code surface to change: temporary env-gated branch in `get_fa_tuning_params_coopmat1`.
2. Guard rails: default route untouched unless `GGML_VK_FA_CM1_BR32_BC32=1`.
3. Rollback path: revert the small host-side env branch if the pp/resource gate fails.

## Benchmark Plan

- Baseline command: q4/q4 pp7488, FlashAttention on, `b8192/ub1024`, `-mmp 0`.
- Candidate command: same plus `GGML_VK_FA_CM1_BR32_BC32=1`.
- First gate: pipeline resource stats for `flash_attn_f32_f16_aligned_f32accq4_0` and pp7488.
- Full 64k server run only if pp/resource gate is positive.

## Metrics

- prompt eval tok/s
- FA route trace (`Br/Bc`, workgroup size, row split)
- driver pipeline stats (VGPR/SGPR/LDS/scratch)
- 64k real-server prompt eval if candidate survives

## Result

- Outcome: regression; no code kept.
- Delta:
  - baseline default `Br16/Bc64`, f32acc: `971.09 tok/s`, `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch`;
  - candidate `Br32/Bc32`, f32acc: `896.97 tok/s`, `133 VGPR / 83 SGPR / 27136 B LDS / 0 scratch`;
  - companion `Br32/Bc32`, f16acc: `922.22 tok/s`, `134 VGPR / 83 SGPR / 25088 B LDS / 0 scratch`.
- Confidence: high for rejecting this route shape. Route trace confirmed coopmat1 q4/q4 stayed active and used `Br=32,Bc=32,row_split=2,workgroup_size=128`.
- Recommendation: reject and revert the env-gated host branch. Do not pursue larger-`Br` FA in cm1 unless a new design reduces per-row live state; the current shader pays for extra rows with high VGPR pressure before it can benefit from KV reuse.

## Notes

- This is intentionally not a repeat of E129 `Bc=32`: the Br change is the actual route change.
- The f16acc companion was tested because the f32acc candidate looked register-bound. It reduced LDS but did not reduce VGPR, so the root issue is the extra live `Of/Lf/Mf/mask` row state rather than f32 accumulator storage alone.
- No full 64k server run was needed because the short pp/resource gate failed decisively.

## Artifacts

- `build_logs/agent-workload/e142-vulkan-fa-br32bc32-baseline-pp7488.log`
- `build_logs/agent-workload/e142-vulkan-fa-br32bc32-candidate-pp7488.log`
- `build_logs/agent-workload/e142-vulkan-fa-br32bc32-f16acc-pp7488.log`
