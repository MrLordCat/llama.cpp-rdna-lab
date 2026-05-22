# E145 Vulkan 64k FA D-Split Route Gate

## Metadata

- Experiment ID: E145
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E144 (`5e237124c`)
- Hypothesis ID: H38 / H05
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: the active coopmat1 q4 FlashAttention route may improve if `D_split` is retuned for HSK/HSV 256, changing the per-thread balance between output accumulators and score columns while keeping the same q4/q4 single-dispatch route.
- Mechanism: default `D_split=8` gives `rows_per_thread=4`, `cols_per_thread=8`, `HSV_per_thread=32`, and `Of=32 vec4` live output elements. `D_split=16` halves the output vector state to `Of=16 vec4` but doubles `Sf` score-column state to `64` scalar cells; `D_split=4` does the opposite (`Of=64 vec4`, `Sf=16`). This is a route-resource probe, not a nearby `Bc/Br` repeat.
- Why now: E129/E142 rejected simple FA tile retunes, E131/E132 mapped resources for the current route, and E138 showed existing split-K is the wrong topology. `D_split` is the next single-dispatch cm1 branch that can change register pressure without adding reduce dispatches or KV dtype changes.

## Math / Theory

- E134 FA share: `0.4160` of the Vulkan 64k traced route.
- FA alone needs about `1.494x` local speedup to close the full Vulkan-vs-ROCm 64k gap.
- `speedup_model.py` projects:
  - `1.05x` local FA -> about `1.37 TPS`;
  - `1.10x` local FA -> about `1.39 TPS`;
  - `1.20x` local FA -> about `1.44 TPS`;
  - `1.494x` local FA -> about `1.55 TPS`.
- Failure conditions:
  - `D_split=16` can lose if extra score-column state or lower columns-per-iteration hurts memory/coalescing more than the reduced output state helps.
  - `D_split=4` can lose if output accumulator live state dominates VGPR pressure.
  - Any fallback to scalar or changed `Br/Bc` invalidates the comparison.

## Implementation Plan

1. Minimal code surface to change: temporary env-gated branch in `get_fa_tuning_params_coopmat1`.
2. Guard rails: default route untouched unless `GGML_VK_FA_CM1_D_SPLIT` is set to `4`, `8`, or `16`.
3. Rollback path: revert the host-side env branch if the pp/resource gate fails.

## Benchmark Plan

- Baseline command: q4/q4 pp7488, FlashAttention on, `b8192/ub1024`, `-mmp 0`, route trace and FA pipeline stats.
- Candidate commands: same with `GGML_VK_FA_CM1_D_SPLIT=4` and `GGML_VK_FA_CM1_D_SPLIT=16`.
- First gate: pipeline resource stats for `flash_attn_f32_f16_aligned_f32accq4_0` and pp7488.
- Full 64k server run only if pp/resource gate is positive.

## Metrics

- prompt eval tok/s
- FA route trace (`D_split`, `Br/Bc`, row split, path)
- driver pipeline stats (VGPR/SGPR/LDS/scratch)
- 64k real-server prompt eval if candidate survives

## Analytic Gate

Commands run before code:

```powershell
python scripts\research\formula_sanity_checks.py
python scripts\research\speedup_model.py --baseline-tps 1.3406 --prefill-share 0.4160 --flash-prefill-speedup 1.10 --decode-kernel-speedup 1.0 --draft-len 1 --accept-rate 0 --spec-overhead 0 --sweep-flash 1.03,1.05,1.10,1.20,1.35,1.494 --sweep-accept 0
python scripts\research\required_acceptance.py --target-wall 1.1596 --draft-len 4 --prefill-share 0.4160 --prefill-speedup 1.10 --decode-kernel-speedup 1.0 --spec-overhead 0.0
```

- `formula_sanity_checks.py`: passed.
- `1.10x` local FA model: `1.0393x` wall, projected `1.3933 TPS`.
- Sensitivity grid: `1.03x -> 1.36`, `1.05x -> 1.37`, `1.10x -> 1.39`, `1.20x -> 1.44`, `1.35x -> 1.50`, `1.494x -> 1.55 TPS`.

## Result

- Outcome: regression; no code kept.
- Delta:
  - baseline `D_split=8`: `978.88 tok/s`, route `Br16/Bc64,row_split=4`, resources `98 VGPR / 76 SGPR / 26112 B LDS / 0 scratch`;
  - candidate `D_split=4`: `953.24 tok/s`, same visible resources, `-2.62%`;
  - candidate `D_split=16`: `951.54 tok/s`, same visible resources, `-2.79%`.
- Confidence: high for rejecting simple `D_split` retuning. Route trace confirmed coopmat1 q4/q4 stayed active and only `D_split` changed.
- Recommendation: reject and revert the env-gated host branch. `D_split` changes alter internal per-thread work distribution, but the current shader does not convert lower output state into lower reported VGPR/LDS, and both directions lose runtime. Future FA work needs body-level shader changes or per-KV tail instrumentation, not another tuning-parameter flip.

## Notes

- This is deliberately not another `Bc`/`Br` sweep. It keeps `Br16/Bc64` and changes the internal distribution of head-dimension work.
- The failed prediction is informative: driver pipeline statistics stayed identical across `D_split=4/8/16`, so VGPR/LDS stats alone are insufficient for this FA subroute. The throughput loss likely comes from the changed column iteration / score buffering cadence and instruction scheduling inside the shader.

## Artifacts

- `build_logs/agent-workload/e145-vulkan-fa-dsplit8-baseline-pp7488.log`
- `build_logs/agent-workload/e145-vulkan-fa-dsplit4-pp7488.log`
- `build_logs/agent-workload/e145-vulkan-fa-dsplit16-pp7488.log`
