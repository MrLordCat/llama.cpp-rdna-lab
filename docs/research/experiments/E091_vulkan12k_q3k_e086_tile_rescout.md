# E091 Vulkan 12k Q3_K E086 Tile Re-Scout

## Metadata

- Experiment ID: E091
- Date: 2026-05-20
- Owner: Copilot
- Branch/Commit: local master after E086 kept and E087-E090 rejected
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, fixed pp7488 prefill gate

## Hypothesis

- Statement: E086's corrected Q3_K loadvec4 may change the best legal AMD large-matmul tile variant.
- Mechanism: A-load grouping affects shared-memory write pressure and may make `WN` variants that were neutral before E086 become useful.

## Scout Results

| Variant | pp7488 r1 tok/s | Decision |
| --- | ---: | --- |
| E086 base | `961.82 ± 25.60` | baseline |
| `wn48` | `976.84` | confirm |
| `wn96` | `975.72` | close backup |
| `block128` | `970.12` | close backup |
| `bm64` | `797.92` | reject |
| `bn64` | `720.07` / `737.21` | reject |
| `bm64-bn64` | `646.10` | reject |
| `block128-bn64` | `736.65` | reject |
| `block128-wm128` | `903.45` | reject |

## Benchmark Plan

1. Confirm `wn48` with fixed pp7488 r3.
2. Validate with 12k prompt-heavy workload r1.
3. Keep as opt-in env/profile only if workload is positive and output is sane.

## Result

- Outcome: downgraded by E093 to `needs-layout-validation`, not an opt-in profile.
- Fixed gate: `wn48` confirmed at `972.31 ± 1.97 tok/s` on pp7488 r3, `+1.09%` vs E086 `961.82 tok/s`.
- Workload validation: 12k prompt-heavy r3 reached `6.7981` aggregate TPS, prompt eval `962.8567 tok/s`, decode `40.0967 tok/s`, prompt tokens `7489`, errors `0`.
- Output sanity: r1 response preview was normal text, not the all-slash corruption seen in the old invalid `wm32-wn32` profile.
- Remaining gap: still below ROCm same-lane control (`7.1936` aggregate TPS, `1129.76` prompt eval TPS). This closes part of the gap but not the target.
- Resource stats: active `matmul_q3_k_f32_f16acc_aligned_l` resources match E086 (`113 VGPR / 45 SGPR / 20480 B LDS`, no scratch), so the win is tile scheduling/geometry rather than lower shader resource use.
- Follow-up correction: E093 static warptile scout marks `wn48` invalid for the current `BN=128` layout (`128 % 48 != 0`). Treat the E091 positive measurement as suspicious/no-promotion unless a separate backend log proves a different active prepared tile. Accepted H31 baseline remains E086 source-only.
- Recommendation: do not use `wn48` as a profile. Future tile env work must pass `scripts/research/vulkan_warptile_static_scout.py` before benchmark claims.

Artifacts:

- `build_logs/agent-workload/e091-vulkan-q3-e086-wn48-pp7488-r3.md`
- `build_logs/agent-workload/e091-vulkan12k-q3-e086-wn48-r1.diagnostics.md`
- `build_logs/agent-workload/e091-vulkan12k-q3-e086-wn48-r3.diagnostics.md`
- `build_logs/agent-workload/e091-vulkan-q3-e086-wn48-pipeline-stats.log`