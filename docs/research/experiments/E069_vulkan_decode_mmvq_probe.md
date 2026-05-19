# E069 - Vulkan decode MMVQ probe

Date: 2026-05-19

## Hypothesis

Vulkan already beats ROCm in decode on the active Qwen3.6-27B-Q3_K_S lane, but pure token generation may still have headroom in the integer-dot MMVQ path. The likely target is Q3_K `MUL_MAT_VEC` rather than FlashAttention, GDN, or large prefill matmul tiles.

## Contract

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- Backend: `build-vulkan/bin/llama-server.exe`
- Env: `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`, `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`
- Task: `quick/triage_diff`
- Decode-biased mode: short prompt, `max_tokens=128` or `256`, `ctx=12288`, `b=4096`, `ub=1024`, `q4_0/q4_0`, FlashAttention on, `spec=none`, no reuse, thinking on

## Profiling Result

Fresh E068 decode-biased baseline with 256 generated tokens:

| Config | Runs | Aggregate TPS | Prompt eval TPS | Decode eval TPS |
| --- | ---: | ---: | ---: | ---: |
| E068 Vulkan decode-biased | 1 | `39.1935` | `709.41` | `40.75` |

Perf logger is intrusive and dropped the measured run to `29.3922` aggregate TPS, so it is diagnostic only. The final per-token decode blocks showed the hot centers:

| Operation | Approx per-token total |
| --- | ---: |
| `MUL_MAT_VEC q3_K m=17408 n=1 k=5120` | `8.7-9.1 ms` |
| `MUL_MAT_ADD MUL_MAT_VEC q3_K m=5120 n=1 k=17408` | `4.7-4.9 ms` |
| `MUL_MAT_VEC q6_K m=248320 n=1 k=5120` | `1.66-1.68 ms` |
| `MUL_MAT_VEC q4_K m=5120 n=1 k=6144` | `1.49-1.51 ms` |
| `GATED_DELTA_NET` | `0.32-0.34 ms` |
| `FLASH_ATTN_EXT` | `0.24-0.27 ms` |

Interpretation: the remaining decode target is Q3_K MMVQ shader/layout. FA/GDN are too small for a first decode-focused lever on this lane.

## No-Code Knob Screen

All runs use 128 generated tokens.

| Config | Aggregate TPS | Decision |
| --- | ---: | --- |
| E068 baseline | `37.76` | reference |
| `GGML_VK_DISABLE_FUSION=1` | `37.63` | reject/noise-negative |
| `GGML_VK_FORCE_MMVQ=1` | `37.86` | neutral/noise |
| `GGML_VK_DISABLE_MMVQ=1` | `34.25` | reject |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | `33.92` | reject |

## Code Probes

Temporary code probes were implemented, built, measured, and then reverted because they did not beat baseline.

| Probe | Aggregate TPS | Result |
| --- | ---: | --- |
| Baseline after probe scaffold | `37.87` | reference |
| Force DMMV large workgroup | `33.16` | reject |
| K-quant integer MMVQ rows-per-workgroup `rm=2` | `35.56` | reject |
| K-quant integer MMVQ rows-per-workgroup `rm=4` | `32.72` | reject |
| Force DMMV large + `rm=2` | `33.37` | reject |
| Q3_K packed32 scale load probe | `37.96` r1 / `37.96` r3 | reject as noise; baseline r3 `37.91` aggregate, baseline median `37.98` |

## Decision

Reject all E069 implementation probes. No speed code kept.

The decode path has real theoretical headroom because Q3_K MMVQ dominates per-token time, but the cheap knobs and scale-load rewrite are not the right levers. The next viable implementation should be a deeper Q3_K MMVQ specialization, not workgroup size forcing or simple packed scale loads.

## Artifacts

- `build_logs/agent-workload/e069-vulkan-decode-e068-r1.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-e068-perflog-r1.server.log`
- `build_logs/agent-workload/e069-vulkan-decode-base128-r1.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-disablefusion128-r1.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-forcemmvq128-r1.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-disablemmvq128-r1.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-disableintdot128-r1.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-postknob-base128-r3.diagnostics.md`
- `build_logs/agent-workload/e069-vulkan-decode-q3scale-packed32-128-r3.diagnostics.md`
