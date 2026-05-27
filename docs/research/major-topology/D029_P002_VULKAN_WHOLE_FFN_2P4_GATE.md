# D029 P002 Vulkan Whole-FFN 2.4 TPS Gate

Date: 2026-05-27

Status: design gate; rejects activation-only and naive streaming whole-FFN routes.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Baseline: D012 `2.0013 TPS`, prompt `1053.1067 tok/s`, decode `42.7233 tok/s`.
- Target: `2.4 TPS`, D028 required wall speedup `1.1992x`.

## Gate Artifact

`scripts/research/vulkan_ffn_wholeblock_2p4_gate.py` writes:

- `build_logs/agent-workload/d029-vulkan-whole-ffn-2p4-gate.md`

## Inputs

The model uses D009/D012 point-route timing for the D012-family q3quad stack:

| Item | Value |
| --- | ---: |
| Dense FFN route share | `59.52%` |
| Required dense-FFN local speedup for `2.4 TPS` | `1.3872x` |
| Gate/up Q3_K point | `2759.96 ms` |
| Down Q3_K point | `1417.34 ms` |
| Dense FFN point | `4177.30 ms` |
| Dense FFN target point time | `3011.38 ms` |
| Required dense FFN point savings | `1165.92 ms` |

## Route Models

| Route model | Lower bound / saving | Local signal | Decision |
| --- | ---: | ---: | --- |
| Activation-only whole-FFN fusion | saves at most `2.09 GiB` hidden write/read traffic across the active prefill graph | needs `1165.9 ms` dense-FFN savings | insufficient ceiling |
| Full streaming with gate/up recompute per `64` down rows | `222214.1 ms` lower bound | `0.0188x` local | blocked by recompute |
| Full streaming with partial output per `128` hidden rows | `83.67 GiB` partial output R/W across the active prefill graph | bandwidth-only gate | blocked by output traffic |

Traffic sketch:

- GLU hidden materialization write+read is `34.00 MiB/layer`, `2.09 GiB` across
  `63` active prefill layers.
- Hidden-tile partial output R/W at `hidden_tile=128` is `1.33 GiB/layer`,
  `83.67 GiB` across `63` active prefill layers.

## Decision

Reject activation-only whole-FFN fusion as a route to `2.4 TPS`. D007 proves the
non-adjacent graph surface exists, but simply avoiding the GLU hidden
materialization cannot produce the required `1.387x` dense-FFN local speedup.

Reject naive full-FFN streaming too. Without a new cross-down-row hidden-sharing
mechanism, it either recomputes gate/up for every down-row tile or writes global
partial outputs for every hidden tile. Both fail before shader code.

Whole-FFN work is only worth reopening if it reduces Q3_K matmul work itself or
becomes part of a broader all-Q3 dataflow. The next Vulkan speed candidate should
therefore move away from launch/activation-only FFN fusion and toward an all-Q3
body/layout proof that can plausibly deliver about `1.26x` local on the D012
baseline.