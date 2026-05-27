# D030 P002 Vulkan All-Q3 2.4 TPS Gate

Date: 2026-05-27

Status: design gate; rejects nearby old all-Q3 body/layout families as first
routes to `2.4 TPS`.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Baseline: D012 `2.0013 TPS`, prompt `1053.1067 tok/s`, decode `42.7233 tok/s`.
- Target: `2.4 TPS`, D028 required wall speedup `1.1992x`.

## Gate Artifact

`scripts/research/vulkan_allq3_2p4_gate.py` writes:

- `build_logs/agent-workload/d030-vulkan-allq3-2p4-gate.md`

## Target Math

| Item | Value |
| --- | ---: |
| All-Q3 wall share | `80.50%` |
| D012 selected Q3 wall share | `77.65%` |
| All-Q3 local speedup needed | `1.2600x` |
| D012 selected Q3 local speedup needed | `1.2722x` |
| D009/D012 all-Q3 point | `5691.67 ms` |
| All-Q3 target point time | `4517.10 ms` |
| All-Q3 point savings needed | `1174.57 ms` |
| Selected-Q3 point savings needed | `1217.68 ms` |

## Candidate Fence

| Candidate family | Evidence | Decision |
| --- | --- | --- |
| Extend current q3quad/tile stack | D008 to D009 all-Q3 point improved `5876.48 -> 5691.67 ms` (`1.0325x`, `184.81 ms` saved) | already in D012; still needs about `1175 ms` more all-Q3 savings |
| Scale-only metadata/helper reuse | E088 pair-scale `-0.20%`; E080 unsigned scale `-0.44%`; E101 scale-int negative | reject scale-only or expression-only probes |
| Persistent signed-nibble Q3_K layout | S001 static SPIR-V improved, but runtime hot5 `1.5186` vs control `1.5798` (`0.9613x`), and all-Q3 storage failed the 130k fit check | reject as-is; do not reopen without a different compute body |
| Q8_1 / integer-dot prefill route | E099 forced `matmul_q3_k_q8_1_l`: pp256 `225.08`, `143 VGPR / 28672 B LDS`; D006 no-coopmat/Q8 prompt about `400 tok/s` | reject Q8_1/int-dot for P002 prompt speed |
| Persistent fp16/int8 expanded layouts | P002 layout gate: FFN fp16 `+25.03 GiB`, FFN int8 `+9.09 GiB`; D026 all-Q3 int8+fp16 expansion `+15.42 GiB` | reject broad expanded layouts |
| Neighbor tile/resource tweaks | D012 already includes `bn256 + lowtile3 + q3quad`; lowtile2/4, down split-K 6, m10240 inclusion, vector-return q3quad, and MMVQ-disable were measured negative | reject as first `2.4 TPS` route |

## Decision

No old all-Q3 body/layout family clears the D028 bar. The next useful D031
candidate needs a new Q3_K compute body or layout-body pair that removes actual
matmul/dequant work, not a storage-only format change, scale/helper expression
rewrite, Q8_1 route switch, or neighboring tile shape.

Acceptance bar for D031:

- explain how it can remove roughly `1175 ms` from the D012 all-Q3 point route
  or show a smaller route with a correspondingly larger local speedup;
- preserve the D012 `bn256 + lowtile3 + q3quad + GLU` baseline stack unless it
  explicitly replaces that body;
- include a residency model for 130k/16 GB before runtime wiring;
- start with a static resource/SPIR-V or standalone point proof before full
  server A/B.