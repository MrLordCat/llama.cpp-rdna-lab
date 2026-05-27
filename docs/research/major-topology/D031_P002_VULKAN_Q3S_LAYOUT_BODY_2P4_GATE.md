# D031 P002 Vulkan Q3S Layout-Body 2.4 Gate

Date: 2026-05-27.

## Lane

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Backend: Vulkan on RX 9070 XT.
- Shape: `ctx=131072,b=512,ub=256,q4_0/q4_0,FlashAttention,spec=none,real-context-chars=24576,max_tokens=16`.
- Baseline: D012 `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3`, `2.0013 TPS` r3, `--no-mmap`, `bn256 + lowtile3 + q3quad + GLU`.
- Target: `2.4 TPS` cold-first, same lane.

## Gate Artifact

- Script: `scripts/research/vulkan_q3s_layout_body_2p4_gate.py`.
- Artifact: `build_logs/agent-workload/d031-vulkan-q3s-layout-body-2p4-gate.md`.

## Target Math

D030 established the all-Q3 point budget for `2.4 TPS`:

| Quantity | Value |
| --- | ---: |
| D012 all-Q3 point time | `5691.67 ms` |
| all-Q3 target point time | `4517.10 ms` |
| required all-Q3 point savings | `1174.57 ms` |

## Candidate Checked

D031 checks the last moderately memory-plausible persistent Q3_K layout family:
compact Q3S/signed-nibble values plus predecoded scale forms. This is broader
than signed-nibble-only storage, but still does not change the matrix-core work.

Evidence:

- S001 static scout reduced SPIR-V op count from `1491` to `1375` (`7.78%`) and
  SPIR-V size from `25128` to `23088` bytes.
- An optimistic linear all-Q3 upper bound from that static drop is only about
  `442.87 ms`, far below the `1174.57 ms` required point savings.
- Same-session signed-nibble runtime evidence was negative: `1.5186 TPS` vs
  `1.5798 TPS` control (`0.9613x`).
- D026 residency gate puts compact Q3S at `+2.980 GiB` all-Q3 raw or
  `+4.206 GiB` all-Q3 aligned over runtime padded Q3_K; even FFN-only is
  `+2.117 GiB` raw or `+2.988 GiB` aligned.

## Decision

Reject compact Q3S/signed-nibble plus predecoded-scale layout-body work as the
next Vulkan `2.4 TPS` route. It cannot explain the required point savings, and
the residency cost is too large for a negative/low-ceiling route on the 16 GB
130k lane.

## Next Gate

D032 should move away from layout-only unpack simplification. The next candidate
needs a true Q3_K compute body or compressed-dot route that removes matrix work
itself, while preserving the D012 stack unless it replaces it with a measured
point/resource proof.