# D028 P002 Vulkan 2.4 TPS Target Gate

Date: 2026-05-27

Status: closed and parked on 2026-08-13; no TPS claim and no source change.
The `2.4 TPS` target is retained as a historical topology gate, not an active
project objective.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,ubatch=256,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, `max_tokens=16`, cold-first, no reuse, no v2 prime, thinking on.
- Historical baseline: D012 `d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3`, `2.0013 TPS`, prompt `1053.1067 tok/s`, decode `42.7233 tok/s`, `7970` prompt tokens.
- Retired target: `2.4 TPS` on the same lane.

## Gate Artifact

`scripts/research/vulkan_2p4_target_gate.py` writes:

- `build_logs/agent-workload/d028-vulkan-130k-2p4-target-gate.md`

## Derived Target

| Item | Value |
| --- | ---: |
| Current wall from aggregate TPS | `7.9948 s` |
| Target wall at `2.4 TPS` | `6.6667 s` |
| Required wall speedup | `1.1992x` (`+19.92%`) |
| Current prompt time estimate | `7.5681 s` |
| Current decode time estimate | `0.3745 s` |
| Residual overhead estimate | `0.0522 s` |
| Prompt eval needed if decode/overhead stay flat | `1277.25 tok/s` (`1.2128x`) |

Route-local requirement against the D012 baseline:

| Touched route | Wall share | Local speedup needed | Source |
| --- | ---: | ---: | --- |
| Dense FFN gate/up only | `34.91%` | `1.908x` | D004 corrected dense-FFN route share |
| Dense FFN down only | `24.61%` | `3.077x` | D004 corrected dense-FFN route share |
| Dense FFN gate/up + down | `59.52%` | `1.387x` | D004 corrected dense-FFN route share |
| All Q3_K MUL_MAT | `80.50%` | `1.260x` | D004 corrected all-Q3 route share |
| D012 selected q3quad point Q3_K | `77.65%` | `1.272x` | D009/D012 point trace: `5691.67 / 7330.07 ms` |

## Decision

The `2.4 TPS` target is a new topology target, not a promotion-hardening target.
The previous D012 follow-up plan to harden `bn256`, lowtile3, and q3quad as
defaults is still useful, but it is lower priority than finding another broad
route because hardening does not move TPS by itself.

Gate/up-only FFN fusion is below the new bar: it would need about `1.9x` local
speedup, while D004's generous dual-A model projected only `1.417x` local before
D012 and did not touch down. Down-only is even less realistic at `3.077x` local.

The next Vulkan candidate should therefore touch either:

- the whole dense FFN Q3_K route (`gate/up + SwiGLU + down`), with a design that
  preserves D005 split-K down behavior and handles D007's non-adjacent graph
  surface; or
- enough of all-Q3 prefill to target about `1.26x` local on the same lane.

Do not reopen ubatch, memory priority, output placement, m10240 q3quad inclusion,
lowtile2/4, down split-K 6, MMVQ disable, no-graphics queue, vector-return
q3quad, or adjacent-only whole-FFN matching as first moves toward `2.4 TPS`.

## Historical Reopen Scout

D029-D033 subsequently rejected the nearest non-adjacent whole-FFN, all-Q3,
Q3S, FA-stack and q3-octa families, and the P002 program was explicitly paused.
There is therefore no required follow-up in the active queue.

If the program is explicitly reopened, the first useful scout must be a new
topology rather than shader code for a previously rejected family. It should
quantify:

- exact graph/node contract from D007's `scan_blocks=64` surface;
- temporary activation residency and write/read traffic for the GLU output;
- whether a fused or scheduled route can reduce enough work beyond D012's q3quad
  and GLU fast path to approach `1.387x` local on dense FFN;
- SPIR-V/resource risks before any runtime prototype;
- fail-closed env guard and rollback path.

Stop condition: if the design cannot beat the D012 point/wall ceiling without
extra residency traffic or accumulator/LDS cliffs, move to a broader all-Q3 body
or layout proof instead of implementing another FFN shader.