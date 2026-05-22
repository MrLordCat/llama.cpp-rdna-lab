# E136 Vulkan 64k FFN Route Model

## Metadata

- Experiment ID: E136
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E135
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: the dense FFN `MUL_MAT + MUL_MAT + GLU` branch is real, but a dual-A/same-B fusion may not be enough unless it also reduces repeated Q3_K A-dequant work.
- Mechanism: current gate/up does two Q3_K matmuls. A fused route can reuse the B/activation tile and write the GLU output directly, but it still needs two different Q3_K A weight tiles and two accumulator sets.
- Why now: E135 proved graph matchability. Before writing a large Vulkan shader and host path, we need a resource/ceiling gate for the whole route.

## Tooling

Added:

```powershell
python scripts\research\vulkan_ffn_route_model.py --shape 17408x1024x5120 --baseline-tps 1.3406 --target-tps 1.5545 --wall-share 0.2491
```

Checks:

```powershell
python -m py_compile scripts\research\vulkan_ffn_route_model.py
python scripts\research\formula_sanity_checks.py
python scripts\research\speedup_model.py --baseline-tps 1.3406 --prefill-share 0.2491 --flash-prefill-speedup 1.417 --decode-kernel-speedup 1.0 --draft-len 1 --accept-rate 0 --spec-overhead 0
```

`formula_sanity_checks.py` passed. `speedup_model.py` projected `1.4466 TPS` for the model's base `1.417x` local ceiling.

## Model Result

Target route: FFN gate/up shape `M=17408,N=1024,K=5120`.

Route target math:

- E128 Vulkan best: `1.3406 TPS`;
- E128 ROCm same-lane target: `1.5545 TPS`;
- total required wall speedup: `1.1596x`;
- FFN gate/up Q3_K share: `24.91%`;
- required local speedup if this route must close the gap alone: `2.234x`.

Best base-shape model:

| Variant | Dual-A LDS | Accumulators | Unchanged A LDS proxy | Removable proxy | Local ceiling with A proxy | Wall projection |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `base` | `29696 B` | `16 -> 32` fragments | `2720.0 MiB` | `1700.0 MiB` | `1.417x` | `1.079x` |
| `wn32` | `31744 B` | `8 -> 16` fragments | `2720.0 MiB` | `1700.0 MiB` | `1.417x` | `1.079x` |
| `bk16-wn32` | `19456 B` | `8 -> 16` fragments | `2720.0 MiB` | `1700.0 MiB` | `1.417x` | `1.079x` |
| `bn256` | `40960 B` | `16 -> 32` fragments | `1360.0 MiB` | `1700.0 MiB` | `1.625x` | `1.106x` |

The optimistic memory-only ceiling reaches `2.250x`, but that ignores unchanged A-dequant and coopmat work. The more useful proxy adds unchanged A LDS writes and drops the base local ceiling to `1.417x`. Real speed can be lower because Q3_K decode ALU and coopmat arithmetic are also unchanged.

## Interpretation

- Dual-A/same-B FFN fusion is not enough as the sole H38 fix. Even an optimistic base proxy projects `1.4466 TPS`, below the ROCm `1.5545 TPS` target.
- It can still be a stack component if it compiles without scratch and without severe VGPR pressure, but it is no longer the first speed-only implementation target.
- The actual Q3_K route blocker is repeated A-side Q3_K dequant across N-blocks. Base gate/up has `713.0M` A pair-dequants for the fused pair and the fusion does not reduce that.
- `bn256` is the only listed variant that reduces A-dequant proxy, but E098 already rejected the single-route family and dual-A LDS is above 32 KiB (`40960 B`), so it is not a valid shortcut.
- Resource-safe profiles such as `bk16-wn32` reduce accumulator/LDS risk but double K-block/barrier count and do not reduce A-dequant. Build only if a shader prototype proves unexpectedly low VGPR and no scratch.

## Decision

- Outcome: diagnostic keep; no speed claim.
- Decision: do not spend the next implementation pass on a full host-integrated FFN fusion unless it is paired with an A-dequant/layout mechanism. The next complex Q3_K branch should target repeated A-dequant across N-blocks or backend-private Q3_K layout/repack. FA long-KV remains the co-primary branch because a modest Q3+FA stack only needs about `1.172x` local across both.
- Workflow correction: graph proof is necessary but not sufficient. Future FFN route candidates need all three gates: graph match, resource proof, and A-side work reduction proof.

## Artifacts

- `scripts/research/vulkan_ffn_route_model.py`
- `build_logs/agent-workload/e136-vulkan64k-ffn-route-model.md`
- `docs/research/experiments/E136_vulkan64k_ffn_route_model.md`
