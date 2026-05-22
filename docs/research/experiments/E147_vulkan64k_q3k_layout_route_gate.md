# E147 Vulkan 64k Q3_K Layout Route Gate

## Metadata

- Experiment ID: E147
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E146 (`faa1062a3`)
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: after the current Q3_K tile families failed, a backend-private Q3_K layout/repack branch might still improve the hot large-prefill route by reducing repeated A-side decode cost without larger tiles, fp16 temp sync, or extra split/reduce dispatches.
- Mechanism: the hot FFN shapes repeatedly consume the same Q3_K A tiles across `N=1024`. A route that either truly reuses A work or makes Q3_K decode much cheaper could move the `MUL_MAT q3_K` share without changing the rest of the graph.
- Why now: E137 rejected dual-N accumulators, E139 rejected per-node fp16 predequant, E143/E146 rejected larger N/M tiles, and E144 rejected BK shrink. The remaining Q3_K path has to be a layout/topology branch, not another local retune.

## Math / Theory

- E134 Q3_K share proxy: `0.5228`; Q3_K alone needs `1.3573x` local speedup to close the Vulkan-vs-ROCm 64k wall.
- Hot FFN proxy from the GGUF-backed route gate:

| shape | count | base workgroups | base A pair-dequants | N-reuse2 A pairs | N-reuse4 A pairs |
|---|---:|---:|---:|---:|---:|
| `17408x1024x5120` gate/up | 2 | 2176 | 713,031,680 | 356,515,840 | 178,257,920 |
| `5120x1024x17408` down | 1 | 320 | 356,515,840 | 178,257,920 | 89,128,960 |
| total hot FFN proxy |  | 2496 | 1,069,547,520 | 534,773,760 | 267,386,880 |

- Correction to the tempting route idea: a single-accumulator sequential N loop cannot reduce A-dequant work. To reuse A across N blocks, partial sums for every N block must stay alive across the full K loop, which means multiple accumulator sets like E137 or a global partial/reduce route.
- Memory gate on the real `Qwen3.6-27B-Q3_K_S.gguf`:

| Tensor group | Current Q3 bytes | Persistent fp16 delta | Persistent int8 delta | Persistent signed-nibble delta |
|---|---:|---:|---:|---:|
| FFN all (`gate/up/down`, 192 tensors) | 6.85 GiB | +25.03 GiB | +9.09 GiB | +1.12 GiB |
| FFN gate/up only | 4.57 GiB | +16.68 GiB | +6.06 GiB | +765.00 MiB |
| FFN down only | 2.28 GiB | +8.34 GiB | +3.03 GiB | +382.50 MiB |
| all Q3_K tensors | 9.64 GiB | +35.23 GiB | +12.79 GiB | +1.58 GiB |

- Speedup model with no speculative decode: local Q3_K/layout gains of `1.05x`, `1.08x`, `1.10x`, `1.20x`, and `1.3573x` project about `1.0255x`, `1.040x`, `1.05x`, `1.10x`, and `1.1596x` wall speed respectively.

## Gate Evidence

- `python scripts\research\formula_sanity_checks.py`: passed.
- `python scripts\research\vulkan_q3k_layout_route_gate.py --model models\Qwen3.6-27B-Q3_K_S.gguf`: produced the memory and A-reuse table above.
- `python scripts\research\speedup_model.py --baseline-tps 1.3406 --prefill-share 0.5228 --draft-len 1 --accept-rate 0 --spec-overhead 0 --sweep-flash 1.05,1.08,1.10,1.20,1.3573`: confirms small local Q3_K wins are not target-closing.
- `python scripts\research\required_acceptance.py --target-wall 1.1596 --draft-len 4 ...`: protocol control only; speculative acceptance is not the target of this route.
- SPIR-V opcode gate:
  - current Q3_K aligned cm1 shader: `247 OpLoad`, `84 OpStore`, `81 OpIMul`, `28 OpUDiv`, `13 OpUMod`, `4 OpShiftRightLogical`, `2 OpShiftLeftLogical`, `4 OpBitwiseAnd`;
  - f16 aligned cm1 shader: `219 OpLoad`, `74 OpStore`, `73 OpIMul`, `20 OpUDiv`, `7 OpUMod`;
  - Q4_K aligned cm1 shader is not obviously lighter than Q3_K (`273 OpLoad`, `97 OpStore`, similar integer ops).

## Result

- Outcome: reject broad persistent fp16/int8 and reject single-accumulator sequential-N analytically. Do not start a signed-nibble implementation yet.
- Delta: no runtime TPS claim; this is a design gate.
- Confidence: high for memory rejections, medium for signed-nibble low-ceiling warning. SPIR-V counts are static, but they align with E088/E090 negative runtime probes.
- Recommendation:
  - Do not build persistent fp16/int8 FFN layout for the 16 GiB 64k lane.
  - Do not implement sequential-N unless it includes a new partial-sum topology and a proof it avoids E137 accumulator pressure.
  - Treat signed-nibble Q3_K layout as only a narrow research branch. It is memory-plausible, but likely low-ceiling unless a later instruction/resource gate proves bit unpack dominates more than the current SPIR-V and E088/E090 suggest.
  - Shift the next complex Vulkan effort toward q4 FlashAttention long-KV shader-body/per-tail work, or a genuinely new Q3_K topology that removes A pair count without fp16 temp, accumulator blowup, or near-limit LDS.

## Workflow Correction

- `vulkan_q3k_prebuild_gate.py` now ignores aliases in a local negative context such as `no larger tile`, so descriptions of new topology candidates do not accidentally match rejected large-tile priors.
- The gate still flags E139 as the closest negative analogue for persistent/repack wording. That is intentional: a future layout candidate must explicitly show why it avoids per-node fp16 temp traffic and sync.

## Artifacts

- `build_logs/agent-workload/e147-vulkan-q3k-layout-route-gate.md`
- `build_logs/agent-workload/e147-vulkan-q3k-layout-speedup-model.md`
- `build_logs/agent-workload/e147-required-acceptance-control.md`
- `build_logs/agent-workload/e147-vulkan-q3k-layout-prebuild-gate.md`
- `build_logs/agent-workload/e147-spirv-q3k-current-aligned-cm1.md`
- `build_logs/agent-workload/e147-spirv-q3k-vs-q4k-current.md`
- `build_logs/agent-workload/e147-spirv-q3k-vs-f16-current.md`
- `scripts/research/vulkan_q3k_layout_route_gate.py`
- `scripts/research/vulkan_q3k_prebuild_gate.py`
