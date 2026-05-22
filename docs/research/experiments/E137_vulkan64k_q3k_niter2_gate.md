# E137 Vulkan 64k Q3_K Dual-N Gate

## Metadata

- Experiment ID: E137
- Date: 2026-05-22
- Owner: Codex
- Branch/Commit: master after E136 (`dc92f278b`)
- Hypothesis ID: H38 / H31
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan `ctx=65536`, q4/q4 KV, FlashAttention on, no reuse

## Hypothesis

- Statement: processing two adjacent `BN=128` N-blocks inside one large Q3_K matmul workgroup might reduce repeated A-side Q3_K dequant for hot `N=1024` prefill shapes.
- Mechanism: E133 shows the dominant Vulkan Q3_K shapes use `N=1024`, which means eight `BN=128` workgroups repeat the same A tile along N. A dual-N route can load/dequant A once, stream two B tiles through the same LDS buffer, and keep two accumulator sets.
- Why now: E136 rejected FFN gate/up launch fusion as a solo route because it does not reduce A-side work. Dual-N is a more direct test of the repeated-A-dequant route class.

## Implementation Probe

Temporary, reverted patch:

- added specialization constant `NITER=2` to `mul_mm.comp`;
- changed large AMD matmul variant `GGML_VK_AMD_LARGE_MATMUL_VARIANT=niter2` to dispatch Y with denom `256`;
- accumulated two adjacent `BN=128` output tiles per workgroup;
- added a generic non-`MUL_MAT_ID` B-load guard for the partial second N tile.

The patch compiled successfully and did not introduce scratch memory, but it was reverted after the gate.

## Benchmark Plan

Short pp gate only:

```powershell
$env:GGML_VK_PIPELINE_STATS='matmul_q3_k'
.\build-vulkan\bin\llama-bench.exe -m models\Qwen3.6-27B-Q3_K_S.gguf -p 7488 -n 0 -r 1 --no-warmup -b 4096 -ub 1024 -ctk q4_0 -ctv q4_0 -ngl 999 -fa 1
```

Candidate:

```powershell
$env:GGML_VK_AMD_LARGE_MATMUL_VARIANT='niter2'
$env:GGML_VK_PIPELINE_STATS='matmul_q3_k'
.\build-vulkan\bin\llama-bench.exe -m models\Qwen3.6-27B-Q3_K_S.gguf -p 7488 -n 0 -r 1 --no-warmup -b 4096 -ub 1024 -ctk q4_0 -ctv q4_0 -ngl 999 -fa 1
```

## Result

| Build state | Variant | pp7488 | VGPR | SGPR | LDS | Scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| temporary probe source | default `NITER=1` with generic B guard | `858.83` | `95` | `45` | `20480 B` | `0` |
| temporary probe source | `niter2` | `855.29` | `120` | `45` | `20480 B` | `0` |
| clean restored source | accepted default | `974.92` | `113` | `45` | `20480 B` | `0` |

Outcome: regression/reject. No full 64k real-server A/B was run because the shader gate failed before the lane benchmark.

## Interpretation

- The route class is still relevant, but this concrete shape is wrong. It avoids extra LDS and scratch, but the second accumulator set pushes the candidate to `120 VGPR`, while the clean accepted route is already register-heavy.
- The generic B-load guard is itself a default-route hazard. Even with `NITER=1`, it changed the shader fingerprint and dropped pp7488 to `858.83`, so future structural variants must use a separate guarded shader path or host-level shape restriction rather than perturbing the default source.
- The expected A-dequant reduction was not enough to offset larger live state and lower effective occupancy. The workgroup count reduction may also reduce useful parallelism on `N=1024`.
- This result explains why prior large-`BN` routes like `bn256` looked mathematically attractive but lost in practice: reducing repeated A work is not enough if the route buys it with accumulator pressure or compiler-unfriendly control flow.

## Decision

- Revert the runtime patch.
- Do not pursue `NITER=2` / dual-N in the current `mul_mm.comp` shape.
- Keep the broader goal: reduce repeated Q3_K A-side work, but prefer backend-private Q3_K repack/layout or a separate shape-specific shader that does not double accumulator footprint.
- Workflow correction: any future whole-route Q3_K candidate must report a clean-restored default control, a candidate resource profile, and a reason why the default shader fingerprint is untouched.

## Artifacts

- `build_logs/agent-workload/e137-vulkan-q3k-niter2-baseline-pp7488.log`
- `build_logs/agent-workload/e137-vulkan-q3k-niter2-candidate-pp7488.log`
- `build_logs/agent-workload/e137-vulkan-q3k-niter2-clean-restored-pp7488.log`
