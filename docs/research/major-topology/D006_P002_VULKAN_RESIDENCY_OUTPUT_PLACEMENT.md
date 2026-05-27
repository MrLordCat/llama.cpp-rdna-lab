# D006 P002 Vulkan Residency Output Placement

Date: 2026-05-26

Status: closed as diagnostic evidence; not promoted as a speed route.

## Lane

- Backend: Vulkan, `build-vulkan`.
- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Contract: `ctx=131072,batch=512,q4_0/q4_0,FlashAttention,spec=none,--no-mmap`.
- Workload: `quick:triage_diff`, `real-context-chars=24576`, cold-first, no reuse, no v2 prime, thinking on.
- Current kept baseline: D005 default split-K, `d005-vulkan-default-splitk-confirm3`, `1.7898 TPS`, prompt `934.81 tok/s`, decode `43.59 tok/s`.
- D006 output-relief probes used `ubatch=512` where noted. They are not a new baseline because the route changes output placement and decode behavior.

## Trigger

After D005, the GUI `--no-mmap` mismatch was fixed and the user restored the GPU power limit to `+10%`. A fresh default/full-output run still fell into a severe 130k residency cliff:

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Vulkan model | Vulkan host model | Compute | Graph splits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D005 default after power restore | `d006-vulkan-130k-d005-default-powerplus-r1` | `512/256` | `0.3252` | `163.67` | `41.65` | `11434.19 MiB` | `521.00 MiB` | `228.27 MiB` | `2` |

That does not invalidate the earlier D005 r3 fast-corridor baseline, but it proves the current 130k Vulkan lane is sensitive to output-layer residency/lifetime. At `ctx=131072`, this sensitivity is part of the target problem, not measurement noise.

## Source Probes

D006 added two default-off diagnostics in `src/llama-model.cpp`:

- `LLAMA_NO_OUTPUT_OFFLOAD`: keep the output layer on CPU and use a CPU output buffer.
- `LLAMA_OUTPUT_HOST_GPU_DEV`: keep output weights in host memory while the logical output device remains the first GPU device, producing a `Vulkan_Host` output buffer.

Measured full runs after the power-limit correction:

| Variant | Label | B/UB | TPS | Prompt tok/s | Decode tok/s | Output buffer | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| CPU output layer | `d006-vulkan-130k-no-output-ub512-powerplus-r1` | `512/512` | `1.7551` | `955.88` | `22.13` | `CPU` | prompt recovered, decode penalty blocks promotion |
| Host output, GPU device | `d006-vulkan-130k-output-host-gpudev-ub512-r1` | `512/512` | `1.7769` | `968.32` | `22.28` | `Vulkan_Host` | best current D006 full run, still below D005 r3 baseline and below 2 TPS |

Both output-relief variants moved model residency to roughly `10430.07 MiB` on
`Vulkan0` and `1515.64 MiB` on `Vulkan_Host`, with `456.28 MiB` Vulkan compute
buffer and `3` graph splits. This recovers prompt eval to the fast corridor, but
decode drops to about half the D005 default decode rate because the final LM head
and logits/output path are no longer the same device-local route.

Earlier same-session output-relief probes reached similar or slightly higher one-run values before the power-limit recheck:

| Variant | Label | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| CPU output, `ub512` | `d006-vulkan-130k-no-output-ub512-full-r1` | `1.8018` | `983.73` | `22.07` | no r3; decode penalty remains |
| CPU output + memory-priority diagnostic | `d006-vulkan-130k-no-output-ub512-memprio-full-r1` | `1.8068` | `986.62` | `22.09` | not a speed route; tied within diagnostic noise |
| CPU output + priority split | `d006-vulkan-130k-no-output-ub512-priosplit-full-r1` | `1.8046` | `985.34` | `22.05` | not a speed route |
| CPU output + graphics queue | `d006-vulkan-130k-no-output-ub512-gfxq-full-r1` | `1.7989` | `985.40` | `21.25` | not a speed route |

Point-only `LLAMA_OUTPUT_HOST_GPU_DEV` variants stayed in the same prompt band:

| Variant | Label | Prompt tok/s | Decision |
| --- | --- | ---: | --- |
| `b512/ub512` | `d006-vulkan-130k-output-host-gpudev-ub512-point-r1` | `966.76` | no better than the full-run signal |
| `b1024/ub512` | `d006-vulkan-130k-output-host-gpudev-b1024-ub512-point-r1` | `967.70` | no new route |
| `b1024/ub512`, `wn32+gfxq` | `d006-vulkan-130k-output-host-gpudev-b1024-ub512-wn32-gfxq-point-r1` | `970.04` | point-only tie; no sweep continuation |

These point runs are useful for residency diagnosis only; they are not full workload TPS claims.

## Rejected Branches

| Branch | Result | Decision |
| --- | --- | --- |
| Force `result_output`/LM-head matmul back to Vulkan while output weight is host-resident | about `1.04 TPS`, prompt `679 tok/s`, decode `4.37 tok/s`, compute buffer grew to about `1005 MiB` | reverted; host-output + forced Vulkan matmul is worse on both prompt and decode |
| CPU output override on the first implementation path | `d006-vulkan-130k-override-output-cpu-ub512-full-r1`: `1.5507 TPS`, prompt `850.62`, decode `17.93` | rejected |
| `--no-kv-offload` | `d006-vulkan-130k-no-kv-offload-ub512-full-r1`: `1.3408 TPS`, decode `9.83` | rejected |
| Partial layer offload (`ngl64`) | `d006-vulkan-130k-ngl64-ub512-full-r1`: `0.2982 TPS` | rejected |
| No-coopmat/Q8_1 route | selected `matmul_q3_k_q8_1_m`, prompt about `400 tok/s` | rejected; do not chase no-coopmat Q8 path |
| Q3 helper-only quad dequant rewrite | prompt about `938 tok/s`, below fast corridor | reverted; helper rewrite not enough |
| Low-tile split-K diagnostic | tied or regressed under output relief | keep only as diagnostic if retained; not a speed route |

## Interpretation

D006 proves a real residency lever: moving the output layer out of device-local
VRAM can restore 130k Vulkan prompt eval from the `160-190 tok/s` cliff to about
`956-986 tok/s`. It does not solve the lane because the route pays back in decode
latency and does not beat the D005 kept r3 baseline.

The root problem is therefore not a simple ubatch or launch flag. It is an
end-to-end topology/lifetime issue: the final output layer, graph split plan,
compute buffers, and Q3_K FFN prompt route interact with 16 GB VRAM residency at
130k. Continuing with batch/ubatch/queue sweeps is below the current evidence bar.

## Decision

- Do not promote `LLAMA_NO_OUTPUT_OFFLOAD` or `LLAMA_OUTPUT_HOST_GPU_DEV` as launch defaults.
- Keep D005 `d005-vulkan-default-splitk-confirm3` as the current kept Vulkan 130k speed baseline.
- Treat the output-placement knobs as default-off diagnostics only while the next major-topology design is selected. Before a final commit, either document their diagnostic role in the source-facing docs or remove them if they are not needed by the next topology prototype.
- Do not run more `ubatch`/queue/output-placement sweeps unless a major-topology design note explains the new mechanism.

## Next Required Artifact

The next Vulkan source work should start with a major-topology design note, not another benchmark sweep. The strongest current target remains a Q3_K/whole-FFN route that stacks with D005 and includes an explicit residency model for output/logits buffers:

1. Lock the D005 fast-corridor control and attach route/residency evidence.
2. Model memory traffic and buffer lifetime for the proposed Q3_K/FFN topology.
3. Define prompt and decode correctness invariants before editing shader/runtime code.
4. Build a default-off prototype only after the design/scout gate passes.
5. Run paired cold P002 A/B with the same `max_tokens=16`, no reuse, and no v2 prime contract.