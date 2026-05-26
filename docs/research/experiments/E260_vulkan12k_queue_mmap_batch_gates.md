# E260 Vulkan 12k No-Code Transfer Gates

## Metadata

- Experiment ID: E260
- Date: 2026-05-26
- Owner: Copilot
- Branch/Commit: local dirty tree after E257/E259; E258 source prototype reverted
- Target lane: Vulkan, Qwen3.6-27B-Q3_K_S, `ctx=12288`, cold/no-reuse/no-prime, thinking on, `quick:triage_diff`, `max_tokens=64`, `spec=none`

## Hypothesis

- Statement: older Vulkan 64k and 12k controls saw occasional benefit from graphics-queue routing, `--no-mmap`, larger prompt batches, or route pivots around f16 support, so the current E257 `b7168/ub1024` 12k profile needed exact-lane transfer checks before more source work.
- Mechanism: graphics queue may improve command scheduling/residency on this AMD driver; `--no-mmap` may reduce paging behavior; `batch=8192` may reduce prompt chunk overhead relative to `b7168`; `GGML_VK_DISABLE_F16=1` may force non-f16acc/non-f16 paths.
- Why now: E259 closed f16 KV and `batch=7680`, but graphics queue/no-mmap and a full `8192` batch had not been measured on the exact E257 dense 27B profile.

## Benchmark Plan

- Baseline: E257 `e257-vulkan12k-shape-b7168-ub1024-r3`, q4/q4 KV, `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s`.
- Candidates:
  - `e260-vulkan12k-graphicsq-b7168-ub1024-r1`: `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `batch=7168`, `ubatch=1024`.
  - `e260-vulkan12k-graphicsq-nommap-b7168-ub1024-r1`: same plus server `--no-mmap`.
  - `e261-vulkan12k-b8192-ub1024-r1`: default queue, `batch=8192`, `ubatch=1024`.
  - `e263-vulkan12k-disable-f16-b7168-ub1024-r1`: `GGML_VK_DISABLE_F16=1`, `batch=7168`, `ubatch=1024`.
- Number of runs: r1 gates only; all candidates were negative enough that no r3 confirmation was warranted.
- Artifacts path: `build_logs/agent-workload/e260-*` and `build_logs/agent-workload/e261-*`.

## Result

| Label | Extra Route | Batch | UBatch | TPS | Prompt tok/s | Decode tok/s | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E257 best | default queue, mmap | 7168 | 1024 | 7.0319 | 999.22 | 40.93 | baseline |
| `e260-vulkan12k-graphicsq-b7168-ub1024-r1` | graphics queue | 7168 | 1024 | 6.8663 | 974.75 | 40.06 | reject |
| `e260-vulkan12k-graphicsq-nommap-b7168-ub1024-r1` | graphics queue + `--no-mmap` | 7168 | 1024 | 6.8743 | 975.84 | 40.10 | reject |
| `e261-vulkan12k-b8192-ub1024-r1` | default queue, mmap | 8192 | 1024 | 6.7312 | 951.24 | 40.10 | reject |
| `e263-vulkan12k-disable-f16-b7168-ub1024-r1` | `GGML_VK_DISABLE_F16=1` | 7168 | 1024 | 5.2700 | 710.49 | 40.92 | reject |

- Outcome: all transfer/no-code gates are negative on the current E257 12k shape.
- Delta vs E257 best: graphics queue `-2.36%` wall, graphics queue + `--no-mmap` `-2.24%`, `batch=8192` `-4.28%`, f16-disable route `-25.06%`.
- Confidence: medium; r1 is sufficient because each candidate is well below the r3 E257 mean and the prompt side regressed clearly.
- Recommendation: keep the GUI preset on q4/q4 `b7168/ub1024`, default queue, mmap. Do not transfer the older 64k graphics-queue/no-mmap profile to this 12k dense 27B lane.

## Notes

- `batch=8192` produced a single `7489/7489` prompt chunk, but prompt eval still fell to `951.24 tok/s`, so fewer chunks did not translate to better wall time.
- Graphics queue and `--no-mmap` both preserved decode around `40 tok/s` but lost prompt throughput, which is the wall-dominant segment of this lane.
- `GGML_VK_DISABLE_F16=1` preserved decode (`40.92 tok/s`) but collapsed prompt eval (`710.49 tok/s`), so a broad f16-disable/f32acc-style pivot is not useful for this lane.
- After E259/E260, nearby no-code transfers are exhausted for the 12k dense Vulkan profile. Further progress needs a source-level Q3_K topology or a separately justified FA shader-body route.
