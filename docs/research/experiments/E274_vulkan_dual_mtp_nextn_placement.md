# E274 Vulkan Dual MTP NextN Placement

## Metadata

- Experiment ID: E274
- Date: 2026-07-09
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`
- Target lane: Vulkan dual `Qwen3.6-27B-Q3_K_S_mtp.gguf`, `ctx=131072`, `b512/ub256`, q4/q4 KV, `real-context-chars=152000`, cold-first, no reuse/no prime

## Hypothesis

- Statement: Vulkan dual MTP is slow because the MTP/NextN tensors land on the second physical GPU and pipeline parallelism adds scheduler copies; long prompts also pay too much target NextN extraction with the old 8192-token tail window.
- Mechanism: Move `LLM_TENSOR_NEXTN_*` tensors to the first Vulkan device, disable Vulkan pipeline parallelism for NextN/MTP models, and reduce the default MTP prefill tail from 8192 to 512 tokens.
- Why now: E272/E273 showed Vulkan dual is the stronger no-MTP dual backend, but first MTP smoke showed n1/n2 slower than baseline until manual `-ot blk\.[0-9]+\.nextn\..*=Vulkan0` removed extra scheduler copies.

## Implementation

1. `src/llama-model.cpp`: route `LLM_TENSOR_NEXTN_*` tensors to the first Vulkan device for layer-split multi-GPU models by default. `LLAMA_VK_MTP_NEXTN_MAIN_DEVICE=0` restores old placement.
2. `src/llama-context.cpp`: disable pipeline parallelism for Vulkan NextN/MTP models by default because it produced `sched copies=4`; `LLAMA_VK_MTP_PIPELINE_PARALLEL=1` force-enables the old route.
3. `tools/server/server-context.cpp`: reduce default `LLAMA_SPEC_PREFILL_WINDOW` from 8192 to 512, keeping the env override.

## Results

Baseline on the same MTP GGUF with `--spec-type none`:

- `vulkan-dual-mtpgguf-130k-big-c152k-mt64-none-r1`: aggregate `1.2044 TPS`, prompt `1111.69 tok/s`, decode `27.76 tok/s`.

Pre-fix MTP:

- `vulkan-dual-mtpgguf-130k-big-c152k-mt64-n1-r1`: aggregate `1.0129 TPS`, prompt `932.85 tok/s`, decode `24.47 tok/s`, acceptance `93.75%`.
- `vulkan-dual-mtpgguf-smoke-n2-mt64-r1`: aggregate `11.8719 TPS`, decode `13.06 tok/s`, acceptance `88.89%`.

Manual proof:

- `vulkan-dual-mtpgguf-smoke-n2-nextn0-mt64-r1`: aggregate `38.5356 TPS`, prompt `442.86 tok/s`, decode `50.79 tok/s`, acceptance `88.89%`, `sched copies=1`.
- Same smoke baseline `spec=none`: aggregate `32.9303 TPS`, prompt `729.84 tok/s`, decode `37.94 tok/s`.

Final code/default result:

- `vulkan-dual-mtpgguf-smoke-n2-autonextn0-nopp-mt64-r1`: aggregate `39.1725 TPS`, prompt `460.81 tok/s`, decode `51.69 tok/s`, acceptance `88.89%`.
- `vulkan-dual-mtpgguf-130k-big-c152k-mt64-n2-auto-default512-r1`: aggregate `1.2481 TPS`, prompt `1143.43 tok/s`, decode `34.57 tok/s`, acceptance `62.50%`, `sched copies=1`.

Final long-prompt delta vs same-GGUF `spec=none`:

- Aggregate: `+3.63%`.
- Prompt: `+2.85%`.
- Decode: `+24.53%`.

## Rejected Depths

- `n_max=3` is stable but much slower: `vulkan-dual-mtpgguf-smoke-n3-auto-default512-mt64-r1` aggregate `16.7882 TPS`, decode `18.68 tok/s`, acceptance `90.20%`.
- `n_max=4` still crashes the server process before timings, but the GPU driver remains healthy. Treat as a separate draft-loop/KV rollback bug, not a placement issue.

## Recommendation

Keep the Vulkan NextN placement guard, Vulkan NextN pipeline-parallel guard, and 512-token MTP prefill window default. For this model/backend, recommend `--spec-type draft-mtp --spec-draft-n-max 2`; avoid n3/n4 until the repeated draft-decode path is redesigned.
