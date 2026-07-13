# D080 P003 Vulkan Layer Stage Balance

## Decision

- Status: keep `-ts 5,6` as the P003 long-prompt baseline.
- Mechanism: Vulkan1 owns the large output tensor but equal `1,1` also gave it half the layers. Moving about three layers to Vulkan0 balances model plus layer-local context and improves pipeline throughput.
- Stop rule: no further tensor-ratio sweep; the remaining target gap is code/topology work.

## Fixed Lane

- Non-MTP `Qwen3.6-27B-Q3_K_S.gguf`, dual Vulkan, output on Vulkan1.
- 56,456 prompt tokens, `ctx=131072`, `b8192/ub1024`, q8/q8 KV.
- FlashAttention, `spec=none`, no warmup/reuse/prime, thinking on.

## Results

| Split | Prompt tok/s | Decode tok/s | Decision |
| --- | ---: | ---: | --- |
| `1,1` | 1276.93 | 14.32 | control |
| `7,9` | 1303.06 | 14.97 | positive but slightly overbalanced |
| `5,6` r1 | 1350.92 | 14.89 | keep candidate |
| `5,6` r3 cold run 1 | 1350.01 | 15.17 | confirmed cold baseline |
| `5,6` r3 mean | 1327.82 | 15.57 | stable, zero errors |

Cold-first delta: `1350.01 / 1276.93 = 1.0572x` (`+5.72%`). The new distance to 2000 is `1.4815x`.

## Residency Evidence

Equal split model buffers were Vulkan1 6384.67 MiB and Vulkan0 5049.53 MiB. At `5,6`:

- Vulkan1: model 5898.30 MiB, context 1975 MiB, accounted self 7881 MiB.
- Vulkan0: model 5535.90 MiB, context 2525 MiB, accounted self 8071 MiB.

The stage footprint difference fell to about 190 MiB without moving all KV to one device or entering a tensor-parallel path.

## Recommendation

Use `-dev Vulkan1,Vulkan0 -sm layer -ts 5,6` plus `LLAMA_OUTPUT_DEVICE=Vulkan1` for P003 comparisons. Do not generalize this ratio to other models or context formats without measuring their layer/output balance.
