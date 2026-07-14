# E284: Matched 49K-Context README Lane

Date: 2026-07-14

## Reason

E283 used different long-prompt sizes for Vulkan and ROCm. That made the two
backend rows useful individually but unsuitable for direct comparison. E284
repeats both backends with the exact GUI `Long ctx 49K` contract.

The GUI's old `Long 50K` name referred to the 49,152-token context capacity,
not the actual prompt. With safe fill 0.88, a 2,048-token reserve, and 128 output
tokens, the runner caps the requested 147,456-character repository snapshot to
106,800 characters. The resulting prompt is 31,997 tokens on both backends.

## Fixed Contract

- model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`;
- context: 49,152;
- actual prompt / output: 31,997 / 128 tokens;
- batch / ubatch: 8,192 / 1,024;
- K/V cache: q8_0 / q8_0;
- FlashAttention on, one slot, all model layers offloaded;
- cold prompt, no warmup, no prompt reuse, no prime pass;
- `--cache-ram 0 --ctx-checkpoints 0`;
- seed 42, temperature 0.2, top-p 0.9;
- two runs per backend and mode;
- MTP: n3, backend-resident NextN, 256-token prefill window.

Placement:

- Vulkan: `-dev Vulkan0,Vulkan1 -sm layer -ts 1,1`, output on Vulkan1,
  AMD large-matmul route enabled;
- ROCm: `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, direct peer copy disabled.

## Results

| Backend | Mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| Vulkan | none | 1488.47 | 32.89 | 5.0245 | - |
| Vulkan | MTP n3 | 1519.98 | 41.97 | 5.2899 | 48.70% |
| ROCm | none | 1338.10 | 22.30 | 4.3035 | - |
| ROCm | MTP n3 | 1337.23 | 32.24 | 4.5733 | 50.33% |

MTP deltas:

| Backend | Prompt | Decode | Aggregate |
| --- | ---: | ---: | ---: |
| Vulkan | +2.12% | +27.59% | +5.28% |
| ROCm | -0.07% | +44.61% | +6.27% |

## Artifact Labels

- `e284-readme-long49ctx-vulkan-none-dev01-r2`;
- `e284-readme-long49ctx-vulkan-mtp-n3-dev01-r2`;
- `e284-readme-long49ctx-rocm-none-dev10-r2`;
- `e284-readme-long49ctx-rocm-mtp-n3-dev10-r2`.

These rows supersede the unequal E283 long-prompt rows in the project README.
