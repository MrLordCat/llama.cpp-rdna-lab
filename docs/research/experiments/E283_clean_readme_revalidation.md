# E283: Clean README Revalidation

Date: 2026-07-14

> The short lanes remain current. The unequal Vulkan/ROCm long rows below were
> superseded by the matched E284 49K-context lane.

## Goal

Rebuild both production backends and replace the README headline numbers with
matched cold runs collected without a foreground GPU workload.

## Locked Configuration

- model: `models/Qwen3.6-27B-Q3_K_S_mtp.gguf`;
- one server slot, FlashAttention enabled;
- `batch=8192`, `ubatch=1024`;
- q8_0 K/V cache;
- prompt cache and context checkpoints disabled;
- repository-snapshot prompt, thinking enabled, temperature 0.2;
- `spec=none` or `draft-mtp`, `n_max=3`;
- no prime pass and no hard task timeout.

Vulkan used `-dev Vulkan0,Vulkan1 -sm layer -ts 1,1` with
`LLAMA_OUTPUT_DEVICE=Vulkan1`. ROCm used
`-dev ROCm1,ROCm0 -sm layer -ts 1,1`.

## Results

| Backend | Lane | Mode | Prompt / output | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Vulkan | short, r3 | none | 7,842 / 128 | 1783.49 | 38.17 | 16.42 | - |
| Vulkan | short, r3 | MTP n3 | 7,842 / 128 | 1724.73 | 51.82 | 17.99 | 60.05% |
| Vulkan | long, r2 | none | 29,540 / 128 | 1536.99 | 34.49 | 5.56 | - |
| Vulkan | long, r2 | MTP n3 | 29,540 / 128 | 1557.06 | 42.82 | 5.81 | 48.70% |
| ROCm | short, r3 | none | 7,729 / 256 | 1706.57 | 25.74 | 17.61 | - |
| ROCm | short, r3 | MTP n3 | 7,729 / 256 | 1656.53 | 34.95 | 21.26 | 44.65% |
| ROCm | long, r1 | none | 43,125 / 128 | 1183.46 | 20.95 | 3.00 | - |
| ROCm | long, r1 | MTP n3 | 43,125 / 128 | 1174.33 | 28.83 | 3.10 | 48.39% |

Matched MTP deltas:

| Lane | Prompt | Decode | Aggregate |
| --- | ---: | ---: | ---: |
| Vulkan short | -3.29% | +35.78% | +9.55% |
| Vulkan long | +1.31% | +24.15% | +4.45% |
| ROCm short | -2.93% | +35.76% | +20.71% |
| ROCm long | -0.77% | +37.61% | +3.36% |

## ROCm Startup Fix

The first ROCm MTP validation aborted immediately after speculative-context
initialization at `ggml_backend_dev_name()` with a null device. The custom
Vulkan MTP warmup detector iterated the null-terminated `params.devices` list
and queried every entry. Vulkan returned early after finding its first device,
which hid the bug; a ROCm-only list reached the null terminator and asserted.

`server_params_use_vulkan()` now skips null entries before querying the backend
name. Both Vulkan and ROCm `llama-server` targets rebuilt successfully, and the
ROCm MTP short and long lanes then completed with device handoff enabled.

## Artifact Labels

- `e283-clean-vulkan-short-none-dev01-r3`;
- `e283-clean-vulkan-short-mtp-n3-dev01-r3`;
- `e283-clean-vulkan-long-none-dev01-r2`;
- `e283-clean-vulkan-long-mtp-n3-dev01-r2`;
- `e283-clean-rocm-short-none-dev10-r3`;
- `e283-clean-rocm-short-mtp-n3-dev10-r3-fixed`;
- `e283-clean-rocm-long-none-dev10-r1-fixed`;
- `e283-clean-rocm-long-mtp-n3-dev10-r1-fixed`.

The canonical rows are stored in
`build_logs/agent-workload/BENCH_RUNS.csv`.
