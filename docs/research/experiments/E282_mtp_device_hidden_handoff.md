# E282 MTP Device Hidden-State Handoff

Date: 2026-07-14

## Question

Can Vulkan MTP retain its decode gain on a long prompt without paying for a
GPU-to-host-to-GPU transfer of every NextN hidden-state row, and without a
material prompt-eval regression?

## Change

- Added backend-resident NextN output staging to `llama_context`.
- Bound contiguous staged rows directly to the MTP hidden-state graph input.
- Kept target and draft hidden tensors on the consuming NextN layer backend.
- Preserved the host extraction route as a fallback and as the
  `LLAMA_MTP_DEVICE_HANDOFF=0` rollback.
- Kept target graph topology stable while limiting real draft prefill to the
  recent prompt tail.

`LLAMA_MTP_DEVICE_HANDOFF_TRACE=1` verified that the measured Qwen3.6 route is
`Vulkan1 -> Vulkan1` for prompt catch-up, target verification, and draft rows.

## Lane

- Model: `Qwen3.6-27B-Q3_K_S_mtp.gguf`
- Backend: Vulkan dual layer, `Vulkan0,Vulkan1`, equal split
- Context: `49152`
- Prompt: `29540` tokens (`repo-snapshot`, 96000 characters)
- Batch/ubatch: `8192/1024`
- KV: `q8_0/q8_0`
- Output: `128` tokens
- Speculative depth: `3`

The final comparison was run while League of Legends was active. Two `none`
controls bracketed the MTP samples so game contention was not compared against
an idle baseline.

## Results

| Route | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| `none` control 1 | 1409.70 | 29.12 | 5.03 | - |
| `MTP`, window 256 | 1399.70 | 40.83 | 5.26 | 50.33% |
| `MTP`, window 128 | 1361.79 | 39.49 | 5.12 | 48.70% |
| `none` control 2 | 1430.52 | 28.95 | 5.09 | - |
| `MTP`, built-in default 256 confirmation | 1409.61 | 35.72 | 5.20 | 50.33% |

Against the mean of the two controls, the mean of both window-256 runs changes
prompt eval by `-1.09%`, decode by `+31.83%`, and aggregate throughput by
`+3.36%`. The matching `76/151` acceptance count in both runs attributes the
decode spread to game contention rather than a different speculative path.

An earlier direct negative control on the same implementation measured the new
device handoff at `1392.85` prompt TPS versus `966.91` with
`LLAMA_MTP_DEVICE_HANDOFF=0`, a `+44.1%` recovery of the full MTP prefill path.

## Decision

Keep backend-resident handoff and reduce the built-in server recent-window
default from 512 to 256 tokens. Window 128 loses acceptance and is slower in
this lane. A prompt-eval delta of about one to two percent is accepted because
decode improves by about forty percent and aggregate throughput remains
positive.

The result is Vulkan runtime evidence. ROCm compilation and runtime validation
remain separate follow-up work; Windows ROCm peer-copy stays disabled.

## Artifacts

- `build_logs/agent-workload/e279-lol-vulkan32k-none-n128.*`
- `build_logs/agent-workload/e279-lol-vulkan32k-mtp-device-window256-n128.*`
- `build_logs/agent-workload/e279-lol-vulkan32k-mtp-device-window128-n128.*`
- `build_logs/agent-workload/e279-lol-vulkan32k-none-repeat-n128.*`
- `build_logs/agent-workload/e282-lol-vulkan32k-mtp-default256-n128.*`
