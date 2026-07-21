# D088 P003 Vulkan No-Peer TKV4 Direct Prefill Gate

Date: 2026-07-20

Status: implemented, built, correctness-validated, and benchmarked as an
opt-in long-context residency route. It is not a replacement for physical
peer memory and it is not the default maximum-quality KV profile.

## Question

Can code compensate for the Windows AMD driver's lack of peer access between
two RX 9070 XT cards, while improving prompt evaluation on the production
Vulkan layer-split lane?

## No-Peer Boundary Gate

The driver exposes the cards as two singleton Vulkan device groups. Therefore
the application cannot construct a peer-capable logical device, and Win32
external memory cannot turn allocations from different physical GPUs into
device-local peer memory.

The exact layer-stage trace also closes a custom relay as a prompt-speed route:

- one layer-split boundary copy costs about `4.4-6.3 ms` per ubatch;
- the adjacent device stages cost about `475-563 ms`;
- even a free boundary copy would recover only about `1-1.5%`.

A pinned-host double buffer, relay thread, or D3D12 cross-adapter bridge would
still pay synchronization and system-memory traffic, so its real ceiling is
below that bound. This is different from tensor split, where 127 reductions per
ubatch make transport dominant; D084/D085 already keep that route opt-in.

## Implemented Route

The useful software compensation is to reduce local KV traffic and residency,
so long contexts do not spill KV to host memory merely because peer VRAM is
unavailable. The new Vulkan TKV4 direct-prefill path adds:

- F32 to TKV4 quantization and TKV4 `SET_ROWS` pipelines;
- a 128-thread workgroup per 128-value TKV block;
- parallel norm reduction and Walsh-Hadamard butterflies in shared memory;
- direct TKV4 dequantization in scalar and coopmat1 Flash Attention;
- explicit Vulkan graph support for the forward and inverse Turbo WHT ops;
- backend coverage for copy, set-rows, and WHT correctness.

Enable the experimental prefill route with:

```text
GGML_TKV_DIRECT_PREFILL=1
```

Use matching cache types, for example `-ctk turbo4 -ctv turbo4`. The active
same-type route is Vulkan coopmat1 Flash Attention and does not fall back to a
CPU KV conversion.

## Correctness

Focused backend tests pass on both RX 9070 XT devices:

| Device | Focused tests | Result |
| --- | ---: | --- |
| Vulkan0 | 6/6 | pass |
| Vulkan1 | 6/6 | pass |

The complete focused invocation reports `3/3 backends passed`. Coverage
includes F32-to-TKV4 copy, TKV4 set-rows, and forward/inverse 128-element WHT.

## Prompt Evaluation

Matched lane:

- model: `models/Qwen3.6-27B-Q3_K_S.gguf`;
- devices: `Vulkan1/Vulkan0`, layer split `5/6`;
- `n_batch=8192`, `n_ubatch=1024`, Flash Attention, no mmap;
- output device: `Vulkan1`.

| Prompt | KV | Repetitions | Prompt tok/s | Delta vs q8 |
| ---: | --- | ---: | ---: | ---: |
| 7488 | q8_0/q8_0 | 3 | 1800.43 | reference |
| 7488 | TKV4/TKV4 | 3 | 1779.87 | -1.14% |
| 43008 | q8_0/q8_0 | 1 | 1247.09 | reference |
| 43008 | TKV4/TKV4 | 1 | 1234.40 | -1.02% |

TKV4 is near parity but does not beat q8 prompt evaluation. It is kept for
residency rather than reported as a direct speed win.

## Residency

`q8_0` stores 34 bytes per 32 values, or 8.5 bpw. TKV4 stores 66 bytes per 128
values, or 4.125 bpw. For the Qwen3.6-27B 131072-context KV shape:

| KV | Size | Delta vs q8 |
| --- | ---: | ---: |
| q8_0/q8_0 | 4352 MiB | reference |
| TKV4/TKV4 | 2112 MiB | -2240 MiB (-51.5%) |

D037 found that full-device q8 needed about 1967 MiB of relief on the 130k
shape. The TKV4 saving exceeds that deficit and can keep the same layer-split
topology without direct host-KV spill. This is the concrete compensation for
missing peer memory: fewer bytes must remain resident on each isolated device.

## Remaining Cost

Vulkan performance logging shows that the two Turbo WHT graph regions total
about `2.13 ms` out of roughly `878 ms`, approximately `0.24%`. TKV4
`SET_ROWS` adds less than another `0.1%`. Fusing WHT into coopmat1 Flash
Attention would require a special graph/pipeline contract and additional
shared memory while its theoretical whole-run ceiling is below the measured
q8 gap. That fusion is rejected for now.

The main prompt-eval target remains the Q3_K compute body. Compressed KV can
remove residency and host-spill constraints, but it cannot by itself close the
2000 tok/s target.

## Decision

- Keep the Vulkan TKV4 direct-prefill implementation.
- Keep `GGML_TKV_DIRECT_PREFILL=1` opt-in until a long-agent quality and
  perplexity gate is complete.
- Use q8_0 as the maximum-quality and slightly faster prompt reference.
- Do not build a custom layer-boundary host relay on this Windows lane; its
  measured Amdahl ceiling is about `1-1.5%` before relay overhead.
- Reopen peer transport only if a driver exposes a real multi-device Vulkan
  group, or on Linux ROCm with a verified RCCL/P2P path.
- Continue prompt-speed work in the Q3_K matmul body rather than WHT fusion or
  additional host-relay scheduling.
