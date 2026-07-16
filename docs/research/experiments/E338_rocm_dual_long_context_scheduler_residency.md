# E338: ROCm dual-GPU long-context scheduler residency

Date: 2026-07-16

## Goal

Explain why Windows reports growing Shared GPU memory for ROCm even when both
cards still show free dedicated VRAM, distinguish this from KV-cache growth,
and recover enough local-memory headroom for long-prompt MTP n3.

## Test contract

- two RX 9070 XT 16 GB cards;
- `ROCm1,ROCm0 -sm layer -ts 1,1`, one server slot;
- FlashAttention, `b8192/ub1024`, q8_0 K/V;
- cold repository-snapshot prompts, no reuse, no warmup, `-fit off`;
- direct HIP peer copy disabled;
- WDDM process counters sampled independently of HIP.

No direct `hipMemGetInfo` query or hard process termination was used.

## What was actually growing

The K/V allocation is fixed when the context is created. In the Q4 98K lane,
the server reports about 1,632 MiB of context storage on each card before
prompt processing, and that value does not grow with the number of processed
tokens.

The larger growth came from the split-graph scheduler. Pipeline parallelism
used four graph copies unconditionally. Representative Q4 allocations were:

```text
ROCm1: 221 + 4 * 384 + 136 + 192 + 204 MiB ~= 2289 MiB
ROCm0: 221 + 4 * 192 + 136 + 198 MiB       ~= 1323 MiB
```

These buffers are created when request graph shapes are first used. On the
Windows HIP/WDDM path they also acquire pageable system-memory backing, which
appears in Task Manager as Shared GPU memory. Therefore Shared can rise while
dedicated allocations are also resident and while the cards still have unused
physical VRAM.

This is not equivalent to an active PCIe spill. The practical indicators are
throughput and residency together. The old Q4 failure measured only 553.50
prompt tok/s at 98K context and 6.25 GiB Shared. The corrected route remains at
about 1,485 prompt tok/s despite a smaller non-zero Shared commitment.

ROCm also sees a lower usable WDDM budget than the physical 16,304 MiB shown by
the device. Desktop/driver allocations and per-process residency budgets mean
that Task Manager's apparent free total cannot all be consumed by one HIP
process before WDDM starts backing allocations.

## Scheduler-copy change

`ggml_backend_sched_new()` now detects a ROCm backend and uses one pipeline
copy by default. `GGML_SCHED_PIPELINE_COPIES=1..4` remains an explicit research
override. Vulkan and CPU defaults are unchanged.

The fork targets a single active request (`-np 1`). In this workload, extra
copies retained large graph arenas without improving throughput.

### Q4 98K copy sweep

Model: Qwen3.6-27B Q4_K_M, 59,071 prompt tokens, 16 output tokens.

| Copies | Prompt TPS | Decode TPS | Prefill dedicated | Prefill Shared |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1477.30 | 16.25 | 23.848 GiB | 5.458 GiB |
| 2 | 1494.82 | 16.77 | 22.712 GiB | 4.045 GiB |
| 1 | 1485.27 | 18.37 | 22.045 GiB | 3.203 GiB |

The 16-token decode samples are too short for ranking decode throughput. The
prompt result is neutral within run variance, while one copy saves 1.803 GiB
of prefill dedicated memory and 2.255 GiB of Shared commitment versus four.

### Q4 98K MTP validation

A final matched pair uses the MTP-enabled Q4_K_M GGUF, 59,045 prompt tokens,
64 output tokens, and one scheduler copy.

| Mode | Prompt TPS | Decode TPS | Wall time | Acceptance | Prefill dedicated | Prefill Shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **1493.21** | 19.15 | **42.98 s** | - | 22.045 GiB | 3.204 GiB |
| MTP n3 | 1435.97 | **35.44** | 43.04 s | **80.00%** | 23.956 GiB | 3.261 GiB |

MTP costs 3.83% prompt throughput and gains 85.1% decode throughput. At only
64 generated tokens the two wall times are effectively equal; longer answers
favor MTP. MTP adds 1.911 GiB Dedicated but only 0.057 GiB Shared during
prefill, so it does not reintroduce the old Q4 spill cliff.

## Q3 near-capacity MTP validation

Model: Qwen3.6-27B Q3_K_S MTP, context 131,072, 72,295 prompt tokens, 64
output tokens. Both rows use one scheduler copy.

| Mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance | Prefill dedicated | Prefill Shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | **1439.89** | 21.90 | **1.20** | - | 19.350 GiB | 3.514 GiB |
| MTP n3 | 1363.95 | **32.53** | 1.16 | 74.14% | 21.709 GiB | 3.576 GiB |

MTP costs 5.27% prompt throughput and gains 48.5% decode throughput. Its
prefill Shared peak is only 62 MiB above the no-MTP row. The additional MTP
working set is predominantly dedicated memory, so n3 does not trigger a new
RAM-residency cliff on this lane. For only 64 generated tokens the request is
still prompt-dominated; longer answers amortize the prefill cost.

Final backend accounting leaves 5,594/4,210 MiB free on ROCm1/ROCm0 for MTP,
versus 5,841/5,491 MiB without MTP. `unaccounted` includes the WDDM/driver and
scheduler allocations that are outside llama's model/context/compute totals.

## Deferred sparse-prefill correctness fix

The first MTP memory run exposed a separate lifecycle bug. A sparse draft batch
could remain deferred while capture stayed enabled. The next 256-token tail
then reused the same batch and target staging. `begin()` attempted to decode
that tail twice and logged inconsistent sequence positions, although the HTTP
request still returned successfully.

The fix flushes any deferred sparse batch before every next active target
capture, including true-to-true gate transitions, and checks failed draft
memory trims instead of ignoring them. The repeated 72,295-token validation
has no warnings, identical 74.14% acceptance, and equivalent throughput.

## Additional long-prompt controls

Before the copy reduction, the bounded Q8 route was also validated at two
prompt sizes with four scheduler copies:

| Backend | Context | Prompt | Prompt TPS | Decode TPS |
| --- | ---: | ---: | ---: | ---: |
| ROCm | 131,072 | 43,081 | 1639.84 | 22.85 |
| Vulkan | 131,072 | 43,081 | 1453.12 | 15.91 |
| ROCm | 131,072 | 72,295 | 1428.26 | 16.88 |
| Vulkan | 131,072 | 72,295 | 1179.30 | 14.65 |

ROCm stayed 12.9-21.1% faster in prompt evaluation. This confirms that the
remaining Shared display was not the old active-spill failure.

## Decision

- Keep bounded Q8 chunked WMMA for long ROCm contexts.
- Use one ROCm pipeline scheduler copy by default for the fork's single-request
  workload.
- Keep `GGML_SCHED_PIPELINE_COPIES` as a controlled multi-request/research
  override.
- Treat Shared as a residency warning, not as proof of active spill; require a
  corresponding throughput cliff or transfer trace before diagnosing RAM-bound
  execution.
- MTP n3 is supported on both the tested 131K/72K Q3 lane and the 98K/59K Q4
  lane.

Key artifacts use the `e338-` prefix under `build_logs/agent-workload`.
