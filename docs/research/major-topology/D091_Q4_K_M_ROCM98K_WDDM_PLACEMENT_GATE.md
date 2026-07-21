# D091 Q4_K_M ROCm 98K WDDM Placement Gate

Date: 2026-07-20

Status: cause confirmed. No backend source change is required. The safe 98K
placement is `-dev ROCm1,ROCm0 -sm layer -ts 1,1`. The remaining approximately
`2.6 GiB` one-adapter Shared peak is classified as WDDM backing/commit, not an
active-spill throughput limiter on the healthy route.

## Trigger

GUI autotune ran `Qwen3.6-27B-Q4_K_M.gguf` at `ctx=98304`, `b8192/ub1024`,
q8 K/V, MTP n2 and a 59145-token prompt. It measured only `947.46 prompt
tok/s`, well below the established E338 98K corridor.

The run used the backend-default physical order. Its model buffers were
`7386.64 MiB` on ROCm0 and `8235.14 MiB` on ROCm1. The MTP draft context and
default output placement also landed on ROCm1. At shutdown the smaller ROCm1
WDDM budget was exhausted while ROCm0 still had substantial headroom:

| Device | Effective total | Self | Unaccounted | Free |
| --- | ---: | ---: | ---: | ---: |
| ROCm0 | `15428 MiB` | `9456 MiB` | `1701 MiB` | `4270 MiB` |
| ROCm1 | `12812 MiB` | `10279 MiB` | `2532 MiB` | `0 MiB` |

Free memory on ROCm0 cannot satisfy an allocation owned by ROCm1. WDDM can
therefore page or back ROCm1 allocations through system memory even though the
sum of free VRAM across both cards looks sufficient. The llama Host row is not
a direct spill counter: it does not include all WDDM pageable backing.

## Causal Repeat

Two clean runs used the production order
`-dev ROCm1,ROCm0 -sm layer -ts 1,1`. No `llama-server` or `Sovereign` process
was active before either launch.

| Run | Prompt tok/s | Decode tok/s | Aggregate TPS | ROCm1 free | ROCm0 free |
| --- | ---: | ---: | ---: | ---: | ---: |
| wrong order, MTP n2 | `947.46` | `29.71` | `1.9148` | `0 MiB` | `4270 MiB` |
| correct order, spec none | `1483.41` | - | - | `1755 MiB` | `4137 MiB` |
| correct order, MTP n2 | `1426.54` | `33.37` | `2.82` | `1584 MiB` | `2866 MiB` |

The corrected MTP run is `+50.56%` faster in prompt evaluation than the GUI
autotune run. It also matches the earlier E338 MTP n3 result of `1435.97
prompt tok/s` within 0.66%. The corrected spec-none result matches E338's
`1493.21 prompt tok/s` within 0.66%.

The effective ROCm1 budget remained reduced to about `12742-12745 MiB` during
the repeats, so recovery came from placement rather than a larger available
budget. Correct ordering moved the smaller `7386 MiB` model share to ROCm1 and
the larger `8235 MiB` share plus output/draft work to ROCm0.

## Why Shared Still Grows During Prefill

Task Manager's Shared GPU memory is not a PCIe-byte counter and does not prove
that kernels are reading their working set from system RAM. On Windows HIP,
WDDM can attach pageable system-memory backing to GPU-addressable allocations
while their active pages remain resident in dedicated VRAM.

The allocations on this lane separate into three relevant classes:

1. Model buffers are fixed at load time (`8235/7387 MiB` across the two GPUs).
2. q8 K/V is fixed at context creation (`1632 MiB` per GPU at `ctx=98304`);
   it does not grow with processed prompt tokens.
3. The PP split-graph scheduler reserves the `ubatch=1024` compute graph and
   its inter-backend tensor copies. WDDM accounts the arenas as they are first
   touched during prefill, so Shared rises with prompt progress and can remain
   committed after the PP scheduler is released or becomes inactive.

The exact approximately `2.6 GiB` observation already exists in the healthy
E338 one-copy run: one adapter peaked at `2.582 GiB` Shared, total prefill
Shared was `3.203 GiB`, and prompt evaluation was `1485.27 tok/s`. A repeated
64-token control peaked at `2.562 GiB` on that adapter and measured
`1493.21 tok/s`.

### Causal Scheduler-Copy Sweep

| Scheduler copies | Prefill Shared | Prompt tok/s |
| ---: | ---: | ---: |
| 4 | `5.458 GiB` | `1477.30` |
| 2 | `4.045 GiB` | `1494.82` |
| 1 | `3.203 GiB` | `1485.27` |

Reducing Shared by `41.32%` from four copies to one changed prompt throughput
by only `+0.54%`. Reducing it another `20.82%` from two copies to one changed
prompt throughput by `-0.64%`. The relationship is neither large nor
monotonic. One scheduler copy is already the minimum practical setting, and
this sweep rejects Shared commitment itself as the missing 1700 tok/s lever.

There is still real host-staged transport at the single layer boundary because
ROCm peer copies are intentionally disabled on Windows. That is separate from
the `2.6 GiB` Shared display: earlier split timing bounded the layer-boundary
copy share at about `1-1.5%` of prompt time. Enabling unsafe peer copy cannot
close a `13.85-19.17%` target gap and is not admitted by this gate.

## 1700 tok/s Ceiling Check

Reaching `1700 tok/s` requires `+14.60%` over the D091 spec-none repeat or
`+19.17%` over MTP n2. The established 49K lane processes about 29561 prompt
tokens at `1778.59 tok/s`; the 98K lane processes about 59045 tokens at
`1493.21 tok/s`, a `16.05%` reduction. The longer prompt increases average
attention span and total Flash Attention work even though the KV allocation
was reserved in advance.

Therefore 1700 remains a valid kernel target, but not a memory-accounting
prediction. It requires a material Flash Attention/MMQ/runtime win. A future
claim of active spill must add transfer or residency evidence (for example
DxgKrnl ETW page-fault/eviction data) alongside a throughput cliff; Task
Manager Shared alone is insufficient.

## Decision

- Keep q8 KV, the one-copy scheduler and bounded Q8 Flash Attention scratch.
- Treat the approximately `2.6 GiB` one-adapter Shared peak as the known
  healthy WDDM backing corridor, not as reclaimable VRAM or a TPS estimate.
- For Q4_K_M at 98K, select `ROCm1,ROCm0 - layer split (recommended)` in the
  GUI benchmark device selector.
- Use `27:37` only for the separately measured 131K placement-stress lane.
- Do not enable unsafe peer copy and do not force the output back to ROCm1.
- Treat the GUI lane estimate of about 84442 prompt tokens as an estimate: the
  recorded run actually processed 59145 tokens.
- The autotune session contained only one configuration and no spec-none or
  device-placement sweep, so it could not identify this placement failure.

## Artifacts

- failing GUI run:
  `gui-autotune-Qwen3.6-27B-Q4_K_M-20260720-223143-cfg01.server.log`;
- corrected control: `d091-q4km-rocm98k-correct-order-none-r1.*`;
- corrected MTP n2: `d091-q4km-rocm98k-correct-order-mtp2-r1.*`.
