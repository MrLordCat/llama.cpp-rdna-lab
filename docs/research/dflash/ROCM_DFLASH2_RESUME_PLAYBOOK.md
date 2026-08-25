# ROCm DFlash2 Resume Playbook

Updated: 2026-08-21
Status: PAUSED. DFlash2 integration is suspended at this checkpoint. No further
tuning, profiling, or benchmarking is scheduled until the pause is lifted.

## Stable checkpoint

- Branch: `dflash2`
- Pushed commit: `85d5f69a3 dflash: promote encoder prefill and long-session benchmarks`
- Local commit: `7bdc4a2b0 dflash: bring up ROCm profiling and device staging`
- The pause commit lands the Q4_K `nwarps=2` tuning, the opt-in multi-scheduler,
  and the measurement logs on top of `7bdc4a2b0`.
- Everything kept is either default-off or a measured win. Every rejected
  experiment was reverted before the commit, so the committed tree matches the
  binary that produced the numbers below.

## Pause checkpoint: 2026-08-21

There was no active `llama-server`, CMake, Ninja, or compiler process when this
checkpoint was written. The interrupted `nwarps=3` build was not used as
evidence, and the source was restored to the last measured winner,
`Q4_K nwarps=2` for `ncols_dst <= 4` on RDNA4.

### Reusable DFlash graph families

The current worktree has an opt-in three-slot scheduler/result cache:

```bash
export LLAMA_DFLASH_MULTI_SCHED=1
```

It separates encoder, KV-injection, and draft-token graph families. A 498+32
trace reused 33 of 45 submissions instead of rebuilding all 45. Artifact:
`build_logs/agent-workload/rocm-dflash2-n2-multisched-trace-ctx4k-out32-r1-20260821.server.log`.

The strict n=3 boundary/parity gate remained stable and bit-exact:
`build_logs/dflash2-lab/20260821T080927Z-rocm-np1-n3-multisched-boundaries-20260821`.

Matched 498+128, three-repeat results:

| Variant | Aggregate tok/s | Decode tok/s | Prompt tok/s |
| --- | ---: | ---: | ---: |
| single-cache control | 29.00 | 36.33 | 749.70 |
| multi-scheduler | 29.59 | 36.90 | 785.58 |

Graph-family reuse is therefore a small but repeatable win on this lane, about
1.6% decode and 2.0% aggregate. It remains default-off pending the final gates.

### Selector experiments

`LLAMA_DFLASH_COMPACT_SELECTOR=1` was an exact opt-in path that projected only
the selected mask rows through the full-vocabulary head. It passed the strict
n=3 gate at
`build_logs/dflash2-lab/20260821T082529Z-rocm-np1-n3-multisched-compact-boundaries-20260821`,
but 498+128 measured 29.70 aggregate and 36.94 decode tok/s, effectively flat
against multi-scheduler alone. It was therefore removed from `common/speculative.cpp`
and `src/models/dflash.cpp`; do not reintroduce it without a cross-backend
measurement that shows a real gain. The strict n=3 gate passed again on the
reverted source at
`build_logs/dflash2-lab/20260821T091602Z-rocm-np1-n3-postcompactremoval-20260821`.

The approximate rank-256 global-selector scout collapsed to zero accepted
draft tokens (`0/251`) and about 16.79 decode tok/s. That code was removed; do
not resume this approximation without a new algorithmic correctness argument.

### Q4_K small-row MMVQ result

The DFlash draft token graph is dominated by the Q4_K output head. On RDNA4,
using two warps for Q4_K MMVQ with `ncols_dst <= 4` was the current best
498+128 result:

| Q4_K warps | Aggregate tok/s | Decode tok/s | Prompt tok/s | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1 | 29.59 | 36.90 | 785.58 | prior control |
| 2 | 30.55 | 38.59 | 770.31 | keep |
| 4 | 27.73 | 34.49 | 771.01 | reject |

Artifacts use labels
`rocm-dflash2-n2-q4k-nwarps2-ctx4k-out128-r3-20260821` and
`rocm-dflash2-n2-q4k-nwarps4-ctx4k-out128-r3-20260821`. The nwarps=3 probe was
interrupted before a valid build or measurement, so it has no result.

This short-lane winner is still about 1.6% below the existing 498+512 MTP n=2
decode result (38.59 versus 39.22 tok/s), and the prompt/output lengths differ.
It is not evidence that DFlash has matched or beaten MTP. The decision requires
an adjacent same-session 498+512 comparison.

### Adjacent 498+512 comparison, 2026-08-21

Both runs used the same rebuilt binary, three repeats of `v2_write_function`,
`--max-tokens 512`, `--no-reuse`, and no other GPU load. DFlash ran with
`LLAMA_DFLASH_MULTI_SCHED=1`; the compact selector was off.

| Mode | Aggregate tok/s | Decode tok/s | Prompt tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| DFlash n=2 | 33.67 | 36.05 | 744.56 | 64.57% |
| MTP n=2 | 34.87 | 37.37 | 677.54 | 59.31% |

Artifacts: `rocm-dflash2-n2-adjacent-ctx4k-out512-r3-20260821` and
`rocm-mtp2-adjacent-ctx4k-out512-r3-20260821`.

MTP still wins by about 3.7% decode and 3.6% aggregate. Both absolute numbers
are below the 2026-08-20 session, which is why only the adjacent pair is
comparable. DFlash keeps the acceptance advantage, 64.57% against 59.31%, so
the remaining deficit is draft-side runtime cost, not draft quality.

## Open problems at the pause

These are the reasons DFlash2 cannot become a default yet, and why the work
stops here instead of continuing to tune.

### 1. MTP is still faster on the matched long lane

On the adjacent 498+512 pair above, MTP n=2 reaches 37.37 decode tok/s while
DFlash n=2 reaches 36.05, even though DFlash accepts more draft tokens. The
deficit is per-step draft runtime cost. Resumed work must attack that cost;
improving acceptance further will not close the gap.

### 2. The draft graph is fragmented across CPU and GPU

`GGML_SCHED_SPLIT_TIMING=1` shows the draft-token graph running as six splits
instead of one. Artifact:
`build_logs/agent-workload/rocm-dflash2-n2-splittiming-r1-20260821.server.log`.

| Split | Backend | Nodes | First node | Total ms |
| ---: | --- | ---: | --- | ---: |
| 1 | CPU | 1 | `GET_ROWS:inp_noise_embd` | 0.015 |
| 2 | ROCm0 | 770 | `RMS_NORM:norm-0` | 5.333 |
| 3 | CPU | 2 | `TOP_K:node_771` | 1.028 |
| 4 | ROCm0 | 3 | `CONT` | 0.236 |
| 5 | CPU | 6 | `GET_ROWS:node_776` | 0.757 |
| 6 | ROCm0 | 22 | `MUL_MAT:node_782` | 0.575 |

The two CPU islands cost about 1.8 ms per draft step and force four extra
device transitions. `TOP_K` has no HIP path at the DFlash row width without
CUB, so it always falls back to the CPU.

### 3. The GPU top-k attempt failed and was reverted

A block-selection `TOP_K` kernel for `k <= 64` was written to remove split 3.
On the 498+128 lane, three repeats, clean environment:

| Variant | Decode tok/s | Acceptance |
| --- | ---: | ---: |
| CPU top-k baseline | 38.59 | 69.81% |
| GPU top-k, first version | 29.81 | 63.96% |
| GPU top-k, second version | 32.05 | 63.96% |

Draft submit time rose from 64.3 ms to 76.9 ms. The acceptance drop is the more
serious signal: the kernel selected different tokens, so it was not a drop-in
replacement for the CPU op. It also changed shared `ggml-cuda` files used by
every model, so it was reverted instead of being left default-off. Artifacts:
`rocm-dflash2-n2-gputopk-ctx4k-out128-r3-20260821`,
`rocm-dflash2-n2-gputopk2-ctx4k-out128-r3-20260821`,
`rocm-dflash2-n2-gputopk-splits-r1-20260821`.

A future attempt must first prove bit-identical output against the CPU op on
the DFlash row shapes, then measure.

### 4. Submission overhead dominates the draft path

The same profiled session reports `encode_submit_ms=65.670`,
`draft_submit_ms=64.344`, and `inject_ms=25.534`, against `encode_sync_ms=2.440`
and `draft_sync_ms=0.144`. Nearly all DFlash time is graph submission and
injection bookkeeping rather than GPU math. Multi-scheduler reuse recovered
about 1.6% decode; the rest needs a structural change such as fusing the
encoder and KV injection into a single graph.

### 5. Vulkan is unvalidated against the current shared code

Shared DFlash code changed after the last successful Vulkan build. Vulkan was
not rebuilt or re-gated in this session, so every number in
`VULKAN_DFLASH2_RESUME_PLAYBOOK.md` is older than this tree.

### 6. Host RAM and VRAM accounting

The latest ROCm nwarps=2 server log shows:

- target: `CPU_Mapped 682.03 MiB`, `ROCm0 8235.14 MiB`, `ROCm1 7386.64 MiB`;
- draft: `CPU_Mapped 68.20 MiB`, `ROCm0 1011.41 MiB`;
- shutdown accounting: Host model 682 MiB, ROCm model 8235 + 7386 MiB.

The small RAM usage is therefore the GGUF file-backed mmap/host mapping, not
evidence that model layers spilled out of VRAM. Windows can count mapped pages
or pinned staging allocations as process RAM/shared GPU memory even while the
corresponding compute weights reside in VRAM. A real offload/spill diagnosis
must distinguish `CPU_Mapped` from a CPU model buffer and must check that all
requested layers and model buffers were placed on ROCm devices.

On resume, if this still looks suspicious, make one isolated `--mmap` versus
`--no-mmap` startup comparison and record private working set plus the server
buffer lines. Do not run that diagnostic concurrently with another GPU server
or hardware discovery.

## Resume checklist when the pause is lifted

The 2026-08-21 gates are complete: ROCm was rebuilt from the nwarps=2 source,
the strict n=3 boundary/parity gate passed twice, the adjacent 498+512 pair was
measured, and both the compact selector and the GPU top-k kernel were removed.
Multi-scheduler stays opt-in behind `LLAMA_DFLASH_MULTI_SCHED`.

1. Rebuild ROCm and Vulkan; the Vulkan binary is older than this tree.
2. Repeat the strict n=3 ROCm boundary/parity gate before any measurement.
3. Work on problem 2 or problem 4 above. Do not tune acceptance or GDN kernels.
4. Re-run the adjacent 498+512 DFlash n=2 versus MTP n=2 pair in one session.
5. Finish with `git diff --check` and confirm graceful server exit.

## Supported production topology

- Backend: ROCm 7.1, Ninja Release build.
- Target devices: `ROCm1,ROCm0`
- Target split: `-sm layer`
- DFlash draft device: `ROCm0`
- Do not set `LLAMA_OUTPUT_DEVICE`.
- `ROCm1`-only draft placement currently aborts during startup because
  `output.weight` is preallocated on ROCm0 while the operation is assigned to ROCm1.

Build:

```bash
export PATH="/c/Program Files/AMD/ROCm/7.1/bin:$PATH"
cmake --build build-rocm-full -j 4 --target llama-server
```

## Verified correctness baseline

The ROCm build completed and the original n=3 strict gate passed:

```bash
python scripts/research/dflash2_lab.py \
  --server-bin build-rocm-full/bin/llama-server.exe \
  --model models/Qwen3.8-27B-Q4_K_M.gguf \
  --draft-model models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --devices ROCm1,ROCm0 --draft-devices ROCm0 \
  --parallel 1 --ctx-size 4096 --spec-n-max 3 --max-tokens 64 \
  --boundary-tokens 1,2,3,7,8,9,15,16 \
  --require-serial-parity --require-identical-slot-stability \
  --require-boundary-parity \
  --label rocm-np1-n3-auto128-boundaries
```

Artifact stem:
`build_logs/dflash2-lab/20260820T182729Z-rocm-np1-n3-auto128-boundaries`

Result: stable and bit-exact, all eight boundary cases passed.

The same strict gate passed again after the local ROCm staging work:
`build_logs/dflash2-lab/20260820T193106Z-rocm-np1-n3-post-handoff-default-boundaries`.

## Long-workload baseline

Matched 498-token prompt plus 512 output tokens, no prompt reuse:

| Mode | Aggregate tok/s | Decode tok/s | Acceptance |
| --- | ---: | ---: | ---: |
| Target only | 24.51 | 25.57 | n/a |
| MTP n=2 | 36.66 | 39.22 | 61.98% |
| DFlash n=3 | 33.57 | 35.65 | 51.67% |
| DFlash n=2 | 33.33 | 36.97 | 64.21% |

DFlash n=2 is the active ROCm depth. It has slightly better acceptance than
MTP but still loses on draft runtime overhead.

## Phase profiler

Set `LLAMA_DFLASH_TIMING=1` to print aggregate DFlash phase timings at
shutdown. The host-path 498+128 profile reported:

- process: 1316.152 ms
- target synchronization: 1146.249 ms
- encoder submit/sync: 67.361 / 26.921 ms
- injection: 68.959 ms
- draft submit/sync: 564.153 / 0.364 ms

The fused GDN kernels themselves accounted for only about 95.8 ms in the
separate HIP synchronization trace. GDN kernel math is not the leading
bottleneck.

## Experimental device layer-input handoff

The local worktree contains an opt-in backend-resident path:

```bash
export LLAMA_DFLASH_DEVICE_HANDOFF=1
```

It stages the five extracted target layers on ROCm0 and feeds the DFlash
encoder without the old host feature buffer. It is default-off because the
current row-packing implementation did not improve end-to-end throughput.

Correctness evidence:

- device handoff and host control produced identical speculative hashes for
  serial, heterogeneous, identical-slot, and all boundary runs;
- both n=2 runs independently reproduce the same known max-token boundary
  mismatch at limits 1 and 2, so that mismatch is not introduced by handoff;
- artifacts:
  - `20260820T191250Z-rocm-np1-n2-device-handoff-boundaries`
  - `20260820T191454Z-rocm-np1-n2-host-control-boundaries`

Performance evidence:

- device handoff reduced measured DFlash `process()` time from about 1316 ms
  to about 171 ms on 498+128;
- total decode throughput remained about 40 tok/s because the old target wait
  mostly represented a real target-compute dependency;
- 498+512 device-handoff scout: 32.41 aggregate, 36.45 decode tok/s, slightly
  below the host baseline.

## Next optimization target

A traced 498+32 run recorded 45 DFlash graph submissions and zero graph reuse.
The alternating encoder, KV-injection, and draft graphs overwrite the single
cached graph result on every step.

Aggregated DFlash-only trace:

- graph build: 2.661 ms
- graph allocation: 12.818 ms
- input setup: 11.831 ms
- compute-call wall time: 152.429 ms
- total: 182.118 ms

Artifact:
`build_logs/agent-workload/rocm-dflash2-n2-graphtrace-ctx4k-out32-r1-20260820.server.log`

The three recurring DFlash graph families now have separate reusable
scheduler/result slots behind `LLAMA_DFLASH_MULTI_SCHED`, which removed most of
that rebuild churn. The remaining structural idea is to fuse the encoder and KV
injection into one graph. Do not re-tune GDN kernels before that is done.

## Required gates before the next commit

1. `cmake --build build-rocm-full -j 4 --target llama-server`
2. strict n=3 ROCm boundary/parity gate above
3. host/device output-hash comparison when touching staging
4. matched 498+512 DFlash n=2 versus MTP n=2, preferably three repeats
5. `git diff --check`
6. confirm no `llama-server` process remains
