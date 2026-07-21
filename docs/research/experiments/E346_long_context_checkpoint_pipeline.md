# E346: Long-context checkpoint pipeline

Date: 2026-07-16

## Goal

Make long-context checkpoint behavior reproducible, separate pending GPU compute
from checkpoint transfer time, and reduce avoidable host-transfer overhead on
the Windows ROCm dual-GPU route.

## Reproduction

The dedicated harness creates an exact append-only prefix through the raw
`/completion` endpoint. The default workload is a 57,000-token root followed
by 3,072, 1,280, and 4,096-token additions. It uses the production Qwen3.6
27B Q4_K_M route, `ROCm1,ROCm0`, `-sm layer -ts 27,37`, q8 KV,
`b8192/ub1024`, MTP n2, and four context checkpoints.

```powershell
python -u scripts\research\checkpoint_long_context_bench.py `
  --label checkpoint-session `
  --delta-tokens 3072,1280,4096 `
  --max-tokens 64 `
  --pinned-staging
```

The script writes JSON, CSV, and the complete server log under
`build_logs/checkpoint-bench`. It starts the server in a new process group and
uses only the existing soft CTRL_BREAK shutdown path.

## Timing correction

The old checkpoint total included synchronization of target compute submitted
before the checkpoint call. This made a roughly 600 ms pending decode look like
a 600 ms host copy. The trace now reports these phases separately:

- state-size calculation;
- checkpoint-vector allocation;
- pending target synchronization;
- state transfer;
- total checkpoint call time.

At this model and context, the target recurrent state is 156,894,356 bytes.
The draft state starts near 16 MiB and grows to roughly 35 MiB. A restore of
both states was only about 37-41 ms before the transfer changes, not hundreds
of milliseconds.

## Implementation

- Large checkpoint vectors are recycled when the checkpoint list is full.
- D2H and H2D state operations are submitted to all owning backends before the
  code waits, allowing both GPU transfers to overlap.
- `LLAMA_CHECKPOINT_PINNED_STAGING=1` enables one reusable backend-pinned host
  staging buffer per target or draft context.
- `--checkpoint-min-step` is available for checkpoint-density experiments, but
  remains zero by default because removing the middle prompt-tail checkpoint
  did not improve end-to-end latency and can reduce rollback coverage.

## Controlled A/B

Both runs used the same rebuilt binary, a 57,000-token root, one 3,072-token
addition, MTP n2, and exact raw-prefix reuse.

```powershell
python -u scripts\research\checkpoint_long_context_bench.py `
  --label checkpoint-ab-pageable --delta-tokens 3072 --no-pinned-staging

python -u scripts\research\checkpoint_long_context_bench.py `
  --label checkpoint-ab-pinned --delta-tokens 3072 --pinned-staging
```

| Route | Root prompt tok/s | Increment prompt tok/s | Increment wall ms | Restore transfer ms | Save transfer ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pageable checkpoint vectors | 1322.50 | 648.94 | 5139.08 | 36.51 | 104.72 |
| Reusable pinned staging | 1333.95 | 653.27 | 5121.07 | 33.90 | 81.67 |

Pinned staging reduced measured save transfer by 22.0% and restore transfer by
7.1%. The end-to-end increment improved by only 18 ms, about 0.35%, because
checkpoint I/O is a small part of the request.

## Session validation

The pinned route completed a 57k root and all three successive branches:

| Stage | Evaluated tokens | Prompt tok/s | Decode tok/s | MTP acceptance | Restore ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Root | 57,000 | 1321.88 | 33.86 | 86.96% | 0.00 |
| +3,072 | 3,077 | 652.73 | 29.96 | 71.15% | 34.99 |
| +1,280 | 1,285 | 500.61 | 33.69 | 90.91% | 34.53 |
| +4,096 | 4,101 | 701.14 | 31.93 | 84.78% | 30.86 |

No speculative-state failures or hard server stops occurred.

## Remaining bottleneck

A matching pinned `spec=none` control measured 1395.90 tok/s for the 57k root
and 698.69 tok/s for the 3,077-token increment. MTP measured 1333.95 and
653.27 tok/s, respectively. The MTP route adds its 256-token recent-history
boundary and draft-state work, while checkpoint rollback adds prompt-tail
boundaries around the final physical ubatch.

After a restore, a small append can therefore be evaluated as batches such as
`2049 + 772 + 252 + 4`. At a 60k-100k existing KV length, those small batches
are attention and launch-overhead limited. This is the primary reason prompt
throughput can fall toward 300-700 tok/s in a live agent session. Faster
checkpoint loading cannot restore cold 8k-batch throughput for a small append.

## Decision

Keep the timing, vector reuse, dual-backend async submission, and pinned
staging implementation. Pinned staging remains opt-in because its real wall
gain is small and it keeps about 185-220 MiB of additional host memory pinned
for the target and draft contexts. Continue prompt-eval work at prompt-tail
batch formation and long-KV attention rather than checkpoint serialization.

Key artifacts:

- `build_logs/checkpoint-bench/checkpoint-ab-pageable.*`
- `build_logs/checkpoint-bench/checkpoint-ab-pinned.*`
- `build_logs/checkpoint-bench/checkpoint-ab-none-pinned.*`
- `build_logs/checkpoint-bench/checkpoint-pinned-session-validate.*`
