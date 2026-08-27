# D132 - RPC prefill inter-ubatch pipeline

Status: Q8_0 transport subroute accepted opt-in; true P2 pipeline remains active (2026-08-26)

## Objective

Raise the Qwen3.8-27B Q4_K_M 94K RPC-first prefill lane by at least 3%
without trading away the rf47 decode result or changing local Vulkan behavior.

Locked comparison lane:

- client: dual Vulkan RX 9070 XT;
- server: RPC0 RTX 3080 over 1 GbE;
- order/split: `RPC0,Vulkan0,Vulkan1`, layer split, `-ts 0.8,1,1.4`;
- context/batch: `98304`, `8192/1024`, q8_0/q8_0 KV;
- model/spec: Qwen3.8-27B-Q4_K_M, MTP n=4;
- historical control: rf47, `1029.64 prompt tok/s`, `39.36 decode tok/s`;
- same-binary adjacent F16 control: rf59, `1016.82 / 37.59`.

## Evidence gate

The rf44/rf45 split trace places `221-238 ms` per ubatch in the boundary
from the RPC head to the first local split. That interval contains three
ordered phases:

1. wait for the server graph that produces `l_out-18`;
2. GET and F16 wire transfer of the boundary activation;
3. expand to F32 and set the Vulkan0 split input.

The server graph and transfer are therefore on the same critical path as the
local Vulkan tail. rf43 proves that merely rotating scheduler input copies
does not overlap the phases (`+0.4%` at 94K). rf46 proves that a detached
copy worker without a real next-ubatch submission also loses performance.

Wall-share lower bound: rf47 evaluates about 58K prompt tokens in 56.56 s,
or roughly 0.99 s per 1024-token ubatch. A 30 ms/ubatch overlap is already a
3% lane win. The measured RPC boundary is over 220 ms/ubatch, so the candidate
has enough Amdahl ceiling if it actually overlaps work rather than moving the
same wait to another thread.

## Accepted transport subroute (rf50-rf64)

P0 showed that the dominant boundary tensor is F32 `l_out-16` with
`20,971,520` bytes (`5,242,880` values). The existing F16 wire uses
`10,485,760` bytes. Block Q8_0 uses one fp16 scale plus 32 signed bytes per
block, or `5,570,560` bytes total: `46.875%` fewer wire bytes than F16.

The retained implementation is protocol `5.0.1`, default-off, and uses:

- `GGML_RPC_ACT_Q8_0=1` for intermediate `l_out-*` only; final logits remain
  F16;
- `GGML_RPC_ACT_THREADS=16` on the client; the scheduled remote server used
  the helper default of 8 threads (its launch environment did not set it);
- `LLAMA_RPC_RUN_AHEAD=1` to rotate scheduler copy slots between prefill
  ubatches;
- `GGML_RPC_ASYNC_GRAPH=1` for the established ordered server graph path.

Measured 94K result on identical binaries and the locked lane:

| Run | Boundary/config | Prompt t/s | Decode t/s |
| --- | --- | ---: | ---: |
| rf59 | adjacent F16 control | 1016.82 | 37.59 |
| rf62 | Q8_0, client/server 16/8 threads, run-ahead | 1065.39 | 40.48 |
| rf63 | exact repeat | 1062.35 | 38.45 |
| rf62/rf63 center | accepted candidate | **1063.87** | **39.47** |

The repeat spread is `0.29%`. Prompt improves `+4.63%` over rf59 and
`3.32%` over the historical rf47 result, while centered decode is unchanged
versus rf47 (`39.47` vs `39.36`). The adjacent local dual-Vulkan control rf64
is `1443.85 / 50.67`, matching rf22 (`1441.99 / 51.19`), so no local
scheduler regression was detected.

The gain is composite: Q8_0 alone (rf58) was only `1036.19 / 38.66`, and
run-ahead alone had previously measured only `+0.4%` at 94K. Reducing the
wire wall makes copy-slot rotation useful enough to clear the acceptance
gate. This does not implement the full P2 overlap described below.

Rejected controls were retained as evidence: ubatch 2048 lost `15.2%`
(rf51), skipping the coarse source barrier was noise-level (`+0.8%`, rf52),
and eliminating an extra response copy was neutral (rf55). Those prototypes
were removed.

## Hypothesis H81

Prefill ubatches write disjoint KV ranges and the scheduler already owns
multiple boundary-copy slots. If RPC head graph N+1 can be submitted while
the local Vulkan tail of N consumes a different slot, the steady-state wall
can approach `max(RPC head + pull, local tail)` instead of their sum.

## Staged plan

### P0 - protocol/copy census

- quantify server graph wait, wire receive, F16 conversion and Vulkan set
  separately on the exact 14K and 94K contracts;
- test `ubatch=2048` as a fusion control: if boundary frequency is material,
  it should move the same wall without any concurrency claim;
- verify that the explicit RPC synchronize before GET is redundant or retain
  it with evidence.

### P1 - source-side RPC pull primitive

- let the scheduler try a source backend asynchronous copy hook after the
  destination hook declines;
- implement the RPC-source hook as an ordered pull into owned host staging;
- keep it behind `GGML_RPC_PREFILL_PIPELINE=1`;
- do not call Vulkan synchronization or Vulkan set from the RPC/socket worker.

P1 is an instrumentation/correctness primitive, not yet a speed claim.

### P2 - two-slot scheduler handoff

- retain two graph/split-input slots for prefill only;
- submit RPC head N+1 before waiting for the staged boundary of N;
- perform Vulkan set and the local tail for N on the scheduler thread;
- fence slot reuse with per-copy events;
- drain all staged work on error, reset, scheduler switch and destruction.

Decode (`n_tokens == 1`), graph callbacks, non-causal graphs, embeddings and
non-RPC topologies must stay on the existing path.

## Correctness and safety gates

- same prompt/decode output sanity and no RPC protocol assertion;
- no detached worker may touch Vulkan compute state;
- no overwrite of a slot until both remote GET and local tail are complete;
- 14K smoke before 94K;
- adjacent local 94K control after any common scheduler change;
- Vulkan build plus `git diff --check` for every retained candidate.

## Acceptance and rollback

Accept only if an adjacent 94K candidate improves prompt throughput by at
least 3% over the same-binary control and decode remains within normal lane
variance. A promising r1 requires a repeat. The whole route is default-off;
rollback is removal or disabling of `GGML_RPC_PREFILL_PIPELINE` with no format
or default behavior change. The accepted Q8_0 subroute clears the performance
gate but remains opt-in until a separate PPL/quality gate covers the lossy
activation boundary. Its rollback is to omit `GGML_RPC_ACT_Q8_0`,
`GGML_RPC_ACT_THREADS`, and `LLAMA_RPC_RUN_AHEAD`; F16 remains the default.
