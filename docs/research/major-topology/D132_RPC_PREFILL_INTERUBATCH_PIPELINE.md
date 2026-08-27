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

## P2 feasibility update (2026-08-27, trace94k-p2a + P1 staged pull)

Profile on the accepted lane (base MMQ tile, `ts 0.8,1,1.4`,
`GGML_RPC_ASYNC_GRAPH=1 GGML_RPC_ACT_Q8_0=1 LLAMA_RPC_RUN_AHEAD=1`):

- client RPC timeline: `GET_TENSOR` totals `46.64 s` per 94K run, 40% of
  wall; mean `256 ms`, late ubatches `420-509 ms`;
- split timing: the boundary copy `l_out-16 RPC0->Vulkan0` is `237.8 ms`
  per ubatch; every other split is <= 7 ms;
- `GRAPH_COMPUTE_ASYNC` is already fire-and-forget (b=0), so the GET wait is
  the server graph finishing before the non-async GET (auto-flush), not the
  transfer.

P1 landed (env `GGML_RPC_PREFILL_PIPELINE=1`, ggml-backend.cpp): the split
worker only pulls into host staging; the Vulkan set stays on the scheduler
thread (no detached GPU access, D132 safety gate). Correctness primitive
only, 14K smoke passed (`p2p1-14k-smoke`).

P2 design conclusion: a single `ggml_backend_sched` cannot hold two ubatch
graphs (`sched->graph` is per-graph; `alloc_graph` overwrites splits), and
two PP schedulers would duplicate the PP compute buffers (multi-GiB VRAM
budget on 2x 16 GiB with 15.9 GiB model). Required P2 path (next block):

1. `ggml_backend_sched_graph_compute_range(sched, split_from, split_to)` —
   execute a subset of the built splits, leaving the rest pending
   (implemented 2026-08-27);
2. llama-context prefill driver: phase A submits the RPC0 head (splits:
   CPU inputs, RPC0 graph, KQ-mask round trip) for ubatch N and N+1;
   phase B (per staged-pull availability) runs the Vulkan tails;
3. `GGML_RPC_PREFILL_PIPELINE=1` gates everything; run-ahead
   (`LLAMA_RPC_RUN_AHEAD=1`) stays required; decode/MTP paths unchanged;
4. validation: 14K smoke, local 94K control (no common-path regression),
   adjacent 94K A/B vs trace94k-p2a (same binary), expect >= 3% (D132 gate).

## P2 validated design (2026-08-27, after RPC command census)

Stream per prefill ubatch (client timeline): `SET_TENSOR*` (graph inputs)
→ `GRAPH_COMPUTE_ASYNC` (layers 0-3) → `SET_TENSOR_MASK_NPAST` →
`GRAPH_COMPUTE_ASYNC` (layers 4-16) → `GET_TENSOR-rsp l_out-16` (rsp =
231-509 ms = server GPU finishing; grows with KV) → local
`VK0 -> VK1 l_out-37` slow copy (470 ms = `ggml_backend_synchronize(VK0)`
CPU block) → next ubatch. Commands for server N+1 can be sent right after
server N (all RPC0 inputs are host CPU data; server buffers are allocated
once per run - `ALLOC_BUFFER` n=19/max-tokens).

Blocking constraint: server output buffers are single-set per ubatch, so a
`GET l_out-16(N)` after `GRAPH N+1` would read overwritten data. Therefore
the inter-ubatch handoff requires:

1. **Server-side double-buffered graph outputs** (alternating copy sets on
   each GRAPH_COMPUTE_ASYNC; GET carries a copy index; protocol 5.0.1 patch)
   - keeps `GET l_out-16(N)` valid after `GRAPH N+1` is queued;
2. **Early head submit in llama-context**: keep one reusable `llm_graph
   result` graph (`can_reuse` kept ON in P2 mode), `mctx->peek_next_ubatch()`
   to prepare ubatch N+1 inputs, then `compute_range(rpc_head)` for N+1
   before the tail of N (`compute_range(tail)`); range API already exists;
3. `GGML_RPC_PREFILL_PIPELINE=1` gates all of it; decode, MTP and non-RPC
   topologies unchanged; `LLAMA_RPC_RUN_AHEAD=1` may be superseded in P2
   mode (graph reuse instead of fresh alloc per ubatch);
4. Acceptance per this doc; expected win ~466 ms/ub (server work hidden
   behind VK0/VK1 tails) -> ~+40-45% on the 94K lane.

## P2 status 2026-08-27 (protocol + scheduler infra in, decode driver NOT yet)

Implemented, builds clean (llama-server + rpc-server, build-vulkan):

- **Part A (server pipeline, `GGML_RPC_PREFILL_PIPELINE=1`)**:
  - `rpc_msg_get_tensor_req.wait_seq` (protocol 5.0.2) - GET can wait for a
    specific queued graph instead of the whole worker queue;
  - server worker queue (`worker_pending`/`worker_seq`/`worker_done`) with a
    release gate: graph N+1 never starts before the GET of graph N is served
    (result buffers are shared between consecutive graphs - this keeps GET
    reads race-free without double-buffering the outputs);
  - data commands (SET/MASK/copy) in P2 mode wait only for the graphs
    enqueued before them (`wait_enqueued_graphs`);
  - client: `rpc_p2_get_seq` global (set via
    `ggml_backend_rpc_set_p2_get_seq`) is read by both sync and async GET;
    `input_embed`/`inp_embd` writes also use the no-flush path in P2 mode.
- **Part B (phase APIs)**:
  - `ggml_backend_sched_p2_head_end()`: exclusive end of the RPC head part
    of the current splits (all splits up to the last RPC split; the
    `GET l_out-*`/`VK0->VK1` copies are the tail part);
  - `graph_compute_range` (llama-context) + `process_ubatch` phase
    parameters (`p2_phase` 1=head, 2=tail, `p2_override`, `p2_skip_apply`);
  - `llama_memory_context_i::peek_next_ubatch()` (kv-cache, iswa, both
    hybrid implementations) - next ubatch without consuming it;
  - extract block moved into `extract_ubatch_results` lambda (parameters
    named like the variables, body unchanged).

**Blocking barrier found while wiring the decode driver**: the P2 order
(head N+1 submitted before the GET/tail of N) requires graph N+1 to be
BUILT+ALLOCATED before tail N runs, but `ggml_backend_sched_alloc_graph`
re-splits the scheduler (`sched->splits`) and overwrites the split copies -
tail N would be lost. Graph reuse cannot be used either: `can_reuse_kq_mask`
requires `kq_mask->ne[0] == n_kv`, which grows every prefill ubatch.
Options for the next block:
  1. two-slot split lists (`ggml_backend_sched_split` copies + compute from
     a saved split set) - the full D132 two-graph scheduler;
  2. fixed-size prefill mask (round n_kv up to the next ubatch boundary)
     plus forced graph reuse, so no alloc happens between head N+1 and
     tail N;
  3. server-side double-buffered result + input copies (VRAM cost
     ~2x l_out-16 16MB + mask on the RTX 3080).

Decision: implement (1) as D132 next block; (2) is a numeric/performance
shortcut that changes the mask shapes and needs a PPL gate; (3) alone does
not remove the client-side alloc barrier.

## P2 status 2026-08-27 (final): correct device order + A/B + network ceiling

- **Device order matters**: the working order is `-dev RPC0,Vulkan0,Vulkan1`
  (NOT Vulkan1,Vulkan0,RPC0): with RPC0 last the split layout puts the RPC
  split at the END (head_end == n_splits, empty tail), so P2 head/tail never
  engaged. With RPC0 first the RPC split is #4 of 6 - head (RPC) runs first,
  tail = local VK splits (verified via GGML_SCHED_SPLIT_TIMING: split=4/6
  backend=RPC0, nodes=801, first=FLASH_ATTN_EXT).
- **A/B (correct order, server 5.0.2, MTP n=4)**:
  - 14K (ts 0.8,1,1.4): base 1218.9/38.7 vs P2 1177.9-1220.3/39.3 (parity).
  - 94K (ts 1,1,1.1, 58K tok, --task-hard-timeout 180): base 925.0/36.5 vs
    P2 942.4/34.7 (+1.9% ptps, within noise); acceptance identical in run 2.
- **Network ceiling investigated (user observation)**: both ends report
  1 Gbps; Task Manager shows only ~15% link utilization (~148 Mbps).
  `SO_RCVBUF=16MB` added on server accept + client connect (commit
  c6807eb23): no regression, no gain - the RPC ceiling is protocol
  serialization (single socket, round-trip waits), not socket buffers or
  the physical link.
- Conclusion: serial P2 (head/tail/extract) is correctness-parity; the
  remaining headroom is NOT the 2-buffer split pipeline alone (old profile
  estimated <=6%) but protocol parallelism/compression (F8 l_out) plus
  server-side NV FATTN/MM compute. Next stage must be measured before
  implementation (F8 l_out or pipelined GET/SET first).

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
