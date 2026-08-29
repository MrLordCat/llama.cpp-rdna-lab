# D133 - RPC topology control matrix

Status: controls complete; RPC bottleneck evidence collected (2026-08-27)

## Objective

Establish a clean 14K smoke reference for the RPC path before changing RPC
runtime code. The next investigation is RPC-only: identify whether the wall is
the remote GPU, the activation transfer, the protocol round trip, or the
local Vulkan tail.

## Locked smoke lane

- model: `Qwen3.8-27B-Q4_K_M.gguf`;
- context/batch: `12288`, `8192/1024`;
- KV: `q8_0/q8_0`, flash attention enabled;
- MTP: `draft-mtp`, `spec-draft-n-max=4`;
- output: `128` tokens, seed `42`, no warmup;
- prompt: `repo-snapshot`, `24576` requested chars;
- fitting/cache: `-fit off`, `--cache-ram 0`, `--ctx-checkpoints 0`;
- one run per topology, same client/server binaries and clean server state.

## Control matrix

| ID | Topology | Device order / split | Purpose |
| --- | --- | --- | --- |
| L0 | local dual Vulkan | `Vulkan1,Vulkan0`, `-ts 1,1` | local control without RPC |
| R1 | remote RTX 3080 + local Vulkan1 | `RPC0,Vulkan1`, `-ts 1,1` | isolate two-device RPC path |
| R2 | remote RTX 3080 + both local Vulkan GPUs | `RPC0,Vulkan0,Vulkan1`, `-ts 0.8,1,1.4` | production RPC topology control |

The RPC-first order is intentional. It places the RPC split before the local
Vulkan splits and is the only order to use for the R1/R2 comparison.

## Measured controls (2026-08-27)

All three runs are one-run, cold-first `quick` smokes on the locked lane. Each
completed both tasks (`2/2`) with exit code 0. Acceptance stayed within the
same range, so no quality regression was observed in this control set.

| ID | Aggregate TPS | Prompt TPS | Decode TPS | MTP acceptance |
| --- | ---: | ---: | ---: | --- |
| L0 | 16.8235 | 1636.5400 | 48.6700 | 54.57% (173/317) |
| R1 | 10.9461 | 933.0500 | 41.1650 | 57.89% (176/304) |
| R2 | 13.0091 | 1217.0150 | 39.8750 | 55.73% (175/314) |

R1 is 43.0% below L0 on prompt evaluation. Adding the second local Vulkan
GPU in R2 recovers 30.4% prompt throughput versus R1, but R2 is still 25.6%
below the local dual-GPU control. R2 decode is 18.1% below L0. This separates
the cost of the RPC boundary from the cost of the local Vulkan tail without
changing the model, context, batching, KV type, MTP or prompt.

## RPC diagnostic evidence

The diagnostic run was `d133-r2-rpcdiag-14k`; its speed result is intentionally
not comparable with the clean controls because RPC, scheduler and ubatch
timing instrumentation was enabled. The client timeline recorded:

- 103 `GET_TENSOR` calls, 8484.7 ms total response wait;
- 14 full `l_out-16` transfers of 20 MiB logical size, 5393.4 ms total,
   385.2 ms average per transfer;
- 206 `GRAPH_COMPUTE` commands. In the current synchronous protocol these
   commands have no response, so the server completes the graph before it can
   service the next command on the same connection.

The matching temporary server trace (`d133-r2-rpcdiag-srv-14k`) measured 206
`GRAPH_COMPUTE` handlers at 7277.0 ms total (35.3 ms average), but only
328.4 ms total inside 103 `GET_TENSOR` handlers (3.2 ms average). The 14 large
GET handlers accounted for 266.0 ms total (19.0 ms average). Therefore the
385.2 ms client-side wait for a large GET includes the serialized protocol
ordering and receive/transfer drain, not just the server tensor-copy handler.
This makes the RPC boundary/receive path the primary next candidate; it does
not by itself prove that raw Ethernet bandwidth is saturated.

The remote server was restored to the normal pre-P2 launch configuration after
the trace and the port was verified ready. No runtime source change was
accepted from this investigation.

## Evidence to collect after the clean controls

1. prompt/decode/aggregate TPS and speculative acceptance for each topology;
2. scheduler split placement and per-ubatch timing on one R2 diagnostic run;
3. RPC operation counts and transferred tensor sizes, with debug tracing kept
   separate from speed results;
4. only then choose between protocol round-trip reduction, activation-wire
   reduction, or remote Vulkan kernel work.

Do not compare diagnostic runs with clean TPS controls. Keep the RPC server on
the same protocol-compatible binary as the local client, and stop it gracefully
between binary changes.