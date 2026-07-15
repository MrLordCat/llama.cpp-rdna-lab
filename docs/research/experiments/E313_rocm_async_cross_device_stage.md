# E313: ROCm event-chained cross-device host staging

Date: 2026-07-14

## Goal

Remove the host synchronization between the two layer-split HIP graphs without
using unstable peer access. The existing safe route waits for ROCm1, performs a
host-staged copy, then submits the ROCm0 graph.

## Design

The opt-in `GGML_ROCM_ASYNC_CROSS_DEVICE_STAGE=1` route uses eight reusable
pinned-host slots. For each cross-device tensor it queues:

1. D2H on the source stream after the source graph;
2. a source event;
3. a destination-stream wait on that event;
4. H2D on the destination stream;
5. a destination completion event protecting slot lifetime.

The scheduler can then submit the second graph while the first graph is still
running. No peer mapping or `hipMemcpyPeerAsync` is used. If the event wait or
H2D enqueue fails, the code synchronizes the queued D2H and returns to the old
synchronous fallback.

This is supported by the HIP 7.1 stream/event contract: stream waits enqueue a
dependency, and graph execution is associated with streams. HIP graphs reduce
launch overhead, but a graph launch is still a stream operation.

## Correctness and route proof

A trace run recorded 270 async cross-device copies and zero fallbacks. The
generated preview, response length, token count, acceptance, and deterministic
temperature-zero output matched the synchronous controls. Five candidate
server launches, including long baseline and MTP runs, exited gracefully with
no driver loss.

## Performance

All long runs used dual `ROCm1,ROCm0`, equal layer split, output on ROCm0,
q8 KV, batch/ubatch 8192/1024, and a 30,099-token prompt.

| Mode | Synchronous | Async stage | Delta |
| --- | ---: | ---: | ---: |
| Baseline prompt, r2 mean | 1767.23 tok/s | 1813.96 tok/s | +2.65% |
| Baseline decode, r2 mean | 26.77 tok/s | 26.65 tok/s | -0.45% |
| Baseline aggregate, r2 mean | 5.855 TPS | 5.965 TPS | +1.88% |
| MTP n3 prompt | 1740.24 tok/s | 1772.67 tok/s | +1.86% |
| MTP n3 decode | 29.49 tok/s | 29.48 tok/s | neutral |
| MTP n3 aggregate | 5.90 TPS | 5.99 TPS | +1.53% |

MTP acceptance was identical at 44.44% (72 accepted of 162 generated). A
6,395-token diagnostic showed a larger prompt gain but noisy decode behavior;
it is not used for the production decision.

## Decision

Keep the route opt-in while extending stability coverage to larger contexts
and repeated GUI sessions. The 30K evidence is positive for the fork's main
prompt-heavy workload and neutral for decode, but the Windows driver history
requires more than one validation session before making cross-device event
staging the default.

Primary artifacts use `e313-rocm-dual-*` prefixes.

## References

- https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/how-to/hip_runtime_api/hipgraph.html
- https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/doxygen/html/group___stream.html
- https://rocm.docs.amd.com/projects/HIP/en/docs-7.1.0/doxygen/html/group___event.html
