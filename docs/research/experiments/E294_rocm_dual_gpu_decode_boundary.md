# E294: ROCm long-context dual-GPU decode boundary

## Scope

- Backend: ROCm 7.1 on Windows, two RX 9070 XT (gfx1201)
- Model: Qwen3.6-27B-Q3_K_S with embedded MTP tensors
- Main lane: layer split `ROCm1,ROCm0`, `-ts 1,1`, q8_0 KV, FlashAttention
- Prompt: 21.6k to 24.2k actual tokens in `ctx=32768`; 128-token decode unless noted

## Root-cause result

ROCm itself can exceed the requested 30 tok/s baseline decode target. A matched
single-GPU ROCm1 run reached 31.61 tok/s at 24.2k prompt tokens. The matched
dual-GPU run reached 25.54 tok/s while prompt evaluation increased from
1082.94 to 1816.21 tok/s. The remaining baseline decode loss is therefore a
dual-GPU boundary/scheduling cost, not an intrinsic RDNA4 decode-kernel limit.

The output layer is already placed on the last device (`ROCm0`), so the graph
does not make an avoidable return trip to the first GPU. The active Windows
path has one layer boundary and disables direct peer copies because earlier
ROCm/RDNA4 tests produced silent tensor corruption and driver instability.

## Measurements

### Decode-kernel probes

- HIP graph replay is active after warm-up on both devices; missing graph replay
  is not the remaining root cause.
- Forcing the WMMA FlashAttention route for one-row decode was correct but did
  not improve speed (24.93 tok/s). Most of the 16-column tile is idle.
- The production vector FlashAttention split heuristic selected 21 parallel
  blocks. Reducing it to 10 was slower (25.10 vs 26.08 tok/s).
- Grouping two GQA heads in one vector-FA block was correct, but register
  pressure reduced occupancy from 8 to 5 blocks/SM. It tied the normal route
  at 26.08 tok/s and is rejected.
- q4_0 KV improved the diagnostic from 26.08 to 27.23 tok/s (+4.4%). KV traffic
  matters, but it does not explain the full dual-GPU gap.

### Cross-device trace

Each decode token crosses the layer boundary with two tensors:

- `l_out-32`: 20,480 bytes
- converted attention mask: 43,520 bytes

The old safe path uses pageable host memory and synchronizes the destination
stream after every H2D copy. A decode crossing costs about 0.85-1.0 ms after
the source half of the model has completed. Prompt chunks also transfer a
20 MiB hidden tensor and a growing converted mask, up to about 44 MiB.

Pinned host memory removes the blocking behavior from the H2D enqueue but does
not remove physical PCIe transfer latency. It remains a useful safe fallback.

### Queued H2D experiment

An opt-in `GGML_ROCM_ASYNC_HOST_STAGE=1` path used an eight-slot pinned-host
ring. It waits for the source stream, performs D2H, queues H2D on the destination
stream, records a lifetime event, and immediately lets the destination graph be
queued behind the copy. It does not enable peer access.

Deterministic generated text matched the synchronous control. Two 128-token
A/B pairs produced:

| Pair | Synchronous control | Queued H2D | Delta |
|---|---:|---:|---:|
| r1 | 26.86 tok/s | 27.94 tok/s | +4.0% |
| r2 | 27.93 tok/s | 28.88 tok/s | +3.4% |

Prompt evaluation stayed neutral in the matched r2 pair (1813.38 vs
1813.68 tok/s). The required 30,075-token validation then reversed the result:
the synchronous controls reached 26.12 and 26.79 tok/s with about 1694 prompt
tok/s, while queued H2D reached 25.36 tok/s and 1681.96 prompt tok/s. The path
was therefore rejected and removed. Short-prompt improvement is not sufficient
for this long-context fork.

## Rejected or incomplete work

- WMMA one-row decode, reduced vector parallel-block count, and two-head GQA
  vector blocks are not production candidates.
- The first async prototype used async D2H followed by source-stream sync. It
  was correct but slower (27.23 tok/s) and was replaced by source sync plus
  synchronous D2H and queued H2D. The replacement was also removed after the
  30k regression.
- Direct ROCm peer-copy remains disabled. Do not infer stability from speed or
  from a successful model load.

## Next safe optimizations

1. Avoid copying the converted attention mask from GPU1 to GPU0. The original
   F32 mask is a host input, but its F16 CAST is currently placed on GPU1 and
   the converted result is staged across the boundary. Duplicate the cheap CAST
   on each consuming GPU or teach the scheduler to rematerialize this input
   conversion locally.
2. Measure the residual cost of two graph submissions and scheduler events.
   After mask localization, this is the likely final host-side gap to 30 tok/s.
3. Re-test layer-boundary placement only after the transfer route is stable;
   it cannot remove the serial dependency but may balance unequal device clocks.

## Future peer-copy research gate

Peer-copy is the highest-upside route because it can replace
`VRAM1 -> RAM -> VRAM0` with a direct PCIe transfer. It must be treated as a
correctness and driver-stability project, not as an environment toggle.

1. Review current official ROCm 7.1 HIP peer-access and Windows support, plus
   upstream llama.cpp cross-device event/copy behavior.
2. Build a standalone, soft-stoppable probe that checks peer capability before
   enabling access. Do not start with a model server.
3. Validate deterministic patterns and checksums in both directions over small,
   decode-sized, and prompt-sized buffers. Detect silent corruption, not only
   API errors.
4. Test repeated async copies with explicit source/destination events and verify
   every iteration. Keep host staging as fail-closed fallback.
5. Only then run short model correctness and speed A/B, followed by repeated
   long-context and MTP runs. Never make P2P the default from a single pass.

### Peer-copy research already established

- HIP 7.1 exposes the required directional APIs:
  `hipDeviceCanAccessPeer`, `hipDeviceGetP2PAttribute`,
  `hipDeviceEnablePeerAccess`, and `hipMemcpyPeerAsync`.
- Capability and access must be checked and enabled independently in both
  directions. A working copy in one direction does not validate the reverse
  route.
- HIP documentation recommends issuing a peer copy on a stream associated with
  the physical source device. Cross-device completion must be represented
  explicitly before the destination graph consumes the tensor.
- Official documentation does not promise that Radeon peer access is stable on
  this Windows/RDNA4 configuration. The previous corruption and driver hangs
  therefore remain relevant evidence even after the OS and motherboard change.
- The current llama.cpp environment gates are not a sufficient validation
  mechanism: allowing the copy path and enabling peer mappings are separate
  decisions. A future implementation should expose one fail-closed route whose
  state is based on capability checks, successful access setup, and a
  deterministic startup self-test.

## Handoff state (2026-07-14)

GPU performance work was paused before tests with a game running in the
background. Do not use runs made under that load to accept or reject a change.

- Keep the pinned host-staging fallback and its opt-in trace. It is correct and
  is the safe production route while peer copy is disabled.
- The queued H2D ring and Qwen3.6 local-mask rematerialization were removed
  after their long-context regressions.
- Experimental FlashAttention environment overrides used for rejected kernel
  probes still need a source cleanup review before the final production build.
- Rebuild `build-rocm-full` before taking the next baseline: the existing binary
  may still contain removed, environment-gated experiment code from an earlier
  incremental build.

Resume in this order:

1. Clean rejected vector-FA probes and rebuild without changing the accepted
   rocWMMA prompt path or the decode correctness test.
2. Add a standalone ROCm peer probe. Its default mode must only report
   directional capability and P2P attributes; actual copies require an explicit
   opt-in flag.
3. Validate deterministic copies in both directions at 20 KiB, 44 KiB, 1 MiB,
   20 MiB, and 44 MiB, first once and then repeatedly. Any API error, checksum
   mismatch, timeout, or device loss permanently selects host staging for that
   run.
4. Only after repeated probe success, integrate the same state machine into an
   opt-in llama.cpp route and run short correctness tests before performance
   measurements.
5. Compare matched 30k+ baseline and MTP lanes. The target is to recover the
   roughly 6 tok/s single-to-dual ROCm decode loss without reducing long-prompt
   evaluation.

The reliability design and standalone test harness are continued in
`E295_rocm_windows_peer_copy_reliability.md`.

## Local-mask follow-up

The first form of item 2 was implemented and rejected. A second Qwen3.6 mask
input plus local F32-to-F16 CAST correctly removed every cross-device mask copy;
the trace contained only `l_out-32`, and deterministic output still matched.
However, direct F32 input upload plus another CAST cost more than staging the
43 KiB converted mask. Combined with queued H2D it reached only 26.84 tok/s and
1807.72 prompt tok/s, versus 28.88 and 1813.68 without rematerialization. The
prototype was removed. Do not repeat per-device mask rematerialization unless
the scheduler can reuse a device-local converted mask without adding another
input upload/CAST to the critical path.
