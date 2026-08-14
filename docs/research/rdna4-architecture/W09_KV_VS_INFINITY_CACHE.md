# W09: KV working set vs Infinity Cache (L2)

Date: 2026-08-14

Sources: model config from a bench log (`build_logs/agent-workload/baseline_preconvert.log`),
kernel/source facts from W00-W03, product specs for the 64 MB L2 (flagged as
product-page data, not from the ISA). No GPU runs.

## Verified KV arithmetic

Qwen3.6-27B Q4_K_M, fp8 E4M3 KV (1 byte per element):

- 64 layers, n_head_kv = 4, head_dim K = V = 256
- KV per token = 64 x 4 x 256 x 2 tensors x 1 B = **128 KiB per token**
- 49K lane: 49,152 x 128 KiB = **6 GiB**; layer split across 2 GPUs = 3 GiB per GPU
- 98K lane: 12 GiB total, 6 GiB per GPU

## Consequences

1. The decode FA re-reads the entire KV every token (all previous tokens
   attend the new query), so at 49K each GPU streams ~3 GiB of KV through L2
   per token. At the D089 baseline 38.9 tok/s that is ~117 GiB/s per GPU of
   KV reads, in parallel with the Q4_K_M weight stream (D102: ~17 GB read
   per token, ~8.5 GB per GPU).
2. L2 (Infinity Cache) is 64 MB per GPU (product spec). The KV working set
   (3 GiB at 49K) exceeds it 48x: best-case L2 KV hit rate is ~2% (only
   cross-head reuse, below).
3. **Cross-head reuse does exist**: blocks with the same `blockIdx.y` (same
   256-row KV chunk) across the 24 query heads re-read the same KV rows
   within one launch; that chunk is 256 rows x 128 B/row = 32 KiB, trivially
   L2-resident, so the per-launch reuse pattern is already cache-friendly.
4. The bulk of the KV stream is use-once per token; retaining it in L2
   displaces the weight-stream working set (which is reused heavily within a
   token). The gfx12 global-load cache-policy bits (`glc`/`slc`/`nv` and the
   cache-policy field) are not set by the compiler on the KV `global_load_u8`
   stream (W02 audit; no `glc`/`slc` modifiers observed in the disasm).

## Open question (phase-2, needs measurement)

H80: marking the KV u8 load stream as streaming/non-retained in L2 (so it
does not evict the weight stream) could improve decode TPS at 49K/98K. The
ISA provides the flags; the effect on L2 retention and TPS is unmeasurable
without a bench (phase 2). Counter-hypothesis: the L2 is already effectively
LRU-streaming this 48x-oversized working set, so hints change little.

## Related lever already on the shelf

The KV tail re-read cost is the reason the 49K/98K lanes are dominated by
memory traffic, not by FA arithmetic; this reinforces the track-level
conclusion that the largest structural levers are cache/streaming behavior
(W09) and the weight-stream path, not the FA kernel's ALU phases.
