# P003 Q4 MetaComp external reference pack

Date: 2026-05-28  
Owner: Copilot/perf workspace

## Purpose

Collect external/upstream evidence relevant to a Q4 memory-reduction track, and
separate directly applicable signals from non-transferable ones.

## High-relevance references

1. External RDNA4/llama.cpp scout summary:
   - `docs/research/experiments/E059_external_rdna4_llama_research.md`
   - Use: shortlist where upstream/external signals justify local experiments.

2. Vulkan prefill external alignment note:
   - `docs/research/experiments/E062_vulkan_prefill_research.md`
   - Mentions external report alignment (`ggml-org/llama.cpp#20934`) and warns
     against blind enablement of risky large-tile defaults.

3. Upstream-derived probes with measured local outcomes:
   - `docs/research/experiments/E063_vulkan_transpose_a_probe.md`
   - `docs/research/experiments/E065_vulkan_q3k_alignment_rocm_level.md`
   - `docs/research/experiments/E066_vulkan_gdn_chunked_probe.md`
   - Use: evidence that upstream ideas must still pass local lane A/B gates.

## Comparative but non-transferable references

1. Stormrage RDNA2 context (comparison only):
   - `docs/research/experiments/E009_H12_stormrage_tq3_direct_fattn.md`
   - Local notes repeatedly state RX6800XT/RDNA2 numbers are not pass/fail
     targets for RX9070XT/RDNA4.

2. Upstream-stock rollback controls (ROCm path):
   - `docs/research/major-topology/D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md`
   - `docs/research/RESULTS_LOG.md` entry D022
   - Shows imported upstream-stock binary can be materially slower on this lane;
     do not treat "upstream stock" as automatic performance baseline.

## Q4-specific observations relevant to P003

1. Existing local Q4 tooling evidence is mostly quality-focused (tool-call
   reliability), not memory-compaction-focused:
   - `docs/research/TOOL_CALL_WORKLOAD_BENCH.md`
   - `docs/research/major-topology/D038_P002_Q4_TOOL_CALL_THINKING_GUARD.md`

2. There is no current upstream-proven path in this workspace that reduces Q4
   memory footprint while keeping Q4 payload semantics; this supports opening
   P003 as a new research track.

## External-signal policy for P003

1. Treat external/upstream ideas as hypothesis seeds only.
2. Require local lane-matched proof before promotion:
   - fit outcome,
   - prompt/decode TPS,
   - quality smoke.
3. Keep references traceable to source docs/experiments above.
