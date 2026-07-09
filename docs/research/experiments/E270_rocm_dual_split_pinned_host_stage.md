# E270 ROCm Dual Split Pinned Host Stage

## Metadata

- Experiment ID: E270
- Date: 2026-07-09
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`, after `d8584069a`
- Hypothesis ID: H55
- Target lane: ROCm dual-GPU baseline without MTP, `Qwen3.6-27B-Q3_K_S_mtp.gguf`

## Hypothesis

- Statement: The remaining two-GPU `-sm layer` decode drop is dominated by
  Windows ROCm cross-device host staging and synchronization overhead.
- Mechanism: With direct HIP peer-copy disabled for correctness, split-boundary
  tensors are copied GPU -> host -> GPU. E269 moved one safe buffer-copy path
  out of the generic fallback and recovered about `+5%`; using page-locked host
  staging for that path may reduce the remaining transfer cost.
- Boundary: This experiment must not enable `GGML_ROCM_ENABLE_PEER_COPY=1` as a
  default. Direct peer-copy remains correctness-rejected until a microtest proves
  GPU0 <-> GPU1 tensor copies are exact on this Windows/RDNA4 setup.

## Baseline

Short decode-heavy lane, no real-context injection:

`Qwen3.6-27B-Q3_K_S_mtp.gguf`, ROCm, `ctx=8192`, `b512/ub128`,
`q4_0/q4_0`, FlashAttention on, `max_tokens=256`, `temperature=0.0`,
`quick:triage_diff`, no reuse/no prime, thinking enabled, `--spec-type none`,
`-dev ROCm1,ROCm0 -sm layer -ts 1,1`.

Fresh clean control after E269:

| Label | Runs | Aggregate TPS | Decode tok/s | Prompt tok/s | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `rocm-dual-split-clean-control-short-mt256-none-r1` | 1 | `25.8288` | `26.60` | `643.98` | graph splits `3`, sched copies `4` |

## Implementation Plan

1. Replace the E269 thread-local pageable staging buffer with a reusable
   thread-local pinned host buffer for the same safe buffer-copy path.
2. Preserve fallback to pageable `std::vector<uint8_t>` if pinned allocation
   fails or `GGML_CUDA_NO_PINNED` is set.
3. Benchmark the same short lane first. Keep only if decode improves beyond
   noise without errors.
4. If positive, consider applying the same helper to older `MemcpyPeerAsync`
   host-staged wrappers; if negative, revert.

## Result

- Outcome: reject pinned staging candidate.
- Control: `rocm-dual-split-clean-control-short-mt256-none-r1` reached
  `25.8288` aggregate TPS / `26.60` decode tok/s.
- Candidate: `rocm-dual-split-pinnedstage-short-mt256-none-r1` reached
  `25.7722` aggregate TPS / `26.49` decode tok/s.
- Decision: revert the pinned staging C++ candidate. It is noise-level/slightly
  worse on the short split decode lane. Continue H55 via copy-path timing,
  scheduler-copy reduction, or direct peer-copy correctness diagnostics only.
