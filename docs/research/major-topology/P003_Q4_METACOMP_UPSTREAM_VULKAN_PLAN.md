# P003 Q4 MetaComp upstream Vulkan plan

Date: 2026-05-28  
Owner: Copilot/perf workspace

## Objective

Prepare an upstreamable path for Q4 metadata compaction that reduces memory
footprint while preserving Q4 payload semantics and avoiding major runtime
regressions.

## Constraints

- Must preserve existing GGUF compatibility for legacy Q4 tensors.
- New format path must be opt-in and versioned.
- Runtime kernels must stay deterministic and testable across CPU/CUDA/Vulkan.
- No speed claims without apples-to-apples lane matching.

## Proposed Upstream Split

### Track A: Format and tooling (upstream-friendly first)

1. Add a formal metadata-compacted Q4 tensor format variant identifier.
2. Add converter and validator tools in Python/C++ with roundtrip checks.
3. Add gguf dump support to display compact-mode stats.

Why first: small review surface, low risk, immediate testability.

### Track B: Runtime correctness path (feature-gated)

1. Loader support for new Q4 compact variant.
2. Reference decode path with strict equivalence tests vs legacy Q4.
3. Guard with env/flag default-off.

Why second: correctness before performance.

### Track C: Vulkan fused performance path

1. Add fused decode+matmul kernels for compact Q4 blocks.
2. Tune access pattern for coalesced reads and low register pressure.
3. Add point-shape microbench and practical lane A/B.

Why third: isolates backend complexity until format is stable.

## PR Sequence (recommended)

1. PR1: spec + reader/writer metadata tags + converter skeleton + docs.
2. PR2: offline converter implementation + roundtrip/consistency tests.
3. PR3: runtime loader + correctness-only decode fallback (default off).
4. PR4: Vulkan fused kernel route + benchmarks + rollout guard.

## Acceptance Gates per PR

- PR1/PR2:
  - stable serialization/deserialization,
  - deterministic converter output,
  - no regressions for existing GGUF files.

- PR3:
  - decode equivalence tolerance defined and met,
  - integration tests pass on representative Q4 tensors.

- PR4:
  - memory reduction confirmed on target models,
  - throughput within agreed guardrail vs legacy Q4,
  - no quality regressions on BFCL-lite smoke baseline.

## Measurement Pack for Reviews

- Model fit outcome on practical big-context lane.
- Prompt/decode TPS pair from agent workload harness.
- Quality smoke summary (BFCL-lite).
- Tensor-level compactness report from estimator/converter.

## Risk Register

1. Metadata compaction savings may be insufficient alone for full-offload fit.
2. Compact decode may increase kernel pressure and reduce throughput.
3. Mixed tensor modes can complicate scheduler/resource behavior.

Mitigations:

- keep mode per tensor and allow fallback,
- use staged rollout with default-off gate,
- pair format work with memory residency controls in practical lane.
