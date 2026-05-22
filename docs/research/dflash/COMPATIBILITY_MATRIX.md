# DFlash Compatibility Matrix (Planning)

Status: planning estimate before local implementation.

## Backend Readiness Estimate

| Backend | Functional DFlash | Optimized DFlash | Notes |
| --- | --- | --- | --- |
| ROCm/HIP via ggml-cuda | likely in Phase 2 | medium risk | local priority backend; needs hook stability and lane A/B |
| CUDA | reference source behavior | high | Bee source path is CUDA-first reference |
| Vulkan | low (initial) | low | likely fallback path first, no immediate optimization plan |
| CPU-only | possible fallback | low | useful for correctness and fail-safe route |

## Model/Route Expectations

| Scenario | Expected DFlash effect | Priority |
| --- | --- | --- |
| repeated coding tasks / structured output | high upside | P1 |
| decode-heavy medium/long outputs | medium to high | P1 |
| cold-first prompt-heavy short runs | low/neutral/negative possible | P0 safety check |
| open-ended prose | variable, often lower | P2 |

## Required Capability Checks Before Enabling by Default

1. DFlash mode does not break non-DFlash speculative flows.
2. reduced verifier path can be toggled without graph reuse corruption.
3. capture-ring fallback path works when GPU ring is unavailable.
4. server rollback logic remains deterministic under partial accept.

## Promotion Criteria

1. correctness: no output corruption in deterministic smoke.
2. stability: no crash across repeated cycles.
3. performance: repeated/decode-heavy lane shows reproducible gain.
4. neutrality: cold-first lane not silently replaced in defaults.

## Known Unknowns to Resolve During Implementation

1. exact DFlash GGUF metadata variants that must be supported first.
2. minimal model architecture subset for phase-1 launch.
3. whether adaptive controller should ship in first public DFlash cut or wait one phase.
4. final ROCm-specific cost of ring sync and reduced verifier path.
