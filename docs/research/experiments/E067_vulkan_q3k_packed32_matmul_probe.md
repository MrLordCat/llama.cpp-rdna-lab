# E067 Vulkan Q3_K Packed32 Matmul Probe

## Metadata

- Experiment ID: E067
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: local working tree after E065/E066 rollback
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan, ctx=12288, b=4096, ub=1024, q4_0/q4_0 KV, FlashAttention on, thinking on, no reuse, `--spec-type none`

## Hypothesis

- Statement: Vulkan prefill is dominated by large Q3_K `MUL_MAT` kernels; using padded 32-bit Q3_K block loads in the non-coopmat2 matmul dequant path can reduce hot unpack/load overhead.
- Mechanism: E065 padded Q3_K device blocks to 4-byte alignment. The current `mul_mm_funcs.glsl` Q3_K path still uses packed16/byte-style scale, hmask, and quant loads. Loading the same data through `block_q3_K_packed32` should reduce load instructions and improve compiler scheduling on AMD RDNA4.
- Why now: E067 perf logger shows Q3_K `MUL_MAT` consumes most prompt chunk time; no-code fusion/int-dot probes did not hold as a stable improvement.

## Math / Theory

- Assumptions: The shader is load/unpack limited enough that wider aligned loads matter; packed32 indexing preserves the exact Q3_K layout after E065 padding.
- Expected speedup corridor: modest, about +2-8% prompt eval if unpack/load pressure is meaningful.
- Failure conditions: Compiler already combines 16-bit loads, extra shifts raise ALU pressure, or the hot path is tensor-core/decode-function limited elsewhere.

## Implementation Plan

1. Minimal code surface to change: `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl` Q3_K branch only.
2. Guard rails: preserve E065 padded layout; compare with pp7488 and active lane; revert if regression.
3. Rollback path: revert the shader branch to packed16/byte loads.

## Benchmark Plan

- Baseline command: E065 Vulkan `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` active lane and pp7488 artifacts.
- Candidate command: same, after shader change and Vulkan rebuild.
- Number of runs: 1 for gate, 3 only if promising.
- Artifacts path: `build_logs/agent-workload/e067-*`

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- prefill/decode split
- pp7488 llama-bench prompt throughput

## Result

- Outcome: regression
- Delta: pp7488 gate fell to `836.22 tok/s` vs restored E065 default `875.25 tok/s` (`-4.5%`).
- Confidence: high for rejection; the cheap prompt gate is directly on the hot Q3_K matmul path and regressed clearly.
- Recommendation: reject and keep the original packed16/byte-style Q3_K `mul_mm_funcs.glsl` branch.

## Notes

- Surprises: `GGML_VK_DISABLE_FUSION=1` looked positive in one run but confirmed at only `6.40 TPS` over 3 runs. Wider packed32 loads increased shift/register pressure enough to lose despite aligned device blocks.
- Follow-up action: Move to large matmul tile-shape tuning (E068), not more Q3_K packed-load rewriting.