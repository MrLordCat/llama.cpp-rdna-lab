# E062 Vulkan prefill research

## Metadata

- Experiment ID: E062
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 plus local experimental `ngram-mtp` worktree
- Target lane: RX 9070 XT, Windows Vulkan driver 26.3.1, `build-vulkan`, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `b=4096`, `ub=512`, `q4_0/q4_0`, thinking on, no reuse

## Hypothesis

- Statement: Vulkan loses the prompt-heavy lane because its Windows AMD proprietary path is slower in K-quant prefill, even though decode is faster.
- Mechanism: likely a mix of conservative Vulkan matmul tile selection on AMD proprietary driver, K-quant memory layout, and MMVQ selector behavior.
- Why now: E061 showed Vulkan decode-biased wall TPS beating ROCm, but prompt-heavy wall TPS losing due prompt eval throughput.

## Research Notes

Local source findings:

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` disables large matmul tiles for AMD proprietary driver: `mul_mat_l = coopmat_support && driver_id != eAmdProprietary`.
- The same file has runtime knobs `GGML_VK_DISABLE_MMVQ` and `GGML_VK_FORCE_MMVQ`.
- `ggml_vk_should_use_mmvq()` disables MMVQ for `Q3_K` and `Q6_K` by default due 2-byte alignment concerns.
- Local history already includes upstream `#21751` (`vulkan: Coalesce Q4_K/Q5_K scale loads`), so that known Q4/Q5 scale-load fix is not missing.

Upstream / internet research leads:

- `ggml-org/llama.cpp#20934`: external RX 7900 XTX report matches our pattern: Vulkan/RADV faster in token generation, ROCm faster in prompt processing.
- `ggml-org/llama.cpp#22970`: open Vulkan PR transposes K-quant A-matrix layout on upload. Reported RDNA4 prompt gains are about `+4%..+11%` on Q4_K/Q6_K models and `+15.2%` on a Q6_K test-backend-op shape. Not present locally (`transpose_a` / `_transa` variants absent).
- `ggml-org/llama.cpp#22951`: open Vulkan PR pads Q3_K/Q6_K for 32-bit alignment and re-enables MMVQ/block-load paths. Huge reported gains on Intel/Battlemage pure Q3_K/Q6_K, but discussion notes this overlaps with other repack approaches and needs cross-device validation.
- `ggml-org/llama.cpp#21024`: broader Vulkan repack PoC. Discussion shows performance can improve or regress depending on hardware/layout.
- `ggml-org/llama.cpp#23106`: large `MUL_MAT_ID` warptile on AMD coopmat was intentionally disabled after maintainer regression tests. It is also more MoE-specific, so it is not the primary dense Qwen3.6-27B prefill path.

## Benchmark Plan

- Keep E061 as the backend baseline.
- Rerun Vulkan default in the same session to control for driver/runtime variance.
- Test existing Vulkan env knobs:
  - `GGML_VK_FORCE_MMVQ=1`
  - `GGML_VK_DISABLE_MMVQ=1`
- Check whether a larger `batch`/`ubatch` helps Vulkan prefill before code changes.

## Results

Prompt-heavy agent workload (`repo-snapshot`, 7489 prompt tokens, 64 generated):

| Backend / env | Wall TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | --- |
| ROCm E061 baseline | `6.3327` | `960.26` | `28.32` | current prompt-heavy winner |
| Vulkan E061 initial | `4.2206` | `573.93` | `30.85` | initial Vulkan baseline |
| Vulkan default rerun | `4.5539` | `607.78` | `38.32` | same-session control |
| Vulkan `GGML_VK_FORCE_MMVQ=1` | `4.6383` | `619.79` | `38.20` | small prefill gain vs rerun |
| Vulkan `GGML_VK_DISABLE_MMVQ=1` | `4.7172` | `639.81` | `35.15` | best prompt-heavy Vulkan in this screen |

Decode-biased sanity (159 prompt tokens, 128 generated):

| Vulkan env | Wall TPS | Prompt eval TPS | Decode eval TPS | Notes |
| --- | ---: | ---: | ---: | --- |
| default E061 | `35.2850` | `518.83` | `38.81` | best decode-biased baseline |
| `GGML_VK_FORCE_MMVQ=1` | `35.0071` | `507.90` | `38.74` | effectively neutral/slightly lower |
| `GGML_VK_DISABLE_MMVQ=1` | `34.6724` | `504.48` | `38.18` | slightly lower wall |

Vulkan prefill shape matrix with `GGML_VK_DISABLE_MMVQ=1`:

| b | ub | pp4096 tok/s | pp8192 tok/s |
| ---: | ---: | ---: | ---: |
| 4096 | 512 | `632.96` | `609.12` |
| 4096 | 1024 | `614.07` | `600.34` |
| 8192 | 512 | `605.81` | `593.61` |
| 8192 | 1024 | `614.31` | `601.60` |

## Decision

- Keep ROCm as the default for the current prompt-heavy target.
- Keep Vulkan as decode-heavy opt-in; E061's external-report alignment still holds.
- For prompt-heavy Vulkan comparisons, `GGML_VK_DISABLE_MMVQ=1` is the best no-code probe found so far, but it is not a universal default because it slightly lowers decode-biased wall TPS and is only `runs=1`.
- Do not blindly enable large AMD proprietary warptiles or `MUL_MAT_ID` large tiles; upstream explicitly notes regression risk.
- If we do code work, the highest-value candidates are the K-quant layout/repack family: start with an opt-in, minimal port of `#22970` transpose-A or a narrower Q3_K/Q6_K alignment probe inspired by `#22951/#21024`, with a Vulkan correctness run before speed claims.

## Artifacts

- `build_logs/agent-workload/e062-vulkan-default-rerun-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-force-mmvq-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-disable-mmvq-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-force-mmvq-decode-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-disable-mmvq-decode-q3ks.diagnostics.md`
- `build_logs/agent-workload/e062-vulkan-disable-mmvq-b-ub-matrix.md`
- `build_logs/agent-workload/e062-vulkan-default-llamabench-pp512-tg128.md`
- `build_logs/agent-workload/e062-vulkan-force-mmvq-llamabench-pp512-tg128.md`
- `build_logs/agent-workload/e062-vulkan-disable-mmvq-llamabench-pp512-tg128.md`
