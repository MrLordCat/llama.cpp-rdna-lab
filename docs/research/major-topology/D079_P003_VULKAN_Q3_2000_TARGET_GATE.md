# D079 P003 Vulkan Q3 2000 tok/s Target Gate

## Decision

- Status: completed evidence gate; continue through D080 and the combined Q3_K+FA design branch.
- Target: `2000 prompt tok/s` cold-first on the fixed 56,456-token lane.
- Fresh no-warmup baseline: `1276.93 prompt tok/s`, so required total speedup is `1.5663x` before D080 balancing.
- Backend/model: dual Vulkan, non-MTP `Qwen3.6-27B-Q3_K_S.gguf`, `spec=none`.

## Fixed Lane

- `ctx=131072`, `b=8192`, `ub=1024`, q8_0 K/V, FlashAttention on.
- `real-context-mode=repo-snapshot`, `real-context-chars=152000`.
- Measured prompt tokens: 56,456.
- `--cache-ram 0 --ctx-checkpoints 0`, no reuse, no v2 prime pass, thinking on.
- `-dev Vulkan1,Vulkan0 -sm layer -ts 1,1 --no-mmap -fit off`.
- `LLAMA_OUTPUT_DEVICE=Vulkan1`; no forced all-KV placement.
- r1 for scouts, r3 only for final or borderline confirmation.

## Baseline Evidence

Artifact: `build_logs/agent-workload/d079-vulkan-q3ks-long55k-none-nowarmup-r1.*`.

- Prompt: `1276.93 tok/s`.
- Decode: `14.32 tok/s` over 16 generated tokens; diagnostic only.
- All `65/65` layers offloaded.
- Pipeline parallelism enabled.
- Static memory breakdown:
  - Vulkan1: 6,685 MiB free; model 6,384 MiB; context 2,253 MiB.
  - Vulkan0: 7,958 MiB free; model 5,049 MiB; context 2,247 MiB.
  - Host model buffer: 521 MiB.
- Runtime WDDM Shared GPU Memory must still be observed during candidate confirmation.

## Required-Speedup Model

For baseline `B=1276.93` and target `T=2000`:

`S_total = T / B = 1.5663x`.

Required local speedup for a center with wall share `p`:

`S_local = p / (1 / S_total - (1 - p))`.

| Measured wall share p | Required local speedup |
| ---: | ---: |
| 0.70 | 2.07x |
| 0.80 | 1.82x |
| 0.85 | 1.74x |
| 0.90 | 1.67x |
| 0.95 | 1.61x |

Conclusion: a low-share micro-optimization cannot close the target. A source prototype must either change a dominant Q3_K/FA body or parallelize a large fraction of prefill across both GPUs.

## Evidence Plan

1. Run one same-lane Vulkan route/perf trace with no source changes.
2. Summarize Q3_K, Q4_K auxiliaries, FlashAttention, GDN/recurrent, scheduler copies, and other wall share.
3. Generate the Vulkan evidence pack and record missing gates explicitly.
4. Compare measured shares with the table above.
5. Select exactly one implementation center:
   - true Q3_K compressed-dot/body route when Q3_K has sufficient ceiling;
   - Q3_K + long-KV FA stack when neither center closes alone;
   - tensor-parallel prefill rehabilitation when kernel-local ceilings are insufficient.

## Tensor-Split Gate

`-sm tensor` is not assumed to be useful. Existing ROCm evidence with F16 KV measured only `185.94 prompt tok/s`, and q8 KV is rejected by the current context guard.

Before any q8 KV implementation:

1. Run a small deterministic Vulkan F16-KV tensor-split trace.
2. Classify Meta graph duplication, reductions/gathers, synchronization, and per-device dispatch.
3. Require at least 80% of the same-lane layer-split prefill speed.
4. Only then design q8 KV split-state handling and a long-lane test.

Promotion requires at least 5% over the best layer-split candidate, output equivalence, no Shared GPU Memory compute spill, and no driver instability.

## Measured Evidence

Full 56,456-token Vulkan perf trace:

- Q3_K matmul: `37229.20 ms`, 46.6% of parsed time.
- FlashAttention: `37074.96 ms`, 46.4%.
- Q4_K matmul: 4.0%; GLU: 1.8%; other parsed work: about 1.2%.
- Hot Q3_K shapes: `17408x1024x5120` (21.8%) and `5120x1024x17408` (11.7%).
- Active Q3 pipelines use 82-86 VGPR, 44-46 SGPR, 31,744 B LDS, and no scratch.
- Main FA route is q8/q8 coopmat1, `Br=16`, `Bc=64`, `D_split=8`, `split_k=1`.

Tensor split gate:

- Small F16 layer control: `1809.02 prompt tok/s`.
- Vulkan tensor split: `540.18`, only 29.9% of control.
- Each `ubatch=1024` graph has 127 fallback allreduces over 2.663 GB logical tensor data.
- Timed main compute is about 0.83-0.90 s and fallback allreduce about 0.98 s per full ubatch.
- Native allreduce is absent; Vulkan cross-device async copy rejects different devices, so the generic path synchronizes and blocks.
- Even a free allreduce would not beat the current layer pipeline because tensor main compute itself is too slow.

Decision: reject tensor split for P003 before q8-KV implementation. Continue with balanced layer pipeline and a combined Q3_K+FA body/dataflow design.

Instrumentation kept from the gate:

- `GGML_META_TIMING=1` prints env-gated Meta rebuild/main/allreduce counts and timings.
- Vulkan perf logger now synchronizes a scheduler-reserve context before resetting its timestamp query pool; normal non-trace execution is unchanged.
- Benchmark `--no-warmup` now has correct semantics; `--warmup` and compatibility alias `--no-no-warmup` explicitly enable warmup.

## Rejection Fence

Do not repeat:

- helper-only Q3 arithmetic/unpack changes;
- q3quad extension, q3-octa, or wider `LOAD_VEC_A` variants;
- signed-nibble/layout-only or scale-only metadata variants;
- broad Q3 predequant-to-F16 staging;
- nearby batch/ubatch, tensor-ratio, queue, mmap, or KV-format sweeps presented as a code solution;
- all-KV-on-one-device placement.

These routes are already closed by D029-D033, E258-E260, and E280 or cannot meet the new `1.551x` ceiling.

## Rollback And Safety

- New runtime/backend behavior must be env-gated until long-lane confirmation.
- Soft server shutdown only.
- Never invoke `hipMemGetInfo`, the Vulkan staging script, or server version/help probes around benchmarks.
- Stop immediately on WDDM Shared GPU Memory growth attributable to active compute.
