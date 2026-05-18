# Performance Research Index

Local navigation map for Qwen3.6 ROCm performance work in this fork. This is not upstream llama.cpp documentation.

## Archived Status

The current acceleration cycle is archived as of 2026-05-18.

Start here before resuming: `docs/research/PERFORMANCE_ARCHIVE_2026-05-18.md`.

Final practical no-spec lane:

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Backend: ROCm/HIP, target `gfx1201`, preferred build `build-rocm-vec`.
- Workload: `scripts/agent_workload_bench.py --tasks quick --task-ids triage_diff,review_bug --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --max-tokens 120 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"`.
- Reference controls: E045 `11.6534 TPS`, E053 `11.7681 TPS`, E056 `11.6726 TPS`, E058 `11.6132 TPS`.
- Same-session r3 `~0.5-1.5%` wins may matter as stackable opt-in gains, but no current candidate survived that bar.

Reopen only for a new upstream signal, changed benchmark/model/route mix, MTP-enabled GGUF, or a new design with a modeled `>=2%` wall ceiling.

## Subsystem Docs

- `src/PERF_RESEARCH_NOTES.md` covers host/runtime shape control: scheduler, graph reuse, batch splitting, logits/sampling, and model graph routing.
- `ggml/src/ggml-cuda/PERF_RESEARCH_NOTES.md` covers ROCm kernels: Gated Delta Net, FlashAttention, MMQ, and HIP build constraints.

## Experimental Build Corridor

- `build-rocm-fa-reduced` is the current compile corridor for aggressive FATTN/GDN experiments after fresh `fattn.cu` edits started hitting `amdgcn-link command failed due to signal`.
- Configure with `GGML_HIP_QWEN_FA_REDUCED=ON` and `GGML_OPENMP=OFF`; the reduced mode is OFF by default and is not a general ROCm build.
- Smoke benchmark `nonmtp-fa-reduced-ub192-noreuse-20260511-r1` completed on the active lane at `8.46 TPS`, so the dispatcher is valid for A/B work but not a speedup by itself.
- The reduced dispatcher supports `GGML_QWEN_FA_REDUCED_FORCE=vec|wmma_f16` for FATTN selector smoke tests; both forced variants failed to improve the active lane.
- MMVQ source edits still hit `amdgcn-link`, even after reducing the switch to Qwen `q3_K/q4_K/q6_K`; keep MMVQ work on hold until it can be split out more deeply.

## Aggressive Patch Ladder

1. Make the benchmark lane hard to misuse.
   - Use `--no-reuse` instead of manually repeating `--cache-ram 0 --ctx-checkpoints 0`.
   - Keep `--server-extra "--spec-type none"` for non-MTP runs when explicitness matters.
2. Build shape-level instrumentation before kernel edits.
   - Log ubatch sequence shapes, GDN token histograms, FATTN selected kernels, and graph reuse decisions.
   - Prefer low-volume summaries, not full per-node graph dumps.
3. Patch scheduler and split policy when possible.
   - Shape-aware prompt split planning can avoid pathological tails while staying under `ubatch <= 256`.
   - Dual PP/TG scheduling already helps decode; next target is a prefill-specific graph reserve that avoids oversized or slow shapes.
   - First local control: `LLAMA_UBATCH_SPLIT_POLICY=tail-avoid` with optional `LLAMA_UBATCH_SHAPE_PREFERRED=192`.
4. Patch kernels only when build reproducibility is restored.
   - Fresh edits to `fattn.cu` and `gated_delta_net.cu` can trigger `amdgcn-link command failed due to signal` on this machine.
   - Kernel work should first reduce compile/link pressure or split experiments into narrower translation units.
   - Current local route: use `build-rocm-fa-reduced` for smoke/A-B validation, then port successful changes back to the normal ROCm build for final measurements.
   - Latest timing traces show host graph overhead is tiny versus device compute, so this is now the primary path for a real 10% breakthrough.
5. Treat MTP separately.
   - MTP diagnostics live in `MTP_RECHECK_2026-05-11.md`.
   - Do not mix MTP claims with the non-MTP lane unless the model, prompt, and metric are identical.

## Known No-Go Areas

- Current archived lane: broad or shape-specific MMQ for Q3_K `ne11=2048`, Q3_K 128-thread dequant, simple half2/unroll store variants, compute16, generic hipBLASLt/Stream-K env sweeps, and GDN chunk-size sweeps.
- Disabling fused GDN chunked prefill hangs at the first large prompt batch on `ub192`.
- `GGML_GDN_CHUNK_SIZE` sweeps around `ub192` were noise only.
- `GGML_CUDA_DISABLE_FUSION=1`, `--backend-sampling`, `build-rocm-exp`, `build-rocm-compare`, and KV `q8_0/q8_0` regressed on the active lane.
- `GGML_CUDA_DISABLE_GRAPHS=1` was only a sub-1% fluctuation, not progress.
- `--spec-type ngram-mod` produced zero draft tokens on `v2-review` and did not improve wall TPS.
