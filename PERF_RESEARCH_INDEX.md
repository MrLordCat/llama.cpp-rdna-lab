# Performance Research Index

Local navigation map for Qwen3.6 ROCm performance work in this fork. This is not upstream llama.cpp documentation.

## Current Status

The 2026-05-18 acceleration cycle is archived in
`docs/research/archive/2026-05-fast-probe-cycle/PERFORMANCE_ARCHIVE_2026-05-18.md`. A new post-E264 research mode
is active for major RDNA4/Vulkan topology work, starting from:

- `docs/research/MAJOR_TOPOLOGY_WORKFLOW.md`
- `docs/research/major-topology/README.md`
- `docs/research/CONTEXT_130K_WORKFLOW.md`
- `docs/research/EXPERIMENTS_DIGEST.md`
- `docs/research/BENCH_HISTORY_POLICY.md`
- `docs/research/HYPOTHESES.md`
- `docs/research/RESULTS_LOG.md`

Use the archive only for historical context. Do not resume the old quick-probe
cycle by inertia.

Current dense 130k lane:

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Context: `ctx=131072` (~130k), cold-first, repo-snapshot real context, thinking enabled, no reuse, no v2 prime pass.
- Baseline status: measured quick 130k baselines exist for Vulkan and ROCm. Existing 12k/32k/64k/128k rows are historical references only.
- Expected constraint: RX 9070 XT 16 GB cannot keep the whole dense 27B + 130k KV/working set purely VRAM-resident; RAM-spill/residency/PCIe behavior is part of the target.
- Quick baseline contract: `real-context-chars=24576`, q4_0/q4_0 KV, `quick:triage_diff`, `max_tokens=16`, cold/no-reuse/no-prime. Vulkan current best uses `b512/ub256`; ROCm uses `b512/ub128`.
- Vulkan baseline: D005 `d005-vulkan-default-splitk-confirm3`, `--spec-type none --no-mmap`, measured `1.7898 TPS` r3, prompt `934.81 tok/s`, decode `43.59 tok/s` at `b512/ub256`.
- ROCm baseline: `p002-rocm-ub128-current-confirm3`, `--spec-type none`, measured `1.5200 TPS` r3, prompt `801.71 tok/s`, decode `29.07 tok/s`.
- Low-level-language experiment: [D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md](docs/research/major-topology/D002_P002_ROCM_LOW_LEVEL_Q3K_BODY.md) scopes the next ROCm route to a standalone HIP/RDNA4 Q3_K MMQ body gate. Fresh P002 trace shows Q3_K MMQ is `~47%` of diagnostic wall; S002A padded 32-bit load-only scout is correct but rejected as standalone because it does not produce a robust LDS-mode gain.

Archived dense Vulkan 12k lane:

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Backend: Vulkan on RX 9070 XT / AMD proprietary driver.
- Workload: `scripts/agent_workload_bench.py --tasks quick --task-ids triage_diff --ctx-size 12288 --batch-size 7168 --ubatch-size 1024 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --max-tokens 64 --server-extra "--spec-type none"`.
- Current best: E257 r3 `7.0319 TPS`, prompt `999.22 tok/s`, decode `40.93 tok/s`.
- Rejected nearby transfers: E258 transpose-A, E259 `b7680`/f16 KV default, E260 graphics queue/no-mmap/`b8192`/f16-disable, E264 FFN F16 src1 casts.

Archived final practical no-spec ROCm lane:

- Model: `models/Qwen3.6-27B-Q3_K_S.gguf`.
- Backend: ROCm/HIP, target `gfx1201`, preferred build `build-rocm-vec`.
- Workload: `scripts/agent_workload_bench.py --tasks quick --task-ids triage_diff,review_bug --ctx-size 12288 --batch-size 6144 --ubatch-size 2048 --cache-type-k q4_0 --cache-type-v q4_0 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --no-disable-thinking --max-tokens 120 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0"`.
- Reference controls: E045 `11.6534 TPS`, E053 `11.7681 TPS`, E056 `11.6726 TPS`, E058 `11.6132 TPS`.
- Same-session r3 `~0.5-1.5%` wins may matter as stackable opt-in gains, but no current candidate survived that bar.

Reopen archived short-context lanes only for a new upstream signal, changed benchmark/model/route mix, MTP-enabled GGUF, or a new design with a modeled `>=2%` wall ceiling.

## Subsystem Docs

- `src/PERF_RESEARCH_NOTES.md` covers host/runtime shape control: scheduler, graph reuse, batch splitting, logits/sampling, and model graph routing.
- `ggml/src/ggml-cuda/PERF_RESEARCH_NOTES.md` covers ROCm kernels: Gated Delta Net, FlashAttention, MMQ, and HIP build constraints.

## Experimental Build Corridor

- `build-rocm-fa-reduced` is the current compile corridor for aggressive FATTN/GDN experiments after fresh `fattn.cu` edits started hitting `amdgcn-link command failed due to signal`.
- Configure with `GGML_HIP_QWEN_FA_REDUCED=ON` and `GGML_OPENMP=OFF`; the reduced mode is OFF by default and is not a general ROCm build.
- Smoke benchmark `nonmtp-fa-reduced-ub192-noreuse-20260511-r1` completed on the archived short-context lane at `8.46 TPS`, so the dispatcher is valid for A/B work but not a speedup by itself.
- The reduced dispatcher supports `GGML_QWEN_FA_REDUCED_FORCE=vec|wmma_f16` for FATTN selector smoke tests; both forced variants failed to improve the archived short-context lane.
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
- `GGML_CUDA_DISABLE_FUSION=1`, `--backend-sampling`, `build-rocm-exp`, `build-rocm-compare`, and KV `q8_0/q8_0` regressed on the archived short-context lane.
- `GGML_CUDA_DISABLE_GRAPHS=1` was only a sub-1% fluctuation, not progress.
- `--spec-type ngram-mod` produced zero draft tokens on `v2-review` and did not improve wall TPS.
