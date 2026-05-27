# Post-C01 Acceleration Scan (2026-05-18)

Archive update: after E053-E059 and external RDNA4 research, the current acceleration cycle is archived in `docs/research/archive/2026-05-fast-probe-cycle/PERFORMANCE_ARCHIVE_2026-05-18.md`. The remaining targets below are parked leads, not an active work queue.

## Scope

This scan closes the C01-default mindset and re-reads the accumulated local research docs for remaining TPS potential on the current bench.

Current practical bench after C01 closeout:
- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- backend/build: ROCm/RDNA4 `build-rocm-vec`, `gfx1201`
- workload: `scripts/agent_workload_bench.py --tasks quick --task-ids triage_diff,review_bug`
- ctx/batch/ubatch: `12288 / 6144 / 2048`
- KV: `q4_0/q4_0`
- speculative mode: `none`
- reuse: off (`--no-reuse`, `--cache-ram 0 --ctx-checkpoints 0`)
- thinking: on (`--no-disable-thinking`)

## Current Baseline Interpretation

E045 recentered the active cold-first prefill lane from old C01 `ubatch=192/1024` work to `ubatch=2048`:
- `ubatch=1024` r3: `11.4240 TPS`, prompt eval `1146.9633 tok/s`.
- `ubatch=2048` r3: `11.6534 TPS`, prompt eval `1197.5567 tok/s`.
- The gain is prompt-led (`+4.41%` prompt eval) with a small decode regression.

E045 trace at `ubatch=2048`:
- prompt node total: `13969.454 ms`
- `MUL_MAT`: `9053.320 ms` (`64.81%` of prompt trace)
- `GATED_DELTA_NET`: `2024.980 ms` (`14.50%`)
- `FLASH_ATTN_EXT`: `668.210 ms` (`4.78%`)
- `MUL_MAT` source type split: `q3_K 84.32%`, `f32 9.80%`, `q4_K 5.83%`

The old C01 `MMQ type=11 ncols_max=192` center is no longer the default target for this bench. The current hot path is large `cublas_backend` / dequant staging for Q3_K prompt matmuls.

## Closed / Do Not Repeat

No default/profile promotion:
- broad `GGML_CUDA_FORCE_MMQ_RUNTIME=1` at `ubatch=2048`: rejected (`10.00 TPS`).
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F=1`: rejected (`11.7908 -> 11.4146 TPS`).
- `ROCBLAS_USE_HIPBLASLT=1` on `ubatch=2048`: neutral/noise (`+0.10%`).
- hipBLASLt/Stream-K env gate on C01: rejected (`-0.97%..-1.22%`).
- GDN chunk sweep at `ubatch=2048`: rejected (`chunk256` best, still `-0.76%`).
- large-prefill shape-specific MMQ override for Q3_K `row_diff=6144, ne10=5120, ncols=2048`: rejected (`1839.27 -> 2529.35 ms`, `+37.52%` slower locally).
- Q3_K 128-thread fp16 dequant variant: rejected (`11.6534` r3 baseline / `11.92` smoke -> `11.46 TPS`).
- C01 scalar selector/resource queue: closed in `decode-hotspots/C01_mul_mat_forward.md`.

## Highest-Potential Remaining Targets

### P1: Q3_K dequant/layout work on large cuBLAS path

Evidence:
- E049 shows Q3_K traced large calls total `10589.99 ms` with stage split:
  - `src0` dequant: `32.29%` (`33.90%` after removing the one-time GEMM outlier),
  - `src1` conversion: `6.74%`,
  - GEMM: `60.97%`.
- E049 also finds one dequant-heavy shape:
  - Q3_K `row_diff=6144, ne10=5120, ncols=2048`
  - total `1839.27 ms`, `src0` dequant `1438.91 ms` (`78.23%`).
- E051 estimates Q3_K dequant effective wall share at about `16.69%`.

Why still open:
- E050 only rejected MMQ as an alternate route for the dequant-heavy shape.
- E051 only rejected a simple 128-thread/two-values-per-thread dequant kernel.
- The broader problem remains: repeated Q3_K -> fp16 conversion is large enough that a different memory/layout strategy could matter.

Implementation entry points:
- `ggml/src/ggml-cuda/convert.cu`
  - current `dequantize_block_q3_K` kernel
  - launch currently uses `<<<nb, 64>>>`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - `ggml_cuda_op_mul_mat_cublas`
  - env-gated split timing: `GGML_TRACE_CUBLAS_SPLIT_TIMING`, `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS`

Promising shapes of work:
- vectorized/coalesced fp16 store variant that keeps the 64-thread occupancy profile instead of repeating E051's 128-thread shape.
- alignment/pitch experiment for the dequant destination buffer to reduce downstream GEMM or store inefficiency.
- dequant-layout variant designed around the exact large Q3_K prompt shapes, measured first with split timing.

Gate before coding:
- rerun split timing on current tree and confirm Q3_K `src0` dequant remains `>=15%` effective wall share.
- reject any candidate with modeled local gain below `~10%`, because `10%` local dequant gain is only about `+1.5%` wall.

### P2: GATED_DELTA_NET specialized prefill kernel, not chunk-size tuning

Evidence:
- E045 `ubatch=2048` trace: GDN is `2024.980 ms`, `14.50%` of prompt trace.
- Effective wall share is roughly `8-9%` on the current lane.
- E047 rejected chunk-size tuning (`256/512/1024/2048`) and larger chunks regressed by up to about `3%` wall.

Why still open:
- E047 only tests chunk policy and fast-exp style knobs.
- `ggml/src/ggml-cuda/PERF_RESEARCH_NOTES.md` still identifies a different idea: a specialized hot path for `S_v=128`, `KDA=false`, `keep_intermediates=false`, `n_seqs=1`.

Implementation entry points:
- `ggml/src/ggml-cuda/gated_delta_net.cu`
  - `launch_gated_delta_net`
  - tracing knobs: `GGML_TRACE_GDN_PATH`, `GGML_TRACE_GDN_TIMING`, `GGML_TRACE_GDN_TIMING_SYNC_HIP`, `GGML_TRACE_GDN_TIMING_PRE_SYNC_HIP`

Promising shapes of work:
- RDNA4-only env-gated specialized kernel for the current hot contract.
- state layout / per-warp token loop changes that do not merely increase `GGML_GDN_CHUNK_SIZE`.

Gate before coding:
- collect current `ubatch=2048` GDN timing histogram.
- require modeled local gain `>=15-20%` for a worthwhile wall result.

### P3: H06 QKV/RoPE kernel-level gate on current `ubatch=2048`

Evidence:
- E037 found an attention/QKV/RoPE-adjacent slice of `575.093 / 3303.800 ms` (`17.41%`) on the older trace and projected a multi-percent local ceiling.
- E038 rejected only a graph-level concat/split prototype (`11.2031 -> 11.1688 TPS`).

Why still open:
- Graph-level fusion overhead is closed, but kernel-level integration is not.
- The current `ubatch=2048` lane may have a different attention-adjacent share; it needs a fresh trace gate before any code.

Implementation entry points:
- `src/llama-graph.cpp`
  - `ggml_mul_mat_aux`
  - Q/K rotation and memory-copy graph paths around `q_cur` / `k_cur`

Gate before coding:
- do not write another graph-concat prototype.
- first compute the current `ubatch=2048` attention/QKV/RoPE share.
- only proceed if local share and modeled ceiling can reach `>=2%` wall.

### P4: Speculative paths as practical profiles, not kernel-default work

Evidence:
- E028/E030 confirm `ngram-mod 24/48/64` as a repeated/steady accelerator, but not a cold-first kernel/default win.
- MTP docs remain high-potential but blocked on a verified MTP-enabled GGUF and wall-time validation.

Why still open:
- This is practical TPS potential for interactive/session use, but it should not be mixed with no-spec cold-first kernel claims.

Gate before work:
- keep separate `spec=none` and speculative baselines.
- for MTP, require an MTP-enabled GGUF and compare aggregate wall time, not only TG.

## Recommended Next Experiment

Archived decision: no next experiment is currently recommended by default. The section below records the historical E053 selection path that was completed before archive closeout.

Run E053 as a no-code selection gate:

1. current `ubatch=2048` no-spec r1 control, no trace.
2. current `ubatch=2048` split trace with:
   - `GGML_TRACE_CUBLAS_SPLIT_TIMING=1`
   - `GGML_TRACE_CUBLAS_SPLIT_TIMING_MIN_NCOLS=1024`
   - optional GDN timing envs in a separate run to avoid over-synchronizing one trace.
3. Parse:
   - Q3_K dequant effective wall share,
   - largest dequant-heavy shapes,
   - GDN timing histogram,
   - QKV/RoPE-adjacent share.
4. Choose one code candidate only if its modeled wall ceiling is at least `2%`.

Most likely first code branch after E053:
- `Q3_K dequant/layout`, because it has the largest measured remaining share and the simple rejected variants do not exhaust the design space.

E053 result:
- Completed on 2026-05-18 as `docs/research/experiments/E053_post_c01_selection_gate.md`.
- Trace-off control: `prefill-e053-control-r1 = 11.7681 TPS`.
- Split timing repeated the stable Q3_K dequant-heavy signature: Q3_K `src0 32.66%`, `src1 6.85%`, `GEMM 60.49%`; Q3_K `6144x5120@ncols2048` stayed `78.23% src0`.
- Kernel-full large-prompt shares: `MUL_MAT 64.50%`, Q3_K `MUL_MAT 54.21%`, `GATED_DELTA_NET 14.76%`, H06 QKV/RoPE-adjacent `18.48%`.
- Decision: proceed first to P1 with two gates. A dequant-only candidate needs roughly `>=25%` local improvement to clear `+2%` aggregate TPS and become default-worthy on its own. Smaller ideas are still useful if they are low-risk, default-off, and survive same-session r3 as stackable `~0.5-1.5%` aggregate wins; r1-only positives are not enough.

## Quick No-Go Map

Do not start with:
- more hipBLASLt / Stream-K env sweeps,
- compute16/fp16-accumulation for current Q3_K cuBLAS route,
- broad or shape-specific MMQ override for the `6144x5120@ncols2048` Q3_K shape,
- GDN chunk-size sweeps,
- C01 `ncols=192` selector/force-x work,
- FATTN selector flips on this lane unless a fresh trace shows FATTN share moved materially above the E045 `4.78%` prompt share.