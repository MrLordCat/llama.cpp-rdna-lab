# ROCm Kernel Performance Research Notes

This file maps ROCm/HIP kernel areas relevant to Qwen3.6 non-MTP acceleration on RDNA4 (`gfx1201`).

## Key Files

- `ggml/src/ggml-cuda/gated_delta_net.cu`
  - Implements `GGML_OP_GATED_DELTA_NET` for CUDA/HIP.
  - RDNA4 uses chunked prefill when `!keep_intermediates`, `cc` is RDNA4, and `n_tokens >= 128`.
  - Current chunk size policy is `96` for `n_tokens <= 256`, otherwise `128`, with `GGML_GDN_CHUNK_SIZE` override.
- `ggml/src/ggml-cuda/fattn.cu`
  - Selects FlashAttention kernels: `vec`, `tile`, `wmma_f16`, `mma_f16`.
  - On active `ub192` traces, prefill routes mostly through `Q1=192`, selected `wmma_f16`; decode routes through tiny `Q1=1/2`, selected `vec`.
- `ggml/src/ggml-cuda/mmq.cuh`
  - Quantized matrix multiply path used heavily by Q3/Q4 model weights.
  - Previous traces showed selector changes were not the cause of the `ub824/832` cliff, but MMQ still matters for decode throughput.

## Current Kernel Facts

- GDN chunk-size sweeps on `ub192` (`64/80/96/128`) were flat around `8.46-8.47 TPS`.
- Disabling chunked GDN is unsafe: it hung after the first `6144/8030` prompt batch.
- Existing `build-rocm-wmma`, `build-rocm-compare`, and `build-rocm-exp` did not beat `build-rocm-vec` on the active `ub192` lane.
- Fresh edits to `fattn.cu`, `gated_delta_net.cu`, or `mmvq.cu` can hit `amdgcn-link command failed due to signal`; after reverting, cached rebuilds link again.

## Aggressive Kernel Patch Ideas

### Build Pressure Reduction First

Goal: make kernel experimentation possible again.

Possible design:

- Split high-template FlashAttention or GDN experiments into narrower translation units.
- Reduce instantiated FATTN cases for local RDNA4 experiments where possible.
- Create a local experimental CMake option for a reduced kernel matrix so A/B patches can link reliably.

Initial control:

- `GGML_HIP_QWEN_FA_REDUCED=ON` builds a local experimental HIP backend with FlashAttention tile/MMA template instances disabled and filters those kernels out at runtime.
- Use it only for Qwen/RDNA4 A/B runs that remain on `vec`/`wmma_f16`; it is not a general ROCm backend configuration.
- Keep normal ROCm builds without this option for final measurements unless the reduced mode itself is the thing being tested.
- Runtime selector override for smoke tests: `GGML_QWEN_FA_REDUCED_FORCE=vec|wmma_f16`.

Validated reduced build command on Windows/ROCm:

```powershell
cmake -B build-rocm-fa-reduced -G Ninja `
  -DGGML_HIP=ON `
  -DGGML_HIP_ROCWMMA_FATTN=ON `
  -DGGML_HIP_ROCWMMA_INCLUDE_DIR="${PWD}/third_party/rocwmma/rocWMMA-rocm-7.1.0/library/include" `
  -DGGML_HIP_QWEN_FA_REDUCED=ON `
  -DGGML_OPENMP=OFF `
  -DAMDGPU_TARGETS=gfx1201 `
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang.exe" `
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm-fa-reduced --target llama-server -j 1
```

Validation result:

- `build-rocm-fa-reduced/bin/llama-server.exe` built successfully after disabling OpenMP in this fresh build dir.
- `nonmtp-fa-reduced-ub192-noreuse-20260511-r1` completed on the active lane at `8.46 TPS`; prompt eval was `820.43 tok/s`, decode eval was `27.47 tok/s`.
- This confirms the reduced dispatcher is usable for controlled FATTN/GDN A/B work, but it is not a performance improvement by itself.
- FATTN selector checks were negative: forcing `vec` for `Q1=192` dropped prompt eval to `580.15 tok/s`; forcing `wmma_f16` for tiny decode stayed at `27.51 tok/s` decode eval.
- MMVQ reduction is not solved by limiting the type switch to Qwen `q3_K/q4_K/q6_K`; `mmvq.cu` still failed `amdgcn-link`, so MMVQ work needs a deeper split than source-specific compile definitions.

Why this comes first:

- Without reliable HIP linking, kernel hypotheses cannot be validated and should not remain in the tree.

### FATTN Selector A/B

Goal: test whether `Q1=192` should use a different kernel than current `wmma_f16`.

Result:

- `GGML_QWEN_FA_REDUCED_FORCE=vec` confirmed that `vec` is worse for `Q1=192` prefill.
- `GGML_QWEN_FA_REDUCED_FORCE=wmma_f16` confirmed that using WMMA for tiny decode does not beat the default vec decode path.

Possible future design:

- Test only genuinely new kernels or selector options; simple `vec`/`wmma_f16` flips are now closed.
- Keep default behavior unchanged.

Validation:

- Rebuild with reduced kernel matrix if needed.
- Run `v2-review --no-reuse` with trace to verify selected kernel actually changes.

### GDN Specialized Prefill Kernel

Goal: specialize the hot `S_v=128`, non-KDA, `keep_intermediates=false`, single-sequence path.

Possible design:

- Keep generic kernel for all models.
- Add a specialized launch path for `S_v=128`, `KDA=false`, `n_seqs=1`, `n_tokens` in the active corridor.
- Experiment with fewer warps, state layout, or multiple columns per warp only behind an env flag.

Validation:

- Trace GDN token histograms and prompt eval TPS.
- Abort and revert immediately if prefill hangs or if the improvement is below 1%.

### MMQ Decode Route

Goal: improve the decode phase, which remains around `27.5 tok/s` on the active lane.

Current finding:

- `GGML_TRACE_MMQ_PATH=1` on the active lane shows MMQ route summaries for prefill only (`Q3_K/Q4_K`, `ncols=192/158`); decode matvec is in MMVQ.
- Direct `mmvq.cu` edits are currently blocked by `amdgcn-link`, even with a reduced Qwen-only type switch.

Possible design:

- Split MMVQ into smaller translation units or isolate Q3_K/Q4_K/Q6_K kernels without including the full current `mmvq.cu`/`mma.cuh` pressure.
- Only then test Q3_K nwarps or rows-per-block variants for `ncols_dst=1`.

Validation:

- Compare `decode_eval_tps` separately from wall TPS.
- Avoid mixing speculative decoding or prompt reuse into MMQ decode measurements.

## Validation Command Template

```powershell
python scripts/agent_workload_bench.py --label <label> `
  --server-bin build-rocm-vec/bin/llama-server.exe `
  --model models/Qwen3.6-27B-Q3_K_S.gguf `
  --tasks v2-review --runs 1 `
  --ctx-size 12288 --batch-size 6144 --ubatch-size 192 `
  --cache-type-k q4_0 --cache-type-v q4_0 `
  --server-extra "--spec-type none" `
  --real-context-mode repo-snapshot --real-context-chars 21872 `
  --no-reuse --no-v2-prime-pass --no-disable-thinking --max-tokens 120
```
