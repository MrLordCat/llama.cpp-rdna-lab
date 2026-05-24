# E201 ROCm Q3_K Padded Storage P1/P2a Prototype

## Metadata

- Experiment ID: E201
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E200 (`ef0f12e49`)
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: ROCm can safely test Vulkan-like Q3_K `110 -> 112` device storage if the first code slice is env-gated and correctness-first.
- Mechanism: add backend-private padded Q3_K storage for non-split CUDA/HIP buffers, add padded-aware Q3_K dequant and MMVQ accessors, and temporarily avoid MMQ under the padded-storage knob until MMQ gets its own accessor pass.
- Why now: E199/E200 show quick padded-layout shortcuts are invalid, but a staged storage route is technically sliceable if `test-backend-ops` catches stride/copy mistakes before real server.

## Math / Theory

- Assumptions:
  - padding replacement cost is `1.818%` of selected Q3_K storage, not a duplicate model copy;
  - `MUL_MAT n=1` exercises the decode-like MMVQ path;
  - `MUL_MAT n=64` can exercise prompt/matrix paths, but P2a may route away from MMQ while MMQ is not padded-aware.
- Expected speedup corridor:
  - no speed claim in P1/P2a;
  - P1/P2a is only a correctness gate for the larger route.
- Failure conditions:
  - build fails;
  - padded-storage env corrupts Q3_K `MUL_MAT`;
  - model load uses unsupported partial Q3_K set/get or split buffer path;
  - disabling MMQ under the knob makes any speed benchmark invalid as a parity claim.

## Implementation Plan

1. Minimal code surface to change:
   - P1a completed:
     - `ggml/src/ggml-common.h`: padded Q3_K block struct;
     - `ggml/src/ggml-cuda/convert.cu/.cuh`: raw-to-padded pack helper and padded Q3_K fp16 dequant helper;
     - `ggml/src/ggml-cuda/ggml-cuda.cu`: default-off `GGML_CUDA_Q3K_PADDED_DEQUANT_PROBE=1` route through cublas fallback for correctness smoke.
  - P1b/P1c completed:
     - `ggml/src/ggml-cuda/ggml-cuda.cu`: default-off non-split Q3_K padded alloc/set/get, CPY fail-closed guard, MMQ guard;
     - `ggml/src/ggml-cuda/convert.cu`: padded-aware Q3_K dequant dispatch from physical padded storage;
     - `ggml/src/ggml-cuda/vecdotq.cuh` and `mmvq.cu`: padded-aware MMVQ vecdot/kernel launch for dense Q3_K decode forms.
   - P2a completed:
     - `ggml/src/ggml-cuda/mmq.cuh`: padded-aware Q3_K MMQ tile loader using `block_q3_K_padded`;
     - `ggml/src/ggml-cuda/mmq.cu`: default-off `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` gate and MMQ args flag;
     - `ggml/src/ggml-cuda/ggml-cuda.cu`: keep MMQ disabled under padded storage unless the P2a MMQ gate is explicitly enabled.
   - Still pending:
     - split-buffer support;
     - partial `set_tensor_2d/get_tensor_2d`;
     - default-on policy and broader MMQ/prefill coverage;
     - MoE MMVQ storage path.
2. Guard rails:
   - P1a env gate `GGML_CUDA_Q3K_PADDED_DEQUANT_PROBE=1`;
  - storage env gate `GGML_CUDA_Q3K_PADDED_STORAGE=1`;
  - padded MMQ env gate `GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1`;
   - default behavior unchanged;
   - no speed benchmark until Q3_K `MUL_MAT` correctness passes;
   - real-server run is sanity only unless MMQ padded accessor is implemented.
3. Rollback path:
   - revert E201 code hunks if build/correctness fails.

## Benchmark Plan

- Baseline command: E200 current-tree Q3_K `MUL_MAT` smokes.
- Candidate command:

```powershell
$env:GGML_CUDA_Q3K_PADDED_DEQUANT_PROBE='1'
build-rocm-vec\bin\test-backend-ops.exe test -b ROCm0 -o MUL_MAT -p "type_a=q3_K,type_b=f32,m=16,n=1,k=256" --output csv
build-rocm-vec\bin\test-backend-ops.exe test -b ROCm0 -o MUL_MAT -p "type_a=q3_K,type_b=f32,m=1,n=64,k=256" --output csv
```

- Number of runs: correctness `r1`; no speed r3 in P1/P2a.
- Artifacts path: `build_logs/agent-workload/e201-rocm-q3k-padded-*`.

## Metrics

- build status
- Q3_K `MUL_MAT` correctness status
- route trace if available
- real-server text sanity only if correctness passes

## Result

- Outcome: P1a padded-dequant correctness probe passed; P1b/P1c backend-private padded storage passed the narrow correctness slice and produced a small opt-in wall win; P2a added padded-aware Q3_K MMQ and produced a confirmed short-lane wall win.
- Correctness:
  - default and `GGML_CUDA_Q3K_PADDED_DEQUANT_PROBE=1` Q3_K `MUL_MAT` smokes passed;
  - `GGML_CUDA_Q3K_PADDED_STORAGE=1` Q3_K `MUL_MAT m=16,n=1,k=256` passed and route traced as `mul_mat_vec_q_direct`;
  - `GGML_CUDA_Q3K_PADDED_STORAGE=1` Q3_K `MUL_MAT m=1,n=64,k=256` passed through `cublas_backend`;
  - `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` Q3_K `MUL_MAT m=1,n=64,k=256` passed and route traced as `mul_mat_q_direct`, `q3k_padded=1`;
  - Q3_K `CPY -> Q3_K` under padded storage is now fail-closed (`supported=0`) to avoid raw 110-byte copies into 112-byte physical storage.
- Point review:
  - before padded-aware MMVQ, storage improved prompt (`7036.66 -> 6463.99 ms`) but collapsed decode (`30.95 -> 14.69 tok/s`) because Q3_K decode fell to cublas;
  - after padded-aware MMVQ, the cublas point run recovered decode (`30.56 tok/s`) and improved prompt (`7036.66 -> 6841.85 ms` in traced runs);
  - hot cublas buckets moved modestly: `17408x5120@2048` `1424.346 -> 1393.866 ms`, `5120x17408@2048` `799.805 -> 785.679 ms`, `6144x5120@2048` `231.192 -> 225.844 ms`.
- Decode MMVQ point review:
  - `LLAMA_GRAPH_REUSE_DISABLE=1` did not disable CUDA graph capture, so initial token-generation timings were enqueue-only and invalid;
  - existing `GGML_CUDA_DISABLE_GRAPHS=1` produced valid `capture=0` timings;
  - Q3_K decode buckets improved locally: fused `nx=5120/gridx=8704` `116.652 -> 115.254 ms`, fused `nx=17408/gridx=2560` `76.090 -> 72.740 ms`, unfused `nx=5120/gridx=5120` `41.021 -> 40.591 ms`, unfused `nx=5120/gridx=6144` `14.687 -> 14.501 ms`.
- P2a MMQ point review:
  - short real-server point lane: `quick/triage_diff`, no real-context, `max_tokens=32`, `GGML_CUDA_DISABLE_GRAPHS=1`;
  - Q3_K MMQ `ncols_max=159`, `mmq_x=80` improved `252.526 -> 231.453 ms` (`+8.34%` faster) with identical call count (`349`) and route proof `q3k_padded=1`;
  - all MMQ timing moved `273.207 -> 250.737 ms` (`+8.22%` faster).
- Wall A/B, no trace, cold-only, no reuse:
  - r1: control `11.6612 TPS`, candidate `11.8871 TPS` (`+1.94%`);
  - r3: control `12.0761 TPS`, candidate `12.1572 TPS` (`+0.67%`);
  - r3 decode eval improved `29.9933 -> 30.4333 tok/s` (`+1.47%`), prompt eval was essentially neutral `1259.06 -> 1260.00 tok/s`.
- P2a wall A/B, no trace, cold-only, no reuse:
  - decode-biased short lane (`quick/triage_diff`, no real-context, `max_tokens=256`) r3 improved `30.2390 -> 30.9884 TPS` (`+2.48%`), decode eval `31.1767 -> 31.9167 tok/s` (`+2.37%`), prompt eval `708.8467 -> 735.3667 tok/s` (`+3.74%`);
  - active prompt-heavy sanity (`repo-snapshot`, `7422` prompt tokens, `max_tokens=120`) r1 improved `11.8483 -> 12.0795 TPS` (`+1.95%`), prompt eval `1209.88 -> 1232.65 tok/s`, decode eval `30.41 -> 30.93 tok/s`;
  - candidate response previews were normal `Thinking Process:` text and all wall runs had `errors=0`.
- Confidence: medium that the storage contract, dense decode MMVQ slice, and padded-aware MMQ slice are correct on the active non-split Qwen lane; low that this is ready as a default because split buffers, partial views, default-on policy, and MoE paths are intentionally not covered.
- Recommendation: keep `GGML_CUDA_Q3K_PADDED_STORAGE=1 GGML_CUDA_Q3K_PADDED_STORAGE_MMQ=1` as opt-in H43 route work, not a default. Next work should either broaden correctness coverage or target the remaining large-prefill cublas/H42 body because the P2a MMQ win only applies where Q3_K MMQ is selected.

## Notes

- This is deliberately not the final performance branch. It proves the storage/copy/stride foundation and shows a real short-lane MMQ win, but the large `ncols=2048` prefill cublas body is still mostly untouched.
- P1a artifact: `build_logs/agent-workload/e201-rocm-q3k-padded-dequant-probe.md`.
- P1b/P1c artifact: `build_logs/agent-workload/e201-rocm-q3k-padded-storage-p1.md`.
- P2a artifact: `build_logs/agent-workload/e201-rocm-q3k-padded-storage-mmq-p2.md`.
- Server MMVQ timing with `GGML_TRACE_MMVQ_TIMING_SYNC=1` needs `GGML_CUDA_DISABLE_GRAPHS=1` for valid decode point data. Without it, token-generation MMVQ launches are inside CUDA graph capture (`capture=1`) and `sync_ms` stays zero. `LLAMA_GRAPH_REUSE_DISABLE=1` did not change this.
