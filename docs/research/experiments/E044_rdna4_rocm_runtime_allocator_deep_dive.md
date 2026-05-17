## Metadata

- Experiment ID: E044
- Date: 2026-05-17
- Owner: Copilot
- Branch/Commit: master / working tree
- Target lane: C01 cold-first, quick tasks triage_diff+review_bug, ctx=12288, b=6144, ub=192, q4_0/q4_0, spec=none, no-reuse

## Hypothesis

- Statement: на RDNA4 часть деградаций и флат-профилей в C01 может сидеть ниже kernel-selector слоя, в runtime/allocator/graph/rocBLAS route, даже когда маршруты MMQ/MMVQ формально не меняются.
- Mechanism: плохая резидентность compute vbuffer, overhead graph-optimizer, или неудачный rocBLAS route для малых F32 GEMM может съедать эффект от локальных kernel-правок.
- Why now: серия code-only кандидатов в MMQ/MMVQ не дала устойчивого cold-first роста выше 9.26 TPS, значит нужно сместить фокус на ROCm-layer причины.

## Math / Theory

- Assumptions:
  - wall-time ускорение ограничено долей времени route-групп, но на RDNA4 существенный штраф может приходить из allocator/runtime, не отражаясь напрямую в route-count.
  - если проблема в runtime/layout, тогда при неизменном topological route будет совместное замедление нескольких op-групп.
- Expected speedup corridor: +0.5%..+3% для C01 cold-first при удачном попадании в runtime bottleneck.
- Failure conditions:
  - candidate меняет только распределение шума между cold спайками без устойчивого выигрыша в wall TPS;
  - candidate улучшает один route, но ухудшает reserve/compute-buffer или F32 backend;
  - build/link instability ломает воспроизводимость.

## Implementation Plan

1. Minimal code surface to change:
   - ggml/src/ggml-alloc.c: ROCm compute vbuffer chunk policy (только ROCm scope).
   - ggml/src/ggml-cuda/ggml-cuda.cu: RDNA4 graph optimizer gating/diagnostic path.
   - ggml/src/ggml-cuda/mmq.cu: stream-k RDNA4 threshold and scope alignment.
   - ggml/src/ggml-cuda/mmvq.cu: RDNA4 small_k/nwarps rows-per-block policy (env-gated).
2. Guard rails:
   - все risky изменения делать env-gated;
   - default behavior менять только при подтвержденном cold-first выигрыше на текущем lane;
   - сравнение строго с cold-first baseline того же lane.
3. Rollback path:
   - git restore --staged --worktree <edited files>
   - повторный контрольный прогон lane без candidate.

## Benchmark Plan

- Baseline command:
  - python scripts/agent_workload_bench.py --label c01-rocm-base-r1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids triage_diff,review_bug --runs 1 --ctx-size 12288 --batch-size 6144 --ubatch-size 192 --cache-type-k q4_0 --cache-type-v q4_0 --server-extra "--spec-type none --cache-ram 0 --ctx-checkpoints 0" --real-context-mode repo-snapshot --no-reuse --background-server-policy fail --no-v2-prime-pass --no-disable-thinking --max-tokens 120
- Candidate command:
  - тот же контракт, только code-change; если Windows tasklist flaps, запускать сервер вручную и runner через --no-start --background-server-policy ignore.
- Number of runs:
  - быстрый цикл: runs=1
  - подтверждение promising/пограничных: runs=3
- Artifacts path:
  - build_logs/agent-workload/

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- task TPS stdev
- server prompt eval tok/s and eval tok/s (из server log)
- trace compare для route-level причинности при спорных дельтах

## Result

- Outcome: tie/regression on tested runtime knobs
- Delta:
  - baseline `c01-e044-runtime-base-r1`: `9.23 TPS`
  - `ROCBLAS_USE_HIPBLASLT=1`: `9.23 TPS` (`0.00%`)
  - `GGML_CUDA_NO_PINNED=1`: `9.24 TPS` (`+0.11%`, noise-level)
  - `ROCBLAS_USE_HIPBLASLT=1 + GGML_CUDA_NO_PINNED=1`: `9.22 TPS` (`-0.11%`)
  - `GGML_CUDA_GRAPH_OPT=1 + GGML_CUDA_ALLOW_RDNA4_GRAPH_OPT=1`: `9.22 TPS` (`-0.11%`)
  - `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1`: `9.22 TPS` (`-0.11%`)
- Confidence: low for tiny +/-0.01 deltas, sufficient to reject as practical win on this lane
- Recommendation: do not promote any tested runtime-only knob as default for C01 cold-first; continue with deeper code-level ROCm/RDNA4 hypotheses

### Phase B candidate R1: RDNA4 MMF-F32 route override (rejected)

- Code probe:
  - temporary env-gated candidate in `ggml/src/ggml-cuda/mmf.cu`: allow `GGML_TYPE_F32` MMF route on RDNA4 when `GGML_RDNA4_MMF_F32=1`.
- A/B contract:
  - baseline `c01-e044b-mmf-f32-base-r1`: `9.24 TPS`.
  - candidate `c01-e044b-mmf-f32-r1` with `GGML_RDNA4_MMF_F32=1`: hard-timeout at `30.01s` on first task.
- Failure signature (server log):
  - repeated runtime errors: `HIP kernel mul_mat_f has no device code compatible with HIP arch 1300`.
- Decision:
  - reject candidate, revert code immediately.
  - post-revert control `c01-e044b-mmf-f32-postrevert-r1`: `9.23 TPS` (baseline corridor restored).

### Phase B candidate R2: disable RDNA4 batched F32 cublas (rejected)

- Code probe:
  - temporary env-gated candidate in `ggml/src/ggml-cuda/ggml-cuda.cu`: when `GGML_RDNA4_NO_BATCHED_CUBLAS_F32=1`, disable `use_batched_cublas_f32` for RDNA4 and fall back to non-batched backend route.
- A/B contract:
  - baseline `c01-e044r2-base-r1`: `9.22 TPS`.
  - candidate `c01-e044r2-nobatchedf32-r1`: `9.22 TPS`.
- Decision:
  - no measurable cold-first gain (tie within noise), reject and revert.

## Notes

- Surprises:
  - harness иногда зависает в find_background_llama_servers/tasklist на Windows MINGW, обход через manual server + --no-start.
  - На `ctx=12288,ub=192` allocator single-chunk negative control не даёт cliff-сигнала (ожидаемо для малого compute buffer), но даёт небольшой минус к TPS.
- Follow-up action:
  - Phase A закрыт для простых runtime knobs: значимых wins не найдено.
  - Phase B: переход к более глубоким code-level изменениям в ROCm-path (allocator/layout internals или RDNA4-specific compute/load path), с обязательным cold-first A/B и immediate revert при нейтральном результате.
