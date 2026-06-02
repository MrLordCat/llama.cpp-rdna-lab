# P004 CPU Performance Plan

Date: 2026-05-30  
Owner: Copilot/perf workspace

## Target

- Model: `Qwen3.6-27B-Q3_K_S.gguf`
- Backend: CPU (AVX2)
- CPU: AMD Ryzen 7 5800X3D (8C/16T, 96MB L3)
- RAM: 64 GB DDR4

## Phase 1: Baseline & Profiling

1. Build CPU server with `-DCMAKE_BUILD_TYPE=Release -DGGML_AVX2=ON`
2. Run quick smoke: `ctx=4096, b=512, ub=512, max_tokens=16, real-context-chars=24576`
3. Run `llama-bench` for per-operation profiling
4. Identify top hotspots (matvec, attention, FFN)

## Phase 2: 3D V-Cache Exploitation

- Map working set sizes to L3 capacity (96MB)
- Study thread pinning: keep all threads on one CCX (cores 0-7)
- Test `OMP_PROC_BIND=close`, `OMP_PLACES=cores`
- Measure L3 hit rate via AMD uProf or perf counters

## Phase 3: Kernel Optimisation

- Q3_K matvec: AVX2 register blocking, prefetch distance
- Multi-threaded matvec for larger batches
- Attention: memory layout, cache blocking

## Phase 4: Pipeline / Runtime

- Thread pool sizing
- Small-batch batching strategy
- NUMA awareness (single socket, less critical)
