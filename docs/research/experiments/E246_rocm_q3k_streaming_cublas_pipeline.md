# E246 ROCm Q3_K Streaming cuBLAS Pipeline

## Metadata

- Experiment ID: E246
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H42 / ROCm hot-shape Q3_K route
- Target lane: ROCm cold-first Qwen3.6-27B-Q3_K_S, `ctx=12288`,
  `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, `spec=none`, no
  reuse, thinking on

## Hypothesis

- Statement: a route can improve the hot Q3_K prefill GEMM path without replacing
  rocBLAS by streaming Q3_K dequantization into temporary fp16 row chunks and
  running rocBLAS on those chunks.
- Mechanism: keep rocBLAS for the high-throughput GEMM body, but split src0
  staging into row chunks so dequantization and GEMM can be scheduled as a
  pipeline with bounded fp16 residency.
- Why now: E245 direct Q3FlashMatmul proved direct scalar/WMMA kernels lose to
  rocBLAS, while P3 showed that reducing sync/dequant overhead is useful but not
  enough. The next route should therefore preserve rocBLAS efficiency and attack
  staging/scheduling around it.

## Implementation

- Extended `scripts/research/rocm_q3flashmatmul_scout.cpp` with P4 streaming
  pipeline mode.
- P4 allocates two fp16 staging buffers, uses separate non-blocking streams for
  Q3_K dequant and rocBLAS GEMM, and writes row chunks into the full f32 output
  with `ldc=m`.
- `--pipe-chunk` controls row chunk size. Tested chunks: `2048`, `4096`, `6144`,
  `8192`, and full chunk.
- Runtime integration is not yet enabled; the scout is the point gate.

## Correctness

- Small pipeline gate passed against in-process `Q3_K -> fp16 -> rocBLAS`:
  `pipeline_max_abs=0`, `pipeline_rmse=0`.
- The pipeline writes chunked column-major output with `ldc=m`, so this covers the
  row-offset contract needed for a future runtime route.

## Results

| Shape | Chunk | Baseline ms | Pipeline ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `17408x5120@n2048` | `6144` | `5.1528` | `4.3862` | `1.1748x` |
| `5120x17408@n2048` | `4096` | `5.3959` | `4.6166` | `1.1688x` |
| `10240x5120@n2048` | `6144` | `3.2504` | `2.5051` | `1.2975x` |
| `6144x5120@n2048` | `4096` | `1.9787` | `1.6113` | `1.2281x` |

Additional controls:

- `17408x5120@n2048`, chunk `2048`: `0.6406x`, too many small GEMMs.
- `17408x5120@n2048`, chunk `4096`: `1.0608x`, positive but weaker.
- `17408x5120@n2048`, chunk `8192`: `1.1474x`, positive.
- `17408x5120@n2048`, full chunk `17408`: `1.2606x` in this scout order. Treat
  this as a timing-path warning, not as proof that full-chunk is the correct
  runtime shape; it may include allocation/handle/warmup effects.

## Decision

- Keep P4 as the first E245/E246 route that passes the standalone point gate on
  multiple hot shapes.
- Do not default. The real-server runtime prototype confirmed that the route can
  be activated, but it did not convert the point win into a useful wall win.
- Safe synchronous runtime route:
  - activation trace showed `path=q3k_pipeline_sync` in `llama-server`;
  - proper stream dependency handling regressed the active lane to about
    `7.19-7.21 TPS` vs `7.56-7.64` controls.
- Async runtime route:
  - removing host sync required persistent staging/lifetime work;
  - real-server attempts timed out at the 30 s task limit, including after
    private-stream and event-lifetime fixes.
- Current verdict: keep the standalone scout and runtime env gate as research
  artifacts only; do not promote the streaming pipeline until a new graph-safe
  async design avoids per-matmul synchronization and queue/backlog cliffs.

## Artifacts

- `scripts/research/rocm_q3flashmatmul_scout.cpp`
- `build_logs/agent-workload/e245-q3flash-hot17408-n2048-pipeline-chunk6144-r10.csv`
- `build_logs/agent-workload/e245-q3flash-hot5120-n2048-pipeline-chunk4096-r2.csv`
- `build_logs/agent-workload/e245-q3flash-hot10240-n2048-pipeline-chunk6144-r2.csv`
- `build_logs/agent-workload/e245-q3flash-hot6144-n2048-pipeline-chunk4096-r2.csv`
- `build_logs/agent-workload/e246-rocm12k-pipeline-overlap-r1.diagnostics.json`
- `build_logs/agent-workload/e246-rocm12k-pipeline-async-r1.diagnostics.json`
