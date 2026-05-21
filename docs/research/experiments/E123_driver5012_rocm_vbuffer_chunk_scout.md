# E123 Driver 5012 ROCm VBuffer Chunk Scout

## Metadata

- Experiment ID: E123
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ 539f64736
- Hypothesis ID: H11 / H35 adjacency
- Target lane: Qwen3.6-27B-Q3_K_S ROCm cold-first prompt-heavy, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: after AMD driver `32.0.31007.5012`, the best ROCm compute vbuffer chunk size may differ from the current default `256 MiB`.
- Mechanism: E008 fixed a residency cliff by chunking ROCm compute virtual buffers. Driver changes can shift the allocator/residency tradeoff. Smaller chunks could improve residency; single-chunk should remain a negative control.
- Why now: E115 rejected batch/ubatch retuning. Before deeper H35 kernel work, cheaply verify that allocator chunking is not leaving prefill TPS on the table.

## Benchmark Plan

- Baseline reference: E113 cold-first q4 `11.9858 TPS`.
- One-run gates:
  - `GGML_COMPUTE_VBUFFER_MAX_CHUNK_SIZE=134217728` (`128 MiB`)
  - `GGML_COMPUTE_VBUFFER_MAX_CHUNK_SIZE=67108864` (`64 MiB`)
  - `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` negative control
- Confirm r3 only if a candidate beats E113 by more than noise.

## Result

- Outcome: reject smaller chunk sizes; keep current default.
- Baseline reference: E113 cold-first `11.9858 TPS`, prompt eval `1272.84 tok/s`, decode `28.9183 tok/s`.
- Gates:
  - `128 MiB`: `11.8684 TPS`, prompt `1250.925 tok/s`, decode `28.945 tok/s`.
  - `64 MiB`: `11.8013 TPS`, prompt `1239.930 tok/s`, decode `28.935 tok/s`.
  - `single-chunk`: `11.7784 TPS`, prompt `1236.165 tok/s`, decode `28.910 tok/s`.
- Delta:
  - `128 MiB`: `-0.98%` vs E113.
  - `64 MiB`: `-1.54%` vs E113.
  - `single-chunk`: `-1.73%` vs E113.

## Interpretation

- The current 12k cold prompt-heavy lane is not sitting in the old catastrophic vbuffer residency cliff.
- Smaller chunks do not improve prefill; they slightly reduce prompt eval.
- `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` remains a useful negative control for future cliff diagnosis, but it is not a speed route here.
- Continue H35 in kernel/route space instead of allocator chunk-size tuning.

## Artifacts

- `build_logs/agent-workload/e123-driver5012-rocm-cold-vbuf128m-r1.csv`
- `build_logs/agent-workload/e123-driver5012-rocm-cold-vbuf128m-r1.diagnostics.md`
- `build_logs/agent-workload/e123-driver5012-rocm-cold-vbuf64m-r1.csv`
- `build_logs/agent-workload/e123-driver5012-rocm-cold-vbuf64m-r1.diagnostics.md`
- `build_logs/agent-workload/e123-driver5012-rocm-cold-vbuf-single-r1.csv`
- `build_logs/agent-workload/e123-driver5012-rocm-cold-vbuf-single-r1.diagnostics.md`
