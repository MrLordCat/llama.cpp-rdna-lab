# E127 CPU Q3_K Prefetch Probe

## Metadata

- Experiment ID: E127
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ `68664bb85`, local Q3_K shuffle preload still present
- Hypothesis ID: H37
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan build, `-ngl 0`, `--no-mmap`, q4 KV, FlashAttention on, `ctx=4096`, `batch=512`, `ubatch=128`, `max_tokens=64`, no reuse, thinking on

## Hypothesis

- Statement: prefetching the next Q3_K/Q8_K blocks in `ggml_vec_dot_q3_K_q8_K` may improve CPU fallback decode.
- Mechanism: E125/E126 show the lane is memory/dequant heavy. A small `_mm_prefetch` for the next blocks could hide memory latency if the hardware prefetcher is not already sufficient.
- Why now: this is a cheap gate before larger Q3_K repack work.

## Implementation Plan

1. Add an AVX2-only prefetch inside the Q3_K block loop:
   - `_mm_prefetch((const char *) (x + i + 2), _MM_HINT_T0)`
   - `_mm_prefetch((const char *) (y + i + 2), _MM_HINT_T0)`
2. Rebuild `build-vulkan --target llama-server`.
3. Run one 64-token real-server gate against the current no-prefetch baseline.
4. Revert immediately unless the result is clearly above noise.

## Benchmark Plan

Command shape:

```powershell
python scripts\agent_workload_bench.py --label <label> --server-bin build-vulkan\bin\llama-server.exe --model models\Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug --runs 1 --ctx-size 4096 --batch-size 512 --ubatch-size 128 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 0 --max-tokens 64 --real-context-mode off --server-extra "--spec-type none --no-mmap" --no-disable-thinking --no-reuse --background-server-policy fail --write-diagnostics
```

## Result

| Label | Route | Aggregate TPS | Prompt eval | Decode eval |
| --- | --- | ---: | ---: | ---: |
| `e127-q3k-prefetch-base64-r1` | baseline, no prefetch | `2.0716` | `31.91 tok/s` | `2.42 tok/s` |
| `e127-q3k-prefetch-candidate64-r1` | prefetch `i+2` | `2.0950` | `32.30 tok/s` | `2.44 tok/s` |

- Outcome: reject/revert as too small for promotion.
- Delta: aggregate `+1.13%`; decode `+0.02 tok/s`.
- Confidence: medium. The direction is positive but the magnitude is well inside normal single-run CPU/server noise.
- Recommendation: do not keep the prefetch patch. Future CPU work needs either a stronger local microbenchmark or structural Q3_K repack/interleaving.

## Notes

- The prefetch patch was reverted after the probe and `llama-server` was rebuilt.
- Why the hypothesis likely missed: the Q3_K/Q8_K streams are regular, so Zen 3 hardware prefetch is probably already doing most of the useful work, while the remaining bottleneck is dequant/scale arithmetic and memory bandwidth.

## Artifacts

- `build_logs/agent-workload/e127-q3k-prefetch-base64-r1.csv`
- `build_logs/agent-workload/e127-q3k-prefetch-base64-r1.diagnostics.md`
- `build_logs/agent-workload/e127-q3k-prefetch-base64-r1.server.log`
- `build_logs/agent-workload/e127-q3k-prefetch-candidate64-r1.csv`
- `build_logs/agent-workload/e127-q3k-prefetch-candidate64-r1.diagnostics.md`
- `build_logs/agent-workload/e127-q3k-prefetch-candidate64-r1.server.log`
