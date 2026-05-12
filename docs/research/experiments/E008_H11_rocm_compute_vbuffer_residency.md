# E008: H11 ROCm Compute VBuffer Residency Cliff

## Metadata

- Experiment ID: E008
- Date: 2026-05-12
- Owner: Copilot
- Branch/Commit: local `master` after E007 output-aware reserve fix, before commit
- Target lane: `Qwen3.6-27B-Q3_K_S`, RX 9070 XT / ROCm, `ctx=32768`, `batch=5120`, `ubatch=904/1024`, `q4_0/q4_0`, `ngram-mod`, repo-snapshot first request, no reuse, no v2 prime

## Hypothesis

- Statement: The remaining native `ubatch` cliff is caused by ROCm compute-buffer allocation/residency, not by a model op route switch.
- Mechanism: A single large ROCm graph compute vbuffer allocation can land in a bad physical placement/residency pocket on Windows + RDNA4. Splitting the virtual compute buffer into multiple backend allocations should preserve graph semantics while avoiding the slow placement pocket.
- Why now: After E007 removed the PP-output reservation cliff, `ctx=32768, ub=904/1024` and `ctx=16384, ub=900` still had a broad prompt-prefill slowdown even with full native `PP reserve outputs N -> 1`.

## Discovery Path

1. Reproduced the native cliff with the old guard disabled:
   - `ctx32768/ub900`: fast, about `1038 tok/s` prompt eval.
   - `ctx32768/ub904`: slow, about `293 tok/s` prompt eval.
   - `ctx32768/ub1024`: slow, about `306 tok/s` prompt eval.
2. Compared full kernel traces for fast and slow pockets.
   - Node count was identical (`17588` vs `17588`).
   - FATTN selected kernels and K/V length buckets matched between fast and slow cases.
   - Slowdown was broad: GLU/RMS_NORM/ADD/SSM_CONV and parts of MUL_MAT/FATTN all slowed together.
3. Checked address/offset traces.
   - Hot activation buffers moved into different offsets/modulo-2M pockets, but routes still did not change.
   - Simple base-offset padding did not fix `ub904`, so absolute pointer color alone was not the root cause.
4. Tested compute-vbuffer chunking.
   - Keeping full `PP reserve` and only limiting ROCm compute vbuffer chunk size restored throughput.
   - Disabling chunking with `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` reproduced the slow path.

## Math / Theory

- This is not a projected speedup formula experiment. The expected corridor was qualitative: if residency was causal, a chunking-only change should recover most of the `ub900` fast-band prefill without reducing requested `ubatch`.
- Failure condition: if chunking only changed reported buffer size but prompt eval remained near `300 tok/s`, the residency hypothesis would be rejected.

## Implementation Plan

1. Add a max-chunk policy for ROCm graph compute vbuffers in `ggml/src/ggml-alloc.c`.
2. Preserve the existing `ggml_dyn_tallocr` virtual address abstraction so tensor offsets and graph semantics remain unchanged from the scheduler's perspective.
3. Keep CPU/CUDA/Vulkan behavior unchanged by applying the default only when the backend buffer type name starts with `ROCm`.
4. Add env controls:
   - `GGML_COMPUTE_VBUFFER_MAX_CHUNK_SIZE=<bytes>` for A/B.
   - `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` to reproduce the old behavior.

## Benchmark Plan

- Baseline command shape:

```bash
python scripts/agent_workload_bench.py \
  --label native-singlechunk-ctx32768-ub904-mt1-r1 \
  --server-bin build-rocm-vec/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --tasks v2-mini --runs 1 \
  --ctx-size 32768 --allow-ctx-above-16k \
  --batch-size 5120 --ubatch-size 904 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --gpu-layers 99 --server-extra "--spec-type ngram-mod" \
  --no-reuse --max-tokens 1 \
  --real-context-mode repo-snapshot --real-context-chars 21872
```

- Candidate command: same shape, default ROCm vbuffer chunking enabled.
- Practical command: same shape with `--ubatch-size 1024 --max-tokens 120`.
- Number of runs: `1` for rapid cold-first validation; repeat only if results are borderline. The delta here was large enough to accept without a 3-run tiebreaker.
- Artifacts path: `build_logs/agent-workload/`.

## Metrics

- prompt eval time and prompt eval tokens/s
- total wall time for `max-tokens=1`
- decode eval tokens/s for the practical `max-tokens=120` run
- server log reserve line (`PP reserve outputs N -> 1`)

## Result

| Label | PP reserve | ROCm0 compute | Prompt eval | Decision |
| --- | ---: | ---: | ---: | --- |
| `native-singlechunk-ctx32768-ub904-mt1-r1` | `904 -> 1` | single chunk | `23524.85 ms / 302.87 tok/s` | baseline slow path |
| `native-defaultchunk-ctx32768-ub904-mt1-r1` | `904 -> 1` | `374.84 MiB` | `6862.92 ms / 1038.19 tok/s` | fixed |
| `native-final-ctx32768-ub1024-mt1-r1` | `1024 -> 1` | `424.53 MiB` | `6392.54 ms / 1114.58 tok/s` | fixed, better than guard-era ub900 |
| `native-defaultchunk-ctx16384-ub900-mt1-r1` | `900 -> 1` | `281.54 MiB` | `6798.72 ms / 1047.99 tok/s` | fixed bad ctx pocket |

Practical run:

| Label | PP reserve | Prompt eval | Decode | Total |
| --- | ---: | ---: | ---: | ---: |
| `native-defaultchunk-ctx32768-ub1024-mt120-r1` | `1024 -> 1` | `6394.28 ms / 1114.28 tok/s` | `120 tok / 25.03 tok/s` | `11188.80 ms` |

Outcome: win.

Delta:

- `ub904` prompt eval improved from `302.87 tok/s` to `1038.19 tok/s` (`+242.8%`).
- Native `ub1024` reached `1114.58 tok/s` prompt eval without lowering requested `ubatch`.

Confidence: high for this machine/lane because single-chunk control reproduces the old cliff and default chunking removes it while preserving full reserve.

Recommendation: keep ROCm compute vbuffer chunking default at `256 MiB`; do not restore the old `ubatch` guard/cap.

## Agent Tips

- If a future RDNA4 ROCm `ubatch` cliff appears, first check whether the slow run has full `PP reserve outputs N -> 1` and whether `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` reproduces it.
- Do not assume FATTN/GDN/MMQ route changes from a throughput cliff. Full trace can show identical node counts and kernel routes while the issue is still allocator/residency.
- `data_mod2m` differences are useful evidence, but base offset padding alone is not enough proof or fix; test backend allocation chunking separately.
- Avoid promoting a smaller physical `ubatch` as a final fix if full reserve can be made fast. Caps are useful discriminators, not root-cause repairs.
- For speed claims, record both the prompt-only probe and one practical decode run (`max-tokens=120`) so PP and TG stay separated.

## Notes

- `GGML_COMPUTE_VBUFFER_MAX_CHUNK_SIZE=268435456`, `201326592`, `134217728`, and `67108864` all recovered `ctx32768/ub904`; `256 MiB` was chosen as the conservative default with low fragmentation.
- `GGML_ROCM_COMPUTE_VBUFFER_SINGLE_CHUNK=1` is intentionally retained as a diagnostic switch.
