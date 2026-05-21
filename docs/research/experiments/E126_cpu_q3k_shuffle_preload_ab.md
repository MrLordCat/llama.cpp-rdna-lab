# E126 CPU Q3_K Shuffle Preload A/B

## Metadata

- Experiment ID: E126
- Date: 2026-05-21
- Owner: Codex
- Branch/Commit: master @ `68664bb85`, with a separate clean worktree for baseline
- Hypothesis ID: H37
- Target lane: Qwen3.6-27B-Q3_K_S, Vulkan build, `-ngl 0`, `--no-mmap`, q4 KV, FlashAttention on, `ctx=4096`, `batch=512`, `ubatch=128`, `max_tokens=32`, no reuse, thinking on

## Hypothesis

- Statement: preloading Q3_K scale-shuffle masks and high-bit masks outside the inner AVX2 dot loop may improve CPU fallback throughput.
- Mechanism: the active CPU decode route calls `ggml_vec_dot_q3_K_q8_K`; avoiding repeated `get_scale_shuffle_q3k(...)` calls and repeated `_mm256_slli_epi16(mone, bit)` construction could reduce hot-loop instruction overhead.
- Why now: E125 identified Q3_K x86 vec-dot as the main CPU fallback code route. The current working tree already contained this micro-change, so it needed an isolated clean-vs-dirty A/B before any claim.

## Benchmark Plan

- Clean baseline: detached worktree at `68664bb85`, built with Vulkan/Ninja and no `quants.c` dirty change.
- Candidate: main worktree `build-vulkan/bin/llama-server.exe` with the local Q3_K mask/shuffle preload change.
- Command shape:

```powershell
python scripts\agent_workload_bench.py --label <label> --server-bin <server> --model models\Qwen3.6-27B-Q3_K_S.gguf --tasks quick --task-ids review_bug --runs 3 --ctx-size 4096 --batch-size 512 --ubatch-size 128 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 0 --max-tokens 32 --real-context-mode off --server-extra "--spec-type none --no-mmap" --no-disable-thinking --no-reuse --background-server-policy fail --write-diagnostics
```

## Result

| Label | Build | Aggregate TPS | Prompt eval | Decode eval |
| --- | --- | ---: | ---: | ---: |
| `e126-q3k-shuf-clean-nommap32-r3` | clean worktree | `1.8067` | `29.97 tok/s` | `2.4833 tok/s` |
| `e126-q3k-shuf-dirty-nommap32-r3` | local shuffle preload | `1.8611` | `32.96 tok/s` | `2.4800 tok/s` |

- Outcome: inconclusive / do not promote as a CPU Q3_K decode win.
- Delta: aggregate `+3.01%`, but decode is unchanged/slightly lower (`2.4833 -> 2.4800 tok/s`). The apparent aggregate gain comes from prompt variance, especially the slower first clean run.
- Confidence: medium that this is not the structural CPU decode improvement we need.
- Recommendation: do not commit or claim the shuffle preload change as a kept speedup without a stronger dedicated microbenchmark or longer r3/r5 gate. Continue H37 toward Q3_K repack/interleaved vec-dot.

## Notes

- The dirty source was not reverted by this experiment because it predated this agent turn. It remains uncommitted and should be treated as a pending local candidate, not accepted code.
- This result is a workflow correction: aggregate TPS can move while decode eval does not, so CPU Q3_K code probes must inspect decode split before promotion.

## Artifacts

- `build_logs/agent-workload/e126-q3k-shuf-clean-nommap32-r3.csv`
- `build_logs/agent-workload/e126-q3k-shuf-clean-nommap32-r3.diagnostics.md`
- `build_logs/agent-workload/e126-q3k-shuf-clean-nommap32-r3.server.log`
- `build_logs/agent-workload/e126-q3k-shuf-dirty-nommap32-r3.csv`
- `build_logs/agent-workload/e126-q3k-shuf-dirty-nommap32-r3.diagnostics.md`
- `build_logs/agent-workload/e126-q3k-shuf-dirty-nommap32-r3.server.log`
