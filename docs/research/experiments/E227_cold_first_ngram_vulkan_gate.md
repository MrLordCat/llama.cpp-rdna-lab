# E227 cold-first ngram/Vulkan gate after GUI profile

## Metadata

- Experiment ID: E227
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H09 / H36 gate, H38 backend boundary check
- Target lane: Qwen3.6-27B Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, FlashAttention on, full offload, thinking on, no prompt reuse
- ROCm binary: `build-rocm-vec/bin/llama-server.exe`
- Vulkan binary: `build-vulkan/bin/llama-server.exe`

## Hypothesis

- Statement: the measured repeated/session `ngram-mod 12/16/32` route and the fast Vulkan decode route might also help the first cold request enough to move the +20% target.
- Mechanism:
  - `ngram-mod` could reduce decode wall on cold tasks if effective acceptance appears before reuse.
  - Vulkan could overcome ROCm if its faster decode compensates for slower prefill.
- Why now: the GUI now defaults server-side ngram launches to the measured E226 profile, so the same profile needed a cold-first gate before further promotion language.

## Math / Theory

- Baseline: E226 same-task cold-control r3 was `7.8890 TPS`, prompt mean `5978.04 ms`, decode mean `30.45 tok/s`.
- +20% cold target from that baseline: `9.4668 TPS`.
- Failure condition: if wall TPS remains around `7.9` for ROCm ngram, or Vulkan remains below ROCm because prefill dominates, these are not the next cold-first route.

## Benchmark Plan

- Common workload:
  - `--tasks quick --task-ids triage_diff,review_bug --runs 1`
  - `--ctx-size 12288 --batch-size 6144 --ubatch-size 2048`
  - `--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 999`
  - `--max-tokens 64 --real-context-mode repo-snapshot`
  - `--no-reuse --no-v2-prime-pass --no-disable-thinking`
- ROCm candidates:
  - `--spec-type ngram-mod --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-max 32`
  - older GUI/profile control `--spec-ngram-mod-n-min 48 --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-max 64`
- Vulkan candidates:
  - `--spec-type none`
  - `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` plus `--spec-type none --no-mmap`

## Measured Results

| Label | Backend | Candidate | Aggregate TPS | Prompt ms mean | Decode tok/s mean | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `e226-rocm12k-cold-two-task-specnone-r3-seq` | ROCm | cold baseline r3 | `7.8890` | `5978.04` | `30.45` | baseline |
| `e227-rocm12k-cold-ngram12-16-32-r1` | ROCm | E226 ngram profile | `7.8987` | `5995.01` | `30.845` | tie |
| `e227-rocm12k-cold-ngram24-48-64-r1` | ROCm | older/wider ngram profile | `7.8976` | `5996.79` | `30.85` | tie |
| `e227-vulkan12k-cold-q4-specnone-r1` | Vulkan | q4/q4, spec none | `6.7552` | `7854.255` | `40.56` | regression |
| `e227-vulkan12k-cold-q4-graphicsq-nommap-r1` | Vulkan | graphics queue + `--no-mmap` | `7.0101` | `7522.98` | `40.88` | still below ROCm |

## Result

- Outcome: tie/regression for cold-first speed.
- Delta:
  - ROCm `ngram-mod 12/16/32`: `+0.12%` vs E226 cold baseline, not meaningful.
  - ROCm `ngram-mod 24/48/64`: `+0.11%`, same neutral result.
  - best Vulkan cold gate: `-11.14%` vs ROCm cold baseline despite `~40.9 tok/s` decode.
- Confidence: medium for route decision. Each candidate was r1, but the deltas are far from the `+20%` target and the bottleneck split is consistent across diagnostics.
- Recommendation:
  - Keep GUI ngram `12/16/32` as the measured repeated/session launch profile, not as a cold-first claim.
  - Do not pivot cold 12k default to Vulkan; decode is faster, but prompt/prefill dominates wall time.
  - Continue cold-first work on structural ROCm prefill/decode routes, starting with post-H43 point-level Q3_K split timing.

## Notes

- The cold ngram result explains the E226 route: its speedup requires prompt reuse/checkpoints and bursty repeated decode spans.
- Vulkan remains useful for long-answer/session decode routes, but not for this cold prompt-heavy lane.
- Next command should be a ROCm split/route trace on `build-rocm-vec`, not another no-code shape/spec sweep.

## Artifacts

- `build_logs/agent-workload/e227-rocm12k-cold-ngram12-16-32-r1.diagnostics.md`
- `build_logs/agent-workload/e227-rocm12k-cold-ngram24-48-64-r1.diagnostics.md`
- `build_logs/agent-workload/e227-vulkan12k-cold-q4-specnone-r1.diagnostics.md`
- `build_logs/agent-workload/e227-vulkan12k-cold-q4-graphicsq-nommap-r1.diagnostics.md`
