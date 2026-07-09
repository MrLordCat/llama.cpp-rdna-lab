# E268 ROCm MTP Windowed NextN Gate

## Metadata

- Experiment ID: E268
- Date: 2026-07-09
- Owner: Codex
- Branch/Commit: `research/cherry-pick-upstream`, dirty working tree
- Target lane: ROCm two-GPU Qwen3.6 dense MTP, `Qwen3.6-27B-Q3_K_S_mtp.gguf`

## Hypothesis

- Statement: The upstream-style MTP path should keep the E266 decode gain while avoiding the E267 large-prompt NextN extraction tax.
- Mechanism: disable target `embeddings_nextn` and skip MTP `process()` for the bulk of long prompt prefill, then re-enable it for a tail window so the draft context has recent KV state for generation.
- Why now: E266 already reached the desired decode-heavy speedup, while E267 showed prompt extraction was the remaining large-prompt blocker.

## Implementation Plan

1. Add `common_speculative_set_process_enabled()` and an MTP override that toggles target NextN extraction.
2. Track MTP `pending_h` validity by position so a skipped prompt prefix starts cleanly at the tail.
3. Gate server-side MTP processing by `LLAMA_SPEC_PREFILL_WINDOW` (`8192` default, `0` disables gating).

## Benchmark Plan

- Baseline: `--spec-type none`
- Candidate: `--spec-type draft-mtp --spec-draft-n-max 8` (`mtp` is accepted as a local alias)
- Shape: ROCm diagnostic single-GPU `-dev ROCm1 -sm none`, practical dual-GPU `-dev ROCm1,ROCm0 -sm layer -ts 1,1`, `b512/ub128`, q4 KV, FA on, temp `0.0`.
- Important: `-sm none` is not a dual-GPU mode; it means "use one GPU only". Use layer split for practical MTP launches so weights/KV are resident across both cards instead of spilling one card into RAM.
- Runs: `r1` because League of Legends may be active in the background; treat as direction/sanity, not final r3.

## Result

| Lane | Mode | Wall TPS | Prompt tok/s | Decode tok/s | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| short, 159 prompt / 128 out | baseline | `28.3266` | `605.19` | `30.32` | - |
| short, 159 prompt / 128 out | MTP n8 | `47.4238` | `514.10` | `54.04` | `71.52%` |
| repo32k, 11327 prompt / 64 out | baseline | `3.4468` | `696.13` | `28.52` | - |
| repo32k, default window 8192 | MTP n8 | `3.4384` | `659.04` | `46.14` | `59.09%` |
| repo32k, window 2048 | MTP n8 | `3.3735` | `683.45` | `27.28` | `31.47%` |
| ROCm1-only diagnostic, 159 prompt / 256 out | baseline | `28.9357` | `598.78` | `29.93` | - |
| ROCm1-only diagnostic, 159 prompt / 256 out | MTP n8 | `42.6461` | `503.76` | `45.31` | `57.89%` |
| dual layer `ROCm1,ROCm0`, 159 prompt / 256 out | baseline | `24.3710` | `618.77` | `25.06` | - |
| dual layer `ROCm1,ROCm0`, 159 prompt / 256 out | MTP n8 | `39.5312` | `516.74` | `41.71` | `57.89%` |

Short decode-heavy delta: `1.674x` wall and `1.782x` decode versus same-run baseline.

Repo32k prompt-heavy delta: prompt overhead is reduced to about `-5.3%` versus baseline while decode remains `1.62x`. Wall is a tie for 64 output tokens because prefill dominates the run.

Window `2048` recovers prompt throughput but hurts acceptance and decode, so keep the default at `8192`.

Practical two-GPU delta on the layer-split run: `1.622x` wall and `1.664x` decode. The absolute baseline is lower than ROCm1-only because Windows ROCm peer copies are disabled and cross-device layer hops are host-staged, but the two-GPU profile keeps the model resident across both cards and avoids the one-card VRAM/RAM-spill path.

## Decision

Keep the windowed NextN gate. It satisfies the local short/decode-heavy target (`~47 wall TPS`, `54 decode tok/s`) and the practical dual-GPU target (`~39.5 wall TPS`, `41.7 decode tok/s`, `1.62x` wall). GUI/default MTP launch should use `--spec-type draft-mtp --spec-draft-n-max 8` plus `-dev ROCm1,ROCm0 -sm layer -ts 1,1`. Do not claim `1.6x` wall on short-output large-prompt cold runs; MTP accelerates decode, while prefill remains the wall limiter unless output is long enough or prompt cache/reuse is active.

## Artifacts

- `build_logs/agent-workload/mtp-windowed-nextn-short-none-r1.*`
- `build_logs/agent-workload/mtp-windowed-nextn-short-n8-r1.*`
- `build_logs/agent-workload/mtp-windowed-repo32k-none-r1.*`
- `build_logs/agent-workload/mtp-windowed-repo32k-n8-r1.*`
- `build_logs/agent-workload/mtp-windowed-repo32k-n8-win2048-r1.*`
- `build_logs/agent-workload/rocm1-mtp-polish-mt256-none-r1.*`
- `build_logs/agent-workload/rocm1-mtp-polish-mt256-n8-r1.*`
- `build_logs/agent-workload/rocm-dual-layer-mtp-polish-mt256-none-r1.*`
- `build_logs/agent-workload/rocm-dual-layer-mtp-polish-mt256-n8-r1.*`
