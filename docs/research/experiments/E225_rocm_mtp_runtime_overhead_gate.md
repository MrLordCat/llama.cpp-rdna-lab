# E225 ROCm MTP runtime overhead gate

## Metadata

- Experiment ID: E225
- Date: 2026-05-25
- Owner: Copilot
- Hypothesis ID: H03 / H10
- Target lane: ROCm Qwen3.6-27B Q3_K_S MTP GGUF, `ctx=12288`, `b=6144`, `ub=2048`, q4/q4 KV, FA on

## Hypothesis

- Statement: the local `Qwen3.6-27B-Q3_K_S_mtp.gguf` might convert high MTP acceptance into a decode/wall win after H43 and the driver update.
- Mechanism:
  - MTP predicts multiple future tokens from the model's own MTP head;
  - if acceptance is high and prompt overhead is modest, decode wall should improve.
- Why plausible:
  - previous docs showed high MTP acceptance on compatible Qwen MTP GGUFs;
  - E225 control uses the same MTP GGUF with `--spec-type none`, avoiding a base-vs-MTP model mismatch.

## Benchmark Plan

- Binary: `build-rocm-vec/bin/llama-server.exe`
- Model: `models/Qwen3.6-27B-Q3_K_S_mtp.gguf`
- Common args:
  - `--tasks quick --task-ids triage_diff --runs 1`
  - `--ctx-size 12288 --batch-size 6144 --ubatch-size 2048`
  - `--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --gpu-layers 999`
  - `--max-tokens 64 --real-context-mode repo-snapshot`
  - `--no-reuse --no-v2-prime-pass --no-disable-thinking`
- Control: `--server-extra "--spec-type none"`
- Candidate: `--server-extra "--spec-type mtp --spec-draft-n-max 3"`

## Measured Results

| Label | Mode | Aggregate TPS | Prompt eval tok/s | Decode tok/s | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `e225-rocm12k-mtpmodel-specnone-control-r1` | `spec=none` | `7.5617` | `1182.62` | `30.50` | same MTP GGUF, MTP disabled |
| `e225-rocm12k-mtpmodel-mtp-d3-cand-r1` | `mtp`, draft `3` | `0.8547` | `106.12` | `14.96` | acceptance high but runtime overhead dominates |

MTP statistics from the candidate:

- draft acceptance: `45 accepted / 53 generated`, local acceptance `0.84906`;
- MTP calls: `#calls(b,g,a) = 1 18 18`;
- MTP draft generation duration: `2295.892 ms`;
- detail duration: `decode=2133.672 ms`, `sample=156.202 ms`.

## Probe Patch Review

Two narrow runtime probes were attempted and reverted:

- target hidden-only route:
  - idea: avoid forcing target vocab logits for every prompt token by exposing full `h_pre_norm` separately;
  - result: ROCm scheduler split/copy assertion before valid prompt timing;
  - conclusion: this requires a correctness-first graph/output contract change, not a hot patch.
- MTP-head reduced-output route:
  - idea: compute MTP-head logits for zero/one prompt mirror rows instead of every prompt token;
  - result: backend scheduler assertion during prompt mirroring;
  - conclusion: Qwen35 MTP graph `out_ids`/zero-output handling is not safe as a quick speed patch.

All code probes were reverted; no MTP runtime change is kept.

## Result

- Outcome: reject as current speed route.
- Delta: `7.5617 -> 0.8547 TPS` (`-88.7%`) for MTP enabled on the same MTP GGUF.
- Classification:
  - not an acceptance problem;
  - bottleneck is runtime overhead, especially prompt mirroring and MTP-head generation/verification path;
  - MTP remains an opt-in correctness/architecture branch, not a near-term +20% speed source for this ROCm lane.

## Artifacts

- `build_logs/agent-workload/e225-rocm12k-mtpmodel-specnone-control-r1.diagnostics.md`
- `build_logs/agent-workload/e225-rocm12k-mtpmodel-mtp-d3-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e225b-rocm12k-mtpmodel-specnone-control-r1.diagnostics.md`
- `build_logs/agent-workload/e225b-rocm12k-mtpmodel-mtp-d3-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e225c-rocm12k-mtpmodel-mtp-d3-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e225d-rocm12k-mtpmodel-mtp-d3-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e225e-rocm12k-mtphead-zeroout-cand-r1.diagnostics.md`
- `build_logs/agent-workload/e225f-rocm12k-mtphead-oneout-cand-r1.diagnostics.md`
