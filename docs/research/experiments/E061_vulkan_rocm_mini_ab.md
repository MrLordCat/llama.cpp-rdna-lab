# E061 Vulkan vs ROCm mini A/B

## Metadata

- Experiment ID: E061
- Date: 2026-05-19
- Owner: Copilot
- Branch/Commit: master @ 8c1195ab4 plus local experimental `ngram-mtp` worktree
- Target lane: RX 9070 XT, Windows Vulkan driver 26.3.1, ROCm build-rocm-vec, Vulkan build-vulkan

## Hypothesis

- Statement: Vulkan may be competitive with or faster than ROCm on short decode-heavy or mixed workloads on RDNA4.
- Mechanism: Vulkan shader path and AMD proprietary driver may have lower generation overhead for some shapes, while ROCm is expected to remain strong in prompt processing.
- Why now: Local performance notes identify ROCm vs Vulkan parity as a useful comparison, and the machine has a working Vulkan 1.4 driver.

## Math / Theory

- Assumptions: Same model, prompt set, ctx/batch/ubatch/KV/spec settings isolate backend effects enough for a first mini screen.
- Expected speedup corridor: Treat deltas below 3% as tie/no default change; investigate if Vulkan wins clearly.
- Failure conditions: Vulkan build fails, Vulkan cannot offload enough layers, or backend defaults differ in a way visible in server logs.

## Implementation Plan

1. Minimal code surface to change: none intended; configure/build `build-vulkan` with `GGML_VULKAN=ON`.
2. Guard rails: no debug/validation/check-results Vulkan flags; use Release + Ninja and same benchmark harness.
3. Rollback path: remove `build-vulkan` artifacts if needed; no runtime code changes planned for this experiment.

## Benchmark Plan

- Baseline command: ROCm `scripts/agent_workload_bench.py --tasks quick --task-ids triage_diff --runs 1` with same ctx/batch/ubatch/KV/spec.
- Candidate command: Vulkan same command, only `--server-bin build-vulkan/bin/llama-server.exe`.
- Number of runs: 1 for mini viability.
- Artifacts path: `build_logs/agent-workload/`.

## Metrics

- aggregate completion TPS (wall)
- mean task TPS
- error rate
- server log backend/offload evidence

## Result

- Outcome: mixed. Vulkan is full-offload and faster on decode-heavy mini, but slower on the active cold prompt-heavy lane because prefill is much slower.
- Prompt-heavy mini (`repo-snapshot`, 7489 prompt tokens, 64 generated): ROCm `6.3327` wall TPS, Vulkan `4.2206` wall TPS (`-33.4%`).
- Prompt-heavy server timings: ROCm prompt eval `960.26 tok/s`, decode `28.32 tok/s`; Vulkan prompt eval `573.93 tok/s`, decode `30.85 tok/s`.
- Decode-biased sanity (159 prompt tokens, 128 generated): ROCm `27.9781` wall TPS, Vulkan `35.2850` wall TPS (`+26.1%`). Server decode eval was ROCm `29.42 tok/s` vs Vulkan `38.81 tok/s` (`+31.9%`).
- Confidence: medium for directionality; each lane is `runs=1`, but the prompt/decode split is large and backend/offload logs match.
- Recommendation: keep ROCm as the default for current prompt-heavy target. Treat Vulkan as an opt-in decode-heavy/backend comparison path, not a replacement default.

## Commands

Prompt-heavy ROCm:

```powershell
python scripts/agent_workload_bench.py --label e061-rocm-mini-ctx12288-q3ks --out-dir build_logs/agent-workload --tasks quick --task-ids triage_diff --runs 1 --server-bin build-rocm-vec/bin/llama-server.exe --model models/Qwen3.6-27B-Q3_K_S.gguf --gpu-layers=-1 --parallel 1 --ctx-size 12288 --batch-size 4096 --ubatch-size 512 --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn --no-disable-thinking --max-tokens 64 --real-context-mode repo-snapshot --no-reuse --no-v2-prime-pass --server-extra "--spec-type none" --startup-timeout 120 --request-timeout 300 --task-hard-timeout 300 --background-server-policy fail --write-diagnostics
```

Prompt-heavy Vulkan used the same command with `--server-bin build-vulkan/bin/llama-server.exe` and label `e061-vulkan-mini-ctx12288-q3ks`.

Decode-biased sanity used the same backend pair without `--real-context-mode repo-snapshot`, with `--max-tokens 128`, labels `e061-rocm-decode-mini-q3ks` and `e061-vulkan-decode-mini-q3ks`.

## Artifacts

- `build_logs/agent-workload/e061-rocm-mini-ctx12288-q3ks.csv`
- `build_logs/agent-workload/e061-rocm-mini-ctx12288-q3ks.server.log`
- `build_logs/agent-workload/e061-rocm-mini-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e061-vulkan-mini-ctx12288-q3ks.csv`
- `build_logs/agent-workload/e061-vulkan-mini-ctx12288-q3ks.server.log`
- `build_logs/agent-workload/e061-vulkan-mini-ctx12288-q3ks.diagnostics.md`
- `build_logs/agent-workload/e061-rocm-decode-mini-q3ks.csv`
- `build_logs/agent-workload/e061-rocm-decode-mini-q3ks.server.log`
- `build_logs/agent-workload/e061-rocm-decode-mini-q3ks.diagnostics.md`
- `build_logs/agent-workload/e061-vulkan-decode-mini-q3ks.csv`
- `build_logs/agent-workload/e061-vulkan-decode-mini-q3ks.server.log`
- `build_logs/agent-workload/e061-vulkan-decode-mini-q3ks.diagnostics.md`

## Notes

- Vulkan SDK/driver smoke: `vulkaninfo --summary` sees `AMD Radeon RX 9070 XT`, Vulkan API `1.4.344`, AMD proprietary driver `26.3.1`.
- Vulkan configure found `glslc`, cooperative matrix, cooperative matrix2, integer dot product, and bfloat16 GLSL support.
- Vulkan build issue: initial build generated an incomplete `mul_mm.comp.cpp` and missed `matmul_id_f16_fp32_data`; deleting the generated `build-vulkan/ggml/src/ggml-vulkan/mul_mm.comp.cpp` and object forced regeneration and linked `build-vulkan/bin/llama-server.exe` successfully. No source change was required.
- Vulkan runtime issue: this MinGW build must run with `C:\Strawberry\c\bin` before MSYS2 `/mingw64/bin` in `PATH`; otherwise `llama-server.exe --help` exits with code `127` due runtime DLL selection.
- Both benchmark lanes offloaded `65/65` layers to GPU. ROCm used `ROCm0`; Vulkan used `Vulkan0`.