# Performance Workspace

This workspace layer is for fast, reproducible TPS work on `llama.cpp-with-GUI`.
It does not replace `AGENTS.md`; it points agents to the shortest safe path for
local ROCm/Vulkan performance research.

Open this repository in VS Code via `llama.cpp-with-GUI.code-workspace` from the
repo root. The workspace file provides the folder, search/watcher exclusions,
file associations, and extension recommendations; `.vscode/tasks.json` provides
the runnable benchmark/check tasks.

## Purpose

- Reduce agent ramp-up time before TPS experiments.
- Keep tool use narrow and relevant to this repository.
- Make benchmark commands discoverable and repeatable from VS Code tasks.
- Keep research docs updated with measured evidence, not memory or vibes.

## Start Here

For performance work, read these in order:

1. `AGENTS.md`
2. `docs/research/PERF_WORKSPACE.md`
3. `docs/research/HYPOTHESES.md`
4. `docs/research/RESULTS_LOG.md`
5. The latest relevant experiment note under `docs/research/experiments/`
6. `build_logs/agent-workload/BENCH_HISTORY_V2.md`

For paused decode/perf work, also read:

1. `docs/research/decode-hotspots/C01_RESUME_PLAYBOOK.md`
2. `docs/research/decode-hotspots/C01_mul_mat_forward.md`
3. `docs/research/decode-hotspots/DECODE_TRACE_CHECKLIST.md`

## Tool Budget

Default TPS agents should use only:

- read/search tools for code and logs
- edit tools for focused patches and docs
- terminal execution for builds, checks, and benchmarks
- todo tracking for multi-step experiments

Do not use browser/UI automation, notebooks, Java/debug tools, extension search,
network tools, or image/page capture for normal TPS work. Use a web/upstream
scout only when the task explicitly needs upstream issues, PRs, or external
research.

Prefer `rg`/`rg --files` for local search. Avoid `cmd.exe` wrappers for long
builds or benchmarks. Use Git Bash or PowerShell directly.

## Agent Roles

Use these workspace agents when available:

- `tps-research`: code + benchmark loop for Vulkan/ROCm TPS experiments.
- `bench-runner`: run fixed controls/candidates without editing source.
- `research-docs`: update experiment notes, result logs, and benchmark summaries.
- `upstream-scout`: read-only local/web research for upstream PRs/issues.

The point is separation: the runner should not invent code, the docs agent should
not make unmeasured speed claims, and the upstream scout should not edit files.

## Fixed Workflow

1. Snapshot state:
   - `git status --short --branch`
   - confirm no background `llama-server`
   - identify the active hypothesis ID or create one in `HYPOTHESES.md`
2. Prepare an experiment note from `docs/research/EXPERIMENT_TEMPLATE.md`.
3. Run cheap gates before expensive lane benchmarks:
   - `python scripts/research/formula_sanity_checks.py`
   - hypothesis-specific model/check script when applicable
4. Run one-run A/B gates first.
5. Use three runs only for final confirmation of borderline or promising deltas.
6. Compare against the current best/history for the same lane shape.
7. Revert negative runtime/shader/code probes unless they are intentionally kept
   behind a documented opt-in gate.
8. Update docs in the same work unit:
   - experiment note
   - `docs/research/RESULTS_LOG.md`
   - `BENCHMARKS.md` for meaningful user-facing benchmark decisions
   - `docs/research/HYPOTHESES.md` if priority/status changes

## Active Lanes

Project policy primary lane:

- `Qwen3.6-27B-Q3_K_S.gguf`
- prompt-heavy cold-first lane below 16k
- current reference context: `ctx=12288`
- `--real-context-mode repo-snapshot`
- no reuse: `--cache-ram 0 --ctx-checkpoints 0`
- thinking enabled: use `--no-disable-thinking`

Current Vulkan-vs-ROCm diagnostic lane:

- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- context: `32768`
- batch/ubatch: `5120/1024`
- KV: `q4_0/q4_0`
- flash attention: on
- max tokens: `120`
- task: `v2-mini`, `v2_write_function`
- Vulkan safe env: `GGML_VK_FORCE_AMD_LARGE_MATMUL=1`
- do not auto-enable `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wm32-wn32`

## Common Commands

Check no background server:

```powershell
if (Get-Process llama-server -ErrorAction SilentlyContinue) {
   Get-Process llama-server -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,Path
   exit 1
} else {
   Write-Host "No llama-server process found"
}
```

Build Vulkan server and bench:

```bash
cmake --build build-vulkan --target llama-server llama-bench --config Release -j 8
```

Configure ROCm on Windows:

```powershell
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
```

Run the safe Vulkan 32k control:

```bash
PATH="/c/Strawberry/c/bin:$PATH" GGML_VK_FORCE_AMD_LARGE_MATMUL=1 \
python scripts/agent_workload_bench.py \
  --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --label vulkan32k-control-r1 \
  --ctx-size 32768 --batch-size 5120 --ubatch-size 1024 \
  --gpu-layers 999 --cache-type-k q4_0 --cache-type-v q4_0 \
  --flash-attn --parallel 1 --max-tokens 120 \
  --tasks v2-mini --task-ids v2_write_function \
  --real-context-mode repo-snapshot --real-context-chars 21872 \
  --no-disable-thinking --no-reuse --allow-ctx-above-16k \
  --runs 1 --background-server-policy fail
```

## Documentation Contract

Every kept or rejected performance experiment needs:

- artifact labels that include experiment ID and lane hint
- CSV/JSONL/server log in `build_logs/agent-workload/`
- experiment note under `docs/research/experiments/`
- result row in `docs/research/RESULTS_LOG.md`
- a benchmark summary in `BENCHMARKS.md` when the result affects presets,
  default behavior, or future user decisions

Never claim a TPS improvement from:

- an invalid/corrupt generation smoke
- a single surprising run without a neighboring control
- a primed run presented as cold-first
- a different context/batch/ubatch/KV/spec/reuse/thinking shape

## Stop Conditions

Stop and document instead of continuing to probe when:

- candidate is a regression or within noise after a neighboring control
- output smoke suggests corruption
- benchmark shape drifted from the lane contract
- a code probe requires broad refactoring before proving a local mechanism
- another agent/user change makes the current A/B incomparable
