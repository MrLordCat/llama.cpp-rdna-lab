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
2. `docs/research/CONTEXT_130K_WORKFLOW.md`
3. `docs/research/PERF_WORKSPACE.md`
4. `docs/research/MAJOR_TOPOLOGY_WORKFLOW.md`
5. `docs/research/major-topology/README.md`
6. `docs/research/EXPERIMENTS_DIGEST.md`
7. `docs/research/HYPOTHESES.md`
8. `docs/research/RESULTS_LOG.md`
9. `docs/research/BENCH_HISTORY_POLICY.md`
10. The latest relevant note under `docs/research/major-topology/`
11. `build_logs/agent-workload/BENCH_RECENT.md` and `build_logs/agent-workload/BENCH_LANES.md`
12. `docs/research/experiments/` only for legacy cross-reference

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
2. Prepare or update a major-topology note first (`P`/`D`/`S`) in
   `docs/research/major-topology/`.
3. Prepare an experiment note from `docs/research/EXPERIMENT_TEMPLATE.md` only
   when needed for a narrow measured ledger entry.
4. Run cheap gates before expensive lane benchmarks:
   - `python scripts/research/formula_sanity_checks.py`
   - `python scripts/research/vulkan_q3k_prebuild_gate.py --candidate "<idea>"` for H31/Vulkan Q3_K shader or tile ideas
   - hypothesis-specific model/check script when applicable
5. Run one-run A/B gates first.
6. Use three runs only for final confirmation of borderline or promising deltas.
7. Compare against the current best/history for the same lane shape.
8. For route or bucket-local kernel changes, run a point-level timing review before promotion:
   - enable sync timing for the touched route (for MMVQ: `GGML_TRACE_MMVQ_TIMING=1` + `GGML_TRACE_MMVQ_TIMING_SYNC=1` + `GGML_TRACE_MMVQ_RESOURCES=1`)
   - compare the same points (`ncols_x` buckets) before/after with mean `total_ms`
   - report a robust view that excludes startup outliers (for example, `total_ms < 10`)
   - always pair point-level deltas with wall metrics (`aggregate_completion_tps`, prompt/decode ms)
9. If local point timing improves but wall does not improve (or regresses), classify as bottleneck shift and move to the next route hotspot instead of iterating the same micro-point.
10. Revert negative runtime/shader/code probes unless they are intentionally kept
   behind a documented opt-in gate.
11. Update docs in the same work unit:
   - owning major-topology note (required)
   - experiment note
   - `docs/research/RESULTS_LOG.md`
   - `docs/research/EXPERIMENTS_DIGEST.md` via `python scripts/research/refresh_experiment_digest.py`
   - `BENCHMARKS.md` for meaningful user-facing benchmark decisions
   - `docs/research/HYPOTHESES.md` if priority/status changes
12. Refresh canonical benchmark history when logs/schema changed:
   - `python scripts/agent_workload_bench.py --refresh-canonical-history`

## Active Lanes

Project policy primary lane:

- `Qwen3.6-27B-Q3_K_S.gguf`
- long-context cold-first lane at `ctx=131072` (~130k)
- `--real-context-mode repo-snapshot`
- no reuse: `--no-reuse` (`--cache-ram 0 --ctx-checkpoints 0`)
- no v2 prime pass: `--no-v2-prime-pass`
- thinking enabled: use `--no-disable-thinking`
- first baseline contract: `quick:triage_diff`, `max_tokens=16`, `q4_0/q4_0`, `b512`, `real-context-chars=24576`; Vulkan current best uses `ub256`, ROCm remains `ub128` until rechecked
- expected constraint: `ctx=131072` will often spill KV/context/working set beyond 16 GB VRAM into system RAM; keep diagnostics/server logs and treat residency/RAM-spill as part of the result

Current Vulkan-vs-ROCm baseline lane:

- model: `models/Qwen3.6-27B-Q3_K_S.gguf`
- context: `131072`
- batch/ubatch: Vulkan `512/256`, ROCm `512/128`
- KV: `q4_0/q4_0`
- flash attention: on
- max tokens: `16`
- task: `quick`, `triage_diff`
- Vulkan current target: `2.4 TPS` on the D012 lane. D012 remains the active baseline at `2.0013 TPS` r3, prompt `1053.11 tok/s`, decode `42.72 tok/s`, with q3quad/GLU opt-in stack plus `--no-mmap`; D005 split-K anchor remains `1.7898 TPS`
- ROCm baseline: `1.5200 TPS` r3, prompt `801.71 tok/s`, decode `29.07 tok/s`, no `HSA_OVERRIDE_GFX_VERSION` on HIP SDK 7.1
- ROCm pause/fence: D013-D027 reject ub256/storage/cublas/no-mmap/src1/y32w2/GLU/current-MMQ/Q3Flash/vbuffer/upstream-stock/streaming-cublas, pair-only FFN SwiGLU, naive whole-FFN streaming, expanded persistent Q3_K layout, and compact signed-nibble unpack-only layout routes. Leave ROCm paused unless a stronger Q3_K/FFN dataflow proof appears; active work is Vulkan `2.4 TPS`.
- rollback/negative control: `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1`
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

Run the Vulkan 130k baseline:

```bash
PATH="/c/Strawberry/c/bin:$PATH" \
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 \
python scripts/agent_workload_bench.py \
  --server-bin build-vulkan/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
   --label vulkan130k-quick-c24k-b512-ub128-r1 \
   --ctx-size 131072 --batch-size 512 --ubatch-size 128 \
  --gpu-layers 999 --cache-type-k q4_0 --cache-type-v q4_0 \
   --flash-attn --parallel 1 --max-tokens 16 \
   --tasks quick --task-ids triage_diff \
   --real-context-mode repo-snapshot --real-context-chars 24576 \
   --no-disable-thinking --no-reuse --no-v2-prime-pass \
   --runs 1 --request-timeout 180 --startup-timeout 900 --task-hard-timeout 45 \
   --background-server-policy fail --server-extra "--spec-type none --no-mmap" \
   --write-diagnostics
```

Run the ROCm 130k baseline:

```bash
python scripts/agent_workload_bench.py \
   --server-bin build-rocm-vec/bin/llama-server.exe \
   --model models/Qwen3.6-27B-Q3_K_S.gguf \
   --label rocm130k-quick-c24k-b512-ub128-r1 \
   --ctx-size 131072 --batch-size 512 --ubatch-size 128 \
   --gpu-layers 999 --cache-type-k q4_0 --cache-type-v q4_0 \
   --flash-attn --parallel 1 --max-tokens 16 \
   --tasks quick --task-ids triage_diff \
   --real-context-mode repo-snapshot --real-context-chars 24576 \
   --no-disable-thinking --no-reuse --no-v2-prime-pass \
   --runs 1 --request-timeout 180 --startup-timeout 900 --task-hard-timeout 45 \
   --background-server-policy fail --server-extra "--spec-type none" \
   --write-diagnostics
```

Use `GGML_VK_DISABLE_AMD_LARGE_MATMUL=1` only to prove rollback behavior or to
diagnose the old slow route. `GGML_VK_FORCE_AMD_LARGE_MATMUL=1` remains useful
for experiments on other devices, but it is no longer required for the local
RX 9070 XT / AMD proprietary coopmat path.

Run the Vulkan 2.4 target gate after retargeting or before a broad route design:

```bash
python scripts/research/vulkan_2p4_target_gate.py > build_logs/agent-workload/d028-vulkan-130k-2p4-target-gate.md
```

D028 shows the new target needs `1.1992x` wall over D012, about `1277 tok/s`
prompt eval if decode and overhead stay flat, `1.387x` local on dense FFN, or
`1.260x` local on all-Q3. Treat gate/up-only and down-only Vulkan routes as too
small unless a new model beats those ceilings.

D029 rejects activation-only whole-FFN fusion and naive full-FFN streaming as the
first `2.4 TPS` route: hidden materialization traffic is too small, while
streaming either recomputes gate/up per down-row tile or creates massive partial
output traffic. The next Vulkan scout should target all-Q3/body-layout work or a
whole-FFN design that reduces Q3_K matmul work itself.

D030 rejects the old all-Q3/body-layout families as first moves: q3quad/tile is
already in D012 and still short by about `1175 ms` all-Q3 point time;
scale-only helpers, signed-nibble-only storage, Q8_1/int-dot, expanded layouts,
and neighboring tile tweaks all have prior negative or residency evidence. D031
also rejects compact Q3S/signed-nibble plus predecoded-scale layout-body work:
static savings are below the target-closing budget, runtime evidence is
negative, and residency cost is high. D032 must start from a true Q3_K compute
body or compressed-dot route with a static resource/residency proof before full
server A/B. D032 stack math says FA-only is not a target route: FA `1.5x` still
requires Q3 `1.1987x` local and FA `2.0x` still requires Q3 `1.1702x` local.
Treat FA as a stack component only after Q3 has point/static evidence near the
`1.18-1.20x` local band. D033 rejects q3-octa/`LOAD_VEC_A=8` as a near-repeat
of E087 (`-1.50%`), so wider per-invocation Q3_K dequant is not the next body
candidate.

Run the Vulkan Q3_K pre-build gate before shader experiments:

```bash
python scripts/research/vulkan_q3k_prebuild_gate.py --candidate "Q3_K shared-memory layout rewrite" --local-gain-pct 20 --require-target-closing
```

Use this gate to avoid repeating measured dead ends. It reads the current Q3_K
`mul_mm.comp`/`mul_mm_funcs.glsl`/shader-generator state, compares candidate text
or an optional diff against H31 history in `RESULTS_LOG.md`, and computes the
local speedup needed to close the current Vulkan-vs-ROCm pp gap. The accepted
source baseline is E082 stride18 + E086 corrected `LOAD_VEC_A=4`; E102 only makes
the fast AMD large-matmul route automatic locally. Current no-env pp7488 is
`983.48`, while the old disabled path is `708.19`. Matching ROCm pp7488 via the
active Q3_K hotspot alone still requires a large local win, so helper-only or
neighboring-stride ideas should be skipped unless the gate shows a new high-share
mechanism. The gate also prints a Q3 dequant-reuse sanity model: scale/helper
reuse alone is calibrated by E088 as non-positive, so a dequant-reuse candidate
needs a separate instruction/load-count argument showing real pair-count, LDS
traffic, or coopmat-work reduction before any build.

Run the warptile static scout before tile/env claims:

```bash
python scripts/research/vulkan_warptile_static_scout.py
```

After E097, the scout matches the local driver resources for base Q3_K:
subgroup `64`, coopmat `16x16x16`, and `20480 B` LDS. It marks `wn48`/`wn96`
invalid for `BN=128`, which is why E091 cannot be promoted. E098 measured the
larger `bm256`/`bn256*` family and rejected it: the dequant proxy looked better,
but LDS near 32 KiB and/or register pressure made all variants slower. The scout
also models BK-depth variants. Treat `BK=64` as `needs-resource-proof`: it halves
K-block/barrier rounds but does not reduce full-K dequant/B traffic and raises Q3
shared memory to `34816 B`. Treat `BK=16` as low priority because it doubles
K-block/barrier rounds.

Before chasing Vulkan coopmat/compiler feature routes, capture the local compiler
and device capability snapshot:

```bash
python scripts/research/vulkan_feature_snapshot.py
```

As of E095 on RX 9070 XT / AMD proprietary `26.3.1 (LLPC)`, `glslc` can compile
coopmat2 feature tests but the device exposes `VK_KHR_cooperative_matrix`, not
`VK_NV_cooperative_matrix2`, so `mul_mm_cm2`/NV coopmat2 is not an active AMD
route on this driver.

For generated shader route fingerprints, use:

```bash
python scripts/research/spirv_op_summary.py build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q3_k_f32_aligned_f16acc_cm1.spv build-vulkan/ggml/src/ggml-vulkan/vulkan-shaders.spv/matmul_q3_k_f32_aligned_f16acc.spv --top 35
```

E096 uses this to distinguish the active KHR coopmat Q3_K binary from the plain
binary before benchmark work. Opcode counts are not speed claims; they are a
mechanism/fallback gate.

For active route diagnostics, use `GGML_VK_MATMUL_ROUTE_TRACE=1`. E099 kept this
default-off trace after proving that the Q8_1/int-dot route is not a local H31
speed path (`matmul_q3_k_q8_1_l` pp256 `225.08`, `143 VGPR / 28672 B LDS`).

## Documentation Contract

Every kept or rejected performance experiment needs:

- artifact labels that include experiment ID and lane hint
- CSV/JSONL/server log in `build_logs/agent-workload/`
- experiment note under `docs/research/experiments/`
- result row in `docs/research/RESULTS_LOG.md`
- refresh of `docs/research/EXPERIMENTS_DIGEST.md` when route-family conclusions change
- a benchmark summary in `BENCHMARKS.md` when the result affects presets,
  default behavior, or future user decisions

Canonical benchmark history lives in:

- `build_logs/agent-workload/BENCH_RUNS.csv`
- `build_logs/agent-workload/BENCH_RECENT.md`
- `build_logs/agent-workload/BENCH_LANES.md`

Legacy `BENCH_HISTORY*.md/.csv` files are compatibility artifacts, not the first
place to look.

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

For post-E264 Vulkan/ROCm Q3_K work, stop the normal E### loop and use
`MAJOR_TOPOLOGY_WORKFLOW.md` when the candidate requires broad storage,
graph-level fusion, a new kernel body, or a route design that cannot be proven by
a narrow patch.
