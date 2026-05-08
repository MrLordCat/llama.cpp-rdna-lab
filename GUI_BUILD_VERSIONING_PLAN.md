# GUI Build Versioning And Benchmark Registry Plan

## Goals

1. Keep multiple ROCm build versions side-by-side and selectable in GUI.
2. Allow creating a stock ROCm reference build from upstream `ggml-org/llama.cpp` (not from this fork).
3. Tie benchmark results to a specific build version.
4. Maintain a single benchmark registry (`BENCH_HISTORY.csv` + `BENCH_HISTORY.md`) with pinned best MTP/non-MTP results.
5. Migrate away from per-run benchmark artifact clutter.

## Scope

In scope:
- GUI changes for build creation, build selection, and build metadata display.
- Build registry storage format and migration.
- Benchmark metadata linking to build IDs.
- Stock-upstream ROCm reference build flow.

Out of scope (for this plan iteration):
- Full benchmark UI visual analytics.
- Automated upstream merge conflict resolution UI.

## Phase 0: Registry And Policy Baseline

Status target: done first.

Tasks:
1. Enforce benchmark history append on every benchmark run.
2. Keep a pinned top section in Markdown for best MTP model and best non-MTP model TPS.
3. Mark best rows in history CSV with a dedicated flag.
4. Keep compatibility with old history rows.

Acceptance:
- Any benchmark run updates both history files.
- `BENCH_HISTORY.md` top section updates automatically.

## Phase 1: Build Version Data Model

Add a persistent registry file, for example `gui/build_versions.json`.

Suggested schema per build record:
- `id`: stable unique ID (timestamp + short hash).
- `name`: user-visible name.
- `backend`: `rocm`, `cpu`, `vulkan`, etc.
- `source_type`: `fork` or `upstream-stock`.
- `source_ref`: git commit/tag/branch used for build.
- `build_dir`: absolute or project-relative path.
- `server_bin`: resolved server executable path.
- `toolchain`: key configure details (generator, compiler, targets).
- `created_at`, `updated_at`.
- `status`: `ready`, `failed`, `building`, `archived`.
- `notes`.
- `bench_best_non_mtp_tps`, `bench_best_mtp_tps`.
- `bench_last_run_at`.

Acceptance:
- Build registry survives GUI restart.
- Existing installed builds can be imported into registry.

## Phase 2: Build Creation UX

### Build & Setup tab

Add controls:
1. "Create Build Version" dialog:
   - Build name.
   - Backend.
   - Source type: `fork` vs `upstream-stock`.
2. Optional advanced fields:
   - Custom CMake flags.
   - ROCm target (`gfx1201` default).
   - Build directory override.

Behavior:
- Each new build writes to a dedicated build directory.
- No overwrite of existing build by default.
- Record all build metadata in registry.

Acceptance:
- Two builds with different settings can coexist and be launched independently.

## Phase 3: Upstream Stock ROCm Reference Build

Add a dedicated flow to create a stock upstream reference build.

Implementation outline:
1. Create/update a clean local mirror/worktree for upstream source.
2. Checkout chosen upstream ref (default `master`).
3. Configure ROCm build with Windows constraints (Ninja + ROCm clang/clang++).
4. Store as `source_type=upstream-stock` in build registry.

Safety rules:
- Never overwrite fork workspace content.
- Keep source and build dirs isolated.

Acceptance:
- GUI can build and register at least one upstream-stock ROCm build.
- Server launch can target this reference build.

## Phase 4: Launch Server Build Selection

### Launch Server tab

Add a build selector for ROCm server executable:
1. Build dropdown by registry entries (`name`, `backend`, `source_type`, short commit).
2. Explicit display of selected build path.
3. Validation before launch (file exists, executable available).

Acceptance:
- User can switch between build versions before launch.
- Selection persists in settings.

## Phase 5: Benchmarks Linked To Build Version

Requirements:
1. Every benchmark run includes build ID in history row.
2. Build Info section shows benchmark summary for selected build:
   - best non-MTP TPS,
   - best MTP TPS,
   - last benchmark time,
   - link/path to relevant history rows.
3. On history update, build-level aggregates refresh.

Acceptance:
- For any build in registry, GUI shows stored benchmark maxima.

## Phase 6: Artifact Consolidation And Cleanup

Migration target:
- Stop generating per-run standalone benchmark files by default.
- Keep only unified registry files and optional compact raw log bundles.

Rollout strategy:
1. Introduce `--artifact-mode unified|full` in benchmark script.
2. Default to `unified` in GUI.
3. Keep `full` mode for troubleshooting only.
4. After validation window, remove old accumulated per-run files.

Cleanup policy:
- Delete historical per-run files after confirming all required benchmark information exists in unified registry.
- Keep curated lock/reference files explicitly marked as protected.

Acceptance:
- New benchmark runs no longer create per-run CSV/JSONL by default.
- Legacy clutter removed safely after migration checklist.

## Migration Checklist

1. Backup current `build_logs/agent-workload`.
2. Implement registry-aware benchmark writing.
3. Backfill build IDs for critical historical runs where possible.
4. Validate pinned best rows for MTP/non-MTP remain correct.
5. Enable unified artifact mode by default.
6. Remove old run artifacts.

## Risks And Mitigations

1. Risk: mixed benchmark conditions produce misleading "best" records.
Mitigation: keep mode/model/context columns visible in registry and GUI.

2. Risk: upstream stock build flow breaks due toolchain drift.
Mitigation: pin tested ROCm configure template and validate with smoke benchmark.

3. Risk: accidental deletion of important evidence files.
Mitigation: explicit protected file list + dry-run cleanup preview before delete.

## Definition Of Done

1. GUI supports multiple build versions and launch-time build selection.
2. GUI can create and use upstream-stock ROCm reference build.
3. Bench history is unified, auto-updated, and pinned best MTP/non-MTP is always visible.
4. Build info surfaces benchmark maxima per build.
5. Per-run artifact sprawl is retired with safe migration cleanup.
