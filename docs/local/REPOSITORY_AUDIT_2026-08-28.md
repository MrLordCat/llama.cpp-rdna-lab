# Repository audit — 2026-08-28

Scope: `master` in `D:\GitHub\llama.cpp-with-GUI`, with emphasis on root
clutter, local generated state, documentation ownership, and safe cleanup.

## Decisions

- Keep the active project entry points and policy documents in the root:
  `README.md`, `AGENTS.md`, `AGENT_WORKFLOW.md`, `PROJECT_PROFILE.md`,
  `BENCHMARKS.md`, `PERFORMANCE.md`, `FORK_DETAILS.md`, `MTP.md`,
  `Q4_K_M_RESULTS.md`, `QWEN_SPEED_RESEARCH.md`, and `UPSTREAM_SYNC.md`.
  They are linked from `README.md`, consumed by benchmark context generation,
  or explicitly protected by repository policy.
- Move the build command reference to `docs/local/BUILD_CHEATSHEET.md`.
- Archive the old MSVC detection note under `docs/local/archive/`; its claim
  about automatic GUI initialization no longer matches the current call graph.
- Keep `run.bat` as an ignored personal launcher.
- Keep `.venv-gui`: it is the only local environment that starts successfully
  and it uses the current `D:` checkout with Python 3.13.14.
- Keep `run.py` as the stable root GUI entry point. Move the still-functional
  PyInstaller helper from `build_exe.py` to `scripts/build_gui_exe.py` and
  update `build-gui-exe.bat`.

## Directory audit

| Directory | Tracked size | Decision |
| --- | ---: | --- |
| `tests/` | 2.42 MiB | Keep: CMake enables tests for standalone builds and repository policy requires `ctest`. Generated model caches remain ignored. |
| `tools/` | 12.25 MiB | Keep: owns `llama-server`, RPC, benchmark, quantization, perplexity, and other production targets. |
| `examples/` | 0.74 MiB | Keep: CMake builds it by default and local Vulkan/research probes live there. |
| `scripts/` | 1.37 MiB | Keep: 162 tracked docs/tasks reference scripts, including the protected benchmark harness. Remove only generated executables and Python caches. |
| `cmake/` | 0.01 MiB | Keep intact: common modules and toolchain files are referenced by presets/workflows. Removing ARM/RISC files would save almost nothing and increase upstream-sync conflicts. |
| `media/` | 0.25 MiB | Remove as requested; update the two documentation references and CODEOWNERS. |
| `subProject_q4/` | 0 tracked | Keep local: the 52 GiB directory is covered by `subProject_q4/` in `.gitignore`. |
| `dist/` | 0 tracked | Remove the empty directory; PyInstaller recreates it on demand. |

## Cleanup performed

- Removed broken `.venv` and `.venv-1` environments (about 272 MiB total).
  Both pointed to missing Python installations under `C:\Users\Chris` and
  failed to start. They are reproducible and contain no source data.
- Moved root `flash-*.log`, `config.json.bak.tmp`, and the malformed
  `D:GitHub...vulkan-shaders-gen.log` into the ignored directory
  `tmp/root-cleanup-2026-08-28/`. The move is reversible and preserves the
  diagnostic evidence without leaving it in the repository root.
- Removed about 154 MiB of ignored server-test model caches under
  `tools/server/tests/tmp`, plus pytest/Python caches.
- Removed about 13 MiB of ignored research harness `.exe` files and Python
  bytecode caches under `scripts/`; their tracked source files remain.
- Removed the empty `dist/` directory and the tracked `media/matmul.png` asset.
  The media asset remains recoverable from Git history.
- The clean CPU configure exposed a pre-existing RPC link regression:
  `llama-context.cpp` called `ggml_backend_is_rpc()` even when `GGML_RPC=OFF`.
  Added a compile-time wrapper that preserves RPC behavior under
  `GGML_USE_RPC` and treats every backend as local otherwise.

## Preserved local state

The following build directories were not removed:

- `build-vulkan-dbg`
- `build-vulkan-symbols`
- `build-vulkan-portability-audit`
- `build-rocm-gfx1100-hardening`

Together they occupy roughly 3.5 GiB, but all four are still registered in
`gui/build_versions.json`. Deleting only the directories would leave stale GUI
entries. A later build-cache cleanup should remove the selected directories
and their registry records as one operation, after confirming they are no
longer needed for debugging or regression comparison.

Likewise, `tmp/`, canonical build trees, model files, benchmark artifacts, and
the working `.venv-gui` remain untouched. They are ignored local state with
known owners rather than unexplained root clutter.

## Documentation audit

- Scanned 611 tracked Markdown files for local file targets.
- Reduced unresolved candidates from 37 to 14 by fixing 23 confirmed broken
  paths in project, research, grammar, parser, security, and server docs.
- The remaining 14 are not broken file links: eight are repository-root
  resource pointers in `.pi/gg/SYSTEM.md`, interpreted from the agent working
  directory, and six are API-symbol references in
  `docs/development/parsing.md` that the simple file-link scanner cannot
  distinguish from paths.
- After moving three tracked documents, the root contains 37 tracked files plus
  the ignored personal `run.bat`; no unexplained scratch files remain there.

Validation:

- `python -m compileall -q gui scripts run.py` passes with bytecode redirected
  to a temporary cache outside the source tree.
- CPU configure and `llama-server` build pass with `GGML_RPC=OFF`.
- Vulkan `llama-server` build passes with `GGML_RPC=ON`.
- `git diff --check` passes, the working `.venv-gui` reports Python 3.13.14,
  and all 12 moved scratch artifacts are present under
  `tmp/root-cleanup-2026-08-28/`.

## Follow-up candidates

1. Add a GUI action that prunes missing or explicitly selected build-version
   records together with their directories.
2. Apply the benchmark retention policy to old ignored raw artifacts under
   `build_logs/`, without touching the canonical generated history.
3. Periodically verify local Markdown links and ensure new project-specific
   maintenance notes go under `docs/local/` instead of the root.