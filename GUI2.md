# GUI 2.0 — handoff

Read this before `AGENTS.md`. It overrides the parts of `AGENTS.md` that
describe the location and purpose of this checkout.

## Where you are

| | |
|---|---|
| This worktree | `D:\GitHub\llama.cpp-gui2` |
| Branch | `gui-2.0`, branched from `master` |
| Other checkout | `D:\GitHub\llama.cpp-with-GUI` on `rpc-vulkan` — **another agent works there** |

`AGENTS.md` says the canonical root is `D:\GitHub\llama.cpp-with-GUI`. That is
true for the performance lab, not for this task. Both directories are git
worktrees of the same repository and coexist on disk.

Never switch or reset the branch in the other checkout, and never revert
changes you did not make there. It is safe to *read* files from it.

## Inherited constraints that still apply

- Everything in `AGENTS.md` about **driver safety** stays in force. GPU work in
  this task is not needed at all: GUI 2.0 development is pure Python + web.
- **The GPUs are busy.** The other agent runs RPC/benchmark work. Do not launch
  `llama-server`, benchmarks, or hardware discovery from this worktree.
- `.vscode/tasks.json` was inherited from the performance lab and contains
  build/benchmark tasks. Do not run them from here.
- Supported backends remain CPU / Vulkan / ROCm. Do not reintroduce others.

## What is being built

A from-scratch rewrite of the PyQt6 GUI as a **local web UI**. Same
capabilities, smaller and cleaner codebase. The old GUI is kept in `gui/` as
reference only; GUI 2.0 does not import from it.

**Stack: FastHTML + HTMX + Jinja partials, SSE for log/metric streams.**
Server-rendered, no npm, no SPA build step, one language.

Rejected, with reasons:

| Option | Why not |
|---|---|
| Keep PyQt6 | no remote access, weak tables/charts, most code is manual widget layout |
| FastAPI + Svelte/React (+Tauri) | best UI control, but *more* total code in two languages plus a build pipeline |
| NiceGUI | good fit, but HTMX gives a smaller surface and no hidden state sync |
| Streamlit | script-rerun model breaks long-lived processes (server, benchmark) |
| Gradio / Reflex / Dash | built for ML demos and dashboards, not dense forms and process control |
| Rust core | would require rewriting the bench runner, log parsing and autotune — the most valuable and messiest part |

## Measurements that drove these decisions (2026-08-23)

Old GUI, `gui/*.py` — 15 458 lines total:

```
UI layer (tabs/widgets)    10 786
logic (builds, HF, monitor) 4 670   ← portable to any stack, keep it
```

Startup profile (`QT_QPA_PLATFORM=offscreen`, cProfile on the main window
constructor):

```
import PyQt6            0.04s
init_dependencies()     0.16s
import gui modules      0.27s
apply_modern_theme      0.09s
LlamaCppGUI()           3.36s
TOTAL                   3.96s
```

3.44 s of that 3.36 s is `subprocess.communicate` — synchronous external
processes started inside widget constructors:

| call | cost | cause |
|---|---|---|
| `hardware_tab.detect_hardware()` | 1.32s | `powershell (Get-CimInstance Win32_Processor)` |
| `server_monitor._query_gpu_counters()` | ~3.0s | GPU counters via external process |
| `builds_info_tab._get_directory_size()` | 0.62s | 35 000 `nt.stat` over seven `build-*` dirs |
| `download_tab.load_popular_models()` | 0.34s | network request at startup |

Conclusion: the stack was never the bottleneck. Blocking I/O in constructors
was. GUI 2.0 must keep **all** discovery lazy and off the first render.

Structural problems to not repeat:

- `server_tab` and `benchmark_tab` each compose almost the same `llama-server`
  argv independently. Adding `--rpc` / `-dev` / `-sm` / `-ts` required editing
  both files.
- State lives in widgets, so the code needs `_loading_settings`,
  `blockSignals`, `_user_overrode_device` guards, and settings migrations
  (`*_migrated_v1`, `*_migrated_v2`) scattered across UI handlers.

## Target architecture

```
core/     no web imports, pytest-covered
          RunSpec (frozen dataclass) — the single description of a run
          to_argv(spec) -> list[str]
          validate(spec) -> list[Problem]      (today's problems/notes)
          param schema — one declaration per llama-server flag; forms and
          argv are both generated from it
          presets/profiles with versioned migrations in one place
proc/     single supervisor owning llama-server / benchmark subprocesses.
          Exactly one GPU owner at a time. Windows: consoles must be hidden.
web/      FastHTML routes + Jinja partials, HTMX swaps, SSE for logs/metrics
```

One `RunSpec` serves Server launch, Bench and Autotune. There must be no second
place that builds a command line.

Paths to models and builds are **configuration**, not constants: build
directories are gitignored and exist only in `D:\GitHub\llama.cpp-with-GUI`.

## Scope

In scope: Server launch, Bench / Autotune, History & Analytics, Models.

Out of scope, moved to CLI and VS Code tasks: building, build info, model
download, dependency install. That is ~3 300 lines of the old GUI
(`build_tab`, `builds_info_tab`, `download_tab`, `dependency_installer`,
`model_downloader`).

## Reference material

- Old GUI: `gui/` in this worktree (master snapshot).
- Newer RPC and device-placement work is **uncommitted** in
  `D:\GitHub\llama.cpp-with-GUI\gui\` — `server_backend_panels.py`,
  `server_tab.py`, `benchmark_tab.py`. Read it for the `--rpc` and
  `-dev/-sm/-ts` contract.
- Benchmark runner CLI: `scripts/agent_workload_bench.py`. Unchanged; GUI 2.0
  calls it the same way the old GUI does (`--server-extra "<args>"`).
- History data, tracked in git: `build_logs/agent-workload/BENCH_RUNS.csv`
  (~1650 rows), `BENCH_RECENT.md`, `BENCH_LANES.md`.

## Facts worth not rediscovering

- `--rpc` must appear **before** `-dev` on the command line: `-dev` resolves
  device names while parsing. RPC devices are named `RPC0`, `RPC1`, … in
  `--rpc` order, one per remote device — *per device*, not per address, so a
  worker with two GPUs takes two of the names and shifts every one after it.
- Never probe `llama-server --help` / `--version` to detect capabilities: it
  starts backend discovery and risks a driver drop. Read `CMakeCache.txt`
  (`GGML_RPC:BOOL=ON`, `GGML_VULKAN:BOOL=ON`, `GGML_HIP:BOOL=ON`) or a sidecar
  `llama-server.exe.capabilities.json` instead.
- Every subprocess on Windows must run with a hidden console (see
  `gui/proc_utils.py::run_hidden` in the old GUI). Otherwise `cmd` windows pop
  up and the UI freezes.
- `-sm layer` without `-ts` makes llama.cpp split by free VRAM, which is the
  right default for a mixed 16 GB + 10 GB set of devices.

## First milestone

**History & Analytics page, read-only.** `BENCH_RUNS.csv` → filterable table +
TPS chart. No process control, no GPU, nothing to break. It proves both the
stack and the code-size claim before anything risky is written. **Done.**

## What exists now

`gui2/` is 6 177 lines of Python plus 1 838 of tests, against the old GUI's
15 458 with none. The suite is 134 tests and runs in about 20 seconds without
touching a GPU.

| Module | What it answers |
|---|---|
| `core/gguf.py` | what a model file says about itself — layers, context, head counts, SSM shape. Header only, a few KB, no binary |
| `core/params.py` | one `Param` per flag; forms and argv both generated from it |
| `core/runspec.py` | the single description of a run, `to_argv`, `validate`, and the slot arithmetic `slot_context` |
| `core/memory.py` | what a run *will* cost: weights + KV + compute, from the header; and `capacity`, the context a model has room for on a given card |
| `core/measured.py` | what a run *did* cost, read from its own log as it is written |
| `core/memstore.py` | the same, kept between runs and rescaled across contexts |
| `core/machine.py` | cores, free ports, this machine's LAN address |
| `core/devices.py` | the device list, from past logs and the registry — never from a driver |
| `core/rpc.py` | the worker command to paste, and the handshake that checks it answered |
| `proc/` | one GPU slot, a bounded log with stable line numbers, a finish callback |
| `web/` | FastHTML routes; every panel is an htmx fragment |

The Server page is written for someone who has not read llama.cpp's help
text. Every section says what it is for; every number that a person cannot be
expected to know is either read from the machine or declared automatic with
the automatic value spelled out.

The Models page answers the question a directory listing cannot: not "is the
file there" but "will it load here, and how much context is left once its
weights are down". Both come from the header and the device list, so the page
costs one stat and a few kilobytes per model and starts nothing.

## Facts about llama.cpp worth not rediscovering

- `--parallel N` **divides** the context: `n_ctx_seq = pad256(n_ctx / N)`, and
  `n_ctx` is then re-derived as `n_ctx_seq * N`. `--kv-unified` pools it
  instead. Source: `src/llama-context.cpp`.
- `--threads-http` left unset becomes `max(n_parallel + 4, hardware_concurrency
  - 1)`; `-t` unset becomes the *physical* core count. Sources:
  `tools/server/server-http.cpp`, `common/common.cpp`.
- In a shutdown memory breakdown, `CPU_Mapped` is the model file mapped into
  RAM and `Vulkan_Host` is pinned host memory. Neither is VRAM; counting
  either lands the total gigabytes high.
- A draft model's weights are missing from the breakdown table's `model`
  column and appear in `unaccounted`. Summing the allocation lines is more
  accurate than the table.
- RPC protocol: `HELLO` is pinned to command 14 by a `static_assert` so a
  client may ask the version before agreeing to anything. Every other command
  number is only valid within a matching major version. Framing is
  `cmd(1) | size(8) | payload` out, `size(8) | payload` back.
- `rpc-server` exposes **all** its accelerators unless `-d` says otherwise, so
  one address can occupy several `RPCn` names.
- A split model is `<stem>-00001-of-00003.gguf`, must be loaded through its
  *first* part, and llama.cpp finds the rest from that name itself
  (`llama_get_list_splits`). The later parts carry no architecture at all, so
  the name is the only way to recognise them; sizing a split model by the file
  it was handed underestimates it by however many parts follow.
- The GGUFs in this lab do not carry `nextn_predict_layers`, so an MTP
  conversion is recognisable only by its name and its one extra layer. Same
  rule as `agent_workload_bench.is_mtp_model_name`.

## Validation

```powershell
python -m compileall -q <new package>
python -m pytest tests/          # core must be testable without the web layer
git diff --check
```
