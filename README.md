# llama.cpp-rdna-lab

> **Project status:** Active development will be limited for the near future, and significant updates are unlikely in the short term. The cost of the AI models used for development—particularly DeepSeek—has increased, while my subscription-based model access is currently being prioritized for other projects. Maintaining several AI-heavy projects in parallel is therefore not financially practical at the moment.
>
> The repository is not abandoned. I will continue to maintain it when necessary and may still make smaller fixes or improvements, but larger optimization and research work will resume when resources allow.

Hardware-focused fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
for local AI on Windows with two AMD RDNA4 GPUs. It combines the llama.cpp
runtime with a local web GUI (GUI 2.0, `gui2/`), reproducible benchmark
tooling, long-context work, and AMD-specific Vulkan and ROCm/HIP
optimizations. The desktop application is branded **RDNA LLM Studio**.

The primary workload is agentic coding with Qwen3.8-27B: large cold prompts,
single-user requests, long contexts, tool use, vision, and speculative decode.
The main performance priority is prompt evaluation. MTP is kept only when its
decode gain does not impose an unacceptable prefill cost.

> Specialized research and production fork, not a drop-in replacement for every
> upstream platform. Results and defaults are tuned for the reference
> dual-RX 9070 XT machine. This file is the presentation and headline results;
> details live in the linked documents below.

## Documents

- [Performance](PERFORMANCE.md) — current peer tables, lane contracts, and
  model quality (PPL)
- [Benchmarking](BENCHMARKS.md) — canonical benchmark methodology and history
- [Fork Details](FORK_DETAILS.md) — fork-only features, backend fixes,
  runtime profiles
- [MTP](MTP.md) — MTP behavior and practical rules
- [Backends & Models](docs/BACKENDS_AND_MODELS.md) — supported backends,
  model/format matrix, vision
- [RPC Backend](docs/RPC_BACKEND.md) — remote GPU stack and measured results
- [Active Branches](docs/BRANCHES.md) — branch status and resume points
- [Build Guide](docs/build.md) — CPU / Vulkan / ROCm requirements and commands
- [Supported backends](docs/SUPPORTED_BACKENDS.md) — backend policy
- [Contributing](CONTRIBUTING.md), [Development](AGENTS.md), [Upstream sync](UPSTREAM_SYNC.md)
- [License & Security](LICENSE) — MIT, with [SECURITY.md](docs/SECURITY.md)

## At a Glance

| Area | Current focus |
| --- | --- |
| Host platform | Windows 11 on AMD AM4 (Linux ROCm 10 also used for research lanes) |
| Accelerators | 2x Radeon RX 9070 XT 16 GB (`gfx1201`) |
| Backends | ROCm/HIP, Vulkan, and CPU (+ remote GPU via RPC) |
| Primary model | Qwen3.8-27B Q4_K_M with MTP; MXFP4 for speed lanes |
| Main objective | Maximum cold prompt evaluation without sacrificing useful decode speed |
| Serving | OpenAI-compatible `llama-server` plus the local web GUI (GUI 2.0) |

Supported backends, model/format matrix and vision:
[docs/BACKENDS_AND_MODELS.md](docs/BACKENDS_AND_MODELS.md).

## Project Goals

- Maximize Qwen3.8 prompt-evaluation throughput for agent workloads.
- Use both GPUs without moving the active working set into system RAM.
- Make MTP improve decode while keeping long-prompt prefill close to baseline.
- Provide a practical GUI for building, launching, monitoring, and autotuning.
- Keep performance claims reproducible through cold, lane-locked benchmarks.
- Keep the fork maintainable by carrying only useful backends and upstream
  changes.

## Reference System

- Windows 11, AMD Ryzen 7 5800X3D, 64 GB RAM
- 2x AMD Radeon RX 9070 XT, 16 GB VRAM each, RDNA4 `gfx1201`
- AMD ROCm/HIP SDK 7.1 (reference) and 7.2 (current builds) for Windows; AMD
  proprietary Vulkan driver
- Main model: `Qwen3.8-27B-Q4_K_M.gguf` (UD variant
  `Qwen3.8-27B-UD-Q4_K_M.gguf` in the tables below); Vision projector:
  `mmproj-F16.gguf`

The two GPUs are normally used with layer split, not tensor split. GPU1 is the
preferred output device because GPU0 also drives the desktop. Device order is
backend- and workload-sensitive; exact routes are recorded with each benchmark.
See [Active Branches](docs/BRANCHES.md) for branch status.

## Performance Summary

Full tables, lane contracts and evidence links live in
[PERFORMANCE.md](PERFORMANCE.md).

### Benchmark levels L1-L3

All rows use one server slot, FlashAttention, cold prompt processing, no
prompt-cache reuse, no prime pass, `batch 8192 / ubatch 1024`, KV
`f8_e4m3 / f8_e4m3`, device route `ROCm1,ROCm0 -sm layer -ts 1,1`, `-ngl 999`,
`seed 42`, temperature 0.2, top-p 0.9, `--no-warmup`. Three-tier agent
workload: L1/L2 use a repository snapshot, L3 uses a deterministic synthetic
context (repo-snapshot is capped at ~53K tokens).

| Lane | Context | Actual prompt | Output | Context source |
| --- | ---: | ---: | ---: | --- |
| L1 | 16,384 | ~8.4K | 128 | repo-snapshot |
| L2 | 49,152 | ~33.9K | 256 | repo-snapshot |
| L3 | 98,304 | ~64.3K | 256 | synthetic |


### Windows results (2026-09-23, ROCm/HIP 7.2 and Vulkan)

Fresh Windows re-runs on the current binary family (`7c49e6212`:
`build-rocm72` HIP 7.2 and `build-vulkan-gcc16`), same lane contract as the
Linux table below: `batch 8192 / ubatch 1024`, KV `f8_e4m3 / f8_e4m3`,
FlashAttention, `-ngl 999`, one slot, cold prompt, `seed 42`, temp 0.2,
top-p 0.9, `--no-warmup`, repo-snapshot L1/L2, synthetic L3. Model
`Qwen3.8-27B-UD-Q4_K_M.gguf` (UD), single run per cell; actual prompt
geometry L1 8386 tok, L2 32996 tok, L3 64287 tok (Linux L1/L2 differ by
<3%: 8450/33865).

| Backend | Lane | Spec | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ROCm 7.2 | L1 | none | **2007.84** | 23.82 | 13.402 | - |
| ROCm 7.2 | L1 | MTP n3 | 1810.81 | 38.22 | 16.040 | 51.0% |
| ROCm 7.2 | L2 | none | **1819.46** | 24.00 | **8.888** | - |
| ROCm 7.2 | L2 | MTP n3 | 1744.44 | 36.10 | 9.844 | 50.0% |
| ROCm 7.2 | L3 | none | **1525.66** | 20.92 | 4.708 | - |
| ROCm 7.2 | L3 | MTP n3 | 1472.01 | **36.37** | **5.048** | 60.3% |
| Vulkan | L1 | none | 1569.17 | **25.42** | 12.332 | - |
| Vulkan | L1 | MTP n3 | 1606.98 | **39.00** | 15.059 | 53.4% |
| Vulkan | L2 | none | 1470.80 | 23.61 | 7.693 | - |
| Vulkan | L2 | MTP n3 | 1580.02 | 35.36 | 9.103 | 48.6% |
| Vulkan | L3 | none | 1243.99 | **22.19** | 4.050 | - |
| Vulkan | L3 | MTP n3 | 1357.37 | 35.14 | 4.685 | 58.3% |

ROCm stays ahead on prompt processing (L1 +28%, L2 +24%, L3 +23% over
Vulkan). Vulkan leads decode on the L1 rows (25.42 vs 23.82, 39.00 vs 38.22)
and on L3 spec-none (22.19 vs 20.92), while ROCm leads the L2 and L3 MTP rows
(36.10 vs 35.36, 36.37 vs 35.14). The `f8_e4m3`
KV policy is used on both backends; MTP rows carry the automatic Vulkan
last-8-f16 tail and the HIP `n_ubatch=256` draft cap from the same commit.

### Linux results (2026-09-09, ROCm 10, current binary)

| Format | Lane | Spec | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MXFP4-requant (dense) | L1 | none | **2253.06** | 29.38 | **15.787** | - |
| Q4_K_M (UD) | L1 | MTP n3 | 1836.97 | 50.13 | 18.674 | 78.1% |
| MXFP4-requant (UD) | L1 | MTP n3 | 1825.13 | 50.99 | 18.716 | 57.6% |
| Q4_K_M (UD) | L2 | none | 1765.82 | 23.53 | 8.517 | - |
| MXFP4-requant (UD) | L2 | none | **2076.46** | 25.92 | **9.777** | - |
| Q4_K_M (UD) | L2 | MTP n3 | 1712.48 | 39.94 | 10.542 | 62.9% |
| MXFP4-requant (UD) | L2 | MTP n3 | 1666.09 | **49.195** | 10.028 | 66.0% |
| MXFP4-hybrid-attnQ6 | L2 | none | 1769.91 | 24.26 | 8.623 | - |
| NVFP4-native | L2 | none | 488.95 | 24.86 | 3.218 | - |
| Q4_K_M (UD) | L3 | none | 1539.60 | 20.80 | 4.735 | - |
| MXFP4-requant (UD) | L3 | none | **1765.49** | 22.60 | **5.362** | - |
| Q4_K_M (UD) | L3 | MTP n3 | 1329.83 | 33.27 | 4.568 | 64.5% |
| MXFP4-requant (UD) | L3 | MTP n3 | 1502.55 | 33.68 | 5.081 | 52.4% |
| MXFP4-hybrid-attnQ6 | L3 | none | 1718.59 | 21.63 | 5.199 | - |
| NVFP4-native | L3 | none | 473.85 | 21.75 | 1.736 | - |

Current binary includes W24 (`nwarps=8` for MXFP4 decode) and W26 (MXFP4
prefill routed to MMQ): MXFP4 prefill +17-21% and decode +3-5% vs before.
MTP n3 is the decode optimum; acceptance profile `0.800/0.600/0.432` (L2),
and all `p_min`/`n_max`/draft-up-quantization attempts are negative. MTP
acceptance falls at L3 (52-64%) and NVFP4-native is not usable for
prompt-heavy lanes; hybrid is quality-first. L1/L2 MTP rows were recorded on
the W20/W24 binary (pre-W26 prefill routing); details and exact artifacts in
[PERFORMANCE.md](PERFORMANCE.md).

### Linux vs Windows (same lanes)

Windows re-runs (2026-09-23) repeat every Linux row on both Windows backends
with the same lane contract. Cells are `prompt / decode` in tok/s.

| Format | Lane | Spec | Linux ROCm 10 | Windows ROCm 7.2 | Windows Vulkan |
| --- | --- | --- | --- | --- | --- |
| MXFP4-requant (dense) | L1 | none | **2253.06** / **29.38** | 1999.23 / 27.05 | 1746.42 / 28.06 |
| Q4_K_M (UD) | L1 | MTP n3 | 1836.97 / **50.13** | 1810.81 / 38.22 | 1606.98 / 39.00 |
| MXFP4-requant (UD) | L1 | MTP n3 | 1825.13 / **50.99** | 1825.71 / 44.77 | 1739.19 / 46.35 |
| Q4_K_M (UD) | L2 | none | 1765.82 / 23.53 | **1819.46** / **24.00** | 1470.80 / 23.61 |
| MXFP4-requant (UD) | L2 | none | **2076.46** / 25.92 | 1813.11 / 25.25 | 1577.90 / **25.95** |
| Q4_K_M (UD) | L2 | MTP n3 | 1712.48 / **39.94** | **1744.44** / 36.10 | 1580.02 / 35.36 |
| MXFP4-requant (UD) | L2 | MTP n3 | 1666.09 / **49.20** | **1751.81** / 41.49 | 1703.11 / 44.99 |
| MXFP4-hybrid-attnQ6 | L2 | none | **1769.91** / 24.26 | 1756.82 / 24.44 | 1542.28 / **24.67** |
| NVFP4-native | L2 | none | 488.95 / 24.86 | 79.05 / 24.65 | **1560.12** / **25.81** |
| Q4_K_M (UD) | L3 | none | **1539.60** / 20.80 | 1525.66 / 20.92 | 1243.99 / **22.19** |
| MXFP4-requant (UD) | L3 | none | **1765.49** / 22.60 | 1528.46 / 22.01 | 1322.88 / **23.48** |
| Q4_K_M (UD) | L3 | MTP n3 | 1329.83 / 33.27 | **1472.01** / **36.37** | 1357.37 / 35.14 |
| MXFP4-requant (UD) | L3 | MTP n3 | 1502.55 / 33.68 | 1482.71 / 36.78 | 1447.82 / **41.64** |
| MXFP4-hybrid-attnQ6 | L3 | none | **1718.59** / 21.63 | 1484.54 / 21.35 | 1295.55 / **22.61** |
| NVFP4-native | L3 | none | 473.85 / 21.75 | 79.71 / 22.35 | **1307.73** / **23.17** |

Read this table together with the caveats:

- **Linux ROCm rows used GPU peer-to-peer (P2P)**; ROCm on Windows does not
  provide P2P, so the Windows ROCm column is a non-P2P baseline. The
  Windows-vs-Linux delta mixes the OS/driver change with the loss of P2P for
  cross-device layer traffic.
- **NVFP4-native is broken on Windows ROCm**: prompt throughput collapses to
  79 tok/s against 489 on Linux (-84%) while decode stays at parity. Windows
  Vulkan runs the same model normally (1560/1307), so the regression is in
  the Windows HIP path, not in the model.
- **MXFP4 prompt throughput is 12-14% below Linux on Windows ROCm**
  (`2076 -> 1813` at L2, `1765 -> 1528` at L3, `2253 -> 1999` at L1 dense);
  Q4_K_M and hybrid are at parity (±3%). MXFP4 decode is within 3%.
- **MTP acceptance is lower on Windows** (47-60% against 52-78% on Linux),
  which costs MTP decode (`Q4_K_M L1: 50.13 -> 38.22`); the L3 MTP rows are
  the exception and land slightly above Linux.
- **Vulkan gives up prompt throughput** (L3 1244 vs ROCm 1526) but wins
  decode on several rows and handles NVFP4-native far better than Windows
  ROCm.
- Single run per cell (r1): treat deltas below ~3% as noise.

### Model quality (PPL, 512 chunks; Linux ROCm 10)

| Format | bpw | PPL | Δ vs Q4_K_M |
| --- | ---: | ---: | ---: |
| Q4_K_M | 5.01 | **6.7802 ± 0.050** | — baseline |
| MXFP4-native | 4.25 | 7.1391 ± 0.054 | +5.29% |
| MXFP4-requant | 4.25 | 7.1937 ± 0.055 | +6.10% |
| MXFP4-hybrid-attnQ6 | 4.85 | **6.9925 ± 0.054** | +3.13% |
| NVFP4-native | 4.50 | **6.9840 ± 0.052** | +3.01% |

Q4_K_M stays the production quality baseline. MXFP4/NVFP4 trade quality for
speed — compare speed only together with this table. imatrix for FP4 formats
is not implemented (`quantize_mxfp4` ignores `quant_weights`).

## Quick Start

Install GUI dependencies and launch from the repository root:

```powershell
python -m pip install -r gui/requirements-gui.txt
python run.py
```

In the GUI: open **Build & Setup** and configure a backend, build or select
`llama-server`, open **Launch Server** and pick a GGUF model. Start with
`Spec: None` for the baseline; for an MTP-enabled GGUF use depth 3 as the
starting point. Use the recommended explicit device order in
**Benchmark / Autotune** — `Auto` is discovery only, not a benchmark
contract. Full build commands and requirements: [docs/build.md](docs/build.md).

## Benchmarking

Canonical methodology, lane contracts and history:
[BENCHMARKS.md](BENCHMARKS.md), `build_logs/agent-workload/BENCH_RUNS.csv`,
`BENCH_RECENT.md`, `BENCH_LANES.md`, `docs/research/RESULTS_LOG.md`,
and the per-experiment notes in [docs/research/](docs/research/). Always
compare neighboring controls (same model, backend, device order, split,
context, prompt/output length, batch/ubatch, KV, spec, cache policy and
background load); record an explicit `-dev` route for every dual-GPU result.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `gui2/` | GUI 2.0: local web UI (FastHTML + HTMX) |
| `src/`, `common/`, `include/` | llama runtime and speculative pipeline |
| `ggml/src/ggml-vulkan/`, `ggml/src/ggml-hip/`, `ggml/src/ggml-cuda/`, `ggml/src/ggml-cpu/` | Backends (CUDA layer is the HIP-compatible kernel source) |
| `scripts/` | Benchmark and autotune runners (`bench2.py`, agent workload) |
| `PERFORMANCE.md` | Current benchmark tables and lane contracts |
| `docs/` | Backends, build, RPC, research notes (accepted/rejected experiments) |

## Development

Read [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) and
[UPSTREAM_SYNC.md](UPSTREAM_SYNC.md) before changing the fork. When reporting
performance, include the model, backend, device order, split, context, actual
prompt/output tokens, batch/ubatch, KV types, speculative mode, cache policy
and background load.

## License and Security

Derived from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
MIT [LICENSE](LICENSE); bundled third-party components retain their own
notices. Security issues are handled privately per
[SECURITY.md](docs/SECURITY.md).
