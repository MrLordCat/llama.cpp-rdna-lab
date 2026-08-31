# HIP SDK 7.2 rollout checklist

Date: 2026-08-30

Experiment: G00

Objective: build the current fork with HIP SDK 7.2 in a separate directory,
prove that the binary loads the 7.2 runtime and `gfx1201` backend correctly,
then compare it with the unchanged HIP 7.1 control. This is a toolchain/runtime
change only; no performance feature is enabled during G00.

## Safety contract

- [x] Work on branch `decode`.
- [x] Preserve the existing `build-rocm` HIP 7.1 control.
- [x] Preserve unrelated working-tree edits in
  `scripts/research/coherence_smoke.py` and `src/models/qwen4exp.cpp`.
- [x] Confirm HIP SDK 7.2 is installed under
  `C:/Program Files/AMD/ROCm/7.2` (`HIP version 7.2.60201`).
- [x] Use a new directory: `build-rocm72`.
- [ ] Do not promote/copy the binary until correctness and L0 gates pass.

## A. Freeze the HIP 7.1 control

Current `build-rocm` contract:

- [x] Ninja / Release / shared libraries.
- [x] HIP SDK 7.1 `clang/clang++`.
- [x] `AMDGPU_TARGETS=gfx1201`.
- [x] `GGML_HIP=ON`.
- [x] `GGML_HIP_MMQ_MFMA=ON`.
- [x] `GGML_HIP_ROCWMMA_FATTN=ON`.
- [x] `GGML_HIP_NO_VMM=ON`.
- [x] `GGML_HIP_FAST_MATH=ON`, FA on, all-quant FA off.
- [x] `GGML_HIP_EXPERIMENT_PROFILE=default`.
- [x] `GGML_OPENMP=OFF`.

## B. Configure `build-rocm72`

- [x] Pin HIP/ROCm paths to 7.2 for the configure process.
- [x] Pin the known-compatible MSVC 14.44 toolset. HIP clang can otherwise
  discover 14.51 and fail in CUDA/HIP math declarations.
- [x] Configure with the same feature switches as the 7.1 control.
- [x] Verify `CMakeCache.txt` names HIP 7.2 compilers and Release mode.

Reproducible bash command:

```bash
export HIP_PATH='C:/Program Files/AMD/ROCm/7.2'
export ROCM_PATH="$HIP_PATH"
export CMAKE_PREFIX_PATH="$HIP_PATH/lib/cmake"
export VCToolsInstallDir='C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207'
export VCINSTALLDIR='C:\Program Files\Microsoft Visual Studio\18\Community\VC'
export PATH="/c/Program Files/AMD/ROCm/7.2/bin:/c/Strawberry/perl/bin:/c/Strawberry/c/bin:$PATH"

cmake -S . -B build-rocm72 -G Ninja \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_FAST_MATH=ON \
  -DGGML_HIP_EXPERIMENT_PROFILE=default \
  -DGGML_OPENMP=OFF \
  -DCMAKE_C_COMPILER="$HIP_PATH/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$HIP_PATH/bin/clang++.exe" \
  -DCMAKE_BUILD_TYPE=Release
```

If Windows API declarations fail, repeat configure with both
`CMAKE_C_FLAGS` and `CMAKE_CXX_FLAGS` set to
`-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00`, recording that as a configuration
difference. Do not add those flags preemptively.

## C. Build

- [x] Build `llama-server` with `-j 4`.
- [x] Confirm `build-rocm72/bin/llama-server.exe` exists.
- [x] Confirm the required shared DLLs were staged into `bin/`.
- [x] Review warnings: only existing source/Clang warnings and missing optional
  OpenSSL/HTTPS support; no HIP compile or link failure.

```bash
cmake --build build-rocm72 -j 4 --target llama-server
```

## D. Binary and backend smoke

- [x] `llama-server.exe --version` exits successfully.
- [x] `llama-server.exe --list-devices` loads the HIP backend.
- [x] Both RX 9070 XT devices appear as `gfx1201`/ROCm devices.
- [x] Runtime resolution selects
  `C:/Program Files/AMD/ROCm/7.2/bin/amdhip64_7.dll`; configure reports SDK
  rocWMMA and the model context enables Flash Attention.
- [x] A short deterministic `Hello` request completes and the server shuts
  down cleanly.

## E. G00 performance/correctness gate

Use the locked contract from [the decode lane](../decode/README.md).

- [x] Adjacent HIP 7.1 dual ROCm1,ROCm0 control (L0/L1, r1).
- [x] HIP 7.2 dual ROCm1,ROCm0 candidate (L0/L1, r1).
- [x] Compare prompt and decode separately.
- [x] Coherence/correct output and graceful shutdown pass (`server_stopped`).
- [x] No additional active Shared-memory spill or residency regression observed
  (identical command line, identical memory breakdown shape).
- [ ] Repeat any apparent gain before promotion: decode delta is slightly
  negative; repeat is required before any promotion decision.

## F. Promotion decision

Promote HIP 7.2 to the decode research control only if:

- it is correct and stable on both topologies;
- prompt does not regress by more than 1%;
- decode is neutral or better after repeat;
- VRAM/Shared-memory behavior is no worse;
- all subsequent experiments identify the binary/toolchain explicitly.

If a gate fails, keep `build-rocm` as the control and record the exact
reopening condition. Do not delete either build directory.

## Result record

| Check | Result |
| --- | --- |
| Configure | pass: HIP 7.2 clang/clang++, MSVC 14.44 compatibility headers, Release/default profile; no extra WINVER flags required |
| Build | pass: `llama-server.exe` 11 MiB, `ggml-hip.dll` 82 MiB; repeat Ninja invocation reports no work |
| Version/backend smoke | pass: commit `75f7e87dc`, two RX 9070 XT `gfx1201`, Wave32, VMM off, host-staged peer route |
| Hello smoke | pass: visible response `Hello! How can I help you today?`, stop finish, 31 tokens, about 27.55 tok/s; clean Ctrl+C shutdown |
| Dual L0 (r1) | HIP 7.1 1498.02 / 27.41 ptps/dtps vs HIP 7.2 1503.94 / 26.89 ptps/dtps |
| Dual L1 (r1) | HIP 7.1 1774.51 / 26.26 ptps/dtps vs HIP 7.2 1772.67 / 26.00 ptps/dtps |
| Decision | deferred: prompt parity within noise, decode slightly negative; repeat needed before promotion |

## G00 benchmark comparison (2026-08-30, dual ROCm1,ROCm0, `-sm layer -ts 1,1`)

Contract: `models/Qwen3.8-27B-UD-Q4_K_M.gguf`, context 8192/16384,
batch/ubatch 8192/1024, KV `f8_e4m3/f8_e4m3`, Flash Attention on,
`spec=none`, `-fit off`, `--no-warmup`, seed 42, temperature 0, one shot
(runs=1). Both builds are the same commit `75f7e87dc`; only the HIP SDK
differs (7.1 vs 7.2).

| Level | Metric | HIP 7.1 | HIP 7.2 | Delta |
| --- | --- | ---: | ---: | ---: |
| L0 (3995 tok) | Prefill tok/s | 1498.02 | 1503.94 | +0.40% |
| L0 (64 tok) | Decode tok/s | 27.41 | 26.89 | -1.90% |
| L1 (7901 tok) | Prefill tok/s | 1774.51 | 1772.67 | -0.10% |
| L1 (128 tok) | Decode tok/s | 26.26 | 26.00 | -0.99% |

Interpretation:

- Prefill is at parity on both levels (`+0.40%` / `-0.10%`), well inside
  run-to-run noise.
- Decode is slightly negative on both levels (`-1.90%` / `-0.99%`). The
  decode-lane `+3%` admission gate is not met, and the `-1%` prompt gate is
  not triggered.
- The toolchain change is therefore not promoted as a decode improvement and
  currently shows no reason to demote HIP 7.1 as the control for the decode
  lane.
- One-shot `r1` deltas of this size are noise-dominated; a repeat (r2) is
  needed only if a real A/B conclusion is required. For the purpose of the
  knowledge base, HIP 7.2 remains a valid alternative build (works, same
  prefill, decode within ~2%) but is not the default control.

Artifacts:

- `build_logs/bench/g00-rocm72-l01-r1/`;
- `build_logs/bench/g00-rocm71-l01-r1/`;
- `build_logs/g00-rocm72-l01-r1.bench.log`;
- `build_logs/g00-rocm71-l01-r1.bench.log`.

Smoke residency at `-c 8192`, `f8_e4m3`, dual layer split:

- model: ROCm0 7,887 MiB, ROCm1 6,791 MiB, CPU-mapped 682 MiB;
- context: ROCm0 199 MiB, ROCm1 205 MiB;
- compute: ROCm0 15 MiB, ROCm1 8 MiB;
- after shutdown: no `llama-server.exe`; both devices report 15,265 MiB free.
