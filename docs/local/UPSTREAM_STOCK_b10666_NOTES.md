# Upstream stock `llama.cpp` — notes (b10666)

> Working notes for the stock clone `D:/GitHub/llama.cpp-upstream-stock`
> (HEAD `b10666` = `4e97ac86e`, refreshed 2026-08-28, ~644 commits ahead).
> Fork: `D:/GitHub/llama.cpp-with-GUI`.

## 1. Builds (recipes)

### Vulkan — `build-vulkan-b10666` ok
- Toolchain: **winlibs** GCC 16.2.0 + mingw-w64 14.0.0 UCRT -> `D:/GitHub/toolchains/winlibs-ucrt/mingw64/bin`.
- PATH for build: `.../winlibs-ucrt/mingw64/bin:/c/VulkanSDK/1.4.350.0/Bin`.
- `-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00` required (new APIs `THREAD_POWER_THROTTLING_STATE` in `ggml-cpu.c`, `CreateFile2` in `cpp-httplib`).
- Old Strawberry MinGW (GCC 13, 2023) does NOT have these APIs — cannot build.
- Result: `build-vulkan-b10666/bin/llama-server.exe`.

### ROCm — `build-rocm-b10666` ok
- SDK: **HIP 7.2.0** (`C:\Program Files\AMD\ROCm\7.2`, clang 21.0.0git AMD-Lightning-Internal).
- CMake: Ninja, `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DGGML_HIP_NO_VMM=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release`, compilers `ROCm/7.2/bin/clang(.++).exe`, `CMAKE_C/CXX_FLAGS="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00"`.
- **Mandatory** before `ninja`: `export VCToolsInstallDir="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207"` (+ `VCINSTALLDIR`) — otherwise HIP-clang picks MSVC 14.51 and fails (`__clang_cuda_math_forward_declares.h: isgreater overload`).
- HIP 7.1 does **not** work: `Invalid dpp_ctrl wavefront shifts GFX10+` in `mmf-instance-ncols_1.cu` (WMMA gfx12). HIP 7.2.0 fixes it.
- CMake reconfiguration: the cache does not overwrite `CMAKE_BUILD_TYPE`/`GGML_OPENMP` — use `-UCMAKE_BUILD_TYPE` and explicit `-D` (otherwise Debug + OpenMP ON -> `__kmpc_*` undefined at link).
- Result: `build-rocm-b10666/bin/llama-server.exe` (shared build; DLLs in same `bin/`).

## 2. Running "like the fork"

Canonical fork profile (Primary Qwen3.8-27B Q4_K_M, safe 49K) and its stock
equivalent (fork-only flags removed):

```
llama-server.exe -m Qwen3.8-27B-Q4_K_M.gguf \
  -c 49152 -b 8192 -ub 1024 -np 1 -ngl 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type none            # see section 3 about MTP
  -dev ROCm1,ROCm0 -sm layer -ts 1,1     # Vulkan: -dev Vulkan1,Vulkan0 ...
  --seed 42 --no-warmup -fit off --cache-ram 0 --ctx-checkpoints 0 \
  --metrics -t 8 --threads-http 4
```

### Fork vs stock flag table
| Flag / param | Fork (rdna-lab profile) | Stock b10666 | Status |
|---|---|---|---|
| `-dev / -sm / -ts` | `ROCm1,ROCm0` / `layer` / `1,1` | default: all GPUs / `layer` / auto | ok |
| `-c -b -ub -np -ngl` | 49152 / 8192 / 1024 / 1 / 999 | 0 / 2048 / 512 / 1 / -1(auto) | must set explicitly |
| KV | `q8_0` (ROCm), `f8_e4m3` (Vulkan) | `f16` default; `f8_e4m3` **not available** | use `q8_0` |
| `--flash-attn` | `on` | `auto` | set `on` |
| `--ctx-checkpoints` | 0 | **32** default | set 0 |
| `--conversation-cache`, `--checkpoint-every-n-tokens` | present | **absent** | do not pass |
| `GGML_VK_FORCE_AMD_LARGE_MATMUL`, `GGML_VK_AMD_LARGE_MATMUL_VARIANT=wn32` | present | **absent** | do not pass |
| `--metrics` | present | present | ok |

## 3. MTP / nextn — SUPPORTED (important!)

The initial claim "stock does not support MTP for Qwen3.8" was **wrong**.
Facts (verified in code and by a run):

- Stock knows arch `qwen35` (`src/llama-arch.cpp:41-42`, `src/llama-model.cpp:320-322`)
  and nextn: `src/models/qwen35.cpp` reads `LLM_KV_NEXTN_PREDICT_LAYERS`
  (`hparams.n_layer_nextn`), `load_block_mtp()` loads the `blk.<nextn>.nextn.*` tensors.
- GGUF `Qwen3.8-27B-Q4_K_M.gguf`: `block_count=65`, `nextn_predict_layers=1`,
  tensors `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm,...}` — present.
- Warnings `unused tensor blk.64.nextn.*` in the **main** context are
  **expected**: nextn weights are `TENSOR_SKIP` when `ml.load_mtp == false`
  and are loaded only into the **draft** (MTP) context. This is not "no support".
- Run with `--spec-type draft-mtp --spec-draft-n-max 3` (ROCm, single server):
  `draft acceptance = 0.037 (2/54), mean len = 1.11`; decode 15.56 tps —
  **low acceptance**, so MTP in stock is slower than its own `spec=none`
  (~25 tps). Any speed claim needs an adjacent `spec=none` baseline and a
  comparison with the fork.
- **Honestly NOT investigated yet: why stock acceptance is so low.**
  Known: mechanics enabled (draft_n>0, accepted printed), GGUF and hparams are
  correct (block_count=65, nextn_predict_layers=1), but acceptance 3.7% vs
  ~60%+ on the fork (fork: local MTP path optimizations, RPC hash fix, etc.).
  Two untested hypotheses: (1) stock lacks the fork's nextn-feed optimizations,
  so it predicts worse; (2) interplay of q8_0 KV / batch 8192 / ubatch 1024
  with the stock graph. Neither has been tested; "why" needs a separate pass
  (diff the nextn paths in `src/llama-*.cpp` + `llama-model.cpp`, or A/B with
  different b/ub/KV).

## 3b. IMPORTANT: stock PEG parser and HTTP 500 on long synthetic prompts (solved)

- Stock b10666 enables an **auto PEG parser** for `/v1/chat/completions`
  (`autoparser::peg_generator` -> `COMMON_CHAT_FORMAT_PEG_NATIVE`,
  `common/chat.cpp`). It is strict: when the model emits "garbage" bytes on
  a long synthetic prompt (e.g. `| 2 | 00666`, `0= 0 | 00...`), the server
  replies **HTTP 500** "model produced output that does not match the expected
  peg-native format" and the benchmark level fails. The fork has no PEG parser,
  so the same requests pass (fork parses as content).
- **`--skip-chat-parsing` does NOT work on the server** (gap in b10666):
  `server-common.cpp:1334 inputs.force_pure_content = opt.force_pure_content`,
  but `server_opt.force_pure_content` (server-common.h:319) is never populated
  from `params_base.force_pure_content_parser` -> always false.
- Working workaround: API field **`"chat_format": 0`** (CONTENT_ONLY) in the
  payload. Stock knows it (`server-schema.cpp:291`); the fork does not — and
  does not need it.
- `bench2.py` gained option `--api-extra '{"chat_format": 0}'`
  (merged into every completion payload; script NOT committed).
- Working stock recipe: `--server-extra "--reasoning off"` +
  `--api-extra '{"chat_format": 0}'`. Without `--reasoning off` the model
  stops early (EOS after ~7 tokens).

## 4. Stopping the server

- Stock has **no** `POST /shutdown` (404).
- Graceful: `AttachConsole` + `GenerateConsoleCtrlEvent(1, windowsPID)`
  (see `gui/bench_runner.py: send_windows_console_break`).
  Take the Windows PID from `tasklist` — **not** the `[N]` job id in git-bash.
- Alternative: `taskkill /PID <pid> /T` (no `/F`).

## 5. Smoke results (2026-08-28, 54-token prompt — NOT a benchmark!)

| | ROCm (HIP 7.2) | Vulkan |
|---|---|---|
| health | ~9–11 s | 11 s |
| prefill | 147.5 tps | 404.7 tps |
| decode | 25.1 tps | 29.1 tps |
| graphs reused | 6 | 6 |

Notes:
- `llama_sampler_backend_support: device 'ROCm0' does not have support for op TOP_K` — warning, not an error.
- `--cache-idle-slots requires --cache-ram, disabling` — harmless with `--cache-ram 0`.

## 5b. Bench: fork vs stock L0–L4 (2026-08-28, bench2)

Setup: Qwen3.8-27B-Q4_K_M, rdna-lab profile (`-c` per level:
8192/16384/49152/98304/131072, b 8192, ub 1024, np 1, ngl 999, FA on,
KV q8_0, spec none, `-sm layer -ts 1,1`, seed 42, no-warmup, fit off,
cache-ram 0, ctx-checkpoints 0). Stock ran from copies
`bench2-bins/stock_vk|stock_rm` (to avoid PATH substitution), all with
`--reasoning off`; stock L2 additionally with `--api-extra '{"chat_format": 0}'`
(see 3b).

### prefill_tps (tok/s)

| L | ctx / prompt | stock-vk | fork-vk | stock-rocm | fork-rocm |
|---|---|---|---|---|---|
| 0 | 8K / 4K    | 1245.0 | 1417.8 | 1632.1 | 1761.2 |
| 1 | 16K / 8K   | 1325.8 | 1592.9 | 1665.4 | 1871.4 |
| 2 | 49K / 31K  | 1203.9 | 1512.2 | 1343.4* | 1684.1 |
| 3 | 98K / 66K  | 1001.5 | 1294.9 |  974.5 | 1404.9 |
| 4 | 131K / 97K |  859.7 | 1160.7 |  785.5 | 1212.6 |

\* stock-rocm L2 — from the separate `stock-rocm-l2-fix` run (predicted = 256).

### decode_tps (tok/s)

| L | stock-vk | fork-vk | stock-rocm | fork-rocm |
|---|---|---|---|---|
| 0 | **30.01** | 27.12 | 23.41 | **24.90** |
| 1 | **29.67** | 27.95 | 23.62 | **24.24** |
| 2 | **27.77** | 27.09 | 21.14* | **21.86** |
| 3 | **25.03** | 24.99 | 18.22 | **18.51** |
| 4 | 23.92 | **24.90** | 15.95 | **16.44** |

Notes:
- Preliminary: **prefill — fork is faster** (Vulkan +14..+35%, ROCm +8..+54%;
  the gap grows with prompt length); **decode — parity** (+/-5%: ROCm fork
  slightly faster, Vulkan short-context stock slightly faster). 1 run/level
  only — not final; needs repeats and adjacent baselines.
- **Vulkan decode: fork is SLOWER than stock on short contexts** (L0 -9.6%,
  L1 -5.8%, L2 -2.5%; parity at L3; fork +4% at L4). Tracked as research:
  `docs/research/D136_STOCK_VULKAN_DECODE_EDGE.md`.
- stock-rocm L1 in the full run stopped early (68 tokens) — decode there is on
  a truncated output; L1 should be re-checked.
- Full runs: `stock-vk-l0-l4`, `fork-vk-l0-l4`, `stock-rocm-l0-l4`,
  `fork-rocm-l0-l4` in `build_logs/bench/`.

## 6. Useful paths / commands

- Model: `D:/GitHub/llama.cpp-with-GUI/models/Qwen3.8-27B-Q4_K_M.gguf`.
- GGUF metadata: `build-rocm-b10666/bin/llama-gguf.exe <file> r n`
  (positional syntax; prints keys only — values need a custom parser).
- PATH to run ROCm: `C:\Program Files\AMD\ROCm\7.2\bin` + winlibs bin.
- HIP 7.1 stays installed (`C:\Program Files\AMD\ROCm\7.1`) — do not remove,
  the fork still builds on it.
