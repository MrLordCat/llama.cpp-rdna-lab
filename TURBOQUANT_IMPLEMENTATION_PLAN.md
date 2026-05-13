# TurboQuant Implementation Plan

Date: 2026-05-13

## Current State

This fork has local TurboQuant-related pieces, a real TurboKV correctness path for selecting `--cache-type-k turboN --cache-type-v turboN`, and an opt-in direct compressed-KV FlashAttention prototype.

Existing local pieces:

- `GGML_TYPE_TBQ3_0` and `GGML_TYPE_TBQ4_0`: CPU-oriented 256-element TurboQuant weight/block formats.
- `GGML_TYPE_TQ3_0`: GPU-oriented 32-element 3-bit PolarQuant/WHT format.
- Stormrage-style KV cache names now resolve to real local `TKV` types:
  - `turbo2` / `turbo2_0` -> `GGML_TYPE_TKV2_0`
  - `turbo3` / `turbo3_0` -> `GGML_TYPE_TKV3_0`
  - `turbo4` / `turbo4_0` -> `GGML_TYPE_TKV4_0`

Implemented in the first pass:

- `GGML_TYPE_TKV2_0`, `GGML_TYPE_TKV3_0`, `GGML_TYPE_TKV4_0` appended after existing local type IDs.
- 128-element WHT block structs for `turbo2_0`, `turbo3_0`, and `turbo4_0`.
- CPU reference quantize/dequantize functions.
- HIP/CUDA `SET_ROWS` support for writing KV cache rows into the new formats.
- HIP/CUDA dequant/conversion support, so the existing attention graph can cast them back through the correctness path.
- CLI and `llama-bench` parsing where `turbo2`, `turbo3`, and `turbo4` now resolve to real local types, not aliases.
- GUI autotune options for `turbo2/3/4`.
- Direct compressed-KV FlashAttention path is now default for eligible local TKV K/V (`TKV2_0/TKV3_0/TKV4_0` with FlashAttention and matching K/V type).
- `GGML_TKV_DIRECT_FATTN=0` provides explicit fallback to graph-dequant path.
- `GGML_OP_TURBO_WHT` graph op for pre-rotating Q and inverse-rotating compressed-V attention output.
- HIP/CUDA FATTN vec instances for same-type `TKV2_0/TKV3_0/TKV4_0` K/V at D=128 and D=256.

Still missing:

- Full logit-level equivalence validation (current check confirms deterministic text-prefix parity on one fixed-seed single-turn sample).
- Kernel/perf tuning needed to close the remaining active-lane gap vs `q4_0`.
- Mixed K/V combinations and `TQ3_0` direct FlashAttention are intentionally not part of the current implementation.

## Constraints

- Do not reuse Stormrage's type IDs. They conflict with this fork:
  - local `41 = Q1_0`
  - local `42 = TBQ3_0`
  - local `43 = TBQ4_0`
  - local `44 = TQ3_0`
- New types must be appended after current local IDs.
- Keep the existing `TBQ*` and `TQ3_0` behavior working.
- Keep existing aliases only after they point to real local types.
- Guard risky fast paths until measured.

## Implementation Strategy

Phase 1: real types and correctness path

Status: implemented. `cmake --build build-rocm-vec --target llama-bench --config Release -j` completed successfully on 2026-05-13.

1. Add new types appended after `GGML_TYPE_TQ3_0`:
   - `GGML_TYPE_TKV2_0` / name `turbo2`
   - `GGML_TYPE_TKV3_0` / name `turbo3`
   - `GGML_TYPE_TKV4_0` / name `turbo4`
2. Add block structs with 128-element groups:
   - `turbo2`: norm + 2-bit indices
   - `turbo3`: norm + 2-bit indices + 1-bit high index
   - `turbo4`: norm + reserved half + 4-bit indices
3. Add CPU reference quantize/dequantize functions for those types.
4. Register the types in `ggml.c` and `ggml-quants.h`.
5. Update CLI/bench parsing so `turbo2/3/4` resolve to the new real types, while `tbq3/tbq4` keep their existing meaning.
6. Add HIP/CUDA set-rows and dequant paths so KV cache can be stored and converted on GPU.

Phase 2: speed path

Status: implemented and enabled by default for eligible local TKV lanes.

1. Add a direct compressed-KV FlashAttention prototype and wire it into runtime selection. Done.
2. Pre-rotate Q to match compressed K. Done with `GGML_OP_TURBO_WHT`.
3. Directly compute KQ from compressed K. Done for same-type `TKV2/3/4` vec FATTN lanes.
4. Inverse-rotate the attention output when V is compressed. Done with `GGML_OP_TURBO_WHT` inverse.
5. Add only the needed HIP template instances for the target lanes. Done for D=128 and D=256.
6. Keep an explicit operator fallback switch. Done with `GGML_TKV_DIRECT_FATTN=0`.

## First Validation

Minimum checks after Phase 1:

```powershell
python -m py_compile gui\llama_gui.py gui\build_manager.py gui\dependency_checker.py gui\hardware_detector.py
cmake -B build-rocm -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --target llama-bench --config Release -j
build-rocm\bin\llama-bench.exe -m models\Qwen3.6-27B-Q3_K_S.gguf -p 512 -n 16 -b 256 -ub 128 -ctk turbo3 -ctv turbo3 -fa 1 -ngl 99 -r 1
```

Benchmark policy:

- Use `--runs 1` for quick iterations.
- Use 3 runs only after a candidate is promising.
- Do not claim speedup until compared against the current best q4_0/q4_0 or q8_0/q8_0 lane with the same context/reuse settings.

## Smoke Benchmark Snapshot

Command shape, all `runs=1`, `--no-warmup`, `-p 64 -n 8 -b 128 -ub 128 -fa 1 -fitt 2048 -fitc 4096`, model `models/Qwen3.6-27B-Q3_K_S.gguf`, ROCm RX 9070 XT, build `9ef08998a` / `9101`:

| KV cache | Path | pp64 tok/s | tg8 tok/s |
| --- | --- | ---: | ---: |
| q4_0/q4_0 | baseline before direct prototype | 224.10 | 26.81 |
| turbo4_0/turbo4_0 | graph dequant fallback | 186.69 | 17.09 |
| turbo4_0/turbo4_0 | `GGML_TKV_DIRECT_FATTN=1` | 227.88 | 24.82 |
| turbo3_0/turbo3_0 | `GGML_TKV_DIRECT_FATTN=1` | 221.67 | 24.60 |
| turbo2_0/turbo2_0 | `GGML_TKV_DIRECT_FATTN=1` | 225.50 | 25.52 |

Interpretation: the direct path removes the major graph-dequant penalty for the small smoke lane and brings TKV decode close to q4_0.

## Active Lane Comparison vs q4_0

Primary comparison for `turbo4` uses the same best-shape lane as q4: `v2-review`, `ctx=12288`, `b=6144`, `ub=1024`, `repo-snapshot chars=21872`, no-reuse, thinking on, `spec=none`, model `Qwen3.6-27B-Q3_K_S.gguf`.

| KV cache | Mode | Runs | Aggregate TPS | Delta vs q4_0 |
| --- | --- | ---: | ---: | ---: |
| q4_0/q4_0 | baseline | 3 | 11.15 | baseline |
| turbo4_0/turbo4_0 | hybrid default (direct decode, F16 prefill) | 3 | 10.02 | -10.1% |
| turbo4_0/turbo4_0 | full direct prefill (`GGML_TKV_DIRECT_PREFILL=1`) | 1 | 7.70 | -30.9% |

Timing breakdown: q4 prompt/decode mean `1149.47/27.85 tok/s`; turbo4 hybrid prompt/decode mean `1013.22/25.80 tok/s`.

Interpretation: the earlier `ub=192` run understated Turbo4. At `ub=1024`, Turbo4 is still slower than q4_0, but the gap is now about 10% instead of 25-26%. Full-direct prefill is slower than the hybrid path; keep direct decode but route prompt/prefill through F16 dequant + WMMA by default.

Initial diagnostic `ub=192` result remains useful for direct-vs-fallback only: `q4_0=9.01 TPS`, direct `turbo4=6.68 TPS`, fallback `turbo4=3.10 TPS`.

## Initial Expected Outcome

Phase 1 should make `turbo2/3/4` real selectable KV cache types. It is correctness-first and may be slower than `q4_0` until Phase 2 removes graph dequant overhead.

Phase 2 is operational and default-enabled for local TKV lanes. The current best Turbo4 path is hybrid default and is about 10% behind q4_0 on the active `ub=1024` lane; further tuning should target Turbo4 decode vec-dot and F16 dequant/prefill overhead.
