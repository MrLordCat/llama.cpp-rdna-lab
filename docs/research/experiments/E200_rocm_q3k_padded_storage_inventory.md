# E200 ROCm Q3_K Padded Storage Inventory

## Metadata

- Experiment ID: E200
- Date: 2026-05-24
- Owner: Codex
- Branch/Commit: master after E199 (`8b9e8e26f`)
- Target lane: H39 ROCm decode parity, Qwen3.6-27B-Q3_K_S, `ctx=12288`, `batch=6144`, `ubatch=2048`, q4/q4 KV, `spec=none`, no reuse, thinking on

## Hypothesis

- Statement: the only plausible padded-layout transfer from Vulkan to ROCm is a backend-private storage route, but it might be possible to stage it as a correctness-first prototype instead of a single high-risk rewrite.
- Mechanism: inventory every code path that assumes host `block_q3_K` stride, then define a cut plan where storage packing, dequant, MMVQ, MMQ, and copy/view helpers are enabled only after cheap correctness gates pass.
- Why now: E199 rejected transient repack, duplicate padded copies, and `vecdotq.cuh`-only packed loads. The remaining route is large enough that the next useful step is a reproducible blast-radius map and a first smoke-test command.

## Math / Theory

- Assumptions:
  - host Q3_K block size is `110` bytes;
  - target padded device Q3_K block size is `112` bytes;
  - E199 measured all-Q3_K replacement padding overhead as `179.47 MiB` (`1.818%`);
  - current ROCm single-GPU model load reports `ROCm0 model buffer size = 11254.73 MiB`, so a replacement layout is memory-plausible while a duplicate copy is not.
- Expected speedup corridor:
  - unchanged from E199: `+2%` wall requires `1.0341x` local at optimistic `59.5%` Q3_K share, and Vulkan parity requires `1.5763x`;
  - E200 makes no speed claim and only decides whether a correctness-first prototype is technically sliceable.
- Failure conditions:
  - the minimal slice still requires split-buffer, async-copy, and all prompt staging code before any cheap correctness test;
  - current correctness tooling cannot isolate Q3_K `MUL_MAT`;
  - the inventory finds hidden Q3_K users outside CUDA/HIP kernels that would make an env-gated prototype unsafe.

## Implementation Plan

1. Minimal code surface to change:
   - add `scripts/research/rocm_q3k_storage_inventory.py`;
   - no runtime C++ behavior change in E200.
2. Guard rails:
   - keep this as an inventory/design gate;
   - use `test-backend-ops` as the first correctness gate before real server;
   - reject decode-only MMVQ changes unless dequant/MMQ/copy paths agree on the same Q3_K storage stride.
3. Rollback path:
   - no C++ rollback needed; remove the inventory script/docs if the route is abandoned.

## Benchmark Plan

- Baseline command: `build-rocm-vec\bin\test-backend-ops.exe test -b ROCm0 -o MUL_MAT -p "type_a=q3_K,type_b=f32,m=16,n=1,k=256" --output csv`.
- Candidate command: none in E200.
- Number of runs: one current-tree smoke for `n=1`, one current-tree smoke for `n=64`.
- Artifacts path: `build_logs/agent-workload/e200-rocm-q3k-*`.

## Metrics

- static touchpoint count
- current Q3_K `MUL_MAT` correctness smoke status
- prototype cut feasibility
- rejected unsafe cuts

## Result

- Outcome: keep inventory; proceed only to a guarded P1/P2 prototype if it starts with correctness.
- Static touchpoints:

| Area | File | Matches | First lines |
| --- | --- | ---: | --- |
| plain CUDA buffer API | `ggml/src/ggml-cuda/ggml-cuda.cu` | `14` | `872, 901, 909, 917, 927, 937, ...` |
| split CUDA buffer API | `ggml/src/ggml-cuda/ggml-cuda.cu` | `8` | `1147, 1197, 1236, 1283, 1285, 1286, ...` |
| async/view/copy helpers | `ggml/src/ggml-cuda/ggml-cuda.cu` | `16` | `1633, 2368, 2382, 3560, 3614, 3625, ...` |
| Q3_K dequant/getrows path | `ggml/src/ggml-cuda/convert.cu` | `3` | `164, 167, 639` |
| Q3_K MMVQ vecdot path | `ggml/src/ggml-cuda/vecdotq.cuh` | `7` | `447, 480, 837, 840, 847, 850, ...` |
| Q3_K MMQ path | `ggml/src/ggml-cuda/mmq.cuh` | `15` | `81, 206, 256, 1965, 1974, 1992, ...` |
| Q3_K MMVQ dispatch/policy | `ggml/src/ggml-cuda/mmvq.cu` | `10` | `23, 52, 124, 145, 162, 192, ...` |
| Q3_K type traits | `ggml/src/ggml-cuda/common.cuh` | `1` | `999` |

- Current correctness smokes:
  - `type_a=q3_K,type_b=f32,m=16,n=1,k=256`: supported/pass row recorded for `ROCm0`;
  - `type_a=q3_K,type_b=f32,m=1,n=64,k=256`: supported/pass row recorded for `ROCm0`.
- Prototype cut:
  - P1: non-split CUDA buffer Q3_K padded set/get and reverse get;
  - P2: padded-aware Q3_K dequant + MMVQ + MMQ accessors, gated by `test-backend-ops` Q3_K `MUL_MAT`;
  - P3: split/async/copy/view offsets and cuBLAS staging;
  - P4: real H39 server A/B and text sanity.
- Confidence: medium. The map is concrete and the current correctness smoke is cheap, but the full route still has broad C++/kernel scope.
- Recommendation: proceed only with an env-gated P1/P2 prototype. Do not run a speed benchmark until Q3_K `MUL_MAT` correctness passes under the padded-storage gate.

## Notes

- Important correction: a decode-only MMVQ padded kernel is not a valid first cut. The model can still touch Q3_K through prompt/prefill, dequant, or test-backend-ops paths before decode speed is measurable.
- Useful first command for the next implementation step:

```powershell
build-rocm-vec\bin\test-backend-ops.exe test -b ROCm0 -o MUL_MAT -p "type_a=q3_K,type_b=f32,m=16,n=1,k=256" --output csv
```

## Artifacts

- `build_logs/agent-workload/e200-rocm-q3k-storage-inventory.md`
- `build_logs/agent-workload/e200-rocm-q3k-mulmat-current-smoke.csv`
- `build_logs/agent-workload/e200-rocm-q3k-mulmat-current-n64-smoke.csv`
- `scripts/research/rocm_q3k_storage_inventory.py`
