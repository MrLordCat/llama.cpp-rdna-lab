# E291: ROCm Long-Context Q3_K Decode and Memory

Date: 2026-07-14

## Scope

This repeats the exact E284 GUI long lane after E289:

- 49,152 context capacity;
- 31,997 actual prompt tokens and 128 output tokens;
- `b8192/ub1024`, q8 K/V, no reuse/prime/warmup;
- `ROCm1,ROCm0`, layer split `1,1`;
- Qwen3.6-27B-Q3_K_S MTP GGUF.

## Performance

| Build / mode | Prompt TPS | Decode TPS | Aggregate TPS | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| E284 pre-fix, none | 1,338.10 | 22.30 | 4.3035 | - |
| E289 packed subtract, none | 1,355.81 | 26.44 | 4.4901 | - |
| E289 packed subtract, MTP n3 | 1,343.57 | 32.34 | 4.5953 | 48.70% |
| E289 packed subtract, MTP n4 | 1,320.58 | 28.33 | 4.44 | 42.02% |

The normal long-context decode gain is `+18.6%`; prompt also moves `+1.3%`
instead of regressing. Current MTP n3 is `+22.3%` faster than current none with
a `-0.9%` prompt cost. MTP n4 is rejected for this lane because lower
acceptance outweighs the deeper draft.

## Matched VRAM Accounting

The llama-owned model/context/compute allocations match Vulkan closely. The
larger ROCm reading is outside those buffers:

| Backend / mode / model share | Self | Unaccounted |
| --- | ---: | ---: |
| Vulkan none, 5.43 GiB | 6,339 MiB | 989 MiB |
| ROCm none, 5.38 GiB | 6,287 MiB | 1,905 MiB |
| Vulkan none, 6.21 GiB | 7,112 MiB | 987 MiB |
| ROCm none, 6.25 GiB | 7,155 MiB | 1,374 MiB |
| Vulkan MTP n3, 5.43 GiB | 6,578 MiB | 1,011 MiB |
| ROCm MTP n3, 5.38 GiB | 6,524 MiB | 1,931 MiB |
| Vulkan MTP n3, 6.21 GiB | 7,328 MiB | 1,264 MiB |
| ROCm MTP n3, 6.25 GiB | 7,373 MiB | 2,291 MiB |

ROCm therefore uses roughly `0.4-1.0 GiB` more untracked device memory per GPU
in this lane. It is HIP/hipBLAS code, graph, and library state rather than a
second model or misplaced KV cache. MTP creates additional context and graph
state and makes the difference most visible on the second device. The host
model mapping remained about `520 MiB` for both backends.

## Decision

- Keep the E289 packed subtraction.
- Use MTP n3 rather than n4 for this exact 32k-prompt/128-output lane.
- Treat ROCm VRAM headroom as about 1 GiB lower per active device when planning
  near-capacity long-context launches, even though llama-owned buffers match.
- The remaining long-MTP limit is FA/KV plus multi-column verify, not the N=1
  packed subtraction fixed by E289.

Primary artifacts:

- `e291-rocm-long49ctx-q3-sub4twiddle-none-r2.*`;
- `e291-rocm-long49ctx-q3-sub4twiddle-mtp-n3-r2.*`;
- `e291-rocm-long49ctx-q3-sub4twiddle-mtp-n4-r1.*`;
- E284 matched Vulkan artifacts for memory comparison.
