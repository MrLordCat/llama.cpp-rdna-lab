# Windows FP8 Reference (bench2 archive, 2026-08-29 .. 2026-09-01)

Reference snapshot of the newest Windows `f8_e4m3` KV runs found in
`D:\GitHub\llama.cpp-with-GUI\build_logs\bench\` (branch `rpc-vulkan`).
Kept at the repo root so Linux results can be compared file-to-file before
any README/PERFORMANCE table update. Every row is exact `metrics.csv`
output; the source directory is the run folder name.

## Evaluation contract

- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf` (Windows UD build, differs from the
  Qwen3.8-27B-Q4_K_M.gguf used in the older README/PERFORMANCE snapshot).
- `bench2` single-level runs, `-c` per level:
  - L1 = `ctx=16384`, prompt ~7901, decode 128;
  - L2 = `ctx=49152`, prompt ~30609, decode 256 (169 in one q8 run);
  - L3 = `ctx=98304`, prompt ~64287, decode 256;
  - L4 = `ctx=131072`, prompt ~94257, decode 256.
- FlashAttention on, `-b 8192 -ub 1024` unless noted, MTP n2 (`--spec-type
  draft-mtp --spec-draft-n-max 2`), `-fit off --cache-ram 0
  --ctx-checkpoints 0`, no warmup.
- `prompt_tps` / `decode_tps` / `aggregate_tps` / `mtp_accepted/mtp_draft_n`
  from `metrics.csv`; acceptance rows are blank when the field was not
  recorded.

## Vulkan `f8_e4m3` (clean `Vulkan0,Vulkan1`)

| Level | Date (2026) | commit | Layout | Prompt tps | Decode tps | Aggregate | Acceptance | Source run folder |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| L1 | 08-31 | `291dc69d7` | `Vulkan0,Vulkan1` | 1650.08 | 38.40 | 15.76 | 68/116 | `vk-qwen3-8-27b-ud-q4-k-m-l1-b8192-u1024-f8_e4m3-mtp-n2--20260831T183641544914Z-5da85c9ce39a4e08a080a1c7828a2bcc` |
| L2 | 08-31 | `75f7e87dc` | `Vulkan0,Vulkan1 ts=100,100` | 1405.21 | 32.68 | 8.64 | - | `vk-qwen3-8-27b-ud-q4-k-m-l2-b8192-u1024-f8_e4m3-mtp-n2--20260831T151437032571Z-21abe17d930a497cb43fdab21d83537d` |
| L3 | 08-29 | `75f7e87dc` | `Vulkan0,Vulkan1 ts=4,6` | 1216.24 | 38.27 | 4.30 | - | `vk-qwen3-8-27b-ud-q4-k-m-b8192-u1024-f8_e4m3-mtp-n2` |

L1 spec-none reference (same binary family): `1450.47/27.28/12.62`
(`vk-qwen3-8-27b-ud-q4-k-m-b8192-u1024-f8_e4m3-none`, 2026-08-29).
L2 spec-none: `1074.16/25.32/6.63`
(`vk-qwen3-8-27b-ud-q4-k-m-l2-b8192-u1024-f8_e4m3-none--20260831T153540927327Z-...`,
2026-08-31).

## Vulkan `f8_e4m3` RPC lane (3 devices, NOT comparable to clean 2-GPU)

| Level | Date (2026) | commit | Layout | Prompt tps | Decode tps | Aggregate | Acceptance |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| L1 | 08-31 | `291dc69d7` | `RPC0,Vulkan0,Vulkan1 ts=27,37,36` | 1048.78 | 30.51 | 10.91 | 74/106 |
| L2 | 08-31 | `291dc69d7` | `RPC0,Vulkan0,Vulkan1 ts=27,37,36` | 1023.87 | 29.43 | 6.63 | 156/198 |
| L3 | 08-31 | `291dc69d7` | `RPC0,Vulkan0,Vulkan1 ts=27,37,36` | 920.00 | 20.79 | 3.11 | 133/243 |
| L4 | 08-31 | `291dc69d7` | `RPC0,Vulkan0,Vulkan1 ts=27,37,36` | 818.08 | 24.86 | 2.04 | 154/201 |

Source: `vk-qwen3-8-27b-ud-q4-k-m-l1234-b8192-u1024-f8_e4m3-mtp-n2--20260831T181732303050Z-d50ad06dda3449cca8aed82a8c71082b`.
The RPC lane row is recorded for completeness only; it is a different
topology (remote head + two local GPUs) and must not be mixed into the
two-GPU lane comparisons.

## Vulkan q8_0 controls (same UD model and b/ub, for delta computation)

| Level | Date (2026) | commit | Layout | Prompt tps | Decode tps | Aggregate | Acceptance |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| L2 | 09-01 | `a9dc4aa3c` | `Vulkan0,Vulkan1 ts=33,34,33` | 1567.23 | 38.28 | 7.06 | 97/142 |
| L3 | 09-01 | `a9dc4aa3c` | `Vulkan0,Vulkan1 ts=33,34,33` | 1366.39 | 36.30 | 4.73 | 147/215 |

Source: `vk-qwen3-8-27b-ud-q4-k-m-l23-b8192-u1024-q8_0-mtp-n2--20260901T093719267024Z-f32d83b343f64772ab41b090e88a6fa8`.
Additional L2 q8 runs on 09-01 (`ts=51,49`, `ts=37,60,3`) exist under
`vk-qwen3-8-27b-ud-q4-k-m-l2-b8192-u1024-q8_0-mtp-n2--20260901T*`; only the
clean L2/L3 pair is tabled here.

## ROCm `f8_e4m3` (Windows, fragmentary)

| Level | Date (2026) | commit | Layout | Prompt tps | Decode tps | Aggregate | Acceptance |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| L0 | 08-30 | `75f7e87dc` | `ROCm0,ROCm1 ts=5,5` | 1527.76 | 25.76 | 12.55 | - |
| L1 | 08-29 | `75f7e87dc` | `ROCm0,ROCm1 ub=128` | 1036.15 | 41.90 | 11.99 | - |
| L1 | 08-29 | `75f7e87dc` | `ROCm0,ROCm1` | 1673.02 | 39.50 | 16.07 | - |
| L3 | 08-29 | `75f7e87dc` | `ts=0.7,0.4` | 1145.95 | 33.97 | 4.02 | - |
| L3 | 08-29 | `75f7e87dc` | `ts=` (single row) | 349.01 | 33.51 | 1.33 | - |

Sources:
- L0/L1: `rocm-qwen3-8-27b-ud-q4-k-m` (L0), `rocm-qwen3-8-27b-ud-q4-k-m-b8192-u1024-f8_e4m3-mtp` (L1),
  `rocm-qwen3-8-27b-ud-q4-k-m-b8192-u128-f8_e4m3-mtp` (L1 ub128).
- L3 n2: `rocm-qwen3-8-27b-ud-q4-k-m-b8192-u1024-f8_e4m3-mtp-n2`.
- L3 n3: `rocm-qwen3-8-27b-ud-q4-k-m-b8192-u1024-f8_e4m3-mtp-n3` (looks like a
  regressed/odd run: prompt 349.01, mark as suspect, do not use).
- Base reference: `rocm-qwen3-8-27b-ud-q4-k-m-b8192-u1024-f16-mtp-n2` (L3 f16,
  control for the same family).

## Gaps to fill before producing a complete Windows L1-L3 table

- Clean (non-RPC) Vulkan L3 `f8_e4m3` with the same `b8192/ub1024/MTP n2`
  contract exists (08-29, `ts=4,6`) but is on an older commit than the
  L1/L2 clean pair (`291dc69d7` vs `75f7e87dc`); no same-commit clean L1-L3
  set.
- ROCm has no clean `ROCm1,ROCm0 -sm layer -ts 1,1` L2 or clean q8_0
  L1-L3 controls; the Windows ROCm rows above use `ROCm0,ROCm1` and
  tensor-split (`ts=0.7,0.4`), which is not the production layer-split
  contract.
- The newer q8_0 L2 runs on 09-01 use `ts=51,49`/three-device orders (D137
  RPC work), not the production `-sm layer -ts 1,1` contract.

## Comparison note

- The full-production Windows reference remains
  `PERFORMANCE.md`/`README.md` `Q4_K_M ROCm q8 vs native FP8` table
  (2026-08-14, Qwen3.8-27B-Q4_K_M non-UD, `b512/ub512`): q8 center
  `1647.17/21.53`, f8 `1713.67/22.20`, MTP q8 `1625.29/35.14/70.7%`, MTP f8
  `1716.79/39.96/78.2%`.
- The rows above are newer but on the UD model and mostly not a matched
  pair; treat them as the newest record of what was measured, not as a
  promotion-ready table.
