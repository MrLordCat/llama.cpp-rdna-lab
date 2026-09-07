# Linux FP8 Results (ROCm 10, P2P ON/OFF, Vulkan; 2026-09-07)

Full Linux `f8_e4m3` KV sweep on the same bench2 contract as the Windows
archive, so the two reference files can be compared directly. Every row is
exact `metrics.csv` output; artifacts live in
`/tmp/bench-linux-fp8-ud/linux-lud-{q8,f8}-{p2pon,p2poff,vk}-l123-mtp--*`.

## Evaluation contract

- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf` (same UD build as the Windows rows).
- Binary: `build-rocm-linux/bin/llama-server` (ROCm 10.0.0, TheRock, HIP
  7.15.26333) and `build-vulkan-linux/bin/llama-server`.
- `-dev ROCm1,ROCm0 -sm layer -ts 1,1` (ROCm) / `-dev Vulkan1,Vulkan0 -sm
  layer -ts 1,1` (Vulkan), `-c 98304 -b 8192 -ub 1024 -ngl 999
  --flash-attn on`, MTP n2 (`--spec-type draft-mtp --spec-draft-n-max 2`),
  `-fit off --cache-ram 0 --ctx-checkpoints 0`, `--seed 42 --no-warmup`.
- Levels (single runs, no reuse, cold):
  - L1 = `ctx=16384`, prompt 7901 / decode 128;
  - L2 = `ctx=49152`, prompt 30609 / decode 256;
  - L3 = `ctx=98304`, prompt 64287 / decode 256.
- P2P-OFF = env `GGML_CUDA_NO_PEER_COPY_RUNTIME=1` (forces host-staged
  cross-device copies; P2P is the default ON path on Linux, unlike Windows
  where RDNA4 peer copies are opt-in).

## ROCm P2P ON (default) - MTP n2

| Level | KV | Prompt tps | Decode tps | Aggregate | Acceptance | Run name |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| L1 | q8_0 | 1758.62 | 45.36 | 17.50 | 79/94 | `linux-lud-q8-p2pon-l123-mtp` |
| L1 | f8_e4m3 | 1775.12 | 39.48 | 16.64 | 71/112 | `linux-lud-f8-p2pon-l123-mtp` |
| L2 | q8_0 | 1585.92 | 38.09 | 9.84 | 154/201 | `linux-lud-q8-p2pon-l123-mtp` |
| L2 | f8_e4m3 | 1644.90 | 39.06 | 10.17 | 153/204 | `linux-lud-f8-p2pon-l123-mtp` |
| L3 | q8_0 | 1268.75 | 30.66 | 4.34 | 151/207 | `linux-lud-q8-p2pon-l123-mtp` |
| L3 | f8_e4m3 | 1315.52 | 30.55 | 4.47 | 147/216 | `linux-lud-f8-p2pon-l123-mtp` |

## ROCm P2P OFF (host-staged) - MTP n2

| Level | KV | Prompt tps | Decode tps | Aggregate | Acceptance | Run name |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| L1 | q8_0 | 1682.71 | 32.66 | 14.86 | 75/103 | `linux-lud-q8-p2poff-l123-mtp` |
| L1 | f8_e4m3 | 1714.14 | 33.64 | 15.21 | 73/108 | `linux-lud-f8-p2poff-l123-mtp` |
| L2 | q8_0 | 1528.11 | 28.89 | 8.86 | 154/201 | `linux-lud-q8-p2poff-l123-mtp` |
| L2 | f8_e4m3 | 1600.78 | 30.02 | 9.26 | 151/208 | `linux-lud-f8-p2poff-l123-mtp` |
| L3 | q8_0 | 1233.36 | 23.94 | 4.08 | 155/197 | `linux-lud-q8-p2poff-l123-mtp` |
| L3 | f8_e4m3 | 1280.18 | 25.58 | 4.25 | 158/193 | `linux-lud-f8-p2poff-l123-mtp` |

## Vulkan - MTP n2

| Level | KV | Prompt tps | Decode tps | Aggregate | Acceptance | Run name |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| L1 | q8_0 | 1496.19 | 33.82 | 14.12 | 64/123 | `linux-lud-q8-vk-l123-mtp` |
| L1 | f8_e4m3 | 1499.12 | 41.61 | 15.34 | 76/101 | `linux-lud-f8-vk-l123-mtp` |
| L2 | q8_0 | 1567.87 | 42.26 | 10.01 | 160/189 | `linux-lud-q8-vk-l123-mtp` |
| L2 | f8_e4m3 | 1557.66 | 40.14 | 9.84 | 155/198 | `linux-lud-f8-vk-l123-mtp` |
| L3 | q8_0 | 1380.52 | 36.48 | 4.78 | 153/204 | `linux-lud-q8-vk-l123-mtp` |
| L3 | f8_e4m3 | 1387.93 | 36.21 | 4.80 | 152/203 | `linux-lud-f8-vk-l123-mtp` |

## Quick deltas (f8 vs q8, same lane and level)

| Lane / level | Prompt Δ | Decode Δ | Aggregate Δ | Acceptance Δ |
| --- | ---: | ---: | ---: | --- |
| ROCm P2P ON L1 | +0.94% | -12.96% | -4.9% | 71/112 vs 79/94 |
| ROCm P2P ON L2 | +3.72% | +2.55% | +3.4% | 153/204 vs 154/201 |
| ROCm P2P ON L3 | +3.69% | -0.36% | +3.0% | 147/216 vs 151/207 |
| ROCm P2P OFF L1 | +1.87% | +3.00% | +2.4% | 73/108 vs 75/103 |
| ROCm P2P OFF L2 | +4.76% | +3.91% | +4.5% | 151/208 vs 154/201 |
| ROCm P2P OFF L3 | +3.80% | +6.85% | +4.2% | 158/193 vs 155/197 |
| Vulkan L1 | +0.20% | +23.04% | +8.6% | 76/101 vs 64/123 |
| Vulkan L2 | -0.65% | -5.01% | -1.7% | 155/198 vs 160/189 |
| Vulkan L3 | +0.54% | -0.74% | +0.4% | 152/203 vs 153/204 |

## Notes

- ROCm FP8 prefill is consistently positive everywhere (+0.9%..+4.8%);
  decode positive on P2P-OFF and L2 P2P-ON, negative on P2P-ON L1/L3.
- P2P ON vs OFF: P2P-ON is much faster for decode on every level and KV
  (e.g. L3 q8 30.66 vs 24.94 tok/s, f8 30.55 vs 25.58 tok/s); prefill is
  also better with P2P-ON. This confirms peer-copy is the correct default
  on Linux (Windows keeps host-staged for RDNA4 safety).
- Vulkan FP8: mixed - decode +23% on L1 (single repeat of 76/101 acceptance
  on a 101-draft sample, wider variance), but regresses L2 and is flat at
  L3. Vulkan acceptance is broadly lower than ROCm in MTP n2.
- Acceptance columns are single-run ratios; n≈94-216 drafts, so differences
  of a few percent are within shot-level variance. Confirm any large claim
  with r2/r3 before promotion.
