# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-14 16:38:38 | cold-first | rocm | r001-mtp-prod-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.5225 | 1756.1900 | 22.7200 | 0 |
| 2026-08-14 16:37:18 | cold-first | rocm | r001-mtp-prod-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.9553 | 1683.5000 | 39.9800 | 0 |
| 2026-08-14 13:57:45 | cold-first | rocm | r001-h77-ab-cand-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.2516 | 1682.7600 | 21.1400 | 0 |
| 2026-08-14 13:56:29 | cold-first | - | r001-h77-ab-ctrl-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.5350 | 1774.3000 | 22.2700 | 0 |
| 2026-08-14 13:54:52 | cold-first | rocm | r001-h77-ab-cand-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.4446 | 1747.9100 | 21.8400 | 0 |
| 2026-08-14 13:53:12 | cold-first | - | r001-h77-ab-ctrl-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.5737 | 1787.1700 | 22.3900 | 0 |
| 2026-08-14 13:16:45 | cold-first | - | r001-h79-ab-cand-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.2784 | 1689.2900 | 21.3700 | 0 |
| 2026-08-14 13:15:07 | cold-first | - | r001-h79-ab-ctrl-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.3601 | 1717.8200 | 21.5900 | 0 |
| 2026-08-14 13:13:13 | cold-first | - | r001-h79-ab-cand-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.3815 | 1718.4000 | 21.8200 | 0 |
| 2026-08-14 13:11:17 | cold-first | - | r001-h79-ab-ctrl-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.2573 | 1672.9000 | 21.8000 | 0 |
| 2026-08-14 12:08:43 | cold-first | rocm | r001-p2-49k-f8-mtp2-baseline-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.5925 | 1575.9400 | 38.5500 | 0 |
| 2026-08-14 12:05:28 | cold-first | rocm | r001-p2-49k-f8-specnone-baseline-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.1726 | 1653.6600 | 21.1200 | 0 |
| 2026-08-14 08:08:23 | cold-first | rocm | d102-g0-rocm49k-fp8-phase-census-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.2551 | 1725.6700 | 19.7000 | 0 |
| 2026-08-14 08:01:41 | cold-first | rocm | d102-g0-rocm49k-fp8-phase-census-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-14 07:53:48 | cold-first | rocm | d102-g0-rocm49k-fp8-phase-census-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-14 06:59:58 | cold-first | rocm | d101-g2-rocm49k-shared-scales-control-close-r8 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6558 | 1809.2400 | 22.8200 | 0 |
| 2026-08-14 06:59:04 | cold-first | rocm | d101-g2-rocm49k-shared-scales-candidate-r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6543 | 1806.4500 | 22.9100 | 0 |
| 2026-08-14 06:58:10 | cold-first | rocm | d101-g2-rocm49k-shared-scales-control-open-r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6639 | 1809.0500 | 22.9100 | 0 |
| 2026-08-14 06:57:03 | cold-first | rocm | d101-g2-rocm49k-shared-scales-route-r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.1167 | 1800.9200 | 37.0900 | 0 |
| 2026-08-14 06:46:23 | cold-first | rocm | d101-g0-rocm49k-fp8-full-close-r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6357 | 1798.2400 | 22.9400 | 0 |
| 2026-08-14 06:45:27 | cold-first | rocm | d101-g0-rocm49k-fp8-reference-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.0092 | 1694.7800 | 17.3100 | 0 |
| 2026-08-14 06:44:30 | cold-first | rocm | d101-g0-rocm49k-fp8-kq-only-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.2934 | 1745.3100 | 19.6000 | 0 |
| 2026-08-14 06:43:31 | cold-first | rocm | d101-g0-rocm49k-fp8-full-open-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6059 | 1791.3000 | 22.6900 | 0 |
| 2026-08-13 22:31:22 | cold-first | rocm | d100-g3-rocm49k-fp8-pb4-auto-close-r11 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6697 | 1811.6700 | 22.9400 | 0 |
| 2026-08-13 22:30:26 | cold-first | rocm | d100-g3-rocm49k-fp8-pb4-forced-r10 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6807 | 1814.2800 | 23.0100 | 0 |
| 2026-08-13 22:29:32 | cold-first | rocm | d100-g3-rocm49k-fp8-pb4-auto-open-r9 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.7040 | 1825.2000 | 22.9400 | 0 |
| 2026-08-13 22:25:24 | cold-first | rocm | d100-g3-rocm49k-fp8-launch-geometry-r8 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.1178 | 1817.6500 | 37.8200 | 0 |
| 2026-08-13 22:20:52 | cold-first | rocm | d100-g3-rocm49k-fp8-launch-geometry-r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0590 | 1817.9900 | - | 0 |
| 2026-08-13 22:19:58 | cold-first | rocm | d100-g3-rocm49k-fp8-pb-auto-close-r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6800 | 1818.7900 | 22.8300 | 0 |
| 2026-08-13 22:19:00 | cold-first | rocm | d100-g3-rocm49k-fp8-pb32-r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6679 | 1819.4800 | 22.5900 | 0 |
| 2026-08-13 22:18:06 | cold-first | rocm | d100-g3-rocm49k-fp8-pb-auto-open-r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6886 | 1818.2700 | 22.9700 | 0 |
| 2026-08-13 22:08:43 | cold-first | rocm | d100-g3-rocm49k-fp8-w8-close-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6575 | 1807.2200 | 22.9400 | 0 |
| 2026-08-13 22:07:38 | cold-first | rocm | d100-g3-rocm49k-fp8-w16-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6358 | 1806.8000 | 22.5400 | 0 |
| 2026-08-13 22:06:43 | cold-first | rocm | d100-g3-rocm49k-fp8-w8-open-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.6757 | 1811.9800 | 22.9900 | 0 |
| 2026-08-13 21:59:47 | cold-first | rocm | d100-g2-rocm98k-fp8-none-devicetrace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 1.6177 | 1604.9600 | 19.9900 | 0 |
| 2026-08-13 21:52:20 | cold-first | rocm | d100-g2-rocm49k-fp8-none-devicetrace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.9972 | 1803.2700 | 22.5200 | 0 |
| 2026-08-13 21:50:34 | cold-first | rocm | d100-g2-rocm49k-fp8-mtp2-devicetrace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 10.7041 | 1724.6400 | 42.2700 | 0 |
| 2026-08-13 21:27:17 | cold-first | rocm | d100-g1-rocm49k-fp8-mtp2-graphdiff-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 0.4422 | 1721.5500 | 37.7800 | 0 |
| 2026-08-13 21:25:47 | cold-first | rocm | d100-g1-rocm49k-fp8-mtp2-graphtrace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 10.5394 | 1743.5300 | 38.5600 | 0 |
| 2026-08-13 21:24:32 | cold-first | rocm | d100-g0-rocm49k-fp8-mtp2-clean-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 10.5700 | 1746.2700 | 38.8900 | 0 |
| 2026-08-13 21:23:30 | cold-first | rocm | d100-g0-rocm49k-fp8-none-clean-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 9.0486 | 1814.3500 | 22.6600 | 0 |
| 2026-08-13 16:42:50 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf16-close-r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4690 | 1569.5000 | 33.9400 | 0 |
| 2026-08-13 16:41:32 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf8-fixed-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4647 | 1566.5200 | 33.9000 | 0 |
| 2026-08-13 16:34:20 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf8-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4563 | 1565.6800 | 33.7200 | 0 |
| 2026-08-13 16:33:04 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf16-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.7928 | 1622.6300 | 40.9100 | 0 |
| 2026-08-13 12:51:55 | cold-first | rocm | d099-rocm49k-q8-close-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 8.7177 | 1739.2200 | 21.9300 | 0 |
| 2026-08-13 12:50:57 | cold-first | rocm | d099-rocm49k-f8-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 9.0811 | 1809.6200 | 22.9600 | 0 |
| 2026-08-13 12:49:59 | cold-first | rocm | d099-rocm49k-q8-open-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 8.7257 | 1741.8000 | 21.9500 | 0 |
| 2026-08-13 11:37:38 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113545 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 3.0656 | 1548.33 | 31.26 | 0 |
| 2026-08-13 11:34:55 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113408 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 16.8156 | 1619.83 | 47.99 | 0 |
| 2026-08-13 11:33:40 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113259 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 16.4508 | 1559.14 | 48.37 | 0 |
| 2026-08-13 11:32:41 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113030 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 6.5217 | 1818.18 | 43.21 | 0 |
| 2026-08-13 11:25:56 | cold-first | rocm | d098-g5-default-smoke-12k-s12r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 2.7627 | 1688.8500 | 25.9900 | 0 |
| 2026-08-13 11:19:35 | cold-first | rocm | d098-g5-98k-mtp2-q8-close-s11r3 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.9036 | 1441.9300 | 33.5700 | 0 |
| 2026-08-13 11:18:22 | cold-first | rocm | d098-g5-98k-mtp2-f8-s11r2 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 2.9919 | 1482.8600 | 35.8200 | 0 |
| 2026-08-13 11:17:09 | cold-first | rocm | d098-g5-98k-mtp2-q8-open-s11r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.9084 | 1445.1400 | 33.4300 | 0 |
| 2026-08-13 11:15:12 | cold-first | rocm | d098-g5-49k-mtp2-q8-close-s10r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9744 | 1678.9000 | 38.2100 | 0 |
| 2026-08-13 11:14:16 | cold-first | rocm | d098-g5-49k-mtp2-f8-s10r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 6.2860 | 1751.5800 | 41.9800 | 0 |
| 2026-08-13 11:13:25 | cold-first | rocm | d098-g5-49k-mtp2-q8-open-s10r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9256 | 1665.1600 | 37.7100 | 0 |
| 2026-08-13 11:09:10 | cold-first | rocm | d098-g4-final-49k-q8-close-s9r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.6167 | 1687.1600 | 21.8100 | 0 |
| 2026-08-13 11:08:13 | cold-first | rocm | d098-g4-final-49k-f8-s9r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0508 | 1769.3700 | 22.9700 | 0 |
| 2026-08-13 11:07:16 | cold-first | rocm | d098-g4-final-49k-q8-open-s9r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.6196 | 1686.3400 | 21.8400 | 0 |
| 2026-08-13 11:04:44 | cold-first | rocm | d098-g4-w8-49k-q8-s8r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.5715 | 1670.8700 | 21.8200 | 0 |
| 2026-08-13 11:01:58 | cold-first | rocm | d098-g4-w8-49k-f16-s8r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 8.8600 | 1691.4300 | 23.3600 | 0 |
| 2026-08-13 11:01:01 | cold-first | rocm | d098-g4-w8-49k-w8-s8r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0191 | 1761.8500 | 22.9200 | 0 |
| 2026-08-13 10:59:58 | cold-first | rocm | d098-g4-w8-49k-w8-s8r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0036 | 1760.5300 | 22.8400 | 0 |
| 2026-08-13 10:59:03 | cold-first | rocm | d098-g4-w8-49k-w4-s8r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.6995 | 1706.4600 | 21.9600 | 0 |
| 2026-08-13 10:57:43 | cold-first | rocm | d098-g4-w8-12k-w8-s8r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.8716 | 1843.6850 | 24.7100 | 0 |
| 2026-08-13 10:56:44 | cold-first | rocm | d098-g4-w8-12k-w4-s8r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.5858 | 1822.2300 | 24.2200 | 0 |
| 2026-08-13 10:47:33 | cold-first | rocm | d098-g4-spillfree-49k-f16-s7r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 8.8413 | 1688.7400 | 23.3300 | 0 |
| 2026-08-13 10:46:34 | cold-first | rocm | d098-g4-spillfree-49k-full-s7r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.6802 | 1699.1900 | 21.9800 | 0 |
| 2026-08-13 10:45:20 | cold-first | rocm | d098-g4-spillfree-49k-full-s7r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.5964 | 1673.2800 | 21.9700 | 0 |
| 2026-08-13 10:44:17 | cold-first | rocm | d098-g4-spillfree-49k-kq-s7r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.3371 | 1720.7200 | 19.6400 | 0 |
| 2026-08-13 10:43:16 | cold-first | rocm | d098-g4-spillfree-12k-full-s7r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.5977 | 1818.5800 | 24.2800 | 0 |
| 2026-08-13 10:42:18 | cold-first | rocm | d098-g4-spillfree-12k-kq-s7r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.2754 | 1817.0450 | 23.6250 | 0 |
| 2026-08-13 09:38:37 | cold-first | rocm | d098-g4-pcvt-49k-full-s6r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 7.4343 | 1468.2400 | 18.5500 | 0 |
| 2026-08-13 09:37:32 | cold-first | rocm | d098-g4-pcvt-49k-kq-s6r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.4408 | 1739.8300 | 19.8400 | 0 |
| 2026-08-13 09:36:35 | cold-first | rocm | d098-g4-pcvt-49k-f16-s6r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 8.8691 | 1694.1600 | 23.3400 | 0 |
| 2026-08-13 09:35:26 | cold-first | rocm | d098-g4-pcvt-12k-full-s6r3 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 15.7483 | 1725.8650 | 23.0000 | 0 |
| 2026-08-13 09:34:26 | cold-first | rocm | d098-g4-pcvt-12k-kq-s6r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.4083 | 1831.4850 | 23.8000 | 0 |
