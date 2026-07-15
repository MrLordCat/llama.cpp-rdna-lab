# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-07-15 18:14:50 | autotune | rocm | gui-autotune-Ternary-Bonsai-27B-PQ2_0-20260715-181422 | Ternary-Bonsai-27B-PQ2_0.gguf | 49152 | sweep/sweep | sweep/sweep | none | 6.1035 | 1872.47 | 33.97 | 0 |
| 2026-07-15 18:13:37 | cold-first | rocm | e330-rocm-dual-q3-12k-mtp-n4-256-r3-refresh | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 26.1161 | 1794.1733 | 41.3867 | 0 |
| 2026-07-15 18:12:45 | cold-first | rocm | e330-rocm-dual-q3-12k-none256-r3-refresh | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 20.0662 | 1850.1300 | 27.6733 | 0 |
| 2026-07-15 18:02:24 | cold-first | rocm | e328-bonsai-pq2-rocm1-mmvq-resources | Ternary-Bonsai-27B-PQ2_0.gguf | 4096 | 512/256 | q8_0/q8_0 | none | 24.2645 | 996.2800 | 52.8500 | 0 |
| 2026-07-15 18:00:19 | cold-first | rocm | e327-bonsai-pq2-rocm10-long30k-mmqx128-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 6.6032 | 1855.9000 | 37.6600 | 0 |
| 2026-07-15 17:59:33 | cold-first | rocm | e327-bonsai-pq2-rocm10-short-mmqx128-r3 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 28.8201 | 1996.9700 | 45.3233 | 0 |
| 2026-07-15 17:58:13 | cold-first | rocm | e326-bonsai-pq2-rocm10-mmqx128-short-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.4228 | 1984.0000 | 43.4800 | 0 |
| 2026-07-15 17:57:54 | cold-first | rocm | e326-bonsai-pq2-rocm10-mmqx112-short-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.1397 | 1842.5900 | 43.5800 | 0 |
| 2026-07-15 17:57:35 | cold-first | rocm | e326-bonsai-pq2-rocm10-mmqx96-short-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.1692 | 1856.7800 | 43.6100 | 0 |
| 2026-07-15 17:57:17 | cold-first | rocm | e326-bonsai-pq2-rocm10-mmqx80-short-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.1647 | 1854.5900 | 43.7800 | 0 |
| 2026-07-15 17:56:58 | cold-first | rocm | e326-bonsai-pq2-rocm10-mmqx64-short-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.0386 | 1796.9300 | 43.4400 | 0 |
| 2026-07-15 17:51:25 | cold-first | rocm | e325-bonsai-pq2-rocm1-mmq-resources-n1024 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 1.6384 | 1300.0900 | 79.9400 | 0 |
| 2026-07-15 17:49:44 | cold-first | rocm | e325-bonsai-pq2-rocm1-mmq-resources | Ternary-Bonsai-27B-PQ2_0.gguf | 4096 | 2048/1024 | q8_0/q8_0 | none | 8.5106 | 1067.9000 | 89.0400 | 0 |
| 2026-07-15 17:49:18 | cold-first | rocm | e324-bonsai-pq2-rocm10-short-force-mmq-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 29.0899 | 2045.3500 | 45.3300 | 0 |
| 2026-07-15 17:48:35 | cold-first | rocm | e324-bonsai-pq2-rocm1-short-force-mmq-r1 | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 25.3438 | 1281.1100 | 50.3600 | 0 |
| 2026-07-15 17:45:44 | cold-first | rocm | e323-bonsai-pq2-rocm1-route-trace | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.6083 | 963.9300 | 54.2600 | 0 |
| 2026-07-15 17:44:17 | cold-first | rocm | e322-bonsai-pq2-rocm10-long30k-none-r1-baseline | Ternary-Bonsai-27B-PQ2_0.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 6.3794 | 1779.5000 | 37.7200 | 0 |
| 2026-07-15 17:43:34 | cold-first | rocm | e322-bonsai-pq2-rocm1-long30k-none-r1-baseline | Ternary-Bonsai-27B-PQ2_0.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.0768 | 1046.0700 | 41.5500 | 0 |
| 2026-07-15 17:42:37 | cold-first | rocm | e322-bonsai-pq2-rocm10-short-none-r3-baseline | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 28.0593 | 1858.6900 | 45.3967 | 0 |
| 2026-07-15 17:41:44 | cold-first | rocm | e322-bonsai-pq2-rocm1-short-none-r3-baseline | Ternary-Bonsai-27B-PQ2_0.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 24.3919 | 1189.1967 | 50.3033 | 0 |
| 2026-07-15 17:25:57 | cold-first | rocm | pq2-run1 | Ternary-Bonsai-27B-PQ2_0.gguf | 4096 | 512/256 | q4_0/q4_0 | none | 16.6424 | 324.4200 | 53.5000 | 0 |
| 2026-07-15 17:22:07 | cold-first | rocm | pq2-single-smoke | Ternary-Bonsai-27B-PQ2_0.gguf | 4096 | 512/256 | q4_0/q4_0 | none | 16.1894 | 313.4000 | 52.8000 | 0 |
| 2026-07-15 13:03:04 | cold-first | - | upstream-f955e394b-rocm-dual-long30k-mtp3-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.2726 | 1102.9200 | 41.5700 | 0 |
| 2026-07-15 12:59:08 | cold-first | - | upstream-f955e394b-rocm-dual-long30k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.4423 | 1285.4200 | 22.3000 | 0 |
| 2026-07-15 12:56:41 | cold-first | - | upstream-f955e394b-vulkan-dual-long30k-mtp3-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 3.0751 | 861.4800 | 17.7700 | 0 |
| 2026-07-15 12:55:20 | cold-first | - | upstream-f955e394b-vulkan-dual-long30k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 3.3850 | 930.1100 | 21.5800 | 0 |
| 2026-07-15 12:09:04 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260715-120826 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.1664 | 1489.53 | 40.42 | 0 |
| 2026-07-15 12:08:11 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260715-120738 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.9169 | 1747.52 | 39.90 | 0 |
| 2026-07-15 11:19:09 | cold-first | rocm | e321-rocm-mtp-defaults-zero-rollback-smoke-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 2.2544 | 1761.8600 | 29.0000 | 0 |
| 2026-07-15 11:16:06 | cold-first | rocm | e321-rocm-dual-long49k-mtp3-sparse-default4k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.3573 | 1597.2300 | 35.9200 | 0 |
| 2026-07-15 11:13:07 | cold-first | rocm | e320-rocm-dual-long30k-mtp3-sparse32kx4k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.3087 | 1721.9700 | 42.0200 | 0 |
| 2026-07-15 11:12:04 | cold-first | rocm | e320-rocm-dual-long49k-mtp3-sparse32kx4k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.3480 | 1593.5500 | 35.9100 | 0 |
| 2026-07-15 11:10:39 | cold-first | rocm | e320-rocm-dual-long49k-mtp3-sparse32kx1k-window4k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.3369 | 1603.0900 | 33.8000 | 0 |
| 2026-07-15 11:09:16 | cold-first | rocm | e320-rocm-dual-long49k-mtp3-sparse16kx1k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.2560 | 1566.9500 | 34.0000 | 0 |
| 2026-07-15 11:08:06 | cold-first | rocm | e320-rocm-dual-long49k-mtp3-full-kvonly-history-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.1156 | 1488.7700 | 37.5400 | 0 |
| 2026-07-15 11:07:08 | cold-first | rocm | e320-rocm-dual-long49k-mtp3-sparse1k-window1024-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.3275 | 1600.1400 | 33.6800 | 0 |
| 2026-07-15 11:06:15 | cold-first | rocm | e320-rocm-dual-long49k-mtp3-sparse1k-window512-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.3550 | 1606.4100 | 34.3400 | 0 |
| 2026-07-15 11:05:12 | cold-first | rocm | e319-rocm-dual-long49k-none-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | none | 4.3037 | 1670.2700 | 25.3400 | 0 |
| 2026-07-15 11:04:16 | cold-first | rocm | e319-rocm-dual-long49k-mtp3-sparse-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.3409 | 1603.7300 | 33.8700 | 0 |
| 2026-07-15 11:03:17 | cold-first | vulkan | e319-vulkan-dual-long30k-mtp3-handoff-default-fix-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.6863 | 1508.0100 | 45.2000 | 0 |
| 2026-07-15 11:01:45 | cold-first | vulkan | e319-vulkan-dual-long12k-mtp3-hosthandoff-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 2.1337 | 1641.0300 | 34.4700 | 0 |
| 2026-07-15 11:00:42 | cold-first | vulkan | e319-vulkan-dual-long12k-mtp3-ubatchtrace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 1.3294 | 995.2000 | 34.4600 | 0 |
| 2026-07-15 10:59:44 | cold-first | vulkan | e319-vulkan-dual-long30k-none-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.6458 | 1556.8900 | 35.4500 | 0 |
| 2026-07-15 10:58:48 | cold-first | vulkan | e319-vulkan-dual-long30k-mtp3-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 3.4044 | 852.4000 | 45.0100 | 0 |
| 2026-07-15 10:57:36 | cold-first | rocm | e319-rocm-dual-long30k-mtp3-sparse-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.3430 | 1731.4500 | 41.9900 | 0 |
| 2026-07-15 10:53:34 | cold-first | vulkan | e318-vulkan-dual-long30k-mtp3-kvonly-prealloc-deferred32kx1k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 3.4182 | 842.1200 | 56.4200 | 0 |
| 2026-07-15 10:49:38 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-prealloc-deferred32kx1k-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.3571 | 1735.5900 | 42.0300 | 0 |
| 2026-07-15 10:47:46 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-prealloc-deferred32kx768-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1847 | 1729.4900 | 36.0800 | 0 |
| 2026-07-15 10:46:42 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-prealloc-deferred32kx512-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1523 | 1719.8200 | 35.9800 | 0 |
| 2026-07-15 10:45:48 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-prealloc-deferred32kx1k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.3475 | 1733.5900 | 41.8700 | 0 |
| 2026-07-15 10:43:52 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-deferred-compact32kx1k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.2317 | 1695.2900 | 42.0600 | 0 |
| 2026-07-15 10:43:08 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-deferred-compact32kx2k-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.2252 | 1697.1500 | 41.4600 | 0 |
| 2026-07-15 10:41:18 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-deferred-compact32kx2k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9433 | 1666.9500 | 34.1700 | 0 |
| 2026-07-15 10:38:51 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-deferred32kx2k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.2310 | 1701.5200 | 41.1400 | 0 |
| 2026-07-15 10:36:06 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-win256-interleaved-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1470 | 1758.7400 | 32.3500 | 0 |
| 2026-07-15 10:35:23 | cold-first | rocm | e318-rocm-dual-long30k-none-interleaved-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.9072 | 1787.9400 | 25.2100 | 0 |
| 2026-07-15 10:33:22 | cold-first | rocm | e318-rocm-dual-long30k-mtp3-kvonly-selective32kx2k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1564 | 1679.4800 | 40.9100 | 0 |
| 2026-07-15 10:29:36 | cold-first | rocm | e317-rocm-dual-long30k-mtp3-kvonly-anchor32kx2k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1710 | 1681.9400 | 41.1800 | 0 |
| 2026-07-15 10:28:40 | cold-first | rocm | e317-rocm-dual-long30k-mtp3-kvonly-anchor32kx4k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1815 | 1680.9000 | 41.7900 | 0 |
| 2026-07-15 10:27:34 | cold-first | rocm | e317-rocm-dual-long30k-mtp3-kvonly-anchor32kx8k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1721 | 1680.9900 | 41.3600 | 0 |
| 2026-07-15 10:25:22 | cold-first | rocm | e317-rocm-dual-long30k-mtp3-kvonly-sparse8192x1024-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8621 | 1580.0800 | 41.7200 | 0 |
| 2026-07-15 10:22:33 | cold-first | rocm | e317-rocm-dual-long30k-none-matched-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.9292 | 1780.1300 | 26.0100 | 0 |
| 2026-07-15 10:21:37 | cold-first | rocm | e317-rocm-dual-long30k-mtp3-kvonly-full-async-handoff-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8502 | 1581.7400 | 40.8900 | 0 |
| 2026-07-15 10:16:55 | cold-first | rocm | e316-rocm-dual-10k-none-ubtrace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 1.4949 | 1124.4500 | 30.8000 | 0 |
| 2026-07-15 10:14:40 | cold-first | rocm | e316-rocm-dual-10k-mtp3-kvonly-full-ubtrace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 1.4551 | 1097.8300 | 28.6000 | 0 |
| 2026-07-15 10:13:26 | cold-first | rocm | e316-rocm-dual-long30k-mtp3-kvonly-full-ctxub4096-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.2196 | 1396.1700 | 38.9100 | 0 |
| 2026-07-15 10:12:13 | cold-first | rocm | e316-rocm-dual-long30k-mtp3-kvonly-full-ctxub2048-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.6181 | 1524.5200 | 38.4000 | 0 |
| 2026-07-15 10:02:59 | cold-first | rocm | e316-rocm-dual-long30k-mtp3-kvonly-win8192-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9710 | 1685.8800 | 33.3100 | 0 |
| 2026-07-15 10:01:53 | cold-first | rocm | e316-rocm-dual-long30k-mtp3-kvonly-fullprefill-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8295 | 1574.2000 | 41.0300 | 0 |
| 2026-07-15 10:00:51 | cold-first | rocm | e316-rocm-dual-long30k-mtp3-kvonly-win256-correct-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1447 | 1758.1700 | 32.3300 | 0 |
| 2026-07-15 10:00:07 | cold-first | rocm | e316-rocm-kvonly-short-contract-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.1631 | 261.5200 | 33.3000 | 0 |
| 2026-07-15 09:56:15 | cold-first | rocm | e316-rocm-kvonly-null-buffer-stack | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-15 09:54:28 | cold-first | rocm | e316-rocm-kvonly-short-outputmark-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-15 09:52:48 | cold-first | rocm | e316-rocm-kvonly-checkpoint-diag-short | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-15 09:50:52 | cold-first | rocm | e316-rocm-dual-long30k-mtp3-kvonly-win256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-15 09:42:34 | cold-first | rocm | e315-rocm-dual-long30k-mtp3-token-trace-production-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.1460 | 1765.3400 | 31.8200 | 0 |
| 2026-07-15 09:41:53 | cold-first | vulkan | e315-vulkan-dual-long30k-mtp3-token-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8053 | 1550.8200 | 43.9800 | 0 |
| 2026-07-15 09:40:57 | cold-first | rocm | e315-rocm-dual-long30k-none-target-production-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.8474 | 1774.6000 | 24.8100 | 0 |
| 2026-07-15 09:39:37 | cold-first | rocm | e315-rocm-dual-long30k-mtp3-token-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.3068 | 1863.5000 | 13.3600 | 0 |
| 2026-07-15 09:37:40 | cold-first | rocm | e315-rocm-dual-long30k-none-target-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5599 | 1887.5600 | 25.4100 | 0 |
