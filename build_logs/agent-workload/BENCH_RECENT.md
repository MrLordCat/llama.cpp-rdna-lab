# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-07-15 09:37:00 | cold-first | vulkan | e315-vulkan-dual-long30k-none-target-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.6373 | 1570.1100 | 33.6500 | 0 |
| 2026-07-15 09:32:43 | cold-first | rocm | e315-rocm-dual-long30k-mtp3-matched-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 10.3731 | 1739.5000 | 33.5700 | 0 |
| 2026-07-15 09:31:46 | cold-first | vulkan | e315-vulkan-dual-long30k-mtp3-matched-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 10.2574 | 1516.3200 | 47.6100 | 0 |
| 2026-07-15 09:25:18 | cold-first | rocm | e314-rocm-dual-rms-rope-control-node-trace | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 3.7495 | 335.4500 | 58.6800 | 0 |
| 2026-07-15 09:24:28 | cold-first | rocm | e314-rocm-dual-rms-rope-control-short128-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 29.7066 | 585.8300 | 31.8800 | 0 |
| 2026-07-15 09:23:49 | cold-first | rocm | e314-rocm-dual-rms-rope-fused-short128-r1b | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 27.8552 | 570.8500 | 29.8100 | 0 |
| 2026-07-14 23:13:48 | cold-first | rocm | e314-rocm-dual-rms-rope-control-short128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 30.3915 | 562.0600 | 32.7700 | 0 |
| 2026-07-14 23:13:05 | cold-first | rocm | e314-rocm-dual-imrope-correctness | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 27.0407 | 657.1800 | 34.7600 | 0 |
| 2026-07-14 23:11:25 | cold-first | rocm | e314-rocm-dual-rms-rope-route-probe2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 3.7679 | 674.8900 | - | 0 |
| 2026-07-14 23:10:00 | cold-first | rocm | e314-rocm-dual-rms-rope-route-probe | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 3.7908 | 671.5200 | - | 0 |
| 2026-07-14 23:08:28 | cold-first | rocm | e314-rocm-dual-rms-rope-correctness | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 27.1003 | 672.3400 | 34.7200 | 0 |
| 2026-07-14 22:59:40 | cold-first | rocm | e313-rocm-dual-async-stage-final-smoke64 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 10.8357 | 1830.7900 | 26.8700 | 0 |
| 2026-07-14 22:56:10 | cold-first | rocm | e313-rocm-dual-sync-stage-mtp3-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8967 | 1740.2400 | 29.4900 | 0 |
| 2026-07-14 22:55:27 | cold-first | rocm | e313-rocm-dual-async-stage-mtp3-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9873 | 1772.6700 | 29.4800 | 0 |
| 2026-07-14 22:54:07 | cold-first | rocm | e313-rocm-dual-sync-stage-long30k-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.8779 | 1766.9500 | 27.3500 | 0 |
| 2026-07-14 22:53:25 | cold-first | rocm | e313-rocm-dual-async-stage-long30k-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.9609 | 1814.5400 | 26.5100 | 0 |
| 2026-07-14 22:52:34 | cold-first | rocm | e313-rocm-dual-sync-stage-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.8251 | 1767.5000 | 26.1900 | 0 |
| 2026-07-14 22:51:49 | cold-first | rocm | e313-rocm-dual-async-stage-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.9720 | 1813.3700 | 26.7900 | 0 |
| 2026-07-14 22:50:58 | cold-first | rocm | e313-rocm-dual-async-stage-short128-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 15.5214 | 1808.8000 | 27.3400 | 0 |
| 2026-07-14 22:50:13 | cold-first | rocm | e313-rocm-dual-sync-stage-short128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 15.4431 | 1652.1500 | 29.1600 | 0 |
| 2026-07-14 22:49:43 | cold-first | rocm | e313-rocm-dual-async-stage-short128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 15.6084 | 1821.3300 | 27.4600 | 0 |
| 2026-07-14 22:39:28 | cold-first | rocm | e312-rocm-dual-output-last-long30k-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.8980 | 1767.6600 | 27.7200 | 0 |
| 2026-07-14 22:38:41 | cold-first | rocm | e312-rocm-dual-output-last-ts17-16-long30k-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.7805 | 1733.2200 | 27.1600 | 0 |
| 2026-07-14 22:37:30 | cold-first | rocm | e312-rocm-dual-output-last-ts9-8-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.6868 | 1698.0800 | 27.0800 | 0 |
| 2026-07-14 22:36:22 | cold-first | rocm | e312-rocm-dual-output-last-order01-ts16-17-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.9380 | 1848.4500 | 24.5400 | 0 |
| 2026-07-14 22:35:30 | cold-first | rocm | e312-rocm-dual-output-last-order01-long30k-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.8799 | 1811.2500 | 25.1200 | 0 |
