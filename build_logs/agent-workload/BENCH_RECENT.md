# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-22 18:34:21 | cold-first | vulkan | rpc3080-16k-nocoop-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7098 | 533.5350 | 21.9000 | 0 |
| 2026-08-22 18:31:04 | cold-first | vulkan | lb16k-full-285814-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 1.1529 | 880.4550 | 25.3700 | 0 |
| 2026-08-22 18:28:18 | cold-first | vulkan | rpc3080-16k-full-285814-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7210 | 540.1200 | 24.2000 | 0 |
| 2026-08-22 18:26:09 | cold-first | vulkan | rpc3080-16k-iso-ts0-8020-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 1.4518 | 1119.3200 | 26.5200 | 0 |
| 2026-08-22 18:23:15 | cold-first | vulkan | rpc3080-16k-iso-ts0-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.1968 | 145.0600 | 12.8900 | 0 |
| 2026-08-22 18:12:02 | cold-first | vulkan | rpc3080-49k-q8-none-b2 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.1907 | 371.3750 | 19.7550 | 0 |
| 2026-08-22 18:07:34 | cold-first | vulkan | rpc3080-49k-q8-none-b1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.1838 | 359.4200 | 15.7750 | 0 |
| 2026-08-22 17:18:21 | cold-first | vulkan | rpc3080-12k-dbg-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 1.1346 | 602.0900 | 23.5250 | 0 |
| 2026-08-22 17:12:11 | cold-first | vulkan | rpc3080-49k-q8-none-ub2048 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/2048 | q8_0/q8_0 | none | 0.1906 | 371.4100 | 19.0050 | 0 |
| 2026-08-22 17:07:37 | cold-first | vulkan | rpc3080-ctl-49k-q8-none-8020 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5069 | 1001.7900 | 24.0100 | 0 |
| 2026-08-22 17:05:54 | cold-first | vulkan | rpc3080-49k-q8-none-ts285220 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.1894 | 369.0100 | 19.5450 | 0 |
| 2026-08-22 17:01:23 | cold-first | vulkan | rpc3080-ctl-49k-q8-none-game | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-08-22 16:57:10 | cold-first | vulkan | rpc3080-49k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.2024 | 395.0650 | 20.0300 | 0 |
| 2026-08-22 16:52:37 | cold-first | vulkan | rpc3080-49k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-08-22 12:03:36 | cold-first | vulkan | vk-mtp2-160k-last12f16-r1-20260822 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 1.3279 | 1075.7300 | 15.4700 | 0 |
| 2026-08-22 11:58:35 | cold-first | vulkan | vk-mtp2-160k-bigprompt-r1-20260822 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 0.0000 | - | - | 1 |
| 2026-08-20 08:23:57 | cold-first | vulkan | d131-r9-mtp128-r9-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.9538 | 1650.6400 | 47.8750 | 0 |
| 2026-08-20 08:22:54 | cold-first | vulkan | d131-r9-mtp128-q8-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9414 | 1639.0200 | 48.8300 | 0 |
| 2026-08-20 08:20:17 | cold-first | vulkan | d131-r9-mtp-r9-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 0.8298 | 1643.3900 | 39.2300 | 0 |
| 2026-08-20 08:19:18 | cold-first | vulkan | d131-r9-mtp-q8-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.8259 | 1634.2900 | 38.8750 | 0 |
| 2026-08-20 08:18:19 | cold-first | vulkan | d131-r9-decode-r9-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.7694 | 1546.0000 | 22.4750 | 0 |
| 2026-08-20 08:17:17 | cold-first | vulkan | d131-r9-decode-q8-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.7701 | 1545.1100 | 23.4550 | 0 |
| 2026-08-19 18:52:42 | cold-first | vulkan | d131-r9-gr9-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 4.0693 | 1629.0400 | 26.2400 | 0 |
| 2026-08-19 18:52:19 | cold-first | vulkan | d131-r9-gctl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 4.0446 | 1630.4200 | 25.4700 | 0 |
| 2026-08-19 18:49:00 | cold-first | vulkan | d131-r9-smoke-r7 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 13.4406 | 1630.9500 | 27.8700 | 0 |
| 2026-08-19 18:41:08 | cold-first | vulkan | d131-r9-greedy-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 4.0826 | 1638.7800 | 25.9300 | 0 |
| 2026-08-19 18:32:04 | cold-first | vulkan | d131-r9-nopc-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 3.4789 | 1015.2300 | 26.4500 | 0 |
| 2026-08-19 18:30:34 | cold-first | vulkan | d131-r9-smoke-r6 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 13.2021 | 1609.9800 | 27.4000 | 0 |
| 2026-08-19 18:27:08 | cold-first | vulkan | d131-r9-trace-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 1.4688 | 1598.7700 | 23.1500 | 0 |
| 2026-08-19 18:26:10 | autotune | vulkan | gui-autotune-Qwen3.8-27B-Q4_K_M-20260819-182542 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | none | 13.2830 | 1572.25 | 28.39 | 0 |
| 2026-08-19 18:25:04 | autotune | vulkan | gui-autotune-Qwen3.8-27B-Q4_K_M-20260819-182409 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | none | 13.3027 | 1604.76 | 27.89 | 0 |
| 2026-08-19 18:24:04 | autotune | rocm | gui-autotune-Qwen3.8-27B-Q4_K_M-20260819-182307 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | none | 12.3933 | 1601.76 | 24.10 | 0 |
| 2026-08-19 18:21:48 | cold-first | vulkan | d131-r9-smoke-r5 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 12.4971 | 1607.9300 | 24.5000 | 0 |
| 2026-08-19 18:13:59 | cold-first | vulkan | d131-r9-smoke-r4 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 13.3417 | 1631.2500 | 27.4400 | 0 |
| 2026-08-19 18:10:13 | cold-first | vulkan | d131-r9-smoke-r3 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-19 18:09:50 | cold-first | vulkan | d131-r9-ctl-r3 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 13.4503 | 1638.3900 | 27.7800 | 0 |
| 2026-08-19 18:01:33 | cold-first | vulkan | d131-r9-ctl-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-19 15:13:20 | cold-first | vulkan | d131-c2-direct-r2 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 7.0304 | 1191.4350 | 24.5450 | 0 |
| 2026-08-19 15:11:50 | cold-first | vulkan | d131-c2-preconv-r2 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.4311 | 1555.4050 | 24.5800 | 0 |
| 2026-08-19 15:10:31 | cold-first | vulkan | d131-c2-direct-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 7.0057 | 1186.0250 | 24.5250 | 0 |
| 2026-08-19 15:09:00 | cold-first | vulkan | d131-c2-preconv-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.4439 | 1553.9500 | 24.7350 | 0 |
| 2026-08-19 15:01:45 | cold-first | vulkan | d104-r3-stock-r3 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.2840 | 832.1650 | 22.8700 | 0 |
| 2026-08-19 14:59:30 | cold-first | vulkan | d104-r3-outdev-r1 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.2533 | 833.1300 | 22.2150 | 0 |
| 2026-08-19 14:57:16 | cold-first | vulkan | d104-r3-wn32only-r1 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.2830 | 831.4300 | 22.9050 | 0 |
| 2026-08-19 14:54:12 | cold-first | vulkan | d104-r3-wn32-r2 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.2717 | 838.1200 | 22.1050 | 0 |
| 2026-08-19 14:51:55 | cold-first | vulkan | d104-r3-stock-r2 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.0797 | 790.0500 | 22.7000 | 0 |
| 2026-08-19 14:49:35 | cold-first | vulkan | d104-r3-wn32-r1 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.2435 | 831.1150 | 22.2450 | 0 |
| 2026-08-19 14:47:17 | cold-first | vulkan | d104-r3-stock-r1 | Qwen3.6-27B-Q6_K.gguf | 49152 | 512/128 | q8_0/q8_0 | none | 5.2383 | 824.7300 | 22.8350 | 0 |
| 2026-08-19 14:34:11 | cold-first | vulkan | d129-wg-lg-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 17.3007 | 1449.9650 | 27.3800 | 0 |
| 2026-08-19 14:33:24 | cold-first | vulkan | d129-wg-sub-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 17.9862 | 1458.2700 | 28.9950 | 0 |
| 2026-08-19 14:32:37 | cold-first | vulkan | d129-wg-lg-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 17.1071 | 1455.1800 | 26.8650 | 0 |
| 2026-08-19 14:31:48 | cold-first | vulkan | d129-wg-sub-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 17.9190 | 1437.6700 | 29.1200 | 0 |
| 2026-08-16 15:03:33 | cold-first | vulkan | d105-p4b-mtp4-128-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | mtp | 15.2014 | 1364.5250 | 52.2150 | 0 |
| 2026-08-16 15:02:55 | cold-first | vulkan | d105-p4b-mtp3-128-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | mtp | 15.8427 | 1433.5900 | 52.5150 | 0 |
| 2026-08-16 15:02:20 | cold-first | vulkan | d105-p4b-mtp2-128-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | mtp | 15.4997 | 1404.3600 | 50.2000 | 0 |
| 2026-08-16 15:01:45 | cold-first | vulkan | d105-p4b-none128-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 12.8445 | 1455.7300 | 28.6550 | 0 |
| 2026-08-16 15:00:01 | cold-first | vulkan | d105-p4-mtp4-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | mtp | 2.6727 | 1413.7850 | 62.3000 | 0 |
| 2026-08-16 14:59:30 | cold-first | vulkan | d105-p4-mtp3-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | mtp | 2.7247 | 1453.2800 | 53.4350 | 0 |
| 2026-08-16 14:58:59 | cold-first | vulkan | d105-p4-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | mtp | 2.7394 | 1457.5450 | 54.0150 | 0 |
| 2026-08-16 14:58:16 | cold-first | vulkan | d105-p3-k16-r1 | Qwen3.6-27B-Q4_K16.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.4197 | 213.6100 | 28.1100 | 0 |
| 2026-08-16 14:50:15 | cold-first | vulkan | d105-split-trace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.5397 | 816.9150 | 29.8050 | 0 |
| 2026-08-16 14:40:45 | cold-first | vulkan | d105-perf-clean-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.3477 | 720.4550 | 22.8500 | 0 |
| 2026-08-16 14:39:08 | cold-first | vulkan | d105-perf-trace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.3348 | 712.5400 | 22.8750 | 0 |
| 2026-08-16 14:37:37 | cold-first | vulkan | d105-outdev-vk1-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 2.6621 | 1476.1300 | 29.5350 | 0 |
| 2026-08-16 14:37:02 | cold-first | vulkan | d105-mmvq-off-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 2.6422 | 1468.5250 | 29.0600 | 0 |
| 2026-08-16 14:36:33 | cold-first | vulkan | d105-mmvq-force-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 2.6496 | 1469.4450 | 29.5200 | 0 |
| 2026-08-16 14:35:25 | cold-first | vulkan | d105-ctrl-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 2.6473 | 1472.6000 | 29.6900 | 0 |
| 2026-08-16 12:39:00 | cold-first | vulkan | d104-r2v3-block512 | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.4924 | 252.0600 | 24.0800 | 0 |
| 2026-08-16 12:36:21 | cold-first | vulkan | d104-r2v3-wm128wn32 | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.4767 | 244.0250 | 23.6200 | 0 |
| 2026-08-16 12:19:02 | cold-first | vulkan | d104-r2v1-q6k-4vec | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.7039 | 924.3750 | 24.0050 | 0 |
| 2026-08-16 12:05:16 | cold-first | vulkan | d104-r1-predequant | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.3955 | 201.6650 | 23.2200 | 0 |
| 2026-08-16 11:50:19 | cold-first | vulkan | q6k-vk-nocoopmat | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.6520 | 336.9050 | 23.5700 | 0 |
| 2026-08-16 11:47:21 | cold-first | vulkan | q6k-vk-variant-b128wn32 | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.8053 | 982.6200 | 24.2050 | 0 |
| 2026-08-16 11:45:50 | cold-first | vulkan | q6k-vk-variant-bn64 | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.4806 | 245.9700 | 23.9300 | 0 |
| 2026-08-16 11:42:46 | cold-first | vulkan | q6k-vk-variant-wn48 | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 0.4932 | 252.4550 | 24.0100 | 0 |
| 2026-08-16 11:39:34 | cold-first | vulkan | q6k-vk-variant-wn32 | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.8106 | 985.7550 | 24.1050 | 0 |
| 2026-08-16 11:34:06 | cold-first | vulkan | q6k-vk-gui-like-noenv | Qwen3.6-27B-Q6_K.gguf | 32768 | 512/128 | f16/f16 | none | 2.6667 | 772.2650 | 16.6900 | 0 |
| 2026-08-16 11:31:08 | cold-first | vulkan | q6k-vk-gui-like-repro | Qwen3.6-27B-Q6_K.gguf | 32768 | 512/128 | f16/f16 | none | 2.8938 | 816.8450 | 24.2400 | 0 |
| 2026-08-16 11:28:25 | cold-first | vulkan | q4km-vk-12k-adjacent-control | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 2.6878 | 1493.3450 | 29.4750 | 0 |
| 2026-08-16 11:26:42 | cold-first | vulkan | q6k-vk-12k-baseline | Qwen3.6-27B-Q6_K.gguf | 12288 | 512/128 | q4_0/q4_0 | none | 1.7103 | 930.1650 | 24.0350 | 0 |
