# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-08-15 10:24:59 | cold-first | rocm | q38-w13-c1b-final-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0116 | 1802.3950 | 22.8750 | 0 |
| 2026-08-15 10:16:38 | cold-first | rocm | q38-w13-c1b-98k-ctrl-r2 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.0955 | 1577.8000 | 19.4550 | 0 |
| 2026-08-15 10:13:58 | cold-first | rocm | q38-w13-c1b-98k-cand-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.1520 | 1570.9500 | 19.9050 | 0 |
| 2026-08-15 10:11:17 | cold-first | rocm | q38-w13-c1b-98k-ctrl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.1158 | 1573.9250 | 19.6400 | 0 |
| 2026-08-15 10:08:29 | cold-first | rocm | q38-w13-c1b-ctrl-r2 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.1840 | 1620.7300 | 21.2700 | 0 |
| 2026-08-15 10:06:38 | cold-first | rocm | q38-w13-c1b-cand-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.3362 | 1643.0200 | 21.6700 | 0 |
| 2026-08-15 10:04:53 | cold-first | rocm | q38-w13-c1b-ctrl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 7.8881 | 1598.5450 | 19.7200 | 0 |
| 2026-08-15 09:50:51 | cold-first | rocm | q38-w13-c1b-trace-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 4.2654 | 1709.6350 | 22.0950 | 0 |
| 2026-08-15 09:38:36 | cold-first | rocm | q38-w13-c1-trace-cand-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 4.2414 | 1695.8200 | 22.1550 | 0 |
| 2026-08-15 09:37:15 | cold-first | rocm | q38-w13-c1-trace-ctrl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 4.2174 | 1695.1200 | 21.4700 | 0 |
| 2026-08-15 09:31:50 | cold-first | rocm | q38-w13-c1-98k-ctrl-r3 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.1416 | 1574.8700 | 19.7800 | 0 |
| 2026-08-15 09:29:11 | cold-first | rocm | q38-w13-c1-98k-cand-r2 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.2393 | 1586.9600 | 20.1300 | 0 |
| 2026-08-15 09:26:34 | cold-first | rocm | q38-w13-c1-98k-ctrl-r2 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 8.2142 | 1591.4100 | 19.8850 | 0 |
| 2026-08-15 09:16:54 | cold-first | rocm | q38-w13-c1-98k-cand-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 2.8552 | 1532.2700 | 19.4300 | 0 |
| 2026-08-15 09:14:23 | cold-first | rocm | q38-w13-c1-98k-ctrl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 2.9317 | 1570.4950 | 19.8500 | 0 |
| 2026-08-15 09:11:57 | cold-first | rocm | q38-w13-c1-ctrl-r4 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.0419 | 1629.8000 | 20.0800 | 0 |
| 2026-08-15 09:10:15 | cold-first | rocm | q38-w13-c1-cand-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.6246 | 1696.5250 | 22.5100 | 0 |
| 2026-08-15 09:06:15 | cold-first | rocm | q38-w13-c1-ctrl-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.4726 | 1687.3900 | 21.6900 | 0 |
| 2026-08-15 09:04:32 | cold-first | rocm | q38-w13-c1-cand-r2 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.4758 | 1674.7100 | 21.8900 | 0 |
| 2026-08-15 09:02:45 | cold-first | rocm | q38-w13-c1-ctrl-r2 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.4211 | 1683.1000 | 21.4200 | 0 |
| 2026-08-15 09:01:08 | cold-first | rocm | q38-w13-c1-cand-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.8043 | 1735.6300 | 22.8250 | 0 |
| 2026-08-15 08:59:36 | cold-first | rocm | q38-w13-c1-ctrl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.5456 | 1695.7100 | 21.9700 | 0 |
| 2026-08-15 08:56:08 | cold-first | rocm | q38-w13-49k-f8-baseline-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.5909 | 1691.4750 | 22.4450 | 0 |
| 2026-08-14 21:13:00 | cold-first | rocm | q38-rb-rc-98k-f8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 2.9455 | 1481.9400 | 31.6050 | 0 |
| 2026-08-14 21:11:01 | cold-first | rocm | q38-rb-rc-98k-q8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.8342 | 1431.4250 | 29.4700 | 0 |
| 2026-08-14 21:08:58 | cold-first | rocm | q38-rb-rc-49k-f8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 6.0276 | 1716.7950 | 39.9550 | 0 |
| 2026-08-14 21:07:41 | cold-first | rocm | q38-rb-rc-49k-q8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.6371 | 1625.2950 | 35.1400 | 0 |
| 2026-08-14 21:06:20 | cold-first | rocm | q38-rb-rc-49k-f8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.6465 | 1713.6700 | 22.2000 | 0 |
| 2026-08-14 21:04:22 | cold-first | rocm | q38-rb-rc-49k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.3455 | 1647.1750 | 21.5300 | 0 |
| 2026-08-14 21:02:16 | cold-first | vulkan | q38-rb-vk-98k-f8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.5470 | 1470.4200 | 39.7750 | 0 |
| 2026-08-14 21:00:04 | cold-first | vulkan | q38-rb-vk-98k-q8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 5.5251 | 1447.9900 | 42.8750 | 0 |
| 2026-08-14 20:57:51 | cold-first | vulkan | q38-rb-vk-98k-f8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 2.6405 | 1366.2650 | 22.8200 | 0 |
| 2026-08-14 20:55:19 | cold-first | vulkan | q38-rb-vk-98k-q8-none-r2 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 2.6415 | 1355.3400 | 24.5500 | 0 |
| 2026-08-14 20:51:29 | cold-first | vulkan | q38-rb-vk-98k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-08-14 20:49:51 | cold-first | vulkan | q38-rb-vk-49k-f8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.9515 | 1648.4400 | 46.6350 | 0 |
| 2026-08-14 20:48:31 | cold-first | vulkan | q38-rb-vk-49k-q8-mtp2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9442 | 1637.4400 | 48.5550 | 0 |
| 2026-08-14 20:47:07 | cold-first | vulkan | q38-rb-vk-49k-f8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.0236 | 1524.5050 | 24.7750 | 0 |
