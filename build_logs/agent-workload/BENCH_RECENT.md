# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-07-16 11:15:12 | cold-first | rocm | e338-rocm-dual-q4km-none-98k-big-q8-copies1-r2-64tok-wddm | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 1.4890 | 1493.2100 | 19.1500 | 0 |
| 2026-07-16 11:13:48 | cold-first | rocm | e338-rocm-dual-q4km-mtp3-98k-big-q8-copies1-r1-wddm | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.4872 | 1435.9700 | 35.4400 | 0 |
| 2026-07-16 11:06:15 | cold-first | rocm | e338-rocm-dual-q3ks-none-131k-nearfull-q8-copies1-r1-wddm | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.2018 | 1439.8900 | 21.9000 | 0 |
| 2026-07-16 11:03:18 | cold-first | rocm | e338-rocm-dual-q3ks-mtp3-131k-nearfull-q8-copies1-r1-wddm | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 1.1614 | 1363.9500 | 32.5300 | 0 |
| 2026-07-16 11:01:26 | cold-first | rocm | e338-rocm-dual-q3ks-mtp3-131k-nearfull-q8-copies2-r3-flushfix | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 1.1361 | 1332.6600 | 32.7000 | 0 |
| 2026-07-16 10:59:53 | cold-first | rocm | e338-rocm-dual-q3ks-mtp3-131k-nearfull-q8-copies2-r2-flushfix | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-16 10:52:09 | cold-first | rocm | e338-rocm-dual-q3ks-mtp3-131k-nearfull-q8-copies2-r1-wddm | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 1.1428 | 1341.7200 | 32.2500 | 0 |
| 2026-07-16 10:49:13 | cold-first | rocm | e338-rocm-dual-q4km-98k-big-q8-copies1-r1-wddm | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.3927 | 1485.2700 | 18.3700 | 0 |
| 2026-07-16 10:48:00 | cold-first | rocm | e338-rocm-dual-q4km-98k-big-q8-copies2-r1-wddm | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.3942 | 1494.8200 | 16.7700 | 0 |
| 2026-07-16 10:45:57 | cold-first | vulkan | e338-vulkan-dual-q4km-98k-big-q8-r1-wddm | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.3096 | 1164.2900 | 20.3200 | 0 |
| 2026-07-16 10:42:25 | cold-first | rocm | e338-rocm-dual-q4km-98k-big-q8-r2-wddm | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.3894 | 1477.3000 | 16.2500 | 0 |
| 2026-07-16 10:40:21 | cold-first | rocm | e338-rocm-dual-q4km-98k-big-q8-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.3900 | 1479.6900 | 16.0800 | 0 |
| 2026-07-16 10:38:12 | cold-first | vulkan | e338-vulkan-dual-q3ks-131k-nearfull-q8-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.2557 | 1179.3000 | 14.6500 | 0 |
| 2026-07-16 10:36:20 | cold-first | rocm | e338-rocm-dual-q3ks-131k-nearfull-q8-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.3095 | 1428.2600 | 16.8800 | 0 |
| 2026-07-16 10:34:28 | cold-first | vulkan | e338-vulkan-dual-q3ks-131k-big-q8-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.5197 | 1453.1200 | 15.9100 | 0 |
| 2026-07-16 10:32:42 | cold-first | rocm | e338-rocm-dual-q3ks-131k-big-q8-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.5913 | 1639.8400 | 22.8500 | 0 |
| 2026-07-16 00:06:34 | cold-first | rocm | e337-rocm0-q3ks-12k-auto-production-r2-latched | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8529 | 1133.4800 | 35.1600 | 0 |
| 2026-07-16 00:05:42 | cold-first | rocm | e337-rocm0-q3ks-49k-auto-production-r2-latched | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5545 | 1045.6100 | 31.3100 | 0 |
| 2026-07-15 23:59:39 | cold-first | rocm | e337-rocm0-q3ks-49k-auto-production-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5605 | 1056.5500 | 31.5200 | 0 |
| 2026-07-15 23:57:10 | cold-first | rocm | e337-rocm0-q3ks-12k-auto-production-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8699 | 1136.7000 | 35.3700 | 0 |
| 2026-07-15 23:48:28 | cold-first | rocm | e337-rocm0-q3ks-49k-default-r2-128 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 3.9926 | 1061.2100 | 30.8800 | 0 |
| 2026-07-15 23:47:18 | cold-first | rocm | e337-rocm0-q3ks-49k-q8chunk-wmma-r2-128 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 3.9356 | 1046.1100 | 30.5400 | 0 |
| 2026-07-15 23:45:30 | cold-first | rocm | e337-rocm0-q3ks-49k-q8chunk-wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5654 | 1066.4800 | 31.6700 | 0 |
| 2026-07-15 23:44:07 | cold-first | rocm | e337-rocm0-q3ks-12k-q8chunk-wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8562 | 1134.2900 | 35.4000 | 0 |
| 2026-07-15 23:32:09 | cold-first | rocm | e337-rocm0-q3ks-12k-q8v-direct-wmma-block-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.7588 | 1107.1100 | 35.3400 | 0 |
| 2026-07-15 23:29:41 | cold-first | rocm | e337-rocm0-q3ks-12k-q8v-direct-wmma-alias-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8357 | 1126.4100 | 35.6500 | 0 |
| 2026-07-15 23:27:39 | cold-first | rocm | e337-rocm0-q3ks-12k-q8v-direct-wmma-pad0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.6642 | 1081.0300 | 35.3900 | 0 |
| 2026-07-15 23:23:02 | cold-first | vulkan | e337-vulkan1-q3ks-49k-none-r2 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.4758 | 893.3000 | 36.3900 | 0 |
| 2026-07-15 23:21:23 | cold-first | rocm | e337-rocm0-q3ks-49k-q8v-direct-wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5033 | 947.1700 | 31.3600 | 0 |
| 2026-07-15 23:19:55 | cold-first | rocm | e337-rocm0-q3ks-12k-q8v-direct-wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8631 | 1130.4800 | 35.6900 | 0 |
| 2026-07-15 23:12:56 | cold-first | rocm | e337-rocm0-q3ks-49k-q8k-direct-wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.4351 | 816.8900 | 31.3200 | 0 |
| 2026-07-15 23:11:25 | cold-first | rocm | e337-rocm0-q3ks-12k-q8k-direct-wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.7386 | 1099.9700 | 35.6400 | 0 |
| 2026-07-15 23:04:21 | cold-first | rocm | e337-rocm0-q3ks-49k-q8kmmq-cols32-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.4738 | 890.7600 | 31.3400 | 0 |
| 2026-07-15 23:03:09 | cold-first | rocm | e337-rocm0-q3ks-12k-q8kmmq-cols32-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8133 | 1120.2400 | 35.6700 | 0 |
| 2026-07-15 23:01:09 | cold-first | rocm | e337-rocm0-q3ks-49k-forcecols16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0352 | 1043.8900 | - | 0 |
| 2026-07-15 22:59:09 | cold-first | rocm | e337-rocm0-q3ks-49k-q8kmmq-layout-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.4813 | 905.1500 | 31.5900 | 0 |
| 2026-07-15 22:57:29 | cold-first | rocm | e337-rocm0-q3ks-12k-q8kmmq-layout-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.7426 | 1100.4800 | 35.3500 | 0 |
| 2026-07-15 22:51:44 | cold-first | rocm | e337-rocm0-q3ks-12k-q8kmmq-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.7787 | 1110.9700 | 35.6300 | 0 |
| 2026-07-15 22:46:36 | cold-first | rocm | e337-rocm0-q3ks-12k-wmma-cols16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8970 | 1141.3800 | 35.3900 | 0 |
| 2026-07-15 22:40:28 | cold-first | rocm | e337-rocm0-q3ks-49k-q8direct-memory-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0218 | 614.8700 | - | 0 |
| 2026-07-15 22:36:41 | cold-first | rocm | e337-rocm0-q3ks-49k-fattnalloc-trace | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.2230 | 773.1500 | - | 0 |
| 2026-07-15 22:32:30 | cold-first | rocm | e337-rocm0-q3ks-12k-q8wmma-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.2130 | 959.9500 | 35.3100 | 0 |
| 2026-07-15 22:26:50 | cold-first | rocm | e337-rocm0-q3ks-12k-q8tile-active-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.3780 | 1000.6100 | 36.5300 | 0 |
| 2026-07-15 22:25:57 | cold-first | rocm | e337-rocm0-q3ks-q8tile-active-smoke | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 3.2010 | 745.9000 | - | 0 |
| 2026-07-15 22:23:16 | cold-first | rocm | e337-rocm0-q3ks-fattnroute-smoke | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q8_0/q8_0 | none | 3.2072 | 752.9100 | - | 0 |
| 2026-07-15 22:22:16 | cold-first | rocm | e337-rocm0-q3ks-49k-q8tile-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5630 | 1061.8500 | 31.5500 | 0 |
| 2026-07-15 22:20:53 | cold-first | rocm | e337-rocm0-q3ks-12k-q8tile-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8735 | 1138.3600 | 35.3100 | 0 |
| 2026-07-15 22:12:12 | cold-first | vulkan | e337-vulkan1-q3ks-49k-prefillretain-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0293 | 870.4000 | - | 0 |
| 2026-07-15 22:10:37 | cold-first | rocm | e337-rocm0-q3ks-49k-buffertrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0352 | 1041.7500 | - | 0 |
| 2026-07-15 22:05:29 | cold-first | vulkan | e337-vulkan1-q3ks-49k-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.4757 | 893.0500 | 36.8900 | 0 |
| 2026-07-15 22:04:04 | cold-first | rocm | e337-rocm0-q3ks-49k-auto-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5537 | 1044.4700 | 31.0700 | 0 |
| 2026-07-15 22:02:13 | cold-first | vulkan | e337-vulkan1-q3ks-12k-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.9092 | 1129.9200 | 39.7900 | 0 |
| 2026-07-15 22:01:06 | cold-first | rocm | e337-rocm0-q3ks-12k-ws0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.9569 | 1159.6400 | 35.0600 | 0 |
| 2026-07-15 22:00:04 | cold-first | rocm | e337-rocm0-q3ks-12k-ws128-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.8821 | 1141.6000 | 35.2000 | 0 |
| 2026-07-15 21:58:36 | cold-first | rocm | e337-rocm0-q3ks-12k-auto-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.9535 | 1160.0800 | 35.4400 | 0 |
| 2026-07-15 21:40:31 | cold-first | rocm | e336-rocm-q4km-49k-pool-trace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.8257 | 1600.6500 | 19.4200 | 0 |
| 2026-07-15 21:33:56 | cold-first | - | e335-stock-f955-rocm-q3ks-49k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0831 | 1235.4700 | 31.1500 | 0 |
| 2026-07-15 21:30:27 | cold-first | rocm | e335-rocm-q4km-98k-maxprompt-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 1.1227 | 553.5000 | 17.6400 | 0 |
| 2026-07-15 21:27:18 | cold-first | rocm | e335-rocm-q4km-98k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 1.2831 | 564.8000 | 18.5500 | 0 |
| 2026-07-15 21:24:21 | cold-first | rocm | e335-rocm-q4km-49k-mtp3-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8580 | 1604.7600 | 38.1000 | 0 |
| 2026-07-15 21:23:31 | cold-first | rocm | e335-rocm-q4km-49k-none-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.3936 | 1716.1650 | 19.9950 | 0 |
| 2026-07-15 21:22:13 | cold-first | rocm | e335-rocm-q4km-12k-mtp3-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 24.2335 | 1444.3500 | 42.1200 | 0 |
| 2026-07-15 21:21:32 | cold-first | rocm | e335-rocm-q4km-12k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 16.2304 | 1522.7000 | 22.2300 | 0 |
| 2026-07-15 21:20:10 | cold-first | rocm | e335-rocm-q3ks-65k-mtp3-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | mtp | 4.2062 | 1546.8800 | 33.9200 | 0 |
| 2026-07-15 21:19:11 | cold-first | rocm | e335-rocm-q3ks-65k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 65536 | 8192/1024 | q8_0/q8_0 | none | 4.2096 | 1630.5900 | 24.9600 | 0 |
| 2026-07-15 21:18:18 | cold-first | rocm | e335-rocm-q3ks-49k-mtp3-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9877 | 1672.0500 | 35.4200 | 0 |
| 2026-07-15 21:17:32 | cold-first | rocm | e335-rocm-q3ks-49k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.7934 | 1734.1400 | 25.7700 | 0 |
| 2026-07-15 21:01:44 | cold-first | rocm | e334-rocm0-q3ks-q8-real-vec-probe-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 1.3476 | 730.5300 | 34.9300 | 0 |
| 2026-07-15 20:58:20 | cold-first | rocm | e334-rocm0-q3ks-q8-direct-default-final-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5526 | 1043.2400 | 30.4400 | 0 |
| 2026-07-15 20:55:09 | cold-first | rocm | e334-rocm0-q3ks-q8-vec-long-r2 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5610 | 1058.9600 | 30.1100 | 0 |
| 2026-07-15 20:54:00 | cold-first | rocm | e334-rocm0-q3ks-q8-mma-long-r2 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5528 | 1043.3400 | 31.0000 | 0 |
| 2026-07-15 20:52:23 | cold-first | rocm | e334-rocm0-q3ks-q8-vec-direct-long-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5646 | 1065.6300 | 30.9300 | 0 |
| 2026-07-15 20:50:46 | cold-first | rocm | e334-rocm0-q3ks-q8-mma-short-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 2.1247 | 1178.4000 | 35.5300 | 0 |
| 2026-07-15 20:49:49 | cold-first | rocm | e334-rocm0-q3ks-q8-vec-direct-probe-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 2.1011 | 1165.9700 | 35.0500 | 0 |
| 2026-07-15 20:42:28 | cold-first | rocm | e334-rocm0-q3ks-q8-post-reserve-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5552 | 1048.0200 | 30.4300 | 0 |
| 2026-07-15 20:38:09 | cold-first | rocm | e334-rocm0-q3ks-q8-pre-reserve-r1 | Qwen3.6-27B-Q3_K_S.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.5678 | 1072.4800 | 30.5800 | 0 |
| 2026-07-15 20:02:43 | cold-first | rocm | e333-q4km-rocm-split27x37-ctx131k-none-prod-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 2.0446 | 1379.1400 | 17.4700 | 0 |
| 2026-07-15 20:01:02 | cold-first | rocm | e333-q4km-rocm-split26x38-ctx131k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.2693 | 1306.6100 | 14.9000 | 0 |
| 2026-07-15 19:59:19 | cold-first | rocm | e333-q4km-rocm-split27x37-ctx131k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.2842 | 1380.9200 | 14.7800 | 0 |
| 2026-07-15 19:57:23 | cold-first | rocm | e333-q4km-rocm-split7x9-ctx131k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.2655 | 1288.9500 | 14.4600 | 0 |
