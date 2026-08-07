# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-06 23:32:22 | cold-first | - | vk-mtp-n2-dual | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.5961 | - | - | 0 |
| 2026-08-06 23:25:50 | cold-first | - | rocm-mtp-n2-dual | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.2934 | - | - | 0 |
| 2026-08-06 23:10:37 | cold-first | - | rocm-mtp-n2-ab2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-08-06 23:03:00 | cold-first | - | vk-mtp-n2-ab2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.5522 | - | - | 0 |
| 2026-08-06 22:06:51 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260806-220607 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.0519 | 1451.28 | 30.54 | 0 |
| 2026-08-06 22:04:51 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260806-220339 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.7655 | 1679.20 | 32.33 | 0 |
| 2026-08-06 22:03:24 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260806-220200 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 6.1358 | 1719.92 | 43.10 | 0 |
| 2026-08-06 21:59:54 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260806-215904 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 17.6262 | 1675.13 | 49.66 | 0 |
| 2026-08-06 21:58:04 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260806-215721 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 15.4068 | 1499.32 | 41.65 | 0 |
| 2026-08-06 20:30:15 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260806-202812 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 3.6300 | 249.40 | 31.94 | 0 |
| 2026-08-06 20:14:09 | cold-first | - | vk-mtp-n2-ab10k | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 1.4136 | - | - | 0 |
| 2026-08-06 20:09:16 | cold-first | - | rocm-mtp-n2-ab10k | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.8402 | - | - | 0 |
| 2026-08-06 20:04:51 | cold-first | - | rocm-mtp-n2-ab | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-08-06 19:44:18 | cold-first | - | vk-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 4.6529 | - | - | 0 |
| 2026-08-06 19:41:35 | cold-first | - | vk-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-08-06 19:34:10 | cold-first | - | vk-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 2.4038 | - | - | 0 |
| 2026-08-06 19:30:32 | cold-first | - | vk-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-08-05 17:48:06 | cold-first | rocm | d094-c8-rocm-mtp-q8-49k | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.9614 | 1424.7000 | 30.0100 | 0 |
| 2026-08-05 17:04:34 | cold-first | vulkan | d094-c7-49k-mtp-kq8vf16 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/f16 | mtp | 0.0000 | - | - | 1 |
| 2026-08-05 16:47:45 | cold-first | vulkan | d094-c7-49k-mtp-q8-f32acc | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.5539 | 1293.8000 | 28.9300 | 0 |
| 2026-08-05 15:45:58 | cold-first | vulkan | d094-c7-49k-mtp-q8-norot | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.5399 | 1310.4400 | 26.7500 | 0 |
| 2026-08-05 12:42:01 | cold-first | vulkan | d094-c7-49k-none-f16 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f16/f16 | none | 5.4511 | 1654.9800 | 26.0100 | 0 |
| 2026-08-05 12:40:49 | cold-first | vulkan | d094-c7-49k-none-q8 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.8198 | 1411.8400 | 26.3800 | 0 |
| 2026-08-05 12:28:06 | cold-first | vulkan | d094-c6-guirepro-f16 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f16/f16 | mtp | 5.4059 | 1499.5900 | 39.7300 | 0 |
| 2026-08-05 12:27:23 | cold-first | vulkan | d094-c6-guirepro-q8 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.5680 | 1295.9800 | 29.3700 | 0 |
| 2026-08-05 12:21:40 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-122009 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.6607 | 1581.22 | 39.96 | 0 |
| 2026-08-05 12:18:51 | cold-first | vulkan | d094-131k-c6b-f16 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | f16/f16 | none | 1.2190 | 1297.8400 | 7.9700 | 0 |
| 2026-08-05 12:15:45 | cold-first | vulkan | d094-131k-c6b-preconv | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.3400 | 1282.4100 | 22.9200 | 0 |
| 2026-08-05 12:13:59 | cold-first | vulkan | d094-131k-c6b-ctl | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.1820 | 1125.8300 | 21.8100 | 0 |
| 2026-08-05 11:58:59 | cold-first | vulkan | d094-131k-c6-f16kv-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | f16/f16 | none | 1.0881 | 1190.2700 | 6.2000 | 0 |
| 2026-08-05 11:56:46 | cold-first | vulkan | d094-131k-c6-ctl-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.2526 | 1198.1400 | 21.6000 | 0 |
| 2026-08-05 11:48:59 | cold-first | vulkan | d094-131k-c6-preconv-r3 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.3589 | 1301.4500 | 23.0300 | 0 |
| 2026-08-05 11:47:16 | cold-first | vulkan | d094-c6-trace2 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 2.8670 | 1626.6500 | 24.3500 | 0 |
| 2026-08-05 10:43:51 | cold-first | vulkan | d094-c6-trace | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 2.7169 | 1533.4900 | 24.1500 | 0 |
| 2026-08-05 10:42:43 | cold-first | vulkan | d094-131k-c6-preconv-r2 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.1292 | 1071.4700 | 22.3400 | 0 |
| 2026-08-05 09:20:02 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-091925 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.6985 | 1667.94 | 31.36 | 0 |
| 2026-08-05 09:18:06 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-091728 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.3154 | 1637.71 | 23.88 | 0 |
| 2026-08-05 09:17:11 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-091633 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.5293 | 1641.74 | 28.58 | 0 |
| 2026-08-05 08:50:16 | cold-first | rocm | d094-rocm-autotune-smoke-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.2912 | 1577.6000 | 26.7900 | 0 |
| 2026-08-05 08:06:58 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-080653 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mixed | 0.0000 | - | - | 1 |
| 2026-08-05 08:06:39 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-080628 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mixed | 0.0000 | - | - | 2 |
| 2026-08-05 08:04:43 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-080432 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mixed | 0.0000 | - | - | 2 |
| 2026-08-05 08:04:21 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260805-080259 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.7456 | 1595.06 | 42.09 | 0 |
| 2026-08-04 22:23:27 | cold-first | vulkan | d094-131k-postint8-revert-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.1328 | 1077.6500 | 21.5400 | 0 |
| 2026-08-04 22:13:57 | cold-first | vulkan | d094-131k-int8-r3 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.9092 | 856.0700 | 20.9600 | 0 |
| 2026-08-04 22:10:21 | cold-first | vulkan | d094-131k-int8-r2 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.9519 | 897.8100 | 20.9200 | 0 |
| 2026-08-04 22:08:19 | cold-first | vulkan | d094-131k-ctl-r2 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.1873 | 1132.3200 | 21.5200 | 0 |
| 2026-08-04 22:06:01 | cold-first | vulkan | d094-131k-int8-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.9829 | 927.4200 | 21.4300 | 0 |
| 2026-08-04 22:03:51 | cold-first | vulkan | d094-12k-int8-on-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.5413 | 1321.0400 | 25.8400 | 0 |
| 2026-08-04 22:02:49 | cold-first | vulkan | d094-12k-int8-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.1417 | 1458.7000 | 26.5100 | 0 |
| 2026-08-04 18:57:53 | cold-first | vulkan | d094-131k-f16kv-r3 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | f16/f16 | none | 1.5776 | 1529.4500 | 22.7533 | 0 |
| 2026-08-04 18:51:47 | cold-first | vulkan | d094-131k-f16kv-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | f16/f16 | none | 1.5627 | 1512.0800 | 22.8700 | 0 |
| 2026-08-04 18:49:47 | cold-first | vulkan | d094-131k-cm1-skipdeq-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.4427 | 1385.6400 | 23.4500 | 0 |
| 2026-08-04 18:48:06 | cold-first | vulkan | d094-131k-cm1-skiploads-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.2903 | 1233.7700 | 22.6600 | 0 |
| 2026-08-04 18:46:17 | cold-first | vulkan | d094-131k-cm1-skipmul-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.3847 | 1328.9300 | 22.7700 | 0 |
| 2026-08-04 18:44:25 | cold-first | vulkan | d094-131k-cm1-emptyj-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.9759 | 1931.0500 | 25.8700 | 0 |
| 2026-08-04 18:42:51 | cold-first | vulkan | d094-12k-cm1-emptyj-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.5053 | 1801.5900 | 27.7000 | 0 |
| 2026-08-04 18:41:33 | cold-first | vulkan | d094-12k-cm1-skiploads-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.7968 | 1619.8200 | 27.0600 | 0 |
| 2026-08-04 18:39:22 | cold-first | vulkan | d094-12k-cm1-skipqktmul-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.1422 | 1716.2300 | 27.0700 | 0 |
| 2026-08-04 18:37:53 | cold-first | vulkan | d094-12k-cm1-skipdeq-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.1214 | 1707.3100 | 27.1900 | 0 |
| 2026-08-04 18:36:02 | cold-first | vulkan | d094-12k-cm1-skipv-diag-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.6428 | 1577.7100 | 27.2100 | 0 |
| 2026-08-04 18:32:59 | cold-first | vulkan | d094-131k-fa-scalar-nomask-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.7655 | 713.4900 | 22.1000 | 0 |
| 2026-08-04 18:26:10 | cold-first | vulkan | d094-131k-fa-scalar-br16stg-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.9581 | 901.8633 | 21.8933 | 0 |
| 2026-08-04 18:21:26 | cold-first | vulkan | d094-12k-fa-scalar-br16-bc64-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 6.3251 | 1146.8600 | 19.9800 | 0 |
| 2026-08-04 18:20:24 | cold-first | vulkan | d094-12k-fa-scalar-br16-sg32-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.1802 | 1244.9600 | 25.0600 | 0 |
| 2026-08-04 18:19:34 | cold-first | vulkan | d094-12k-fa-scalar-br16-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.5499 | 1556.3400 | 26.9300 | 0 |
| 2026-08-04 18:08:50 | cold-first | vulkan | d094-12k-fa-scalar-stg1-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.7178 | 1370.4600 | 25.6800 | 0 |
| 2026-08-04 18:05:38 | cold-first | vulkan | d094-12k-fa-scalar-ds4-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 4.0189 | 590.0900 | 25.0300 | 0 |
| 2026-08-04 18:03:18 | cold-first | vulkan | d094-12k-fa-scalar-bc64-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 5.0394 | 770.4000 | 26.2500 | 0 |
| 2026-08-04 18:00:01 | cold-first | vulkan | d094-12k-fa-scalar-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 5.0306 | 770.0800 | 26.0600 | 0 |
| 2026-08-04 16:26:38 | cold-first | vulkan | d094-12k-postc3-revert-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.8211 | 1641.0600 | 26.5500 | 0 |
| 2026-08-04 16:22:36 | cold-first | vulkan | d094-12k-4dstaging-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.1164 | 1454.8900 | 26.2400 | 0 |
| 2026-08-04 16:15:29 | cold-first | vulkan | d094-12k-pvreuse-lite-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.1835 | 1489.6800 | 25.8800 | 0 |
| 2026-08-04 16:13:46 | cold-first | vulkan | d094-12k-pvreuse2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.9818 | 1424.9700 | 26.2000 | 0 |
| 2026-08-04 16:08:26 | cold-first | vulkan | d094-12k-pvreuse-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.9617 | 1427.1000 | 25.8900 | 0 |
| 2026-08-04 15:43:42 | cold-first | vulkan | d094-12k-postfa-revert-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.8343 | 1647.7400 | 26.7600 | 0 |
| 2026-08-04 15:37:08 | cold-first | vulkan | d094-12k-fa-dualacc-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.3290 | 1501.2500 | 26.8400 | 0 |
| 2026-08-04 15:28:12 | cold-first | vulkan | d094-12k-fa-bc32-nomaskopt-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.6966 | 1617.9200 | 26.2000 | 0 |
| 2026-08-04 14:53:32 | cold-first | vulkan | d094-12k-final-sanity-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.7190 | 1624.0600 | 26.2600 | 0 |
| 2026-08-04 14:51:18 | cold-first | vulkan | d094-131k-p60k-cleanperf-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.6143 | 570.9000 | 18.9900 | 0 |
