# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-04 08:54:35 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-085347 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 4.6769 | 1339.47 | 33.63 | 0 |
| 2026-08-04 08:52:47 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-085204 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.1991 | 1553.25 | 29.79 | 0 |
| 2026-08-04 08:48:44 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-084816 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | sweep/sweep | sweep/sweep | mtp | 12.7296 | 1277.50 | 36.68 | 0 |
| 2026-08-04 08:48:06 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-084701 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 2.6650 | 1399.90 | 22.72 | 0 |
| 2026-08-04 08:46:17 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-084458 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 2.2250 | 1112.02 | 30.12 | 0 |
| 2026-08-04 08:36:52 | cold-first | vulkan | validate-20260804-q4km-vulkan-long131k-p60k-default-wn32-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.1916 | 1171.9400 | 21.3100 | 0 |
| 2026-08-04 08:35:03 | cold-first | vulkan | validate-20260804-q4km-vulkan-short12k-default-wn32-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.4155 | 1563.5700 | 26.3500 | 0 |
| 2026-08-04 08:26:08 | cold-first | vulkan | diag-20260804-q4km-vulkan-long131k-p60k-wn32-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.1781 | 1157.3200 | 21.6400 | 0 |
| 2026-08-04 08:24:52 | cold-first | vulkan | diag-20260804-q4km-vulkan-short12k-wn32-mt64-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.0562 | 1470.5100 | 26.5100 | 0 |
| 2026-08-04 08:23:30 | cold-first | vulkan | diag-20260804-q4km-vkroute-wn32-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.1612 | 1094.5300 | - | 0 |
| 2026-08-04 08:23:05 | cold-first | vulkan | diag-20260804-q4km-vkroute-wmiter1-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.0470 | 316.8800 | - | 0 |
| 2026-08-04 08:22:26 | cold-first | vulkan | diag-20260804-q4km-vkroute-bn64-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.0400 | 269.9900 | - | 0 |
| 2026-08-04 08:21:42 | cold-first | vulkan | diag-20260804-q4km-vkroute-block64-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.0485 | 327.3600 | - | 0 |
| 2026-08-04 08:21:04 | cold-first | vulkan | diag-20260804-q4km-vkroute-block128-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.0490 | 330.4500 | - | 0 |
| 2026-08-04 08:20:27 | cold-first | vulkan | diag-20260804-q4km-vkroute-base-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.0469 | 316.7500 | - | 0 |
| 2026-08-04 08:18:28 | cold-first | vulkan | diag-20260804-q4km-vulkan-short12k-disable-large-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.1351 | 915.3900 | - | 0 |
| 2026-08-04 08:06:48 | cold-first | vulkan | diag-20260804-q4km-vulkan-dual-short12k-perflog-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 0.0349 | 235.0300 | - | 0 |
| 2026-08-04 08:05:08 | cold-first | vulkan | diag-20260804-q4km-vulkan-dual-short12k-e272route-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 3.2042 | 382.2300 | 27.7100 | 0 |
| 2026-08-04 08:03:30 | cold-first | vulkan | diag-20260804-qwen35-9b-vulkan-dual-r1 | Qwen3.5-9B-Q6_K.gguf | 8192 | 8192/1024 | q8_0/q8_0 | none | 3.7742 | 1104.0000 | 54.7200 | 0 |
| 2026-08-04 08:02:54 | cold-first | vulkan | diag-20260804-q4km-vulkan-dual-short12k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 3.1897 | 381.5100 | 27.0500 | 0 |
| 2026-08-04 07:57:22 | cold-first | vulkan | diag-20260804-qwen35-9b-vulkan1-single-r1 | Qwen3.5-9B-Q6_K.gguf | 8192 | 8192/1024 | q8_0/q8_0 | none | 2.6939 | 759.9400 | 70.6800 | 0 |
| 2026-08-04 07:57:02 | cold-first | vulkan | diag-20260804-qwen35-9b-vulkan0-single-r1 | Qwen3.5-9B-Q6_K.gguf | 8192 | 8192/1024 | q8_0/q8_0 | none | 2.3053 | 645.9000 | 63.5900 | 0 |
| 2026-08-04 07:56:17 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-075537 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 5.7520 | 355.67 | 38.95 | 0 |
| 2026-08-04 07:54:29 | cold-first | vulkan | diag-20260804-q4km-vulkan-b9311-graphicsq-ctx131k-p60k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4261 | 401.9200 | 23.4800 | 0 |
| 2026-08-04 07:51:31 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-075107 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 14.4600 | 1312.90 | 34.75 | 0 |
| 2026-08-04 07:50:41 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260804-075002 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 5.7409 | 355.30 | 38.78 | 0 |
| 2026-08-04 07:49:22 | cold-first | vulkan | diag-20260804-q4km-vulkan-b9311-preserved-ctx131k-p60k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4102 | 389.2600 | 17.0000 | 0 |
| 2026-08-04 07:43:53 | cold-first | vulkan | diag-20260804-q4km-vulkan-current-output1-ctx131k-p60k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4233 | 401.1900 | 18.5300 | 0 |
| 2026-08-04 07:37:17 | cold-first | - | diag-20260804-q4km-vulkan-pre-d4-ctx131k-p60k-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4171 | 396.0500 | 16.9400 | 0 |
| 2026-08-03 21:40:13 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260803-213822 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 1.5035 | 399.56 | 19.71 | 0 |
| 2026-08-03 21:32:30 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260803-213048 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 1.5183 | 400.16 | 22.05 | 0 |
| 2026-08-03 21:21:34 | cold-first | vulkan | resume-20260803-q4km-131k-p60k-vulkan-gui-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4059 | 382.4300 | 23.8100 | 0 |
| 2026-08-03 21:15:51 | cold-first | vulkan | resume-20260803-q4km-131k-p60k-vulkan-auto-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4342 | 409.2900 | 24.9800 | 0 |
| 2026-08-03 21:10:44 | cold-first | vulkan | resume-20260803-q4km-131k-p60k-vulkan-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.4255 | 403.6500 | 18.0300 | 0 |
| 2026-08-03 21:05:36 | cold-first | rocm | resume-20260803-q4km-131k-p60k-rocm-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 1.4071 | 1411.9800 | 18.6500 | 0 |
| 2026-07-29 10:12:10 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260729-100553 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 0.3602 | 176.09 | 6.41 | 0 |
| 2026-07-20 22:45:10 | cold-first | rocm | d091-q4km-rocm98k-correct-order-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.8195 | 1426.5400 | 33.3700 | 0 |
| 2026-07-20 22:43:32 | cold-first | rocm | d091-q4km-rocm98k-correct-order-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 0.0250 | 1483.4100 | - | 0 |
| 2026-07-20 22:33:03 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260720-223143 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 1.9148 | 947.46 | 29.71 | 0 |
| 2026-07-20 22:13:36 | cold-first | rocm | d090-idle-control-postD-r1-20260720-221301 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0577 | 1721.7500 | - | 0 |
| 2026-07-20 22:12:32 | cold-first | rocm | d090-idle-q4-half2-D-r1-20260720-221156 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0519 | 1545.6200 | - | 0 |
| 2026-07-20 22:11:29 | cold-first | rocm | d090-idle-control-preD-r1-20260720-221056 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0566 | 1689.0600 | - | 0 |
| 2026-07-20 22:10:12 | cold-first | rocm | d090-idle-control-postC-r1-20260720-220937 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0576 | 1716.2300 | - | 0 |
| 2026-07-20 22:09:03 | cold-first | rocm | d090-idle-cols32-C-r1-20260720-220825 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0497 | 1482.6800 | - | 0 |
| 2026-07-20 22:07:24 | cold-first | rocm | d090-idle-control-preC-r1-20260720-220653 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0575 | 1713.3300 | - | 0 |
| 2026-07-20 22:06:25 | cold-first | rocm | d090-idle-control-postA-r1-20260720-220554 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0576 | 1717.8900 | - | 0 |
| 2026-07-20 22:05:33 | cold-first | rocm | d090-idle-chunk8192-A-r1-20260720-220501 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0559 | 1666.3200 | - | 0 |
| 2026-07-20 22:04:27 | cold-first | rocm | d090-idle-control-preA-r1-20260720-220351 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0567 | 1690.0400 | - | 0 |
| 2026-07-20 20:38:43 | cold-first | rocm | d090-q4km-q4-half2-sovereign-candidate-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0060 | 178.1300 | - | 0 |
| 2026-07-20 20:32:52 | cold-first | rocm | d090-q4km-q4-half2-postrevert-control-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.9330 | 246.1300 | 7.7800 | 0 |
| 2026-07-20 20:25:31 | cold-first | rocm | d090-q4km-q4-half2-candidate-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-07-20 20:21:48 | cold-first | rocm | d090-q4km-half2-mmq-resources-exact-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0060 | 178.4100 | - | 0 |
| 2026-07-20 20:17:47 | cold-first | rocm | d090-q4q5-half2-resource-smoke-r1 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 2.9214 | 653.0900 | - | 0 |
| 2026-07-20 20:06:22 | cold-first | rocm | d090-q4km-rocm49k-mmq-resources-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-07-20 19:53:16 | cold-first | rocm | d090-q4km-rocm49k-fa-cols32-route-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0061 | 181.3100 | - | 0 |
| 2026-07-20 19:46:46 | cold-first | rocm | d090-q4km-rocm49k-fa-phase-timing-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0056 | 165.1800 | - | 0 |
| 2026-07-20 19:33:54 | cold-first | rocm | d090-q4km-rocm49k-fa-chunk4096-postcontrol-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 3.4638 | 1516.0000 | 7.4000 | 0 |
| 2026-07-20 19:32:48 | cold-first | rocm | d090-q4km-rocm49k-fa-chunk8192-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 3.4426 | 1495.9600 | 7.4200 | 0 |
| 2026-07-20 19:31:11 | cold-first | rocm | d090-q4km-rocm49k-fa-chunk8192-route-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0109 | 324.9600 | - | 0 |
| 2026-07-20 19:11:48 | cold-first | rocm | d090-q4km-rocm49k-kernelfull-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.0219 | 650.3600 | - | 0 |
| 2026-07-20 19:10:03 | cold-first | rocm | d090-q4km-rocm49k-adjacent-control-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.0015 | 1598.4800 | 18.5700 | 0 |
| 2026-07-20 19:07:55 | cold-first | rocm | d090-q4km-rocm49k-adjacent-control-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.7682 | 1514.2200 | 17.8900 | 0 |
| 2026-07-20 12:37:14 | cold-first | vulkan | d087-vulkan-q5_1-original-long43k-r3 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q5_1/q5_1 | none | 0.4887 | 1368.0233 | 14.3700 | 0 |
| 2026-07-20 12:29:04 | cold-first | vulkan | d087-vulkan-q5_1-packedbits-long43k-r3 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q5_1/q5_1 | none | 0.5042 | 1411.6033 | 14.7200 | 0 |
| 2026-07-20 12:24:55 | cold-first | vulkan | d087-vulkan-q8_0-long43k-recontrol-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.5329 | 1498.0000 | 13.9700 | 0 |
| 2026-07-20 12:23:32 | cold-first | vulkan | d087-vulkan-q5_1-packedbits-long43k-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q5_1/q5_1 | none | 0.5161 | 1448.7300 | 13.9900 | 0 |
| 2026-07-20 12:11:11 | cold-first | vulkan | d087-vulkan-q5_1-long43k-fit-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q5_1/q5_1 | none | 0.4947 | 1383.2100 | 14.8000 | 0 |
| 2026-07-20 12:08:58 | cold-first | vulkan | d087-vulkan-q8_0-long43k-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 0.5310 | 1491.3900 | 14.1900 | 0 |
| 2026-07-20 12:07:20 | cold-first | vulkan | d087-vulkan-q5_0-long55k-fit-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 8192/1024 | q5_0/q5_0 | none | 0.4847 | 1356.7800 | 14.1400 | 0 |
| 2026-07-19 22:19:55 | cold-first | vulkan | vulkan-mtp49k-fa-bc32-control-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.7008 | 1443.1100 | 34.7400 | 0 |
| 2026-07-19 22:19:17 | cold-first | vulkan | vulkan-mtp49k-fa-bc32-candidate-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.6827 | 1403.9200 | 35.9600 | 0 |
| 2026-07-19 22:18:04 | cold-first | vulkan | vulkan-mtp49k-fa-bc32-candidate-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.6946 | 1428.4000 | 35.3300 | 0 |
| 2026-07-19 22:17:26 | cold-first | vulkan | vulkan-mtp49k-fa-bc32-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.6879 | 1417.1600 | 33.9100 | 0 |
| 2026-07-19 22:15:26 | cold-first | vulkan | vulkan-mtp49k-fa-twoquery-bc32-resource-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.0432 | 1393.5000 | - | 0 |
| 2026-07-19 22:04:23 | cold-first | vulkan | vulkan-mtp49k-fa-twoquery-resource-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.0361 | 1163.6900 | - | 0 |
| 2026-07-19 21:55:01 | cold-first | vulkan | vulkan-mtp49k-q3packed-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.0865 | 1454.9700 | 42.2700 | 0 |
| 2026-07-19 21:45:51 | cold-first | vulkan | vulkan-mtp49k-lowtile3-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.0814 | 1453.4300 | 42.2200 | 0 |
| 2026-07-19 21:45:11 | cold-first | vulkan | vulkan-mtp49k-lowtile-auto-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1686 | 1483.0000 | 42.2300 | 0 |
| 2026-07-19 21:40:58 | cold-first | vulkan | vulkan-mtp49k-post-rgp-sanity-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1559 | 1480.8200 | 41.6700 | 0 |
| 2026-07-19 21:30:03 | cold-first | vulkan | vulkan-mtp49k-rgp-dispatch-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1160 | 1467.1100 | 41.9100 | 0 |
