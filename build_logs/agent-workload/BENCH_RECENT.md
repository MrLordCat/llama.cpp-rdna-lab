# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-07-19 21:27:13 | cold-first | vulkan | vulkan-mtp49k-rgp-dispatch-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1085 | 1475.2400 | 39.9500 | 0 |
| 2026-07-19 21:25:37 | cold-first | vulkan | vulkan-mtp49k-rgp-dispatch-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1093 | 1473.5300 | 40.2300 | 0 |
| 2026-07-19 21:05:35 | cold-first | vulkan | gui-vulkan-mtp-exact-cli-replay-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1419 | 1486.1800 | 40.4100 | 0 |
| 2026-07-19 20:57:00 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260719-205617 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 4.9933 | 1444.96 | 38.39 | 0 |
| 2026-07-16 13:16:26 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260716-131457 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 4.6949 | 1334.39 | 40.79 | 0 |
| 2026-07-16 13:14:16 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260716-131304 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.5683 | 1669.63 | 34.64 | 0 |
| 2026-07-16 13:12:12 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260716-131028 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.6637 | 1672.18 | 38.35 | 0 |
| 2026-07-16 13:00:03 | cold-first | rocm | e357-q4km-rocm-dual-short-mtp3-current-r3 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 27.4421 | 1671.6900 | 47.1733 | 0 |
| 2026-07-16 12:58:32 | cold-first | rocm | e357-q4km-rocm-dual-long30k-mtp3-current-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 6.2802 | 1731.7100 | 39.5750 | 0 |
| 2026-07-16 12:57:07 | cold-first | rocm | e356-q4km-rocm-dual-short-q5-nwarps4-r3 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.9298 | 1806.5933 | 23.9800 | 0 |
| 2026-07-16 12:54:10 | cold-first | rocm | e355-q4km-rocm-dual-short-q5-default-control-r3 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 18.1141 | 1802.4100 | 24.3400 | 0 |
| 2026-07-16 12:53:04 | cold-first | rocm | e355-q4km-rocm-dual-short-q5-smallk-candidate-r3 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.7337 | 1806.5200 | 23.6400 | 0 |
| 2026-07-16 12:50:22 | cold-first | rocm | e354-q4km-rocm-dual-q5-mmvq-resource-trace-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 5.8980 | 1579.1000 | 24.0300 | 0 |
| 2026-07-16 12:47:25 | cold-first | rocm | e353-q4km-rocm-dual-long30k-q6-new-nosmallk-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.6829 | 1778.5900 | 21.9750 | 0 |
| 2026-07-16 12:46:19 | cold-first | rocm | e353-q4km-rocm-dual-long30k-q6-old-smallk-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.6159 | 1777.2850 | 21.0450 | 0 |
| 2026-07-16 12:44:44 | cold-first | rocm | e352-q4km-rocm-dual-short-q6-new-nosmallk-r3 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.9169 | 1798.0267 | 24.0000 | 0 |
| 2026-07-16 12:43:30 | cold-first | rocm | e352-q4km-rocm-dual-short-q6-old-smallk-r3 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.1514 | 1800.2200 | 22.6600 | 0 |
| 2026-07-16 12:36:42 | cold-first | rocm | e351-rocm0-q6k-nosmallk-nwarps4-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 29.0214 | 3259.1500 | 68.4533 | 0 |
| 2026-07-16 12:34:36 | cold-first | rocm | e350-rocm0-q6k-force-old-smallk-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 29.0372 | 3268.2133 | 66.7433 | 0 |
| 2026-07-16 12:34:13 | cold-first | rocm | e350-rocm0-q6k-default-nosmallk-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 29.0777 | 3263.8467 | 68.5133 | 0 |
| 2026-07-16 12:32:18 | cold-first | rocm | e349-rocm0-q6k-4k-ub1024-control-repeat-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 28.7796 | 3251.3500 | 66.6100 | 0 |
| 2026-07-16 12:31:51 | cold-first | rocm | e349-rocm0-q6k-4k-ub1024-disable-smallk-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 28.7013 | 3236.0867 | 68.3233 | 0 |
| 2026-07-16 12:30:18 | cold-first | rocm | e348-rocm0-q6k-decode-mmvq-trace-r1 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 20.7927 | 331.2900 | 60.7400 | 0 |
| 2026-07-16 12:29:12 | cold-first | rocm | e347-rocm0-q6k-4k-ub1024-force-mmq-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 22.6747 | 2045.8033 | 66.1333 | 0 |
| 2026-07-16 12:28:32 | cold-first | rocm | e347-rocm0-q6k-4k-ub1024-force-mmq-r1 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 22.8669 | 2049.5000 | 65.9000 | 0 |
| 2026-07-16 12:27:26 | cold-first | rocm | e346-rocm0-q6k-4k-ub1024-route-mmq-r1 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 13.7374 | 2064.3700 | 65.6000 | 0 |
| 2026-07-16 12:26:45 | cold-first | rocm | e345-rocm0-q6k-4k-ub1024-prompt-control-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 28.2802 | 3219.1933 | 65.9267 | 0 |
| 2026-07-16 12:25:30 | cold-first | rocm | e345-rocm0-q6k-4k-ub1024-control-r3 | Qwen3.5-9B-Q6_K.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 39.9950 | 1319.8000 | 64.3467 | 0 |
| 2026-07-16 12:00:26 | cold-first | rocm | e343-rocm0-q3ks-4k-ub1024-geometry-negative-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 0.4856 | 948.6100 | - | 0 |
| 2026-07-16 12:00:08 | cold-first | rocm | e343-rocm0-q4ks-4k-ub1024-q4q5-wide-kernelfull-r1 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 5.5774 | 915.9600 | 22.1300 | 0 |
| 2026-07-16 11:59:05 | cold-first | rocm | e343-rocm0-q4ks-4k-ub1024-q4q5-wide-r3 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 11.5169 | 1178.9067 | 29.0533 | 0 |
| 2026-07-16 11:58:44 | cold-first | rocm | e343-rocm0-q4ks-4k-ub1024-q4q5-wide-path-r1 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 0.5855 | 1145.6900 | - | 0 |
| 2026-07-16 11:54:11 | cold-first | rocm | e342-rocm0-q4ks-4k-ub1024-y64w4-control-r3 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 11.1575 | 1126.1100 | 28.7167 | 0 |
| 2026-07-16 11:53:11 | cold-first | rocm | e342-rocm0-q4ks-4k-ub1024-y128w8-r3 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 11.2848 | 1148.9300 | 28.5367 | 0 |
| 2026-07-16 11:52:30 | cold-first | rocm | e342-rocm0-q4ks-4k-ub1024-y128w8-r1 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 11.5257 | 1176.6300 | 28.7400 | 0 |
| 2026-07-16 11:49:10 | cold-first | rocm | e341-rocm0-q4ks-4k-ub1024-x112-r1 | Qwen3.6-27B-Q4_K_S.gguf | 4096 | 4096/1024 | q4_0/q4_0 | none | 11.0075 | 1101.5400 | 28.6800 | 0 |
