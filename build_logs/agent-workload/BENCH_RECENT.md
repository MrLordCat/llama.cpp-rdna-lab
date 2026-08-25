# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-25 13:37:57 | cold-first | vulkan | r44-3gpu-160k-long-mtp2-output-vk1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 0.6731 | 512.0600 | 25.1900 | 0 |
| 2026-08-25 13:33:35 | cold-first | vulkan | r43-3gpu-160k-none-output-vk1-trace | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2620 | 787.4100 | 18.8400 | 0 |
| 2026-08-25 13:28:33 | cold-first | vulkan | r42-3gpu-160k-none-trace | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2671 | 804.1600 | 16.7800 | 0 |
| 2026-08-25 13:24:11 | cold-first | vulkan | r41-local-160k-long-none | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2503 | 187.2600 | 22.1100 | 0 |
| 2026-08-25 13:14:07 | cold-first | vulkan | r40-3gpu-160k-long-none | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 1.0260 | 809.3200 | 16.9100 | 0 |
| 2026-08-25 13:00:25 | cold-first | vulkan | r39-3gpu-160k-long-mtp2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 1.0477 | 816.9400 | 21.0000 | 0 |
| 2026-08-25 12:57:27 | cold-first | vulkan | r38-local-160k-long-mtp2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 1.5643 | 1213.8800 | 35.5000 | 0 |
| 2026-08-25 12:54:55 | cold-first | vulkan | r37-local-160k-mtp2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 15.2102 | 1540.7500 | 48.5900 | 0 |
| 2026-08-23 14:04:35 | cold-first | vulkan | r36-loop-hashfix-clean | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.1879 | 1384.7500 | 40.0250 | 0 |
| 2026-08-23 14:03:09 | cold-first | vulkan | r35-loop-hashfix | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.1176 | 1367.6600 | 22.8550 | 0 |
| 2026-08-23 13:56:40 | cold-first | vulkan | r34-loop-dbgcache | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 5.8876 | 1377.6300 | 8.0550 | 0 |
| 2026-08-23 13:52:29 | cold-first | vulkan | r33-loop-nocache | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.0521 | 1375.8950 | 39.3900 | 0 |
| 2026-08-23 13:22:26 | cold-first | vulkan | r32-loop-specnone | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 12.1378 | 1415.9250 | 26.3150 | 0 |
| 2026-08-23 13:14:27 | cold-first | vulkan | r31-local-no-rpc | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.4219 | 1594.8200 | 40.1450 | 0 |
| 2026-08-23 13:12:30 | cold-first | vulkan | r30-loop-nopin | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 9.2192 | 1366.6600 | 16.0650 | 0 |
| 2026-08-23 13:09:53 | cold-first | vulkan | r29-loop-nohandoff | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 8.7244 | 1271.1600 | 15.3350 | 0 |
| 2026-08-23 13:03:58 | cold-first | vulkan | r28-loop-devd | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 8.2902 | 1301.8450 | 13.8300 | 0 |
| 2026-08-23 12:54:53 | cold-first | vulkan | r27-loop-dbg | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 3.3135 | 1360.6200 | 8.7100 | 0 |
| 2026-08-23 12:51:55 | cold-first | vulkan | r26-loop-dbg | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 2.0533 | 785.8350 | 6.1550 | 0 |
| 2026-08-23 12:44:16 | cold-first | vulkan | r25-loop-mtp | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 9.0351 | 1365.2500 | 15.5000 | 0 |
| 2026-08-23 12:34:36 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r23-check | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 5.6563 | 1080.0850 | 8.4200 | 0 |
| 2026-08-23 12:29:45 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r22-dbg | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 4.4000 | 1074.9200 | 9.0900 | 0 |
| 2026-08-23 12:25:53 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r21-check | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 5.6901 | 1087.4450 | 8.4800 | 0 |
| 2026-08-23 12:19:41 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r20-fix | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 5.6791 | 1080.3100 | 8.4750 | 0 |
| 2026-08-23 12:11:26 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r19-trace | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 5.5934 | 1056.8300 | 8.3800 | 0 |
| 2026-08-23 12:07:41 | cold-first | vulkan | q38-solo-12k-q8-mtp4-r18-trace | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.4913 | 1601.9600 | 40.3900 | 0 |
| 2026-08-23 12:04:11 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r17-diag | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 5.2708 | 1079.3700 | 7.6000 | 0 |
| 2026-08-23 12:02:00 | cold-first | vulkan | q38-solo-12k-q8-mtp4-r16 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.6217 | 1607.4550 | 41.2350 | 0 |
| 2026-08-23 11:56:34 | cold-first | vulkan | q38-3gpu-12k-q8-mtp4-r15 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.5018 | 317.4700 | 8.5100 | 1 |
| 2026-08-23 11:45:14 | cold-first | vulkan | q38-3gpu-12k-q8-none-r14-final | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 11.0629 | 1306.4150 | 23.6100 | 0 |
| 2026-08-23 11:42:51 | cold-first | vulkan | q38-3gpu-12k-q8-none-r13-ts0906 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 10.9377 | 1296.4950 | 23.2650 | 0 |
| 2026-08-23 11:40:10 | cold-first | vulkan | q38-3gpu-12k-q8-none-r12 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 10.0731 | 1204.0400 | 21.2100 | 0 |
| 2026-08-23 11:37:38 | cold-first | vulkan | q38-3gpu-12k-q8-none-r11-alloccache | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.1476 | 1025.4100 | 21.4950 | 0 |
| 2026-08-23 11:32:11 | cold-first | vulkan | q38-3gpu-12k-q8-none-r10-ubtiming | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.2460 | 838.8050 | 21.4250 | 0 |
| 2026-08-23 11:30:15 | cold-first | vulkan | q38-3gpu-12k-q8-none-r9-ts0907 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.3374 | 848.4700 | 21.6350 | 0 |
| 2026-08-23 11:27:31 | cold-first | vulkan | q38-3gpu-12k-q8-none-r8-sndbuf | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.8625 | 840.8700 | 18.9450 | 0 |
| 2026-08-23 11:20:33 | cold-first | vulkan | q38-solo-12k-q8-none-r3sync | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 9.9739 | 962.4450 | 28.3900 | 0 |
| 2026-08-23 11:18:04 | cold-first | vulkan | q38-3gpu-12k-q8-none-r7 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.7844 | 825.2800 | 18.9900 | 0 |
| 2026-08-23 11:15:22 | cold-first | vulkan | q38-solo-12k-q8-none-r2diag | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 13.4211 | 1616.2850 | 28.1100 | 0 |
| 2026-08-23 11:10:42 | cold-first | vulkan | q38-3gpu-12k-q8-none-r6 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.6695 | 806.2500 | 18.9500 | 0 |
| 2026-08-23 11:05:23 | cold-first | vulkan | q38-3gpu-12k-q8-none-r5 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.7441 | 822.9000 | 18.8550 | 0 |
| 2026-08-23 10:54:18 | cold-first | vulkan | q38-3gpu-12k-q8-none-r4 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.8046 | 814.2200 | 19.5000 | 0 |
| 2026-08-23 10:47:53 | cold-first | vulkan | q38-3gpu-12k-q8-none-r3 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.9606 | 834.4300 | 19.7250 | 0 |
| 2026-08-23 10:35:32 | cold-first | vulkan | q38-3gpu-12k-dbg-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 8.0657 | 841.0950 | 20.1550 | 0 |
| 2026-08-23 10:33:27 | cold-first | vulkan | q38-3gpu-12k-q8-none-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 7.9161 | 839.4800 | 19.3350 | 0 |
| 2026-08-23 10:32:05 | cold-first | vulkan | q38-solo-12k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 13.1151 | 1596.7950 | 27.1200 | 0 |
| 2026-08-23 09:36:56 | cold-first | vulkan | q38-3gpu-49k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 2.7438 | 770.4650 | 19.6950 | 0 |
| 2026-08-23 09:33:47 | cold-first | vulkan | q38-3gpu-12k-q8-none-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 6.4083 | 647.1750 | 19.5200 | 0 |
| 2026-08-22 22:28:59 | cold-first | vulkan | q9-16k-split5050-b2-f16act-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 2.9375 | 2312.5050 | 40.4450 | 0 |
| 2026-08-22 22:14:58 | cold-first | vulkan | q9-16k-split5050-b1-fapin-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 2.3979 | 1878.4200 | 34.7300 | 0 |
| 2026-08-22 21:59:48 | cold-first | vulkan | q9-16k-split5050-vk1-rpc3080-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.6461 | 1125.8050 | 22.1650 | 0 |
| 2026-08-22 21:47:23 | cold-first | vulkan | q9-16k-local3080-q5-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7347 | - | - | 0 |
| 2026-08-22 21:23:09 | cold-first | vulkan | q9-16k-local9070xt-q5-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 4.1149 | 3214.1150 | 70.6050 | 0 |
| 2026-08-22 21:16:23 | cold-first | vulkan | q9-16k-rpc3080-q5-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 1.9925 | 1657.2850 | 27.7100 | 0 |
| 2026-08-22 21:14:11 | cold-first | vulkan | q9-16k-rpc9070xt-q5-r1 | Qwen3.5-9B-Q5_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 3.6113 | 2846.1200 | 53.5850 | 0 |
| 2026-08-22 20:50:09 | cold-first | vulkan | rpc3080-16k-c2-r2 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.4789 | 403.1300 | 3.5000 | 0 |
| 2026-08-22 20:40:07 | cold-first | vulkan | rpc3080-16k-c2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.4773 | 401.3650 | 3.5100 | 0 |
| 2026-08-22 19:25:54 | cold-first | vulkan | rpc3080-16k-rpcdebug-r3 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7287 | 546.9050 | 23.1900 | 0 |
| 2026-08-22 19:21:23 | cold-first | vulkan | rpc3080-16k-rpcdebug-r2 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7280 | 545.9700 | 23.8700 | 0 |
| 2026-08-22 19:13:25 | cold-first | vulkan | rpc3080-16k-perf-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7178 | 538.8000 | 22.9800 | 0 |
| 2026-08-22 19:10:17 | cold-first | vulkan | rpc3080-16k-tile-t4-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7329 | 553.0700 | 22.5550 | 0 |
| 2026-08-22 19:07:40 | cold-first | vulkan | rpc3080-16k-tile-t3-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7097 | 534.5350 | 23.9200 | 0 |
| 2026-08-22 19:05:00 | cold-first | vulkan | rpc3080-16k-tile-t2-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7312 | 551.8950 | 23.4550 | 0 |
| 2026-08-22 19:02:20 | cold-first | vulkan | rpc3080-16k-tile-t1-r2 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7553 | 571.4450 | 20.8050 | 0 |
| 2026-08-22 18:58:37 | cold-first | vulkan | rpc3080-16k-tile-t1-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7171 | 538.1950 | 23.2450 | 0 |
| 2026-08-22 18:41:15 | cold-first | vulkan | rpc3080-16k-trace-r1 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 0.7145 | 536.1150 | 23.2850 | 0 |
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
