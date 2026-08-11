# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-11 15:00:22 | cold-first | vulkan | d098-vk35b-32k-f8-mtp2-r1 | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | 32768 | 8192/8192 | f8_e4m3/f8_e4m3 | mtp | 1.8892 | 325.1650 | 67.3650 | 0 |
| 2026-08-11 14:57:14 | cold-first | vulkan | d098-vk35b-32k-q8-mtp2-r1 | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | 32768 | 8192/8192 | q8_0/q8_0 | mtp | 1.8473 | 317.7950 | 71.0400 | 0 |
| 2026-08-11 14:53:36 | cold-first | vulkan | d098-vk35b-32k-f8-none-r1 | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | 32768 | 8192/8192 | f8_e4m3/f8_e4m3 | none | 10.3598 | 2048.0550 | 70.1000 | 0 |
| 2026-08-11 14:52:08 | cold-first | vulkan | d098-vk35b-32k-q8-none-r1 | Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | 32768 | 8192/8192 | q8_0/q8_0 | none | 9.7268 | 1962.9700 | 61.4450 | 0 |
| 2026-08-11 13:31:43 | cold-first | vulkan | d097-smoke-98k-explicit-n8 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 0.1864 | 1629.6600 | - | 0 |
| 2026-08-11 13:31:18 | cold-first | vulkan | d097-smoke-49k-auto-n8 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 0.1898 | 1655.2600 | - | 0 |
| 2026-08-11 13:30:11 | cold-first | vulkan | d097-auto-98k-f8-n12-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.6569 | 1473.7400 | 42.7800 | 0 |
| 2026-08-11 13:26:07 | cold-first | vulkan | d097c-98k-q8-b-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 5.4432 | 1414.5300 | 41.8300 | 0 |
| 2026-08-11 13:25:00 | cold-first | vulkan | d097c-98k-f8bridge-m6-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.5687 | 1427.4500 | 47.0500 | 0 |
| 2026-08-11 13:23:53 | cold-first | vulkan | d097c-98k-f8-n12-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.7618 | 1510.9500 | 41.7900 | 0 |
| 2026-08-11 13:22:48 | cold-first | vulkan | d097c-98k-q8-a-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 5.5039 | 1430.8900 | 42.0900 | 0 |
| 2026-08-11 13:20:16 | cold-first | vulkan | d097b-98k-q8-n8-control-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 5.3810 | 1395.9800 | 41.8200 | 0 |
| 2026-08-11 13:19:08 | cold-first | vulkan | d097b-98k-f8bridge-m6-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4891 | 1401.9300 | 47.7400 | 0 |
| 2026-08-11 13:18:01 | cold-first | vulkan | d097b-98k-f8bridge-m4-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.3359 | 1389.8000 | 40.4200 | 0 |
| 2026-08-11 13:16:53 | cold-first | vulkan | d097b-98k-f8bridge-m2-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 0.8677 | 199.6800 | 41.5900 | 0 |
| 2026-08-11 13:11:37 | cold-first | vulkan | d097b-98k-f8bridge-m1-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.2171 | 1359.5900 | 39.2300 | 0 |
| 2026-08-11 13:10:27 | cold-first | vulkan | d097b-98k-f8bridge-m0-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4328 | 1432.9900 | 37.9900 | 0 |
| 2026-08-11 13:06:06 | cold-first | vulkan | d097-98k-q8-n8-b-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 5.5195 | 1435.8200 | 42.0500 | 0 |
| 2026-08-11 13:04:59 | cold-first | vulkan | d097-98k-f8-n8-b-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.5478 | 1468.9300 | 38.0400 | 0 |
| 2026-08-11 13:03:53 | cold-first | vulkan | d097-98k-f8-n16-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.9875 | 1557.4100 | 45.8800 | 0 |
| 2026-08-11 13:02:48 | cold-first | vulkan | d097-98k-f8-n14-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.8706 | 1533.7900 | 43.6700 | 0 |
| 2026-08-11 13:01:44 | cold-first | vulkan | d097-98k-f8-n12-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.7652 | 1509.3600 | 42.2700 | 0 |
| 2026-08-11 13:00:39 | cold-first | vulkan | d097-98k-f8-n10-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.5616 | 1489.0300 | 35.7700 | 0 |
| 2026-08-11 12:59:33 | cold-first | vulkan | d097-98k-f8-n8-a-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.5434 | 1471.0400 | 37.4600 | 0 |
| 2026-08-11 12:58:26 | cold-first | vulkan | d097-98k-q8-n8-a-mt256 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 5.5692 | 1448.5900 | 42.4300 | 0 |
| 2026-08-11 12:35:44 | cold-first | vulkan | d095-refresh-vk-q4km-98k-f8-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 2.9494 | 1465.9800 | 32.4000 | 0 |
| 2026-08-11 12:34:39 | cold-first | vulkan | d095-refresh-vk-q4km-98k-q8-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.9407 | 1440.0200 | 37.7900 | 0 |
| 2026-08-11 12:33:35 | cold-first | vulkan | d095-refresh-vk-q4km-98k-f8-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 2.8522 | 1480.1800 | 21.9800 | 0 |
| 2026-08-11 12:32:32 | cold-first | vulkan | d095-refresh-vk-q4km-98k-q8-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 2.6090 | 1314.5300 | 25.0700 | 0 |
| 2026-08-11 12:30:22 | cold-first | vulkan | d095-refresh-vk-q4km-49k-f8-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 6.0176 | 1679.7600 | 44.5100 | 0 |
| 2026-08-11 12:29:42 | cold-first | vulkan | d095-refresh-vk-q4km-49k-q8-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9288 | 1656.1000 | 43.5900 | 0 |
| 2026-08-11 12:29:01 | cold-first | vulkan | d095-refresh-vk-q4km-49k-f8-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 5.4627 | 1677.4200 | 25.6000 | 0 |
| 2026-08-11 12:28:21 | cold-first | vulkan | d095-refresh-vk-q4km-49k-q8-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.1146 | 1519.4200 | 27.2100 | 0 |
| 2026-08-11 12:27:02 | cold-first | vulkan | d095-refresh-vk-q4km-12k-f8-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 17.7862 | 1702.9400 | 50.6900 | 0 |
| 2026-08-11 12:26:37 | cold-first | vulkan | d095-refresh-vk-q4km-12k-q8-mtp2-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 17.0068 | 1686.7400 | 45.5500 | 0 |
| 2026-08-11 12:26:12 | cold-first | vulkan | d095-refresh-vk-q4km-12k-f8-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 14.2068 | 1791.6400 | 28.0300 | 0 |
| 2026-08-11 12:25:47 | cold-first | vulkan | d095-refresh-vk-q4km-12k-q8-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 13.7959 | 1672.1700 | 28.4700 | 0 |
| 2026-08-11 10:04:53 | cold-first | vulkan | d095-r5-d-n2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.5200 | 1602.4950 | 54.8450 | 0 |
| 2026-08-11 10:04:05 | cold-first | vulkan | d095-r5-c-n4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.8158 | 1603.5450 | 48.8100 | 0 |
| 2026-08-11 10:03:11 | cold-first | vulkan | d095-r5-b-n3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.5597 | 1608.9350 | 45.1350 | 0 |
| 2026-08-11 10:02:24 | cold-first | vulkan | d095-r5-a-n2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.6172 | 1605.3100 | 55.6250 | 0 |
| 2026-08-11 09:59:22 | cold-first | vulkan | d095-r4-n16-two-task | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 17.0252 | 1627.6750 | 58.8150 | 0 |
| 2026-08-11 09:58:31 | cold-first | vulkan | d095-r4-n10-two-task | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.1076 | 1611.8750 | 49.8000 | 0 |
| 2026-08-11 09:57:45 | cold-first | vulkan | d095-r4-n9-two-task | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.1884 | 1603.9200 | 51.2850 | 0 |
| 2026-08-11 09:56:58 | cold-first | vulkan | d095-r4-n8-two-task | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.6092 | 1605.7850 | 55.4950 | 0 |
| 2026-08-11 09:52:36 | cold-first | vulkan | d095-r3-small-cm1-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.1254 | 1589.0000 | 51.9900 | 0 |
| 2026-08-11 09:51:54 | cold-first | vulkan | d095-r3-control-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.2031 | 1595.8700 | 52.2200 | 0 |
| 2026-08-11 09:50:44 | cold-first | vulkan | d095-r3-route-small-cm1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 2.7368 | 1577.5800 | 45.7700 | 0 |
| 2026-08-11 09:34:35 | cold-first | vulkan | d095-r2-bracket-d-last | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.8752 | 1569.6500 | 50.8500 | 0 |
| 2026-08-11 09:33:53 | cold-first | vulkan | d095-r2-bracket-c-first | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.6181 | 1622.4700 | 45.1400 | 0 |
| 2026-08-11 09:33:13 | cold-first | vulkan | d095-r2-bracket-b-interleaved | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.9418 | 1616.8700 | 48.1900 | 0 |
| 2026-08-11 09:32:28 | cold-first | vulkan | d095-r2-bracket-a-last | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.3704 | 1606.4400 | 52.9600 | 0 |
| 2026-08-11 09:26:56 | cold-first | vulkan | d095-r2-f8-hybrid-n12-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.9734 | 1623.1800 | 48.0700 | 0 |
| 2026-08-11 09:25:25 | cold-first | vulkan | d095-r1-f8-scalar-direct-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.4313 | 1612.8800 | 53.4300 | 0 |
| 2026-08-11 09:24:45 | cold-first | vulkan | d095-r1-control-postbuild-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.4664 | 1613.0800 | 53.4000 | 0 |
| 2026-08-11 09:23:42 | cold-first | vulkan | d095-r1-route-f8-direct | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 2.7792 | 1600.8100 | 48.1100 | 0 |
| 2026-08-11 09:18:53 | cold-first | vulkan | d096-r-route-f8mtp-control | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 2.7931 | 1605.9500 | 48.4200 | 0 |
| 2026-08-10 23:11:09 | cold-first | vulkan | d096-ab-f16-mtp-ctl | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f16/f16 | mtp | 17.2897 | 1673.4850 | 58.1400 | 0 |
| 2026-08-10 23:10:06 | cold-first | vulkan | d096-ab-mtp-f8hyb-vs-f16 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.3080 | 1618.6750 | 51.6900 | 0 |
| 2026-08-10 22:59:16 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260810-225514 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 3.1590 | 1531.49 | 45.95 | 0 |
| 2026-08-10 16:34:19 | cold-first | vulkan | d096-dec-mtp-default-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.2292 | 1602.7600 | 51.6800 | 0 |
| 2026-08-10 16:16:57 | cold-first | vulkan | d096-enc-p5-49k-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 512/256 | f8_e4m3/f8_e4m3 | none | 0.5344 | 1398.9900 | 24.1600 | 0 |
| 2026-08-10 16:15:32 | cold-first | vulkan | d096-enc-p5-12k-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | none | 2.6926 | 1621.0700 | 27.3500 | 0 |
| 2026-08-10 16:14:15 | cold-first | vulkan | d096-enc-p5-12k-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | none | 2.3057 | 1364.8700 | 27.7900 | 0 |
| 2026-08-10 15:58:17 | cold-first | vulkan | d096-dec-mtp-lf16-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 16.4191 | 1602.4000 | 53.9800 | 0 |
| 2026-08-10 15:57:26 | cold-first | vulkan | d096-dec-mtp-ctl-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 13.3235 | 1607.2500 | 30.4800 | 0 |
| 2026-08-10 15:56:37 | cold-first | vulkan | d096-dec-mtp-lf8-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.9143 | 1609.2400 | 48.2700 | 0 |
| 2026-08-10 15:55:48 | cold-first | vulkan | d096-dec-mtp-lf4-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 15.6901 | 1603.1300 | 46.7800 | 0 |
| 2026-08-10 15:53:21 | cold-first | vulkan | d096-dec-mtp-lastf16-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 14.9992 | 1604.8200 | 41.0400 | 0 |
| 2026-08-10 15:46:04 | cold-first | vulkan | d096-dec-mtp-lastf16-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 13.4632 | 1592.8500 | 31.4900 | 0 |
| 2026-08-10 15:14:57 | cold-first | vulkan | d096-rt-p5-12k-b | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | none | 2.6730 | 1605.8300 | 28.2100 | 0 |
| 2026-08-10 15:13:26 | cold-first | vulkan | d096-rt-p5-12k | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | none | 2.6744 | 1606.3200 | 27.5800 | 0 |
| 2026-08-10 15:12:19 | cold-first | vulkan | d096-rt-f16-12k | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f16/f16 | none | 2.7307 | 1640.3500 | 28.1800 | 0 |
| 2026-08-10 15:05:17 | cold-first | vulkan | d096-dec-f16kv-49k-r1 | Qwen3.6-27B-Q4_K_M.gguf | 131072 | 512/128 | f16/f16 | none | 0.5271 | 1376.1300 | 26.1000 | 0 |
| 2026-08-10 15:03:50 | cold-first | vulkan | d096-dec-ctl-mtp-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 13.4529 | 1403.8700 | 38.2800 | 0 |
| 2026-08-10 15:03:05 | cold-first | vulkan | d096-dec-p4-mtp-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 14.1086 | 1435.6300 | 42.4600 | 0 |
| 2026-08-10 14:59:13 | cold-first | vulkan | d096-dec-p5-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | none | 17.9265 | 1630.0300 | 28.6200 | 0 |
| 2026-08-10 14:58:22 | cold-first | vulkan | d096-dec-f16kv-none-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f16/f16 | none | 17.8957 | 1607.6400 | 28.8200 | 0 |
| 2026-08-10 14:56:07 | cold-first | vulkan | d096-dec-p5-mtp-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f8_e4m3/f8_e4m3 | mtp | 13.5318 | 1608.0200 | 31.6000 | 0 |
| 2026-08-10 14:55:05 | cold-first | vulkan | d096-dec-f16kv-mtp-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/256 | f16/f16 | mtp | 16.1414 | 1578.8500 | 52.6000 | 0 |
