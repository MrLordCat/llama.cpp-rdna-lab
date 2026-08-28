# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-28 08:55:06 | cold-first | - | bisect-8d7909b33-rc-49k-q8-none-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.2571 | 1638.6500 | 21.1800 | 0 |
| 2026-08-28 08:37:50 | cold-first | - | kanon-a1e2e9eb9-rc-49k-q8-none-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.2630 | 1635.7650 | 21.2633 | 0 |
| 2026-08-28 08:21:22 | cold-first | rocm | q38-fix-rc-49k-q8-none-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.0241 | 1551.3450 | 21.4467 | 0 |
| 2026-08-28 08:14:18 | cold-first | vulkan | q38-recheck-vk-49k-q8-mtp2-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9106 | 1647.4500 | 45.3017 | 0 |
| 2026-08-28 08:11:20 | cold-first | vulkan | q38-recheck-vk-49k-q8-none-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.2028 | 1565.3417 | 26.7283 | 0 |
| 2026-08-28 08:07:24 | cold-first | rocm | q38-recheck-rc-49k-q8-mtp2-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.3534 | 1535.2767 | 34.2417 | 0 |
| 2026-08-28 08:04:11 | cold-first | rocm | q38-recheck-rc-49k-q8-none-r3 | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 7.9227 | 1538.6867 | 21.0217 | 0 |
| 2026-08-27 22:25:01 | cold-first | vulkan | d135-200k-116k-rpc3080-r1 | Qwen3.8-27B-Q4_K_M.gguf | 204800 | 8192/1024 | q8_0/q8_0 | mtp | 0.6796 | 635.2800 | 29.8300 | 0 |
| 2026-08-27 22:19:45 | cold-first | vulkan | d135-200k-160k-local-r1 | Qwen3.8-27B-Q4_K_M.gguf | 204800 | 8192/1024 | q8_0/q8_0 | mtp | 0.1994 | 183.7050 | 21.3300 | 0 |
| 2026-08-27 21:49:29 | cold-first | vulkan | d135-smoke-14k-rpc3080-3eq-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 10.6110 | 897.0150 | 42.8800 | 0 |
| 2026-08-27 21:47:25 | cold-first | vulkan | d135-smoke-14k-rpc3080-3eq-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 10.6058 | 894.8950 | 42.8850 | 0 |
| 2026-08-27 21:39:35 | cold-first | vulkan | d135-dx-14k-rpc3080-tlser-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 10.8254 | 974.2500 | 35.8150 | 0 |
| 2026-08-27 21:28:24 | cold-first | vulkan | d135-smoke-14k-rpc3080-q8emb-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.6145 | 1182.0850 | 38.7900 | 0 |
| 2026-08-27 21:26:53 | cold-first | vulkan | d135-smoke-14k-rpc3080-q8emb-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.6887 | 1185.9500 | 39.1150 | 0 |
| 2026-08-27 21:18:38 | cold-first | vulkan | d135-smoke-14k-rpc3080-nof16emb-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.0739 | 1229.1900 | 50.0400 | 0 |
| 2026-08-27 21:11:03 | cold-first | vulkan | d135-smoke-14k-rpc3080-q8-tl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3320 | 1130.7000 | 39.1750 | 0 |
| 2026-08-27 21:09:45 | cold-first | vulkan | d135-smoke-14k-rpc3080-q8-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.7507 | 1132.3250 | 43.5550 | 0 |
| 2026-08-27 21:08:07 | cold-first | vulkan | d135-smoke-14k-rpc3080-tl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.1981 | 1089.5150 | 41.6500 | 0 |
| 2026-08-27 21:06:39 | cold-first | vulkan | d135-smoke-14k-rpc3080-pinh-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.4268 | 1079.2200 | 45.5950 | 0 |
| 2026-08-27 20:58:48 | cold-first | vulkan | d135-local-rpc-mtconv-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.8541 | 1558.3200 | 52.8050 | 0 |
| 2026-08-27 20:57:40 | cold-first | vulkan | d135-local-rpc-mtconv-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.7580 | 1545.8450 | 52.9000 | 0 |
| 2026-08-27 20:53:25 | cold-first | vulkan | d135-local-rpc-masknull-pinh-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 9.5743 | 1555.0950 | 15.6100 | 0 |
| 2026-08-27 20:50:08 | cold-first | vulkan | d135-local-rpc-pinhhost-tl-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.5406 | 1505.0650 | 44.5150 | 0 |
| 2026-08-27 20:44:25 | cold-first | vulkan | d135-local-rpc-pinhhost-tl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.7030 | 1526.1600 | 44.6350 | 0 |
| 2026-08-27 20:33:27 | cold-first | vulkan | d135-smoke-14k-rpc3080-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.5410 | 1083.2650 | 46.8150 | 0 |
| 2026-08-27 20:23:09 | cold-first | vulkan | d135-local-rpc-nopin-control-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.4704 | 1023.2450 | 53.1100 | 0 |
| 2026-08-27 20:21:40 | cold-first | vulkan | d135-local-norpc-control-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.9215 | 1643.1650 | 48.8600 | 0 |
| 2026-08-27 20:20:28 | cold-first | vulkan | d135-local-rpc-pinhhost-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.4335 | 1513.7600 | 51.9000 | 0 |
| 2026-08-27 20:18:55 | cold-first | vulkan | d135-local-rpc-pinhhost-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.6118 | 1517.0250 | 53.6350 | 0 |
| 2026-08-27 20:12:32 | cold-first | vulkan | d135-local-rpc-masknull-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 8.2378 | 1045.7600 | 16.2500 | 0 |
| 2026-08-27 19:55:01 | cold-first | vulkan | d135-local-rpc-split-tl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.8129 | 1004.0000 | 45.0550 | 0 |
| 2026-08-27 19:48:00 | cold-first | vulkan | d135-local-norpc-split-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.4262 | 1613.2400 | 46.3600 | 0 |
| 2026-08-27 19:34:54 | cold-first | vulkan | d135-local-rpc-f16emb-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3384 | 1014.8700 | 52.5450 | 0 |
| 2026-08-27 19:15:03 | cold-first | vulkan | d135-local-rpc-tl-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3537 | 1022.3500 | 50.7600 | 0 |
| 2026-08-27 19:12:34 | cold-first | vulkan | d135-local-norpc-control-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.9107 | 1642.8500 | 49.0000 | 0 |
| 2026-08-27 19:06:02 | cold-first | vulkan | d134-r2-q8-threads1-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 13.3486 | 1151.2600 | 49.1700 | 0 |
| 2026-08-27 19:04:35 | cold-first | vulkan | d134-r2-q8-threads4-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.1433 | 1237.7500 | 49.7550 | 0 |
| 2026-08-27 19:03:02 | cold-first | vulkan | d134-r2-q8-threads16-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 13.9699 | 1236.0900 | 48.0500 | 0 |
| 2026-08-27 18:46:43 | cold-first | vulkan | d134-r2-q8-async-only-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.0282 | 1229.9050 | 49.2850 | 0 |
| 2026-08-27 18:45:36 | cold-first | vulkan | d134-r2-q8-runahead-only-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 13.8406 | 1238.5900 | 46.3650 | 0 |
| 2026-08-27 18:44:24 | cold-first | vulkan | d134-r2-q8-runahead-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 13.7120 | 1223.7100 | 45.9750 | 0 |
| 2026-08-27 18:39:02 | cold-first | vulkan | d134-r2-q8-wire-diag | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.0370 | 1245.6800 | 47.8400 | 0 |
| 2026-08-27 18:37:23 | cold-first | vulkan | d134-r2-f8-wire-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 13.0656 | 1167.9500 | 44.1600 | 0 |
| 2026-08-27 18:35:37 | cold-first | vulkan | d134-r2-q8-wire-r2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.3348 | 1278.7850 | 48.1850 | 0 |
| 2026-08-27 18:34:11 | cold-first | vulkan | d134-r2-q8-wire-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.4432 | 1296.1050 | 48.1750 | 0 |
| 2026-08-27 18:32:53 | cold-first | vulkan | d134-r2-f16-control-r1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.9731 | 1219.3950 | 39.1850 | 0 |
| 2026-08-27 18:18:00 | cold-first | vulkan | d133-r2-rpcdiag-srv-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.9834 | 1190.5000 | 32.7550 | 0 |
| 2026-08-27 18:07:13 | cold-first | vulkan | d133-r2-rpcdiag-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.2635 | 1227.4150 | 32.9900 | 0 |
| 2026-08-27 18:02:48 | cold-first | vulkan | d133-r2-rpc3080-dualvulkan-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 13.0091 | 1217.0150 | 39.8750 | 0 |
| 2026-08-27 18:01:11 | cold-first | vulkan | d133-r1-rpc3080-vulkan1-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 10.9461 | 933.0500 | 41.1650 | 0 |
| 2026-08-27 17:59:00 | cold-first | vulkan | d133-l0-local-dual-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.8235 | 1636.5400 | 48.6700 | 0 |
| 2026-08-27 16:19:32 | cold-first | vulkan | d132-revert-smoke | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.7894 | 1198.7800 | 38.9750 | 0 |
| 2026-08-26 18:54:42 | cold-first | vulkan | mmapfix-27b-load-r2 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 2.6571 | 1636.5050 | 26.8800 | 0 |
| 2026-08-26 18:51:36 | cold-first | vulkan | mmapfix-27b-load-smoke | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | none | 2.6444 | 1628.6550 | 26.7750 | 0 |
| 2026-08-26 18:35:54 | cold-first | vulkan | mmapfix-smoke-9b-ngl10 | Qwen3.5-9B-Q5_K_M.gguf | 8192 | 2048/256 | q8_0/q8_0 | none | 2.9889 | 299.8650 | 6.1800 | 0 |
| 2026-08-26 18:35:06 | cold-first | vulkan | mmapfix-smoke-9b-r2 | Qwen3.5-9B-Q5_K_M.gguf | 8192 | 2048/256 | q8_0/q8_0 | none | 18.6133 | 1661.2650 | 61.8050 | 0 |
| 2026-08-26 18:30:18 | cold-first | vulkan | mmapfix-smoke-9b-ngl999 | Qwen3.5-9B-Q5_K_M.gguf | 8192 | 2048/256 | q8_0/q8_0 | none | 18.5917 | 1656.1850 | 61.6600 | 0 |
| 2026-08-26 17:55:49 | cold-first | vulkan | rf64-local-94k-final-control | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.7216 | 1443.8450 | 50.6700 | 0 |
| 2026-08-26 17:52:08 | cold-first | vulkan | rf63-94k-q8-t16-runahead-r2 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.1292 | 1062.3450 | 38.4550 | 0 |
| 2026-08-26 17:49:25 | cold-first | vulkan | rf62-94k-q8-t16-runahead | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.1334 | 1065.3900 | 40.4850 | 0 |
| 2026-08-26 17:46:44 | cold-first | vulkan | rf61-14k-q8-t16-runahead | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 4.0496 | 1239.9050 | 42.5400 | 0 |
| 2026-08-26 17:45:32 | cold-first | vulkan | rf60-14k-q8-wire-t16 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 4.0350 | 1232.7300 | 43.4700 | 0 |
| 2026-08-26 17:44:27 | cold-first | vulkan | rf59-94k-q8-adjacent-f16-control | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.0816 | 1016.8150 | 37.5900 | 0 |
| 2026-08-26 17:41:40 | cold-first | vulkan | rf58-94k-q8-wire-t8 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.1001 | 1036.1900 | 38.6650 | 0 |
| 2026-08-26 17:38:52 | cold-first | vulkan | rf57-14k-q8-wire-t8 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 4.0047 | 1234.0850 | 40.0550 | 0 |
| 2026-08-26 17:35:41 | cold-first | vulkan | rf56-14k-q8-wire | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 3.6748 | 1119.0750 | 40.7250 | 0 |
| 2026-08-26 17:27:41 | cold-first | vulkan | rf55-14k-direct-response | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 3.9369 | 1196.4900 | 44.7000 | 0 |
| 2026-08-26 17:24:38 | cold-first | vulkan | rf53-14k-source-copy-trace | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 0.1343 | 1191.9700 | - | 0 |
| 2026-08-26 17:23:36 | cold-first | vulkan | rf52-14k-source-copy-sync | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 3.9586 | 1206.8850 | 43.8200 | 0 |
| 2026-08-26 17:20:44 | cold-first | vulkan | rf51-14k-p0-ub2048 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/2048 | q8_0/q8_0 | mtp | 3.3345 | 1016.0250 | 40.4550 | 0 |
| 2026-08-26 17:19:40 | cold-first | vulkan | rf50-14k-p0-control-ub1024 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 3.9407 | 1197.5350 | 44.1350 | 0 |
| 2026-08-26 15:18:48 | cold-first | vulkan | rf49-94k-ts075145 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.1006 | 1041.7850 | 31.5150 | 0 |
| 2026-08-26 15:16:01 | cold-first | vulkan | rf48-94k-ts0715 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.1057 | 1047.1350 | 30.8950 | 0 |
| 2026-08-26 15:13:19 | cold-first | vulkan | rf47-94k-ts0814 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.0960 | 1029.6400 | 39.3600 | 0 |
| 2026-08-26 15:08:59 | cold-first | vulkan | rf46-14k-asynccopy | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 3.1628 | 958.3800 | 35.8000 | 0 |
| 2026-08-26 15:07:07 | cold-first | vulkan | rf45-14k-maskfix | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 0.6465 | 731.0450 | 28.9500 | 0 |
| 2026-08-26 15:03:44 | cold-first | vulkan | rf44-14k-splittiming | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 0.7349 | 831.9750 | 30.9650 | 0 |
| 2026-08-26 15:01:58 | cold-first | vulkan | rf43-94k-runahead | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.0449 | 984.6350 | 33.7000 | 0 |
| 2026-08-26 14:59:08 | cold-first | vulkan | rf42-14k-runahead | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 6.9101 | 1167.1000 | 40.2150 | 0 |
| 2026-08-26 14:53:47 | cold-first | vulkan | rf41-14k-tl | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 1.0177 | 1166.2150 | 31.0050 | 0 |
