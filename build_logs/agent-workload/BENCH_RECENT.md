# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-07-11 12:12:31 | cold-first | vulkan | e278-vulkan-49k-n3-final-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.1209 | 1399.8500 | 38.6100 | 0 |
| 2026-07-11 12:11:41 | cold-first | vulkan | e278-vulkan-49k-n3-final-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.1194 | 1400.2900 | 38.4500 | 0 |
| 2026-07-11 12:06:57 | cold-first | vulkan | e278-vulkan-49k-none-mt128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.0979 | 1449.4100 | 29.1500 | 0 |
| 2026-07-11 12:05:59 | cold-first | vulkan | e278-vulkan-49k-n3-mt128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.1211 | 1401.1000 | 38.7800 | 0 |
| 2026-07-11 12:04:35 | cold-first | vulkan | e278-vulkan-49k-n3-warmup-keep-pp-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5496 | 1373.7600 | 22.5100 | 0 |
| 2026-07-11 12:03:19 | cold-first | vulkan | e278-vulkan-49k-n3-postspec-warmup-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5575 | 1403.5900 | 17.7100 | 0 |
| 2026-07-11 12:00:48 | cold-first | vulkan | e278-vulkan-49k-n3-shaped-warmup-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5558 | 1399.2200 | 17.6400 | 0 |
| 2026-07-11 11:57:52 | cold-first | vulkan | e278-vulkan-49k-n2-server-phase-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5480 | 1377.0500 | 18.3800 | 0 |
| 2026-07-11 11:56:39 | cold-first | vulkan | e278-vulkan-49k-n3-warmup-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5567 | 1401.5500 | 17.6200 | 0 |
| 2026-07-11 11:55:13 | cold-first | vulkan | e278-vulkan-49k-n3-keep-pp-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5499 | 1376.3700 | 21.1300 | 0 |
| 2026-07-11 11:53:57 | cold-first | vulkan | e278-vulkan-49k-n3-server-phase-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5492 | 1381.1300 | 17.9600 | 0 |
| 2026-07-11 11:50:48 | cold-first | vulkan | e278-vulkan-49k-n3-sync-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.3191 | 790.8200 | 18.0300 | 0 |
| 2026-07-11 11:49:05 | cold-first | vulkan | e278-vulkan-n3-ubatch-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 30.8241 | 325.6300 | 40.9500 | 0 |
| 2026-07-11 11:44:22 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-114341 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | none | 0.5824 | 1451.17 | 27.13 | 0 |
| 2026-07-11 11:43:17 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-114235 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.5549 | 1395.68 | 18.02 | 0 |
| 2026-07-11 11:41:31 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-114046 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.4984 | 1236.93 | 26.69 | 0 |
| 2026-07-11 11:40:01 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113919 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | ngram-mod | 0.5116 | 1276.94 | 21.13 | 0 |
| 2026-07-11 11:39:00 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113817 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | none | 0.5109 | 1276.85 | 20.27 | 0 |
| 2026-07-11 11:37:57 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113740 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | none | 2.9263 | 1535.25 | 25.61 | 0 |
| 2026-07-11 11:37:23 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113706 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | none | 3.3753 | 1752.54 | 34.73 | 0 |
| 2026-07-11 11:36:24 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113351 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 2.8988 | 1549.07 | 22.98 | 0 |
| 2026-07-11 11:33:11 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113237 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 2.7979 | 1402.56 | 38.00 | 0 |
| 2026-07-11 11:31:11 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-113003 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 2.7929 | 1403.31 | 38.06 | 0 |
| 2026-07-11 11:29:23 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-112826 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 2.1079 | 1038.13 | 37.27 | 0 |
| 2026-07-11 11:25:19 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-112500 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 1.9940 | 1006.61 | 24.36 | 0 |
| 2026-07-11 11:13:48 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-111127 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 2.8212 | 1409.00 | 41.93 | 0 |
| 2026-07-11 11:11:13 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-110947 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 2.8267 | 1413.30 | 39.67 | 0 |
| 2026-07-11 11:08:30 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-110746 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.4945 | 1229.53 | 23.55 | 0 |
| 2026-07-11 11:06:52 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-110607 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.4999 | 1242.11 | 24.61 | 0 |
| 2026-07-11 11:05:20 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-110433 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.5006 | 1245.04 | 23.71 | 0 |
| 2026-07-11 10:39:39 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-103852 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.4831 | 1202.17 | 23.09 | 0 |
| 2026-07-11 10:38:34 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-103748 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 0.4916 | 1222.56 | 23.79 | 0 |
| 2026-07-11 10:37:12 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260711-103628 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | ngram-mod | 0.5124 | 1279.82 | 20.85 | 0 |
| 2026-07-11 10:27:19 | cold-first | rocm | e277-rocm-n3-deferred-phase-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.8766 | 1396.1200 | 38.4500 | 0 |
| 2026-07-11 10:26:48 | cold-first | rocm | e277-rocm-n3-immediate-128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.9402 | 1399.8900 | 38.4900 | 0 |
| 2026-07-11 10:26:08 | cold-first | rocm | e277-rocm-n3-deferred-128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.9617 | 1399.6400 | 38.6500 | 0 |
| 2026-07-11 10:22:59 | cold-first | rocm | h60-rocm-n3-phase-process-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.7712 | 1380.1500 | 38.3600 | 0 |
| 2026-07-11 10:21:19 | cold-first | rocm | h60-rocm-n3-phase-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.7243 | 1383.0300 | 37.9300 | 0 |
| 2026-07-11 10:16:44 | cold-first | rocm | d078-rocm-131k-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 2.1682 | 1047.8400 | 24.6100 | 0 |
| 2026-07-11 10:15:19 | cold-first | rocm | d078-rocm-n2-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 21.7054 | 1381.0100 | 39.0600 | 0 |
| 2026-07-11 10:14:13 | cold-first | rocm | d078-rocm-131k-mtp-n3-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 2.1799 | 1045.6200 | 26.8500 | 0 |
| 2026-07-11 10:12:51 | cold-first | rocm | d078-rocm-131k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 2.1859 | 1088.6700 | 19.0200 | 0 |
| 2026-07-11 10:11:17 | cold-first | rocm | d078-rocm-n4-default-hybrid-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.2870 | 1367.8800 | 35.0600 | 0 |
| 2026-07-11 10:10:42 | cold-first | rocm | d078-rocm-n3-default-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 22.2686 | 1367.6000 | 41.3500 | 0 |
| 2026-07-11 10:05:46 | cold-first | rocm | d078-rocm-none-baseline-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.6056 | 1694.8200 | 25.0167 | 0 |
| 2026-07-11 10:04:22 | cold-first | rocm | d078-rocm-n3-control-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 21.2145 | 1545.7500 | 34.9167 | 0 |
| 2026-07-11 10:03:26 | cold-first | rocm | d078-rocm-n3-dp4a-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 23.4187 | 1547.2100 | 41.2500 | 0 |
| 2026-07-11 10:02:21 | cold-first | rocm | d078-rocm-n3-dp4a-256 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 22.2427 | 1369.9200 | 41.2300 | 0 |
| 2026-07-11 10:01:50 | cold-first | rocm | d078-rocm-n3-control-256 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.1896 | 1373.5200 | 34.6100 | 0 |
| 2026-07-11 10:00:30 | cold-first | rocm | d078-rocm-clean-dp4a-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 21.6685 | 1552.1200 | 36.0467 | 0 |
| 2026-07-11 09:59:30 | cold-first | rocm | d078-rocm-clean-control-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 21.4490 | 1558.0733 | 35.3567 | 0 |
| 2026-07-11 09:58:18 | cold-first | rocm | d078-rocm-clean-dp4a-256 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.6820 | 1378.7100 | 35.9100 | 0 |
| 2026-07-11 09:57:44 | cold-first | rocm | d078-rocm-clean-control-256 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.1990 | 1373.9000 | 34.5700 | 0 |
| 2026-07-11 09:54:12 | cold-first | rocm | d078-rocm-wmma-control-smoke | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.2596 | 1245.1700 | 21.8200 | 0 |
| 2026-07-11 09:53:25 | cold-first | rocm | d078-rocm-smalln-dp4a-smoke | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3466 | 1329.3100 | 24.0300 | 0 |
| 2026-07-11 09:38:59 | cold-first | rocm | e276-rocm-n4-pair-256 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.7051 | 1317.7800 | 26.1000 | 0 |
| 2026-07-11 09:38:20 | cold-first | rocm | e276-rocm-n4-control-256 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.7764 | 1323.1400 | 26.1700 | 0 |
| 2026-07-11 09:37:04 | cold-first | rocm | e276-rocm-pair-smoke | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.5943 | 786.2700 | 27.1400 | 0 |
| 2026-07-11 09:27:25 | cold-first | rocm | e276-rocm-n4-mmq-components | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.8283 | 920.5000 | 23.3200 | 0 |
| 2026-07-11 09:24:30 | cold-first | rocm | e276-rocm-n4-trace | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.5672 | 790.2800 | 19.2500 | 0 |
| 2026-07-11 09:16:45 | cold-first | rocm | e275-rocm-dual-49k-mtp-n4-clean-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 1.8828 | 1233.7900 | 26.7500 | 0 |
| 2026-07-11 09:15:43 | cold-first | rocm | e275-rocm-dual-mtp-n6-default-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 19.9504 | 1375.5200 | 33.8000 | 0 |
| 2026-07-11 09:15:05 | cold-first | rocm | e275-rocm-dual-mtp-n5-default-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.4643 | 1382.7900 | 35.1700 | 0 |
| 2026-07-11 09:14:20 | cold-first | rocm | e275-rocm-dual-mtp-n4-default-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.5689 | 1381.1600 | 35.5800 | 0 |
| 2026-07-11 09:13:34 | cold-first | rocm | e275-rocm-dual-mtp-n8-default-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 18.6803 | 1376.4200 | 30.3800 | 0 |
| 2026-07-11 09:12:26 | cold-first | vulkan | e275-vulkan-dual-49k-mtp-n2-clean-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 2.1627 | 1407.5200 | 34.0600 | 0 |
| 2026-07-11 09:11:28 | cold-first | vulkan | e275-vulkan-dual-49k-none-clean-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 2.1948 | 1448.2900 | 28.8000 | 0 |
| 2026-07-11 09:10:17 | cold-first | rocm | e275-rocm-dual-49k-mtp-n2-clean-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 1.8712 | 1232.1900 | 24.8400 | 0 |
| 2026-07-11 09:09:23 | cold-first | rocm | e275-rocm-dual-49k-none-clean-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 1.9027 | 1275.4200 | 20.8800 | 0 |
| 2026-07-11 09:08:25 | cold-first | rocm | e275-rocm-dual-mtp-n2-default-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 19.9214 | 1368.1200 | 33.8500 | 0 |
| 2026-07-11 09:06:26 | cold-first | rocm | e275-rocm-dual-mtp-n2-mmvq2warp-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.8339 | 1383.1600 | 23.4100 | 0 |
| 2026-07-11 09:04:19 | cold-first | rocm | e275-rocm-dual-mtp-n3-mmqverify-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.0174 | 1304.3500 | 35.4100 | 0 |
| 2026-07-11 09:03:35 | cold-first | rocm | e275-rocm-dual-mtp-n1-mmqverify-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 17.1566 | 1381.2000 | 26.4400 | 0 |
| 2026-07-11 09:02:39 | cold-first | rocm | e275-rocm-dual-mtp-n2-mmqverify-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 20.0600 | 1384.7700 | 34.0100 | 0 |
| 2026-07-11 09:02:02 | cold-first | rocm | e275-rocm-dual-none-clean-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.1585 | 1521.1300 | 25.2300 | 0 |
| 2026-07-11 09:00:57 | cold-first | rocm | e275-rocm-dual-mtp-n2-mmqverify-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 10.8491 | 956.2300 | 30.8900 | 0 |
| 2026-07-11 08:57:24 | cold-first | rocm | e275-rocm-dual-mtp-n2-nopp-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 9.6489 | 969.5800 | 22.3900 | 0 |
| 2026-07-11 08:55:31 | cold-first | vulkan | e275-vulkan-dual-mtp-n2-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.7868 | 1004.3600 | 47.1900 | 0 |
| 2026-07-11 08:54:55 | cold-first | vulkan | e275-vulkan-dual-none-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 12.3170 | 1039.7600 | 38.3100 | 0 |
| 2026-07-11 08:53:50 | cold-first | rocm | e275-rocm-dual-mtp-n2-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 9.5264 | 957.9800 | 22.0800 | 0 |
