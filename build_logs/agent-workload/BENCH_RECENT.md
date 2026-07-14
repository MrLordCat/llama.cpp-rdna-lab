# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-07-14 10:54:40 | cold-first | rocm | e283-clean-rocm-long-mtp-n3-dev10-r1-fixed | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | mtp | 3.1014 | 1174.3300 | 28.8300 | 0 |
| 2026-07-14 10:53:35 | cold-first | rocm | e283-clean-rocm-long-none-dev10-r1-fixed | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 8192/1024 | q8_0/q8_0 | none | 3.0006 | 1183.4600 | 20.9500 | 0 |
| 2026-07-14 10:52:25 | cold-first | rocm | e283-clean-rocm-short-mtp-n3-dev10-r3-fixed | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 21.2573 | 1656.5267 | 34.9500 | 0 |
| 2026-07-14 10:44:05 | cold-first | rocm | e283-clean-rocm-short-none-dev10-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 17.6101 | 1706.5667 | 25.7433 | 0 |
| 2026-07-14 10:42:21 | cold-first | vulkan | e283-clean-vulkan-long-mtp-n3-dev01-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.8093 | 1557.0600 | 42.8200 | 0 |
| 2026-07-14 10:41:12 | cold-first | vulkan | e283-clean-vulkan-long-none-dev01-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.5616 | 1536.9900 | 34.4900 | 0 |
| 2026-07-14 10:39:59 | cold-first | vulkan | e283-clean-vulkan-short-none-dev01-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 16.4189 | 1783.4900 | 38.1667 | 0 |
| 2026-07-14 10:39:11 | cold-first | vulkan | e283-clean-vulkan-short-mtp-n3-dev01-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 17.9875 | 1724.7333 | 51.8233 | 0 |
| 2026-07-14 10:38:10 | cold-first | vulkan | e283-clean-vulkan-short-mtp-n3-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.9586 | 1043.4167 | 56.2667 | 0 |
| 2026-07-14 10:37:16 | cold-first | vulkan | e283-clean-vulkan-short-none-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 16.3477 | 1782.3600 | 37.8900 | 0 |
| 2026-07-14 10:12:55 | cold-first | vulkan | e282-lol-vulkan32k-mtp-default256-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1982 | 1409.6100 | 35.7200 | 0 |
| 2026-07-14 10:09:06 | cold-first | vulkan | e279-lol-vulkan32k-none-repeat-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.0871 | 1430.5200 | 28.9500 | 0 |
| 2026-07-14 10:08:17 | cold-first | vulkan | e279-lol-vulkan32k-mtp-device-window128-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.1164 | 1361.7900 | 39.4900 | 0 |
| 2026-07-14 10:07:21 | cold-first | vulkan | e279-lol-vulkan32k-mtp-device-window256-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.2621 | 1399.7000 | 40.8300 | 0 |
| 2026-07-14 10:06:31 | cold-first | vulkan | e279-lol-vulkan32k-none-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.0336 | 1409.7000 | 29.1200 | 0 |
| 2026-07-14 10:04:09 | cold-first | vulkan | e279-vulkan32k-mtp-device-window256-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.4511 | 1478.0200 | 37.4600 | 0 |
| 2026-07-14 10:03:10 | cold-first | vulkan | e279-vulkan32k-none-output1-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 5.3232 | 1515.1400 | 28.6600 | 0 |
| 2026-07-14 10:02:22 | cold-first | vulkan | e279-vulkan32k-mtp-device-window512-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.4375 | 1456.3000 | 40.4100 | 0 |
| 2026-07-14 10:00:31 | cold-first | vulkan | e279-vulkan-mtp-window512-output-local-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.9390 | 1626.0900 | 40.4200 | 0 |
| 2026-07-14 09:59:06 | cold-first | vulkan | e279-vulkan-none-pipeline-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 15.3492 | 1642.7100 | 36.3700 | 0 |
| 2026-07-14 09:58:30 | cold-first | vulkan | e279-vulkan-mtp-window512-pipeline-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 14.9512 | 1516.7000 | 38.2300 | 0 |
| 2026-07-14 09:56:31 | cold-first | vulkan | e279-vulkan-mtp-window512-no-output-retain-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.8764 | 1609.4300 | 40.6500 | 0 |
| 2026-07-14 09:52:49 | cold-first | vulkan | e279-vulkan-mtp-window512-pp-reuse-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-14 09:50:18 | cold-first | vulkan | e279-vulkan-mtp-window512-capture-gate-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 16.2346 | 1647.9700 | 41.6400 | 0 |
| 2026-07-14 09:48:51 | cold-first | vulkan | e279-vulkan-mtp-window512-stable-graph-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.2160 | 1510.1000 | 40.2300 | 0 |
| 2026-07-14 09:47:45 | cold-first | vulkan | e279-vulkan-none-output1-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 16.3420 | 1770.6400 | 38.0300 | 0 |
| 2026-07-14 09:46:57 | cold-first | vulkan | e279-vulkan-mtp-device-window512-n128 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 15.6678 | 1555.7600 | 41.4300 | 0 |
| 2026-07-14 09:45:39 | cold-first | vulkan | e279-vulkan-none-output1-control | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 3.2028 | 1764.1800 | 35.0500 | 0 |
| 2026-07-14 09:45:01 | cold-first | vulkan | e279-vulkan-mtp-device-h-on-layer | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 2.7108 | 1435.9300 | 48.0000 | 0 |
| 2026-07-14 09:43:41 | cold-first | vulkan | e279-vulkan-mtp-device-output1-trace | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 2.6324 | 1393.1900 | 46.8700 | 0 |
| 2026-07-14 09:41:29 | cold-first | vulkan | e279-vulkan-none-control | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 3.1121 | 1711.3800 | 35.2800 | 0 |
| 2026-07-14 09:40:52 | cold-first | vulkan | e279-vulkan-mtp-host-control | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.8609 | 966.9100 | 46.1300 | 0 |
| 2026-07-14 09:39:41 | cold-first | vulkan | e279-vulkan-mtp-device-handoff-smoke | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 2.6448 | 1392.8500 | 51.7900 | 0 |
| 2026-07-14 09:17:13 | cold-first | vulkan | e282-vulkan-none-long-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 4.9703 | 1471.1600 | 32.6400 | 0 |
| 2026-07-14 09:16:09 | cold-first | vulkan | e282-vulkan-mtp-windowfull-long-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 2.8790 | 767.1500 | 47.9200 | 0 |
| 2026-07-14 09:14:53 | cold-first | vulkan | e282-vulkan-mtp-window8192-long-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.2311 | 1175.1700 | 43.4500 | 0 |
| 2026-07-14 09:13:45 | cold-first | vulkan | e282-vulkan-mtp-window2048-long-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.8694 | 1392.7300 | 39.5900 | 0 |
| 2026-07-14 09:12:42 | cold-first | vulkan | e282-vulkan-mtp-window512-long-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.0840 | 1447.3100 | 42.8300 | 0 |
| 2026-07-14 09:09:43 | cold-first | rocm | e281-rocm-peer-copy-control-off-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 24.7774 | 676.9300 | 27.4700 | 0 |
| 2026-07-14 09:09:03 | cold-first | rocm | e281-rocm-peer-copy-smoke-on-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 22.3074 | 718.7200 | 24.3300 | 0 |
| 2026-07-14 08:52:20 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260714-084902 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | ngram-mtp | 4.4943 | 1315.31 | 31.46 | 0 |
| 2026-07-14 08:48:27 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260714-084532 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 5.1818 | 1486.50 | 41.59 | 0 |
| 2026-07-13 21:23:52 | cold-first | vulkan | modular-vulkan-mtp-smoke | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 55.7880 | 344.2300 | 70.7600 | 0 |
| 2026-07-13 21:22:54 | cold-first | vulkan | modular-vulkan-none-smoke | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 33.1434 | 529.8900 | 40.2500 | 0 |
| 2026-07-13 17:42:18 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260713-174042 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | ngram-mtp | 3.6345 | 1238.79 | 32.55 | 0 |
| 2026-07-13 17:40:20 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260713-173849 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | sweep/sweep | sweep/sweep | ngram-mtp | 4.1291 | 1394.62 | 40.11 | 0 |
| 2026-07-13 17:19:00 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260713-171732 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 14.0256 | 1400.62 | 32.88 | 0 |
| 2026-07-13 16:48:44 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260713-164756 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | none | 16.6843 | 1721.45 | 37.80 | 0 |
| 2026-07-13 16:47:36 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260713-164649 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 16.4476 | 1490.10 | 44.86 | 0 |
| 2026-07-13 16:46:25 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260713-164534 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 15.1961 | 1442.58 | 38.43 | 0 |
| 2026-07-13 11:56:29 | cold-first | vulkan | p003-gui-build-vulkan12k-mtp-n4-final-smoke-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.6076 | 1560.3100 | 39.8100 | 0 |
| 2026-07-13 11:54:32 | cold-first | vulkan | p003-gui-build-vulkan12k-mtp-n4-smoke-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.6069 | 1559.2700 | 40.2700 | 0 |
| 2026-07-13 11:53:25 | cold-first | vulkan | p003-gui-build-vulkan12k-mtp-n4-smoke-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3336 | 1563.7000 | 6.4900 | 0 |
| 2026-07-13 11:45:23 | cold-first | - | p003-vulkan48k-lol-none-split41-max64-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 2.2406 | 1457.3600 | 33.7000 | 0 |
| 2026-07-13 11:44:32 | cold-first | - | p003-vulkan48k-lol-mtp-n4-split41-max64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 2.2020 | 1408.4900 | 43.5200 | 0 |
| 2026-07-13 11:43:32 | cold-first | - | p003-vulkan12k-lol-mtp-n4-split41-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.6017 | 1553.5100 | 40.2800 | 0 |
| 2026-07-13 11:37:56 | cold-first | - | p003-vulkan12k-lol-mtp-n4-route-trace-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3740 | 1349.7800 | 26.7100 | 0 |
| 2026-07-13 11:30:34 | cold-first | - | p003-vulkan48k-lol-mtp-n4-tgcache2-max64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 1.6372 | 1102.3000 | 16.6300 | 0 |
| 2026-07-13 11:29:25 | cold-first | - | p003-vulkan48k-lol-mtp-n2-tgcache2-max64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 1.8776 | 1206.0800 | 34.2300 | 0 |
| 2026-07-13 11:26:07 | cold-first | - | p003-vulkan48k-lol-none-tgcache2-max128-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 3.4912 | 1231.9400 | 24.6800 | 0 |
| 2026-07-13 11:25:04 | cold-first | - | p003-vulkan48k-lol-mtp-n3-tgcache2-max128-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 3.5193 | 1185.3300 | 35.0100 | 0 |
| 2026-07-13 11:23:44 | cold-first | - | p003-vulkan48k-lol-none-tgcache2-max16-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 0.4821 | 1196.8700 | 23.1200 | 0 |
| 2026-07-13 11:22:44 | cold-first | - | p003-vulkan48k-lol-mtp-n3-tgcache2-max16-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.5019 | 1239.6200 | 32.4200 | 0 |
| 2026-07-13 11:21:37 | cold-first | - | p003-vulkan12k-lol-mtp-n3-tgcache2-debug-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.4070 | 1359.2600 | 39.0800 | 0 |
| 2026-07-13 11:19:50 | cold-first | - | p003-vulkan12k-lol-mtp-n3-tgcache-debug-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3247 | 1359.4600 | 14.3000 | 0 |
| 2026-07-13 11:18:58 | cold-first | - | p003-vulkan48k-lol-mtp-n3-tgcache-max16-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 0.4887 | 1223.6500 | 16.8300 | 0 |
| 2026-07-13 11:12:07 | cold-first | - | p003-vulkan48k-lol-none-shapewarm-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 2.1837 | 1436.9100 | 28.3400 | 0 |
| 2026-07-13 11:10:36 | cold-first | - | p003-vulkan48k-lol-mtp-n3-shapewarm-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 2.1701 | 1410.5600 | 33.0900 | 0 |
| 2026-07-13 11:09:33 | cold-first | - | p003-vulkan12k-lol-mtp-n3-shapewarm-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.4745 | 1529.4400 | 14.4400 | 0 |
| 2026-07-13 11:07:37 | cold-first | - | p003-vulkan12k-lol-mtp-n3-copygate-only-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.0481 | 1055.4200 | 13.5600 | 0 |
| 2026-07-13 11:07:02 | cold-first | - | p003-vulkan12k-lol-none-dualtopo-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 1.7388 | 1715.5400 | 31.6700 | 0 |
| 2026-07-13 11:06:36 | cold-first | - | p003-vulkan12k-lol-mtp-n3-dualtopo-warm-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.0894 | 1074.0300 | 18.9900 | 0 |
| 2026-07-13 11:04:58 | cold-first | - | p003-vulkan12k-lol-mtp-n3-copygate-warm-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.0294 | 1011.3700 | 19.0200 | 0 |
| 2026-07-13 10:52:26 | cold-first | - | p003-vulkan12k-lol-mtp-n3-reuse-debug-runs2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3588 | 1368.6050 | 18.1650 | 0 |
| 2026-07-13 10:48:55 | cold-first | - | p003-vulkan12k-lol-mtp-n3-keep-sched-runs2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3314 | 1337.2800 | 18.3550 | 0 |
| 2026-07-13 10:47:41 | cold-first | - | p003-vulkan12k-lol-mtp-n3-ubatch-runs2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.2921 | 1322.2400 | 14.5600 | 0 |
| 2026-07-13 10:45:00 | cold-first | - | p003-vulkan12k-lol-mtp-n3-pipeline-trace | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 1.3060 | 1343.7600 | 13.8100 | 0 |
| 2026-07-13 10:43:26 | cold-first | - | p003-vulkan48k-lol-mtp-n3-phase-runs2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 1.6578 | 1112.9900 | 20.6550 | 0 |
| 2026-07-13 10:41:15 | cold-first | - | p003-vulkan48k-lol-none-control-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 1.9362 | 1267.4400 | 26.7000 | 0 |
| 2026-07-13 10:40:21 | cold-first | - | p003-vulkan48k-lol-mtp-n3-phase-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 1.8682 | 1306.5900 | 14.2100 | 0 |
