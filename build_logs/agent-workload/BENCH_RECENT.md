# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-07-13 10:28:13 | cold-first | - | p003-vulkan12k-lol-q3-phasedn-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 3.0087 | 1596.0000 | 24.7100 | 0 |
| 2026-07-13 10:27:19 | cold-first | - | p003-vulkan12k-lol-q3-phasedn-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 2.7550 | 1449.3000 | 23.8900 | 0 |
| 2026-07-13 10:13:33 | cold-first | - | p003-vulkan12k-lol-submit200-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2388 | 1783.8800 | - | 0 |
| 2026-07-13 10:12:40 | cold-first | - | p003-vulkan12k-lol-submit200-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2405 | 1797.4900 | - | 0 |
| 2026-07-13 10:10:23 | cold-first | - | p003-vulkan12k-lol-q3-aprefetch-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 3.4184 | 1804.4400 | 29.1400 | 0 |
| 2026-07-13 10:09:40 | cold-first | - | p003-vulkan12k-lol-q3-aprefetch-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 3.0848 | 1609.3100 | 28.7100 | 0 |
| 2026-07-13 10:01:29 | cold-first | - | p003-vulkan12k-lol-q3-wave64-paired-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 2.9319 | 1551.9900 | 24.4100 | 0 |
| 2026-07-13 10:00:42 | cold-first | - | p003-vulkan12k-lol-q3-wave32-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 2.3189 | 1195.0600 | 23.6300 | 0 |
| 2026-07-13 09:57:05 | cold-first | - | p003-vulkan12k-lol-q3-pipeline-create-scout | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.1924 | 1435.3400 | - | 0 |
| 2026-07-13 09:54:20 | cold-first | - | p003-vulkan12k-lol-scoped-splitk-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 2.9396 | 1558.3100 | 24.3300 | 0 |
| 2026-07-13 09:48:51 | cold-first | - | p003-vulkan12k-lol-baseline-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 2.9932 | 1589.8200 | 24.3500 | 0 |
| 2026-07-13 09:46:36 | cold-first | - | p003-vulkan12k-coopmat-q3q8-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.1441 | 1072.7800 | - | 0 |
| 2026-07-13 09:41:43 | cold-first | - | p003-vulkan12k-layer-production-env-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2116 | 1580.6400 | - | 0 |
| 2026-07-13 09:40:31 | cold-first | - | p003-vulkan12k-layer-ts5_6-nooutputoverride-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2139 | 1599.9100 | - | 0 |
| 2026-07-13 09:39:00 | cold-first | - | p003-vulkan12k-layer-ts5_6-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2318 | 1735.9600 | - | 0 |
| 2026-07-13 09:38:08 | cold-first | - | p003-vulkan12k-layer-ts33_31-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2293 | 1715.5000 | - | 0 |
| 2026-07-13 09:37:02 | cold-first | - | p003-vulkan12k-layer-ts33_31-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2286 | 1713.6600 | - | 0 |
| 2026-07-13 09:35:46 | cold-first | - | p003-vulkan12k-layer-ts1_1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2330 | 1746.2100 | - | 0 |
| 2026-07-13 09:34:12 | cold-first | - | p003-vulkan12k-layer-stage-trace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.1378 | 1025.2300 | - | 0 |
| 2026-07-13 01:25:05 | cold-first | - | p003-vulkan12k-tensor-bf16-upstream-submit-out16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 1.6974 | 1031.9900 | 7.1700 | 0 |
| 2026-07-13 01:24:15 | cold-first | - | p003-vulkan12k-layer56-upstream-submit-out16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 2.9771 | 1826.4700 | 12.3900 | 0 |
| 2026-07-13 01:20:14 | cold-first | - | p003-vulkan12k-tensor-native-bf16-zerocopy-out16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 1.7183 | 1040.2100 | 7.3600 | 0 |
| 2026-07-13 01:16:54 | cold-first | - | p003-vulkan12k-tensor-partial-trace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.1392 | 1035.4300 | - | 0 |
| 2026-07-13 00:27:14 | cold-first | - | p003-vulkan12k-tensor-native-bf16-out16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 1.7134 | 1042.6700 | 7.2200 | 0 |
| 2026-07-13 00:26:26 | cold-first | - | p003-vulkan12k-tensor-native-f32-out16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 1.3385 | 809.0300 | 5.7400 | 0 |
| 2026-07-13 00:23:12 | cold-first | - | p003-vulkan12k-tensor-native-bf16-avx2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.1336 | 994.2900 | - | 0 |
| 2026-07-13 00:20:37 | cold-first | - | p003-vulkan12k-tensor-native-bf16-fixed-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0694 | 514.8000 | - | 0 |
| 2026-07-13 00:18:15 | cold-first | - | p003-vulkan12k-tensor-native-bf16-debug2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0000 | - | - | 1 |
| 2026-07-13 00:15:51 | cold-first | - | p003-vulkan12k-tensor-native-bf16-debug-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0000 | - | - | 1 |
| 2026-07-12 23:19:53 | cold-first | - | p003-vulkan12k-tensor-native-bf16-large-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0000 | - | - | 1 |
| 2026-07-12 23:07:07 | cold-first | - | p003-vulkan12k-tensor-native-phases-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0964 | 716.5600 | - | 0 |
| 2026-07-12 23:02:21 | cold-first | - | p003-vulkan12k-tensor-native-host-parupload-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0938 | 696.9900 | - | 0 |
| 2026-07-12 22:59:22 | cold-first | - | p003-vulkan12k-tensor-native-host-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0842 | 625.5700 | - | 0 |
| 2026-07-12 22:54:15 | cold-first | - | p003-vulkan12k-tensor-q3quad-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0869 | 645.2200 | - | 0 |
| 2026-07-12 22:49:49 | cold-first | - | p003-vulkan12k-tensor-route-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | f16/f16 | none | 0.0835 | 619.3900 | - | 0 |
| 2026-07-12 22:47:35 | cold-first | - | p003-vulkan12k-split65-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2253 | 1685.3200 | - | 0 |
| 2026-07-12 22:42:52 | cold-first | - | d083-vulkan12k-q3-niter2-resources-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.1519 | 1131.2100 | - | 0 |
| 2026-07-12 22:31:47 | cold-first | - | d082-vulkan12k-bn512-resources-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.1278 | 950.3500 | - | 0 |
| 2026-07-12 22:18:49 | cold-first | vulkan | p003-vulkan12k-q3quad-resources-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.2464 | 1840.5500 | - | 0 |
| 2026-07-12 22:15:40 | cold-first | vulkan | p003-vulkan12k-perf-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 0.1411 | 1050.3300 | - | 0 |
| 2026-07-12 22:14:44 | cold-first | vulkan | p003-vulkan12k-nonmtp-b1024-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 1024/1024 | q8_0/q8_0 | none | 3.4021 | 1821.1300 | 26.2800 | 0 |
| 2026-07-12 22:14:06 | cold-first | vulkan | p003-vulkan12k-nonmtp-b8192-r1 | Qwen3.6-27B-Q3_K_S.gguf | 12288 | 8192/1024 | q8_0/q8_0 | none | 3.3071 | 1754.9200 | 27.0500 | 0 |
| 2026-07-12 22:09:34 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260712-220804 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 15.6134 | 1525.07 | 38.09 | 0 |
| 2026-07-12 20:12:29 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q3_K_S_mtp-20260712-201020 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 13.7357 | 1356.50 | 32.79 | 0 |
