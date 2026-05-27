# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-05-27 15:13:37 | cold-first | vulkan | d037-vulkan130k-kq4-vq8-mixedwarn-postpatch2-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 15:11:45 | cold-first | vulkan | d037-vulkan130k-kq4-vq8-mixedwarn-postpatch-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 15:10:36 | cold-first | vulkan | d037-vulkan130k-q8kv-autoq8-last8-smoke-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q8_0/q8_0 | none | 0.0225 | 184.4200 | 1000000.0000 | 0 |
| 2026-05-27 15:09:17 | cold-first | vulkan | d037-vulkan130k-q4default-postpatch-smoke-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9480 | 1054.2800 | 40.1500 | 0 |
| 2026-05-27 15:05:35 | cold-first | vulkan | d037-vulkan130k-q8kv-directkv-last8-route-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q8_0/q8_0 | none | 0.0229 | 187.5600 | 1000000.0000 | 0 |
| 2026-05-27 15:03:20 | cold-first | vulkan | d037-vulkan130k-kq4-vq8-directkv-last3-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 15:00:15 | cold-first | vulkan | d037-vulkan130k-kq8-vq4-control-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q8_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 14:57:25 | cold-first | vulkan | d037-vulkan130k-kq4-vq8-control-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q8_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 14:55:04 | cold-first | vulkan | d037-vulkan130k-q8kv-directkv-vonly16-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q8_0/q8_0 | none | 0.3557 | 184.8200 | 25.4200 | 0 |
| 2026-05-27 14:53:48 | cold-first | vulkan | d037-vulkan130k-q8kv-directkv-konly16-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q8_0/q8_0 | none | 0.3601 | 186.4000 | 34.7400 | 0 |
| 2026-05-27 14:52:19 | cold-first | vulkan | d037-vulkan130k-q8kv-directkv-last8-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q8_0/q8_0 | none | 0.3630 | 187.9400 | 34.3600 | 0 |
| 2026-05-27 14:36:09 | cold-first | vulkan | d036-vulkan130k-default-directkv-last3-b512-ub256-r3 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9410 | 1049.2800 | 40.2033 | 0 |
| 2026-05-27 14:34:55 | cold-first | vulkan | d036-vulkan130k-default-directkv-last3-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9512 | 1055.3800 | 40.1900 | 0 |
| 2026-05-27 14:33:42 | cold-first | vulkan | d036-vulkan130k-directkv-last2-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3510 | 181.3500 | 40.7300 | 0 |
| 2026-05-27 14:32:16 | cold-first | vulkan | d036-vulkan130k-directkv-last3-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9487 | 1053.7800 | 40.1700 | 0 |
| 2026-05-27 14:31:31 | cold-first | vulkan | d036-vulkan130k-directkv-last4-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9440 | 1051.7700 | 39.6500 | 0 |
| 2026-05-27 14:30:47 | cold-first | vulkan | d036-vulkan130k-directkv2-kvhost4-first-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9378 | 1048.1800 | 39.6300 | 0 |
| 2026-05-27 14:20:57 | cold-first | vulkan | d036-vulkan130k-nohost-current-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3552 | 183.5100 | 41.4400 | 0 |
| 2026-05-27 14:18:02 | cold-first | vulkan | d036-vulkan130k-kvhost4-first-no-fa-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9159 | 1038.9800 | 37.4100 | 0 |
| 2026-05-27 14:17:16 | cold-first | vulkan | d036-vulkan130k-kvhost4-first-b512-ub256-mt64-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 6.5907 | 1037.9800 | 35.9900 | 0 |
| 2026-05-27 14:16:04 | cold-first | vulkan | d036-vulkan130k-khost8-first-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8536 | 1004.3400 | 36.8300 | 0 |
| 2026-05-27 14:14:23 | cold-first | vulkan | d036-vulkan130k-kvhost3-first-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8228 | 1025.5000 | 21.2900 | 0 |
| 2026-05-27 14:13:33 | cold-first | vulkan | d036-vulkan130k-kvhost4-first-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9349 | 1049.1300 | 37.8100 | 0 |
| 2026-05-27 14:11:19 | cold-first | vulkan | d036-vulkan130k-kvhost5-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8821 | 1020.7800 | 36.5300 | 0 |
| 2026-05-27 14:10:30 | cold-first | vulkan | d036-vulkan130k-kvhost3-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8348 | 1032.1400 | 21.4900 | 0 |
| 2026-05-27 13:19:46 | cold-first | vulkan | vscode-vulkan130k-defaultguard-b512-ub256-r2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8736 | 1014.6100 | 37.5900 | 0 |
| 2026-05-27 13:17:04 | cold-first | vulkan | vscode-vulkan130k-defaultguard-b512-ub512-lowtile2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9045 | 1078.0000 | 20.9400 | 0 |
| 2026-05-27 13:16:03 | cold-first | vulkan | vscode-vulkan130k-defaultguard-b512-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 0.3155 | 162.9900 | 37.1700 | 0 |
| 2026-05-27 13:14:08 | cold-first | vulkan | vscode-vulkan130k-defaultguard-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8727 | 1014.0800 | 37.4700 | 0 |
| 2026-05-27 13:13:33 | cold-first | vulkan | vscode-vulkan130k-quick-c24k-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8816 | 1019.7100 | 37.2200 | 0 |
| 2026-05-27 13:08:12 | cold-first | vulkan | p002-resume-d012-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3582 | 185.1100 | 41.3700 | 0 |
| 2026-05-27 12:37:14 | cold-first | vulkan | fork-d012-vulkan130k-tuned-q3-b512-ub256-r1-freshcompare | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3876 | 200.4500 | 41.5900 | 0 |
| 2026-05-27 12:34:47 | cold-first | - | upstream-b9254-vulkan130k-default-q3-b512-ub256-r1-longtimeout | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.2205 | 114.9200 | - | 0 |
| 2026-05-27 12:32:39 | cold-first | - | upstream-b9254-vulkan130k-default-q3-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 11:04:36 | cold-first | vulkan | d034-vulkan-130k-kvhost14-fulltile-lowtile1-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9695 | 1070.5400 | 37.0200 | 0 |
| 2026-05-27 11:03:31 | cold-first | vulkan | d034-vulkan-130k-kvhost14-fulltile-lowtile2-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9826 | 1078.7200 | 36.9800 | 0 |
| 2026-05-27 11:02:05 | cold-first | vulkan | d034-vulkan-130k-kvhost14-fulltile-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9718 | 1071.7200 | 37.0800 | 0 |
| 2026-05-27 11:00:04 | cold-first | vulkan | d034-vulkan-130k-kvhost16-fulltile-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9716 | 1071.7300 | 36.9800 | 0 |
| 2026-05-27 10:58:38 | cold-first | vulkan | d034-vulkan-130k-vhost32-fulltile-b1024ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 1024/512 | q4_0/q4_0 | none | 1.9695 | 1071.5900 | 36.5000 | 0 |
| 2026-05-27 10:57:03 | cold-first | vulkan | d034-vulkan-130k-vhost24-fulltile-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 0.8581 | 450.9800 | 37.1600 | 0 |
| 2026-05-27 10:55:57 | cold-first | vulkan | d034-vulkan-130k-vhost16-fulltile-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 10:54:21 | cold-first | vulkan | d034-vulkan-130k-vhost32-fulltile-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9697 | 1071.5700 | 36.5000 | 0 |
| 2026-05-27 10:52:34 | cold-first | vulkan | d034-vulkan-130k-v-host-fulltile-ub512-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/512 | q4_0/q4_0 | none | 1.9312 | 1056.0500 | 32.7100 | 0 |
| 2026-05-27 10:50:52 | cold-first | vulkan | d034-vulkan-64k-fulltile-store-r1 | Qwen3.6-27B-Q3_K_S.gguf | 65536 | 512/256 | q4_0/q4_0 | none | 1.9640 | 1060.9600 | 41.4900 | 0 |
| 2026-05-27 10:49:51 | cold-first | vulkan | d034-vulkan-130k-v-host-fulltile-store-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8199 | 991.5100 | 32.6700 | 0 |
| 2026-05-27 10:48:56 | cold-first | vulkan | d034-vulkan-130k-q3-fulltile-store-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3897 | 201.5800 | 40.8500 | 0 |
| 2026-05-27 10:12:44 | cold-first | vulkan | d034-vulkan-130k-k-dev-v-host-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.8052 | 983.2000 | 32.4700 | 0 |
| 2026-05-27 10:12:05 | cold-first | vulkan | d034-vulkan-130k-k-host-v-dev-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.7692 | 961.7000 | 32.8500 | 0 |
| 2026-05-27 10:10:10 | cold-first | vulkan | d034-vulkan-130k-kv-host-gpu-dev-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.7369 | 952.2900 | 28.2100 | 0 |
| 2026-05-27 10:07:31 | cold-first | vulkan | d034-vulkan-130k-d012-iq4nlkv-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | iq4_nl/iq4_nl | none | 0.0000 | - | - | 1 |
| 2026-05-27 10:05:53 | cold-first | vulkan | d034-vulkan-130k-setmempriority-th4096-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3680 | 191.0100 | 28.5100 | 0 |
| 2026-05-27 10:03:10 | cold-first | vulkan | d034-vulkan-130k-pageable-priority-th4096-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3676 | 190.7800 | 28.5700 | 0 |
| 2026-05-27 10:00:38 | cold-first | vulkan | d034-vulkan-130k-d012-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3590 | 186.2800 | 28.5000 | 0 |
| 2026-05-27 09:57:14 | cold-first | vulkan | d034-vulkan-130k-d012-ngl58-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.9074 | 497.2000 | 14.5200 | 0 |
| 2026-05-27 09:56:22 | cold-first | vulkan | d034-vulkan-130k-d012-ngl64-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.4087 | 212.0500 | 32.7200 | 0 |
| 2026-05-27 09:55:07 | cold-first | vulkan | d034-vulkan-130k-d012-fit-target1536-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3688 | 191.4200 | 28.5100 | 0 |
| 2026-05-27 09:52:36 | cold-first | vulkan | d034-vulkan-130k-d012-no-kv-offload-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.2095 | 712.3600 | 9.4600 | 0 |
| 2026-05-27 09:51:41 | cold-first | vulkan | d034-vulkan-64k-d012-shape-residency-probe-r1 | Qwen3.6-27B-Q3_K_S.gguf | 65536 | 512/256 | q4_0/q4_0 | none | 1.9212 | 1037.1200 | 41.0400 | 0 |
| 2026-05-27 09:51:03 | cold-first | vulkan | d034-vulkan-130k-d012-c22k-length-probe-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.4125 | 190.2900 | 28.7500 | 0 |
| 2026-05-27 09:49:39 | cold-first | vulkan | d034-vulkan-130k-memory-priority-featurefix-th4096-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3675 | 190.7400 | 28.4600 | 0 |
| 2026-05-27 09:46:20 | cold-first | vulkan | d034-vulkan-130k-d012-model-priority-th4096-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3656 | 189.7400 | 28.5300 | 0 |
| 2026-05-27 09:44:15 | cold-first | vulkan | d034-vulkan-130k-d012-perflog-fullprompt-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-05-27 09:42:52 | cold-first | vulkan | d034-vulkan-130k-d012-fitoff-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3675 | 190.7500 | 28.5600 | 0 |
| 2026-05-27 09:41:05 | repeated/steady | vulkan | d034-vulkan-130k-d012-cache-default-single-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3668 | 190.3400 | 28.7700 | 0 |
| 2026-05-27 09:39:49 | cold-first | vulkan | d034-vulkan-130k-d012-layerdisable-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3652 | 189.5300 | 28.6600 | 0 |
| 2026-05-27 09:35:11 | cold-first | vulkan | d034-vulkan-130k-d012-memory-priority-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3574 | 185.4900 | 28.2500 | 0 |
| 2026-05-27 09:30:40 | cold-first | vulkan | d034-vulkan-130k-d012-force-route-smoke-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.9356 | 198.1900 | - | 0 |
| 2026-05-27 09:29:53 | cold-first | vulkan | d034-vulkan-130k-d012-force-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3681 | 191.0700 | 28.4700 | 0 |
| 2026-05-27 09:28:10 | cold-first | vulkan | d034-vulkan-130k-d012-fresh-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3660 | 189.9200 | 28.5600 | 0 |
| 2026-05-27 01:25:25 | cold-first | rocm | d022-rocm130k-upstream-stock-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.5720 | 294.4000 | 21.9600 | 0 |
| 2026-05-27 00:54:01 | cold-first | rocm | d020-rocm130k-vbuffer-singlechunk-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 1.5067 | 798.4800 | 28.7800 | 0 |
| 2026-05-27 00:29:04 | cold-first | rocm | d018-rocm130k-q3k-dualy-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0919 | 735.6800 | 1000000.0000 | 0 |
| 2026-05-27 00:24:14 | cold-first | rocm | d016-rocm130k-q3k-padded-b4loads-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0949 | 760.2300 | 1000000.0000 | 0 |
| 2026-05-27 00:20:20 | cold-first | rocm | d015-rocm130k-dense-q3k-staging-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0855 | 684.4500 | 1000000.0000 | 0 |
| 2026-05-26 23:57:18 | cold-first | rocm | d014-rocm130k-glu-contig-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 1.5059 | 795.6200 | 28.6400 | 0 |
| 2026-05-26 23:56:31 | cold-first | rocm | d014-rocm130k-glu-contig-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0994 | 795.9200 | 1000000.0000 | 0 |
| 2026-05-26 23:53:17 | cold-first | rocm | d013-rocm130k-y32w2-postrevert-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0990 | 792.8800 | 1000000.0000 | 0 |
| 2026-05-26 23:52:04 | cold-first | rocm | d013-rocm130k-y32w2-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0828 | 662.4800 | 1000000.0000 | 0 |
| 2026-05-26 23:45:18 | cold-first | rocm | d013-rocm130k-q3k-src1quant-presync-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0809 | 648.8200 | 1000000.0000 | 0 |
| 2026-05-26 23:43:37 | cold-first | rocm | d013-rocm130k-q3k-src1quant-trace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0862 | 690.2800 | 1000000.0000 | 0 |
