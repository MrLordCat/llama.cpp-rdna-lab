# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-05-30 13:13:32 | cold-first | vulkan | d076-fa-br32-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1033 | 434.5600 | 19.3700 | 0 |
| 2026-05-30 13:03:11 | cold-first | vulkan | d076-fa-nomaskopt-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1321 | 556.3700 | 19.7100 | 0 |
| 2026-05-30 12:14:02 | cold-first | vulkan | d076-fa-shmemstage-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1032 | 433.9000 | 19.2900 | 0 |
| 2026-05-30 12:01:38 | cold-first | vulkan | d076-q3k-bigprompt-route-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.0082 | 546.5300 | 1000000.0000 | 0 |
| 2026-05-30 11:40:55 | cold-first | vulkan | d076-fa-bigprompt-cand-bc96-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1022 | 429.9000 | 18.5500 | 0 |
| 2026-05-30 11:15:57 | cold-first | vulkan | d076-fa-bigprompt-cand-bc128-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1023 | 430.5200 | 18.5900 | 0 |
| 2026-05-30 11:04:31 | cold-first | vulkan | d076-fa-bigprompt-baseline-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1367 | 575.8900 | 19.7100 | 0 |
| 2026-05-29 08:04:30 | cold-first | vulkan | vln75k-max16-cand-b480-ub256-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 480/256 | q4_0/q4_0 | none | 0.1286 | 542.0100 | 18.6000 | 0 |
| 2026-05-29 08:01:46 | cold-first | vulkan | vln75k-max16-cand-b544-ub256-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 544/256 | q4_0/q4_0 | none | 0.1279 | 538.9400 | 19.1600 | 0 |
| 2026-05-29 07:58:57 | cold-first | vulkan | vln75k-max16-cand-b512-ub248-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/248 | q4_0/q4_0 | none | 0.1229 | 517.3600 | 19.2900 | 0 |
| 2026-05-29 07:56:01 | cold-first | vulkan | vln75k-max16-cand-b512-ub240-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/240 | q4_0/q4_0 | none | 0.1227 | 516.7300 | 19.2900 | 0 |
| 2026-05-28 21:30:17 | cold-first | vulkan | vln75k-max16-cand-b512-ub288-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/288 | q4_0/q4_0 | none | 0.1166 | 490.9600 | 19.7000 | 0 |
| 2026-05-28 21:27:07 | cold-first | vulkan | vln75k-max16-cand-b576-ub256-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 576/256 | q4_0/q4_0 | none | 0.1284 | 540.8000 | 20.1100 | 0 |
| 2026-05-28 21:24:18 | cold-first | vulkan | vln75k-max16-cand-b512-ub256-mmap-r2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1385 | 584.0100 | 20.0400 | 0 |
| 2026-05-28 21:21:38 | cold-first | vulkan | vln75k-max16-cand-b512-ub256-mmap-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1386 | 584.2400 | 20.0700 | 0 |
| 2026-05-28 21:18:46 | cold-first | vulkan | vln75k-max16-cand-b512-ub320-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/320 | q4_0/q4_0 | none | 0.1160 | 488.2900 | 20.3100 | 0 |
| 2026-05-28 21:16:02 | cold-first | vulkan | vln75k-max16-cand-b512-ub224-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/224 | q4_0/q4_0 | none | 0.1160 | 488.4500 | 19.9400 | 0 |
| 2026-05-28 21:12:48 | cold-first | vulkan | vln75k-max16-cand-b640-ub256-r2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 640/256 | q4_0/q4_0 | none | 0.1264 | 532.6900 | 19.4000 | 0 |
| 2026-05-28 21:10:15 | cold-first | vulkan | vln75k-max16-base-b512-ub256-r2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1368 | 576.6400 | 19.8900 | 0 |
| 2026-05-28 20:58:35 | cold-first | vulkan | vln75k-cand-b384-ub192-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 384/192 | q4_0/q4_0 | none | 0.0071 | 476.0000 | 1000000.0000 | 0 |
| 2026-05-28 20:55:50 | cold-first | vulkan | vln75k-baseline-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.0086 | 576.6300 | 1000000.0000 | 0 |
| 2026-05-28 17:43:20 | cold-first | vulkan | q4fitauto-vulkan130k-big-c152k-b512-ub256-r2 | Qwen3.6-27B-Q4_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1166 | 423.2600 | 4.2700 | 0 |
| 2026-05-28 13:04:07 | cold-first | vulkan | q4fitauto-vulkan130k-big-c152k-b512-ub256-r1 | Qwen3.6-27B-Q4_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1178 | 427.8600 | 4.1500 | 0 |
| 2026-05-28 11:00:45 | cold-first | vulkan | d047-vulkan130k-big-c152k-lowtile3-noq3quad-noreuse-mt16-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1748 | 622.4000 | 21.3600 | 0 |
| 2026-05-27 21:31:47 | cold-first | vulkan | d046-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b640-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 640/256 | q4_0/q4_0 | none | 0.1640 | 583.4700 | 21.9200 | 0 |
| 2026-05-27 21:28:09 | cold-first | vulkan | d045-vulkan130k-big-c152k-lowtile3-nobn256-noreuse-mt16-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1669 | 593.8500 | 21.6600 | 0 |
| 2026-05-27 21:15:05 | cold-first | vulkan | d043-vulkan130k-big-c152k-lowtile3-noreuse-mt16-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1787 | 636.0900 | 21.5300 | 0 |
| 2026-05-27 20:55:54 | cold-first | vulkan | d042-vulkan130k-big-c152k-first4-noreuse-mt16-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1768 | 629.6100 | 20.9300 | 0 |
| 2026-05-27 20:49:18 | cold-first | vulkan | d041-vulkan130k-big-c152k-noreuse-mt64-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.6911 | 631.7300 | 20.0900 | 0 |
| 2026-05-27 18:38:31 | repeated/steady | rocm | p002-rocm130k-big-c152k-b512-ub128-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.1023 | 363.8100 | 13.8200 | 0 |
| 2026-05-27 18:33:57 | repeated/steady | vulkan | p002-vulkan130k-big-c152k-b512-ub256-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1758 | 626.0600 | 21.6000 | 0 |
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
