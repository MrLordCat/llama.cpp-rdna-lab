# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-07-10 09:45:35 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n4-guard-ht120-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.2160 | 1138.6100 | 21.3800 | 0 |
| 2026-07-10 09:42:25 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n4-guard-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 9.8872 | 1369.2600 | 26.5900 | 0 |
| 2026-07-10 08:57:40 | cold-first | rocm | rocm-dual-mtpgguf-130k-big-c152k-mt64-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | mtp | 0.5789 | 524.9200 | 20.8500 | 0 |
| 2026-07-10 08:55:02 | cold-first | rocm | rocm-dual-mtpgguf-130k-big-c152k-mt64-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | mtp | 0.5805 | 525.1300 | 22.7200 | 0 |
| 2026-07-10 08:52:40 | cold-first | rocm | rocm-dual-mtpgguf-130k-big-c152k-mt64-none-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.5988 | 543.8300 | 20.3400 | 0 |
| 2026-07-09 22:47:50 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n3-auto-default512-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 16.7882 | 447.9700 | 18.6800 | 0 |
| 2026-07-09 22:46:56 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n2-auto-default512-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.2481 | 1143.4300 | 34.5700 | 0 |
| 2026-07-09 22:44:26 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n2-auto-w512-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.2462 | 1141.9300 | 34.1800 | 0 |
| 2026-07-09 22:42:53 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n2-autonextn0-nopp-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 39.1725 | 460.8100 | 51.6900 | 0 |
| 2026-07-09 22:41:34 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-none-nextn0-manual-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | none | 34.3735 | 757.5800 | 39.4000 | 0 |
| 2026-07-09 22:40:28 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n2-autonextn0-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 34.7637 | 331.2800 | 47.9200 | 0 |
| 2026-07-09 22:37:54 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n2-nextn0-w1024-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.2362 | 1130.9300 | 35.4600 | 0 |
| 2026-07-09 22:36:20 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n2-nextn0-w512-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.2413 | 1137.1300 | 34.8700 | 0 |
| 2026-07-09 22:34:48 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n2-nextn0-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.0617 | 965.7500 | 36.6000 | 0 |
| 2026-07-09 22:32:38 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n4-nextn0-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-09 22:31:53 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n2-nextn0-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 38.5356 | 442.8600 | 50.7900 | 0 |
| 2026-07-09 22:31:12 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n1-nextn0-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 35.0493 | 441.1900 | 44.7900 | 0 |
| 2026-07-09 22:28:53 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n1-devd0-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 7.8802 | 880.4100 | 35.1200 | 0 |
| 2026-07-09 22:25:41 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | mtp | 1.0129 | 932.8500 | 24.4700 | 0 |
| 2026-07-09 22:23:38 | cold-first | vulkan | vulkan-dual-mtpgguf-130k-big-c152k-mt64-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.2044 | 1111.6900 | 27.7600 | 0 |
| 2026-07-09 22:21:34 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n4-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-09 22:21:03 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n2-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 11.8719 | 348.8800 | 13.0600 | 0 |
| 2026-07-09 22:20:28 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-none-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | none | 32.9303 | 729.8400 | 37.9400 | 0 |
| 2026-07-09 22:19:54 | cold-first | vulkan | vulkan-single-gpu0-mtpgguf-smoke-n1-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 18.7711 | 417.0300 | 21.4900 | 0 |
| 2026-07-09 22:19:27 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n1-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 28.3738 | 349.2000 | 36.2700 | 0 |
| 2026-07-09 22:18:30 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-n8-mt32-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | mtp | 3.5920 | 206.2000 | 3.9500 | 0 |
| 2026-07-09 22:17:51 | cold-first | vulkan | vulkan-dual-mtpgguf-smoke-none-mt32-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/256 | q4_0/q4_0 | none | 29.2638 | 726.0300 | 38.1500 | 0 |
| 2026-07-09 22:08:37 | cold-first | rocm | rocm-dual-layer-130k-big-c152k-mt64-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.4089 | 370.0500 | 15.7600 | 0 |
| 2026-07-09 21:50:49 | cold-first | vulkan | vulkan-single-gpu0-130k-big-c152k-mt64-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.6667 | 639.6100 | 8.2900 | 0 |
| 2026-07-09 21:48:09 | cold-first | vulkan | vulkan-dual-layer-130k-big-c152k-mt64-none-r2 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.5543 | 504.9100 | 17.4500 | 0 |
| 2026-07-09 21:44:21 | cold-first | vulkan | vulkan-single-gpu1-130k-big-c152k-mt64-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.5224 | 506.7000 | 5.7400 | 0 |
| 2026-07-09 20:59:08 | cold-first | vulkan | vulkan0-single-short-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 8.2558 | 53.2000 | 9.1500 | 0 |
| 2026-07-09 20:50:27 | cold-first | - | rocm-dual-split-pinnedstage-short-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.7722 | 643.8300 | 26.4900 | 0 |
| 2026-07-09 20:48:02 | cold-first | - | rocm-dual-split-clean-control-short-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.8288 | 643.9800 | 26.6000 | 0 |
| 2026-07-09 20:46:59 | cold-first | - | rocm-dual-split-clean-control-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 18.0610 | 907.6700 | 25.9800 | 0 |
| 2026-07-09 20:37:03 | cold-first | rocm | rocm-dual-split-bufferhostcopy-mt256-none-r3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.6137 | 749.8933 | 26.2633 | 0 |
| 2026-07-09 20:35:54 | cold-first | rocm | rocm-dual-split-bufferhostcopy-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.6938 | 646.8700 | 26.4400 | 0 |
| 2026-07-09 20:35:08 | cold-first | rocm | rocm-dual-split-bufferhostcopy-dev01-mg1-ts1_3-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.4359 | 606.5600 | 26.1700 | 0 |
| 2026-07-09 20:33:24 | cold-first | rocm | rocm-dual-split-dev01-mg1-ts1_3-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.1269 | 598.5800 | 25.8500 | 0 |
| 2026-07-09 20:32:45 | cold-first | rocm | rocm-dual-split-dev01-mg1-ts1_2-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.0514 | 604.9500 | 25.8100 | 0 |
| 2026-07-09 20:30:52 | cold-first | rocm | rocm-dual-split-peercopy-dev01-mg1-mt64-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 4.2176 | 742.9500 | - | 0 |
| 2026-07-09 20:30:09 | cold-first | rocm | rocm-dual-split-dev01-mg0-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 24.6180 | 633.5400 | 25.3200 | 0 |
| 2026-07-09 20:29:21 | cold-first | rocm | rocm-dual-split-dev01-mg1-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 25.1185 | 635.4900 | 25.8100 | 0 |
| 2026-07-09 20:28:27 | cold-first | rocm | rocm-dual-split-layer-mg1-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 24.9866 | 645.0900 | 25.6900 | 0 |
| 2026-07-09 20:25:40 | cold-first | rocm | rocm-dual-split-hoststage-async-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 19.7278 | 477.9800 | 20.2900 | 0 |
| 2026-07-09 20:19:37 | cold-first | rocm | rocm-dual-split-layer-ts1_2-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 21.7593 | 455.3400 | 22.4800 | 0 |
| 2026-07-09 20:18:46 | cold-first | rocm | rocm-dual-split-layer-ts2_1-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 19.3897 | 544.6300 | 19.9000 | 0 |
| 2026-07-09 20:17:25 | cold-first | rocm | rocm-dual-split-row-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-07-09 20:07:46 | cold-first | rocm | rocm-dual-layer-mtp-polish-mt256-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 39.5312 | 516.7400 | 41.7100 | 0 |
| 2026-07-09 20:07:07 | cold-first | rocm | rocm-dual-layer-mtp-polish-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 24.3710 | 618.7700 | 25.0600 | 0 |
| 2026-07-09 20:05:42 | cold-first | rocm | rocm1-mtp-polish-mt256-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 42.6461 | 503.7600 | 45.3100 | 0 |
| 2026-07-09 20:05:07 | cold-first | rocm | rocm1-mtp-polish-mt256-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 28.9357 | 598.7800 | 29.9300 | 0 |
| 2026-07-09 20:01:12 | cold-first | rocm | rocm1-mtp-polish-short-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 46.9966 | 513.9300 | 53.9100 | 0 |
| 2026-07-09 20:00:40 | cold-first | rocm | rocm1-mtp-polish-short-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 28.3424 | 615.1700 | 30.3000 | 0 |
| 2026-07-09 19:35:47 | cold-first | rocm | gui-bench-Qwen3.6-27B-Q3_K_S_mtp-20260709-193517 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | mtp | 1.3676 | 728.5900 | 36.9800 | 0 |
| 2026-07-09 18:08:23 | cold-first | rocm | mtp-windowed-repo32k-n8-win2048-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 32768 | 512/128 | q4_0/q4_0 | mtp | 3.3735 | 683.4500 | 27.2800 | 0 |
| 2026-07-09 18:07:13 | cold-first | rocm | mtp-windowed-repo32k-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 32768 | 512/128 | q4_0/q4_0 | none | 3.4468 | 696.1300 | 28.5200 | 0 |
| 2026-07-09 18:06:06 | cold-first | rocm | mtp-windowed-repo32k-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 32768 | 512/128 | q4_0/q4_0 | mtp | 3.4384 | 659.0400 | 46.1400 | 0 |
| 2026-07-09 18:04:39 | cold-first | rocm | mtp-windowed-nextn-short-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 28.3267 | 605.1900 | 30.3200 | 0 |
| 2026-07-09 18:03:25 | cold-first | rocm | mtp-windowed-nextn-short-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 47.4232 | 514.1000 | 54.0400 | 0 |
| 2026-07-09 17:26:23 | cold-first | rocm | upstream-port-rocm1-dflash-stablereserve-mt128-notrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 17:24:58 | cold-first | rocm | upstream-port-rocm1-dflash-stablereserve-mt32-inputcopytrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 5.9522 | 194.9100 | 6.9300 | 0 |
| 2026-07-09 17:20:43 | cold-first | rocm | upstream-port-rocm1-dflash-stablereserve-mt32-inputtrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 17:19:31 | cold-first | rocm | upstream-port-rocm1-dflash-stablereserve-mt32-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 17:15:40 | cold-first | rocm | upstream-port-rocm1-dflash-notg-mt32-ub5-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 17:13:13 | cold-first | rocm | upstream-port-rocm1-dflash-notg-mt32-inputtrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 17:08:25 | cold-first | rocm | upstream-port-rocm1-dflash-notg-mt128-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 17:06:20 | cold-first | rocm | upstream-port-rocm1-dflash-notg-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 5.5264 | 193.9000 | 7.5100 | 0 |
| 2026-07-09 17:01:59 | cold-first | rocm | upstream-port-rocm1-dflash-draftckpt-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:48:17 | cold-first | rocm | upstream-port-rocm1-dflash-rsseq-mt16-notrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:47:13 | cold-first | rocm | upstream-port-rocm1-dflash-rsseq-mt1-sanity-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 1.3180 | 195.6900 | - | 0 |
| 2026-07-09 16:45:56 | cold-first | rocm | upstream-port-rocm1-dflash-trimbeforeinject-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:42:47 | cold-first | rocm | upstream-port-rocm1-dflash-rsseq-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:38:55 | cold-first | rocm | upstream-port-rocm1-dflash-accepttrim-u5-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:37:39 | cold-first | rocm | upstream-port-rocm1-dflash-accepttrim-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:34:46 | cold-first | rocm | upstream-port-rocm1-dflash-kvtrim-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:31:53 | cold-first | rocm | upstream-port-rocm1-dflash-ctx5-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:30:30 | cold-first | rocm | upstream-port-rocm1-dflash-ctxubfix-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:26:12 | cold-first | rocm | upstream-port-rocm1-dflash-encode-reservefix-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
