# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-06-30 18:36:21 | cold-first | - | probe-mtp-dbg-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 1.6577 | 505.9100 | 15.6300 | 0 |
| 2026-06-30 18:33:24 | cold-first | - | probe-mtp-greedy-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 0.8847 | 507.2450 | 9.3000 | 0 |
| 2026-06-30 18:29:35 | cold-first | - | probe-mtp-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 4.2323 | 506.9250 | 15.2950 | 0 |
| 2026-06-30 18:28:38 | cold-first | - | probe-mtp-nothink-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 8.8395 | 542.0850 | 11.0200 | 0 |
| 2026-06-30 18:26:38 | cold-first | - | cmp-mtp-256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 1.4994 | 505.2450 | 27.7150 | 0 |
| 2026-06-30 18:25:54 | cold-first | - | cmp-base-256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 1.2908 | 1059.3900 | 31.1200 | 0 |
| 2026-06-30 18:23:21 | cold-first | - | mtp-fixed-2gpu-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 1.2962 | 513.5750 | 13.2800 | 0 |
| 2026-06-30 18:14:27 | cold-first | - | diag-2gpu-mtp-postreboot-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 3.0710 | 513.0850 | 11.3450 | 0 |
| 2026-06-30 18:13:30 | cold-first | - | diag-2gpu-mtp-postreboot-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 1.2539 | 513.5050 | 8.4500 | 0 |
| 2026-06-30 18:11:57 | cold-first | - | diag-2gpu-none-postreboot-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 2.5470 | 1098.8050 | 33.7600 | 0 |
| 2026-06-30 17:54:23 | cold-first | - | diag-1gpu-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 14.8316 | 792.2400 | 30.7950 | 0 |
| 2026-06-30 17:03:25 | cold-first | rocm | vscode-rocm-mtp-fix-none | Qwen3.6-27B-Q3_K_S_mtp.gguf | 32768 | 512/1024 | q4_0/q4_0 | none | 1.2881 | 1202.2600 | 29.0300 | 0 |
| 2026-06-30 17:02:05 | cold-first | rocm | vscode-rocm-mtp-fix-d3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 32768 | 512/1024 | q4_0/q4_0 | mtp | 0.2142 | 567.5200 | 1000000.0000 | 0 |
| 2026-06-30 16:35:04 | cold-first | rocm | vscode-dualgpu-mtpmodel-mtp-d3-v2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 2.0044 | 496.3300 | 14.2800 | 0 |
| 2026-06-30 16:34:45 | cold-first | rocm | vscode-dualgpu-mtpmodel-none-v2-t02-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 18.0851 | 1141.2800 | 22.8600 | 0 |
| 2026-06-30 16:34:00 | cold-first | rocm | vscode-dualgpu-mtpmodel-mtp-d3-t0-v2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 1.1321 | 481.9000 | 1000000.0000 | 0 |
| 2026-06-30 16:33:30 | cold-first | rocm | vscode-dualgpu-mtpmodel-none-v2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 2.5202 | 1133.1100 | 1000000.0000 | 0 |
| 2026-06-30 16:32:35 | cold-first | rocm | vscode-dualgpu-regular-none-r3-t96 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 14.4290 | 793.5467 | 29.0733 | 0 |
| 2026-06-30 16:32:02 | cold-first | rocm | vscode-dualgpu-mtpmodel-mtp-d3-t0-r3-t96 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 8.1609 | 439.8767 | 333349.1967 | 0 |
| 2026-06-30 16:31:32 | cold-first | rocm | vscode-dualgpu-mtpmodel-none-r3-t96 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 14.3998 | 791.9300 | 333351.1433 | 0 |
| 2026-06-30 16:29:49 | cold-first | rocm | vscode-dualgpu-mtpmodel-mtp-d3-t0-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 2.2517 | 374.5600 | 1000000.0000 | 0 |
| 2026-06-30 16:29:21 | cold-first | rocm | vscode-dualgpu-mtpmodel-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 3.5336 | 610.4200 | 1000000.0000 | 0 |
| 2026-06-30 16:11:27 | cold-first | vulkan | vscode-dualgpu-vulkan-130k-nohostkv-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.0253 | 356.9500 | 23.7700 | 0 |
| 2026-06-30 16:02:24 | cold-first | vulkan | vscode-dualgpu-vulkan-130k-sm-layer-ts1-1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.5136 | 176.6200 | 16.2700 | 0 |
| 2026-06-30 15:45:37 | cold-first | rocm | vscode-dualgpu-sm-layer-ts1-1-v2-r3 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 20.9592 | 792.5400 | 28.2333 | 0 |
| 2026-06-30 15:44:50 | cold-first | rocm | vscode-dualgpu-sm-row-ts1-1-r3 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 3.2798 | 38.0833 | 4.5967 | 0 |
| 2026-06-30 15:43:05 | cold-first | rocm | vscode-dualgpu-130k-sm-layer-ts1-1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0103 | 550.7800 | 27.7800 | 0 |
| 2026-06-30 15:38:53 | cold-first | rocm | vscode-dualgpu-rocm-sm-layer-ts1-1-r3 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 17.9956 | 760.8933 | 33.8200 | 0 |
| 2026-06-02 19:43:56 | cold-first | vulkan | vscode-cherrypick-vulkan-r22887-r3 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/256 | q4_0/q4_0 | none | 30.6690 | 740.9700 | 35.2600 | 0 |
| 2026-06-02 19:25:01 | cold-first | vulkan | vscode-cherrypick-vulkan-baseline-r3 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/256 | q4_0/q4_0 | none | 33.8449 | 739.3200 | 40.6733 | 0 |
| 2026-06-02 19:21:01 | cold-first | vulkan | gitbash-vulkan-fixed-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/256 | q4_0/q4_0 | none | 33.3983 | 769.7900 | 39.9000 | 0 |
| 2026-06-02 19:06:14 | cold-first | rocm | vscode-cherrypick-rocm-r23646-r3 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 26.0870 | 663.1900 | 30.3667 | 0 |
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
