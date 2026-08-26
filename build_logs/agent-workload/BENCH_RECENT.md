# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-26 11:57:54 | cold-first | vulkan | rf23-local-94k-specnone | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 2.6121 | 1335.2100 | 24.7000 | 0 |
| 2026-08-26 11:55:28 | cold-first | vulkan | rf22-local-94k-control | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.7202 | 1441.9850 | 51.1900 | 0 |
| 2026-08-26 11:45:50 | cold-first | vulkan | rf21-14k-ub2048b | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/2048 | q8_0/q8_0 | mtp | 10.2179 | 1205.8200 | 25.4600 | 0 |
| 2026-08-26 11:44:21 | cold-first | vulkan | rf20-14k-ub2048 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/2048 | q8_0/q8_0 | mtp | 9.2317 | 1261.1350 | 27.0950 | 0 |
| 2026-08-26 11:42:42 | cold-first | vulkan | rf19-49k-async-get | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.2859 | 1251.3950 | 25.2400 | 0 |
| 2026-08-26 11:39:54 | cold-first | vulkan | rf18-94k-act-f8 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0851 | 1093.4150 | 16.3600 | 0 |
| 2026-08-26 11:32:59 | cold-first | vulkan | rf17-94k-ts-1.1-0.9 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0526 | 1023.9950 | 24.8500 | 0 |
| 2026-08-26 11:23:37 | cold-first | vulkan | rf16-94k-async-get | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1440 | 1104.2250 | 19.2000 | 0 |
| 2026-08-26 11:20:32 | cold-first | vulkan | rf15-14k-async-get | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3587 | 1387.7700 | 28.0300 | 0 |
| 2026-08-26 11:11:31 | cold-first | vulkan | rf14-94k-split-timing | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1236 | 1091.8950 | 19.1250 | 0 |
| 2026-08-26 11:04:17 | cold-first | vulkan | rf13-49k-stable | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.2010 | 1230.4750 | 24.3050 | 0 |
| 2026-08-26 11:01:30 | cold-first | vulkan | rf12-49k-async | Qwen3.8-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 4.2826 | 1264.8100 | 23.9600 | 0 |
| 2026-08-26 10:54:27 | cold-first | vulkan | rf11-94k-copy-trace | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1583 | 1110.8650 | 19.2400 | 0 |
| 2026-08-26 10:48:33 | cold-first | vulkan | rf10-14k-copy-trace | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.9348 | 1273.7950 | 29.0100 | 0 |
| 2026-08-26 10:29:13 | cold-first | vulkan | rf9-94k-raw | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1714 | 1116.7750 | 19.5150 | 0 |
| 2026-08-26 10:21:22 | cold-first | vulkan | rf7-14k-clean | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.1745 | 1363.0950 | 27.6750 | 0 |
| 2026-08-26 10:18:02 | cold-first | vulkan | rf6-14k-raw | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 8.1227 | 1372.2950 | 12.9250 | 0 |
| 2026-08-26 10:12:31 | cold-first | vulkan | rf5-14k-raw | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 7.8646 | 1338.2450 | 12.4650 | 0 |
| 2026-08-26 10:04:15 | cold-first | vulkan | stable-check-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.6545 | 1322.9550 | 26.0850 | 0 |
| 2026-08-26 09:29:07 | cold-first | vulkan | rf2-14k-raw | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-26 09:10:14 | cold-first | vulkan | revert-14k-check | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.9484 | 1335.8850 | 27.3550 | 0 |
| 2026-08-25 23:02:55 | cold-first | vulkan | ab5-14k-pp1 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3713 | 1377.6500 | 28.2900 | 0 |
| 2026-08-25 22:59:22 | cold-first | vulkan | ab4-94k-async | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0916 | 1070.7200 | 19.3950 | 0 |
| 2026-08-25 22:52:12 | cold-first | vulkan | ab3-14k-async | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-25 22:48:57 | cold-first | vulkan | diag2-14k-async | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.2049 | 1373.4300 | 27.6100 | 0 |
| 2026-08-25 22:45:51 | cold-first | vulkan | ab2-14k-base | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.5125 | 1381.0250 | 28.9450 | 0 |
| 2026-08-25 22:44:56 | cold-first | vulkan | ab2-14k-async | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.2791 | 1380.3800 | 27.7950 | 0 |
| 2026-08-25 22:30:57 | cold-first | vulkan | ab-rpc-94k-base | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0869 | 1065.9900 | 19.7700 | 0 |
| 2026-08-25 22:28:19 | cold-first | vulkan | ab-rpc-94k-async | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0757 | 1065.8200 | 18.8600 | 0 |
| 2026-08-25 22:24:26 | cold-first | vulkan | ab-rpc-14k-async | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3179 | 1371.1750 | 28.2300 | 0 |
| 2026-08-25 22:23:31 | cold-first | vulkan | ab-rpc-14k-base | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3563 | 1378.0150 | 28.3150 | 0 |
| 2026-08-25 22:18:37 | cold-first | vulkan | rpc3080-14k-async-tmr | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3356 | 1372.5000 | 28.2300 | 0 |
| 2026-08-25 22:16:25 | cold-first | vulkan | rpc3080-14k-async-cpy4 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.1325 | 1338.8200 | 28.1250 | 0 |
| 2026-08-25 22:14:43 | cold-first | vulkan | rpc3080-14k-async-cpy3 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.2204 | 1357.9700 | 28.0850 | 0 |
| 2026-08-25 22:11:13 | cold-first | vulkan | rpc3080-14k-async-cpy2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-25 22:02:27 | cold-first | vulkan | rpc3080-94k-balance11 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0850 | 1066.8100 | 19.6100 | 0 |
| 2026-08-25 21:58:42 | cold-first | vulkan | rpc3080-94k-balance | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0901 | 1054.3100 | 23.1650 | 0 |
| 2026-08-25 21:49:57 | cold-first | vulkan | rpc3090-tmr-14k | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 8.6344 | 1199.1350 | 15.8900 | 0 |
| 2026-08-25 21:47:20 | cold-first | vulkan | rpc3080-94k-asyncgraph | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.9567 | 977.3150 | 24.1300 | 0 |
| 2026-08-25 21:44:15 | cold-first | vulkan | rpc3080-94k-asyncgraph | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 1 |
| 2026-08-25 21:34:35 | cold-first | vulkan | ctl-rpc-14k-nograph | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.3462 | 1278.5500 | 31.4550 | 0 |
| 2026-08-25 21:30:08 | cold-first | vulkan | rpc3080-14k-clean | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 12.4821 | 1384.9800 | 28.7600 | 0 |
| 2026-08-25 21:27:46 | cold-first | vulkan | rpc3080-14k-nosplitcopy | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 9.8930 | 1378.3400 | 17.9950 | 0 |
| 2026-08-25 21:17:07 | cold-first | vulkan | rpc3080-14k-dbg | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.2521 | 1382.2450 | 7.2300 | 0 |
| 2026-08-25 21:09:02 | cold-first | vulkan | sched-debug-lo2 | Qwen3.8-27B-UD-Q4_K_M.gguf | 4096 | 4096/1024 | q8_0/q8_0 | mtp | 7.3990 | 475.6400 | 9.9050 | 0 |
| 2026-08-25 20:13:55 | cold-first | vulkan | final-rpc-14k-pipeline-off | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4557 | 1162.7300 | 14.9900 | 0 |
| 2026-08-25 20:11:52 | cold-first | vulkan | ab14k-async-copy-nosync | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 20:10:59 | cold-first | vulkan | ab14k-sync-copy-nosync | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4668 | 1192.1300 | 14.9400 | 0 |
| 2026-08-25 20:08:54 | cold-first | vulkan | ab14k-sync-copy-clean | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4100 | 1044.3900 | 13.7800 | 0 |
| 2026-08-25 20:05:13 | cold-first | vulkan | diag-tl-async-14k | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4076 | 1040.9100 | 12.9800 | 0 |
| 2026-08-25 20:04:06 | cold-first | vulkan | diag-tl-base-14k | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.3966 | 1016.3200 | 11.5700 | 0 |
| 2026-08-25 20:02:42 | cold-first | vulkan | ab14k-asyncpipe-copies4 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4050 | 1035.5300 | 12.4400 | 0 |
| 2026-08-25 20:00:47 | cold-first | vulkan | ab14k-asyncpipectl-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.3998 | 1016.9800 | 14.0600 | 0 |
| 2026-08-25 19:59:27 | cold-first | vulkan | smoke-async-pipe-4k | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 1.7253 | 992.7300 | 14.3000 | 0 |
| 2026-08-25 19:48:46 | cold-first | vulkan | fifo-nosync-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:48:20 | cold-first | vulkan | fifo-nosync-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:46:58 | cold-first | vulkan | fifo-fix-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0270 | 133.9200 | 3.6400 | 0 |
| 2026-08-25 19:44:08 | cold-first | vulkan | fifo-fix-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0480 | 239.4300 | 4.5300 | 0 |
| 2026-08-25 19:41:28 | cold-first | vulkan | loop-faf-4 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:37:05 | cold-first | vulkan | loop-faf-3 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:32:56 | cold-first | vulkan | loop-faf-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:30:30 | cold-first | vulkan | loop-faf-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:26:39 | cold-first | vulkan | fifo-nd-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:25:08 | cold-first | vulkan | fifo-nd-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:19:06 | cold-first | vulkan | fifo-dbg2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:17:23 | cold-first | vulkan | fifo-tl-4 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:16:32 | cold-first | vulkan | fifo-tl-3 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:15:41 | cold-first | vulkan | fifo-tl-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:14:50 | cold-first | vulkan | fifo-tl-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:13:00 | cold-first | vulkan | fifo-faf-dbg | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2049 | 1034.5000 | 10.2000 | 0 |
| 2026-08-25 19:11:07 | cold-first | vulkan | fifo-faf-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:10:14 | cold-first | vulkan | fifo-faf-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 19:07:59 | cold-first | vulkan | cand-rpc-14k-drain-tl | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2014 | 1020.4800 | 9.1300 | 0 |
| 2026-08-25 19:05:59 | cold-first | vulkan | tl-repro-2 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2022 | 1021.6800 | 9.7700 | 0 |
| 2026-08-25 19:05:03 | cold-first | vulkan | tl-repro-1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.2009 | 1016.7700 | 9.2600 | 0 |
| 2026-08-25 19:03:16 | cold-first | vulkan | cand-rpc-8k-debug-tl | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.1939 | 979.1500 | 10.0000 | 0 |
| 2026-08-25 18:58:35 | cold-first | vulkan | cand-rpc-14k-async-copy-b1 | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-25 18:32:06 | cold-first | vulkan | diag-rpc-14k-splitsum | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4501 | 1150.6800 | 14.4700 | 0 |
| 2026-08-25 18:30:59 | cold-first | vulkan | diag-local-14k-ts072-splitsum | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.5783 | 1485.7100 | 15.8200 | 0 |
| 2026-08-25 18:26:37 | cold-first | vulkan | diag-rpc-14k-ub-current | Qwen3.8-27B-Q4_K_M.gguf | 163840 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 0.4442 | 1136.2700 | 14.2000 | 0 |
