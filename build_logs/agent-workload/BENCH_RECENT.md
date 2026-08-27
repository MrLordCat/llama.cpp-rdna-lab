# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-27 12:22:03 | cold-first | vulkan | p2p2t-serial | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.6456 | 1298.9100 | 26.7250 | 0 |
| 2026-08-27 12:05:21 | cold-first | vulkan | p2p2s-final | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.5565 | 1303.3750 | 26.1200 | 0 |
| 2026-08-27 11:59:52 | cold-first | vulkan | p2p2r-p2x | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:57:31 | cold-first | vulkan | p2p2q-phase2 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:51:13 | cold-first | vulkan | p2p2p-dbg4 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:49:09 | cold-first | vulkan | p2p2o-dbg3 | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:43:40 | cold-first | vulkan | p2p2n-pod | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:40:18 | cold-first | vulkan | p2p2m-trace | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:36:59 | cold-first | vulkan | p2p2l-p2smoke | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 0.0000 | - | - | 2 |
| 2026-08-27 11:35:49 | cold-first | vulkan | p2p2k-inline | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.6515 | 1304.3600 | 26.5300 | 0 |
| 2026-08-27 11:28:08 | cold-first | vulkan | p2p2j-stable | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.7166 | 1301.2400 | 26.9650 | 0 |
| 2026-08-27 11:13:57 | cold-first | vulkan | p2p2f-pre | Qwen3.8-27B-Q4_K_M.gguf | 12288 | 8192/1024 | q8_0/q8_0 | mtp | 11.7809 | 1299.7000 | 27.4000 | 0 |
| 2026-08-27 09:29:35 | cold-first | vulkan | p2p1b-14k-smoke | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 5.5982 | 1200.7450 | 37.4650 | 0 |
| 2026-08-27 09:11:56 | cold-first | vulkan | p2p1-14k-smoke | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 5.6078 | 1202.4150 | 37.4450 | 0 |
| 2026-08-27 09:03:05 | cold-first | vulkan | trace94k-p2a | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1972 | 1070.4300 | 34.5350 | 0 |
| 2026-08-27 08:54:22 | cold-first | vulkan | rf81-94k-bm128bn256 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.2140 | 1069.3150 | 39.9850 | 0 |
| 2026-08-27 08:50:28 | cold-first | vulkan | rf80-94k-base-ts11 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.9932 | 1010.6500 | 28.3450 | 0 |
| 2026-08-27 08:45:52 | cold-first | vulkan | rf79-94k-t5-enable1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.2439 | 1162.4750 | 16.7350 | 0 |
| 2026-08-27 08:32:03 | cold-first | vulkan | rf78-94k-base-check | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.2197 | 1072.9150 | 40.2800 | 0 |
| 2026-08-27 08:26:40 | cold-first | vulkan | rf77-94k-t5-f32acc | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.1680 | 1080.9050 | 18.5100 | 0 |
| 2026-08-27 08:14:30 | cold-first | vulkan | rf76-94k-t5-small-base | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.2934 | 1210.0800 | 16.6850 | 0 |
| 2026-08-26 23:31:27 | cold-first | vulkan | rf75-94k-tile-bm128bn64bk64 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.0876 | 1229.6250 | 9.3800 | 0 |
| 2026-08-26 23:26:03 | cold-first | vulkan | rf74-94k-route-trace | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.2917 | 1207.8950 | 16.7400 | 0 |
| 2026-08-26 23:19:28 | cold-first | vulkan | rf73-94k-tile-bm128bn128-smallm-gate | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.2924 | 1210.9250 | 16.6700 | 0 |
| 2026-08-26 23:11:09 | cold-first | vulkan | rf72-94k-tile-bm128bn256bk32 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.9228 | 945.2300 | 42.2000 | 0 |
| 2026-08-26 23:06:25 | cold-first | vulkan | rf71-94k-tile-bm128bn128bk64-es1 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.2941 | 1207.9550 | 17.3850 | 0 |
| 2026-08-26 23:02:19 | cold-first | vulkan | rf70-94k-tile-bm128bn128bk64 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.2464 | 1164.3500 | 17.5200 | 0 |
| 2026-08-26 22:57:34 | cold-first | vulkan | rf69-94k-tile-bm64bn128bk64 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1909 | 1311.0950 | 9.3000 | 0 |
| 2026-08-26 22:50:22 | cold-first | vulkan | rf68-94k-q8-mask-tile0814 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.2350 | 1339.9850 | 9.3825 | 0 |
| 2026-08-26 22:43:58 | cold-first | vulkan | rf67-94k-q8-mask-ts0715 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1221 | 1130.2800 | 31.9875 | 0 |
| 2026-08-26 22:38:52 | cold-first | vulkan | rf66-94k-q8-mask-npast | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.2987 | 1109.1175 | 41.6400 | 0 |
| 2026-08-26 22:33:31 | cold-first | vulkan | rf65-94k-ctl-q8-runahead | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1531 | 1058.0875 | 30.8400 | 0 |
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
| 2026-08-26 14:49:57 | cold-first | vulkan | rf40-94k-rpc-first-f8lout | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.0191 | 961.8050 | 31.8150 | 0 |
| 2026-08-26 14:36:46 | cold-first | vulkan | rf39-14k-rpc-first-ts091112 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 6.6256 | 1103.8750 | 40.9600 | 0 |
| 2026-08-26 14:35:35 | cold-first | vulkan | rf38-14k-rpc-first-ts09115115 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 6.6340 | 1105.3100 | 41.1600 | 0 |
| 2026-08-26 14:34:13 | cold-first | vulkan | rf37-94k-rpc-first-ts0913 | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.0399 | 980.7650 | 32.7850 | 0 |
| 2026-08-26 14:31:18 | cold-first | vulkan | rf36-14k-rpc-first-ts0715 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 7.1931 | 1204.9500 | 43.0300 | 0 |
| 2026-08-26 14:30:15 | cold-first | vulkan | rf35-14k-rpc-first-ts0814 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 7.0672 | 1168.0300 | 45.0850 | 0 |
| 2026-08-26 14:29:07 | cold-first | vulkan | rf34-14k-rpc-first-ts0913 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 6.9327 | 1126.8100 | 47.9650 | 0 |
| 2026-08-26 14:23:56 | cold-first | vulkan | rf33-94k-rpc-first | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 0.8616 | 819.7350 | 29.9600 | 0 |
| 2026-08-26 14:20:21 | cold-first | vulkan | rf32-14k-rpc-first-vk1tail | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 6.0879 | 1003.9150 | 40.1350 | 0 |
| 2026-08-26 14:18:40 | cold-first | vulkan | rf31-14k-rpc-first | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 5.8757 | 974.3300 | 39.8550 | 0 |
| 2026-08-26 14:04:54 | cold-first | vulkan | rf30-14k-kv3080 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 0.8232 | 277.2100 | 1.4700 | 0 |
| 2026-08-26 13:59:28 | cold-first | vulkan | rf29-14k-kv3080 | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 6.8068 | 1306.2800 | 26.2350 | 0 |
| 2026-08-26 13:44:36 | cold-first | vulkan | trace-94k-rpc-deep | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 1.0840 | 1063.9000 | 15.8450 | 0 |
| 2026-08-26 12:44:32 | cold-first | vulkan | rf27-94k-noflush | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.1652 | 1113.3400 | 19.4050 | 0 |
| 2026-08-26 12:41:30 | cold-first | vulkan | rf26-14k-noflush-verify | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 7.0707 | 1351.2900 | 27.4150 | 0 |
| 2026-08-26 12:31:06 | cold-first | vulkan | trace-14k-rpc-deep | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 5.8173 | 1319.9600 | 16.4350 | 0 |
| 2026-08-26 12:25:02 | cold-first | vulkan | rf25-local-94k-specnone | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 2.6411 | 1351.1850 | 25.0550 | 0 |
| 2026-08-26 12:22:33 | cold-first | vulkan | rf24-14k-rpc-split-verify | Qwen3.8-27B-Q4_K_M.gguf | 16384 | 8192/1024 | q8_0/q8_0 | mtp | 10.7281 | 1328.6550 | 25.1250 | 0 |
| 2026-08-26 11:57:54 | cold-first | vulkan | rf23-local-94k-specnone | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | none | 2.6121 | 1335.2100 | 24.7000 | 0 |
| 2026-08-26 11:55:28 | cold-first | vulkan | rf22-local-94k-control | Qwen3.8-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.7202 | 1441.9850 | 51.1900 | 0 |
