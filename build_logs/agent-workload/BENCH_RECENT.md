# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-07-08 21:27:57 | cold-first | rocm | mtp-deferred-rocm0-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.4771 | 157.7500 | 10.1600 | 0 |
| 2026-07-08 21:25:47 | cold-first | rocm | mtp-deferred-rocm0-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.7334 | 378.5400 | 10.6500 | 0 |
| 2026-07-08 21:15:09 | cold-first | rocm | base-rocm1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.7278 | 773.6700 | 29.9500 | 0 |
| 2026-07-08 17:41:41 | cold-first | rocm | dflash-gpu1-default-draft-device-smoke | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 3.3251 | 695.2800 | 7.4000 | 0 |
| 2026-07-08 17:35:34 | cold-first | rocm | mtpmodel-gpu1-temp0-mtp-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 17.7271 | 473.4600 | 35.5800 | 0 |
| 2026-07-08 17:34:59 | cold-first | rocm | mtpmodel-gpu1-temp0-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.6678 | 770.2500 | 29.8800 | 0 |
| 2026-07-08 17:34:07 | cold-first | rocm | mtpmodel-gpu1-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 14.7624 | 472.2600 | 25.3100 | 0 |
| 2026-07-08 17:33:27 | cold-first | rocm | mtpmodel-gpu1-mtp-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 16.0451 | 470.8400 | 29.5000 | 0 |
| 2026-07-08 17:32:50 | cold-first | rocm | mtpmodel-gpu1-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.6729 | 773.9800 | 29.9100 | 0 |
| 2026-07-08 17:31:44 | cold-first | rocm | dflash-targetgpu1-draftgpu0-clone-rs1-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 13.8608 | 696.0200 | 18.9300 | 0 |
| 2026-07-08 17:30:49 | cold-first | rocm | dflash-targetgpu1-draftgpu0-clone-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.7579 | 701.4100 | 6.4700 | 0 |
| 2026-07-08 17:29:27 | cold-first | rocm | dflash-gpu1-draftgpu1-rs1-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 13.3368 | 692.5400 | 18.0000 | 0 |
| 2026-07-08 17:28:19 | cold-first | rocm | dflash-gpu1-draftgpu1-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.6304 | 692.4500 | 6.3200 | 0 |
| 2026-07-08 17:27:01 | cold-first | rocm | dflash-gpu1-draftgpu1-cross64-n4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.5519 | 695.3000 | 4.9900 | 0 |
| 2026-07-08 17:25:07 | cold-first | rocm | dflash-gpu1-draftgpu1-n4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.5437 | 687.2700 | 4.9900 | 0 |
| 2026-07-08 17:18:03 | cold-first | rocm | pre3fc4-dual-gpu1gpu0-layer-7to1-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 17.3071 | 735.5500 | 25.4100 | 0 |
| 2026-07-08 17:17:09 | cold-first | rocm | pre3fc4-dual-gpu1gpu0-layer-equal-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 15.1002 | 668.8800 | 21.6600 | 0 |
| 2026-07-08 17:16:15 | cold-first | rocm | pre3fc4-single-gpu1-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.4331 | 766.1600 | 29.4500 | 0 |
| 2026-07-08 17:14:12 | cold-first | rocm | upstream3fc4-dual-gpu1gpu0-tensor-equal-fitoff-f16kv-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 0.0000 | 112.4200 | 6.3400 | 1 |
| 2026-07-08 17:08:48 | cold-first | rocm | upstream3fc4-dual-gpu1gpu0-layer-7to1-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 17.2644 | 739.9400 | 25.1300 | 0 |
| 2026-07-08 17:07:47 | cold-first | rocm | upstream3fc4-dual-gpu1gpu0-layer-3to1-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 16.1667 | 750.4600 | 22.7300 | 0 |
| 2026-07-08 17:06:56 | cold-first | rocm | upstream3fc4-dual-gpu1gpu0-layer-equal-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 14.7666 | 648.3500 | 21.2600 | 0 |
| 2026-07-08 17:05:58 | cold-first | rocm | upstream3fc4-single-gpu1-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.3810 | 762.9300 | 29.4500 | 0 |
| 2026-07-08 17:02:54 | cold-first | rocm | upstream3fc4-dual-layer-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 14.2175 | 632.7800 | 20.3400 | 0 |
| 2026-07-08 17:02:07 | cold-first | rocm | upstream3fc4-single-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 6.5864 | 454.5900 | 8.1700 | 0 |
| 2026-07-08 16:51:17 | cold-first | rocm | val-mtp-n2-2gpu-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 14.1641 | 455.5100 | 24.1900 | 0 |
| 2026-07-08 16:50:19 | cold-first | rocm | val-mtp-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.2654 | 184.1900 | 15.2500 | 0 |
| 2026-07-08 16:49:02 | cold-first | rocm | val-dflash-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 9.5023 | 734.2100 | 11.4900 | 0 |
| 2026-07-08 16:47:58 | cold-first | rocm | val-base-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.6391 | 773.5000 | 29.9100 | 0 |
| 2026-07-08 16:42:08 | cold-first | rocm | sanity-now-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 5.3740 | 405.3600 | 6.5300 | 0 |
| 2026-07-08 16:35:52 | cold-first | rocm | plain-gstate-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 2.4552 | 418.6200 | 7.4800 | 0 |
| 2026-07-08 16:34:42 | cold-first | rocm | plain-base-1gpu-r2 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 6.1493 | 432.3900 | 7.5900 | 0 |
| 2026-07-08 16:33:43 | cold-first | rocm | mtpmodel-base-1gpu-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 6.5163 | 442.4000 | 8.1200 | 0 |
| 2026-07-08 16:27:05 | cold-first | rocm | mtpmodel-base-1gpu-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 6.6276 | 458.5900 | 8.2300 | 0 |
| 2026-07-08 16:25:57 | cold-first | rocm | mtp-stablekey-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.2911 | 130.1700 | 7.6400 | 0 |
| 2026-07-08 16:19:09 | cold-first | rocm | dflash-stablekey-rs4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.8739 | 520.7400 | 6.9200 | 0 |
| 2026-07-08 16:17:31 | cold-first | rocm | dflash-stablekey-noskpt-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-08 16:16:08 | cold-first | rocm | dflash-ubtime2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.1238 | 723.3200 | 11.8500 | 0 |
| 2026-07-08 16:14:59 | cold-first | rocm | dflash-stablekey-diag | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.0779 | 714.6100 | 11.8200 | 0 |
| 2026-07-08 16:14:01 | cold-first | rocm | dflash-stablekey-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 9.3334 | 722.3800 | 11.2800 | 0 |
| 2026-07-08 16:06:27 | cold-first | rocm | dflash-gdiff2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 2.1767 | 425.9100 | 5.2300 | 0 |
| 2026-07-08 16:04:31 | cold-first | rocm | dflash-gstate-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 2.1881 | 422.4100 | 5.3500 | 0 |
| 2026-07-08 16:01:32 | cold-first | rocm | dflash-ubtime-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 3.4860 | 501.1300 | 5.7400 | 0 |
| 2026-07-08 16:00:06 | cold-first | rocm | dflash-graphdiff-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 3.1584 | 453.3000 | 5.2100 | 0 |
| 2026-07-08 15:51:39 | cold-first | rocm | pl95-base-2gpu-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 18.8922 | 947.0900 | 25.9200 | 0 |
| 2026-07-08 15:50:40 | cold-first | rocm | pl95-base-1gpu-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.5399 | 768.2400 | 29.6300 | 0 |
| 2026-07-08 12:32:20 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-stats2-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 9.7084 | 743.1300 | 11.7800 | 0 |
| 2026-07-08 12:30:10 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-statsfix-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 9.6891 | 740.0200 | 11.7600 | 0 |
| 2026-07-08 12:27:28 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-nockpt-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-08 12:26:25 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-nockpt-cross16-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-08 12:23:28 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-rs1-cross8-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 13.7076 | 743.2400 | 18.2000 | 0 |
| 2026-07-08 12:22:35 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-rs1-cross16-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 13.9917 | 745.6800 | 18.7400 | 0 |
| 2026-07-08 12:21:42 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-rs1-cross32-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 13.1744 | 743.0100 | 17.2700 | 0 |
| 2026-07-08 12:19:58 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-rs2-cross64-n2-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 10.9044 | 745.5100 | 13.5600 | 0 |
| 2026-07-08 12:18:29 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-rs1-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 12.6183 | 740.3100 | 16.3500 | 0 |
| 2026-07-08 12:13:54 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-cross64-n1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 9.7052 | 742.1900 | 11.7700 | 0 |
| 2026-07-08 12:11:53 | cold-first | rocm | dflash-p3-target0-draft1-cloneout-cross64-n4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.0937 | 740.2700 | 8.1500 | 0 |
| 2026-07-08 12:04:06 | cold-first | rocm | dflash-p2-dual-layer-output-rocm0-cross64-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.9095 | 479.9100 | 5.6900 | 0 |
| 2026-07-08 12:01:01 | cold-first | rocm | dflash-p2-dual-layer-cross64-n4-draftrocm1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.3620 | 668.4100 | 6.0100 | 0 |
| 2026-07-08 11:59:31 | cold-first | rocm | dflash-p2-dual-row-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-07-08 11:57:49 | cold-first | rocm | dflash-p2-dual-layer-cross64-n4-draftdual-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.3744 | 671.4800 | 6.0200 | 0 |
| 2026-07-08 11:55:51 | cold-first | rocm | dflash-p2-dual-layer-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 19.3243 | 976.7700 | 26.3400 | 0 |
| 2026-07-08 11:54:19 | cold-first | rocm | dflash-p2-cross64-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.3816 | 743.5700 | 8.5100 | 0 |
| 2026-07-08 11:53:10 | cold-first | rocm | dflash-p2-cross128-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.3369 | 742.5000 | 8.4500 | 0 |
| 2026-07-08 11:50:34 | cold-first | rocm | dflash-p2-rsseq-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 5.6232 | 743.1000 | 6.2600 | 0 |
| 2026-07-08 11:48:36 | cold-first | rocm | dflash-p2-rsseq-graphdiff-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.0585 | 735.0100 | 5.9000 | 0 |
| 2026-07-08 11:45:23 | cold-first | rocm | dflash-p2-graphdiff-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.7745 | 733.6900 | 7.5600 | 0 |
| 2026-07-08 11:40:54 | cold-first | rocm | dflash-p2-undervolt-n3-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 6.9267 | 741.0300 | 7.9200 | 0 |
| 2026-07-08 11:39:56 | cold-first | rocm | dflash-p2-undervolt-n2-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.4928 | 737.9800 | 8.6700 | 0 |
| 2026-07-08 11:38:57 | cold-first | rocm | dflash-p2-undervolt-n1-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.8825 | 742.1800 | 9.1900 | 0 |
| 2026-07-08 11:37:07 | cold-first | rocm | dflash-p2-graphtrace-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 9.5969 | 784.8900 | 31.1800 | 0 |
| 2026-07-08 11:36:13 | cold-first | rocm | dflash-p2-graphtrace-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.8063 | 739.8600 | 7.6000 | 0 |
| 2026-07-08 11:35:00 | cold-first | rocm | dflash-p2-undervolt-n4-draftrocm0-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 6.6806 | 744.3000 | 7.5900 | 0 |
| 2026-07-08 11:33:22 | cold-first | rocm | dflash-p2-undervolt-none-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 20.1100 | 795.1500 | 30.5800 | 0 |
| 2026-07-08 11:32:24 | cold-first | rocm | dflash-p2-undervolt-n4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 6.5816 | 728.0700 | 7.4900 | 0 |
| 2026-07-08 11:02:09 | cold-first | rocm | dflash-p2-validate | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 7.9093 | 751.4700 | 9.2100 | 0 |
| 2026-07-08 10:51:56 | cold-first | - | p2cap-baseline-sanity | Qwen3.6-27B-Q3_K_S.gguf | 2048 | 256/128 | q4_0/q4_0 | none | 7.6538 | 303.8100 | 8.3000 | 0 |
| 2026-07-08 10:41:49 | cold-first | - | dflash-n4-r2 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.4905 | 679.3600 | 4.9300 | 0 |
| 2026-07-08 10:40:11 | cold-first | - | dflash-fixed-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 4.5223 | 691.1600 | 4.9600 | 0 |
| 2026-07-08 10:38:27 | cold-first | - | dflash-diag-r4 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 3.5847 | 686.7600 | 5.1000 | 0 |
