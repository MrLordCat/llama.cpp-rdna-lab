# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-07-08 10:30:55 | cold-first | - | dflash-diag-r2 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 2.5711 | 689.1600 | 4.5000 | 0 |
| 2026-07-08 10:28:58 | cold-first | - | dflash-diag-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 3.3549 | 692.0700 | 4.6300 | 0 |
| 2026-07-08 10:22:57 | cold-first | - | mtpwin-basebig-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 1.5253 | 363.7800 | 20.0900 | 0 |
| 2026-07-08 10:07:23 | cold-first | - | mtpwin-big-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-08 10:05:31 | cold-first | - | mtpwin-sanity-r2 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 18.4775 | 604.5800 | 23.6250 | 0 |
| 2026-07-08 09:48:02 | cold-first | rocm | mtp-bigprompt-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | mtp | 1.5997 | 386.5400 | 18.1500 | 0 |
| 2026-07-08 09:44:46 | cold-first | rocm | mtp-bigprompt-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 2.1774 | 537.9300 | 20.1900 | 0 |
| 2026-07-07 23:03:01 | cold-first | rocm | mtp-repoctx-temp0-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 29.3322 | 620.3100 | 41.5400 | 0 |
| 2026-07-07 23:02:37 | cold-first | rocm | mtp-repoctx-temp0-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 22.3927 | 943.1400 | 26.3600 | 0 |
| 2026-07-07 23:01:49 | cold-first | rocm | mtp-temp0-postbuild-n8-confirm3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 42.1258 | 515.8067 | 44.6767 | 0 |
| 2026-07-07 23:01:15 | cold-first | rocm | mtp-temp0-postbuild-none-confirm3 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 25.6412 | 744.7233 | 26.2900 | 0 |
| 2026-07-07 23:00:06 | cold-first | rocm | mtp-temp0-postbuild-n12-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 37.2331 | 403.9600 | 39.7300 | 0 |
| 2026-07-07 22:59:42 | cold-first | rocm | mtp-temp0-postbuild-n10-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 37.8592 | 410.0000 | 40.4100 | 0 |
| 2026-07-07 22:59:20 | cold-first | rocm | mtp-temp0-postbuild-n6-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 21.5639 | 403.3900 | 22.3400 | 0 |
| 2026-07-07 22:58:06 | cold-first | rocm | mtp-temp0-postbuild-n8-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 41.5827 | 406.4600 | 44.5500 | 0 |
| 2026-07-07 22:57:43 | cold-first | rocm | mtp-temp0-postbuild-n4-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 24.9690 | 410.2600 | 26.0000 | 0 |
| 2026-07-07 22:57:01 | cold-first | rocm | mtp-temp0-postbuild-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 25.9222 | 616.2400 | 26.6700 | 0 |
| 2026-07-07 22:52:19 | cold-first | rocm | mtp-1gpu-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 23.2232 | 360.8600 | 25.1100 | 0 |
| 2026-07-07 22:51:23 | cold-first | rocm | mtp-1gpu-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 25.1823 | 360.2200 | 27.2600 | 0 |
| 2026-07-07 22:50:39 | cold-first | rocm | mtp-1gpu-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 28.8048 | 610.8500 | 30.4300 | 0 |
| 2026-07-07 22:49:53 | cold-first | rocm | mtp-row-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 0.0000 | - | - | 1 |
| 2026-07-07 22:48:32 | cold-first | rocm | mtp-peer-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 6.8823 | 714.5600 | 42.9300 | 0 |
| 2026-07-07 22:47:04 | cold-first | rocm | mtp-temp0-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 23.7786 | 393.6800 | 25.5000 | 0 |
| 2026-07-07 22:46:20 | cold-first | rocm | mtp-temp0-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 25.4130 | 639.7600 | 26.7000 | 0 |
| 2026-07-07 22:39:02 | cold-first | rocm | mtp-combined-hooktrace-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 16.7201 | 399.1000 | 23.7600 | 0 |
| 2026-07-07 22:31:54 | cold-first | rocm | mtp-bulkhidden-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 23.9205 | 407.4400 | 25.5400 | 0 |
| 2026-07-07 22:31:00 | cold-first | rocm | mtp-bulkhidden-hooktrace-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 18.3528 | 393.4700 | 27.3200 | 0 |
| 2026-07-07 22:28:42 | cold-first | rocm | mtp-nologits-n2-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 22.6150 | 396.2000 | 23.9400 | 0 |
| 2026-07-07 22:27:44 | cold-first | rocm | mtp-nologits-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 23.8784 | 404.7900 | 25.5600 | 0 |
| 2026-07-07 22:26:56 | cold-first | rocm | mtp-nologits-none-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | none | 25.3661 | 642.0600 | 26.7000 | 0 |
| 2026-07-07 22:26:07 | cold-first | rocm | mtp-nologits-hooktrace-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 18.1242 | 385.5200 | 27.0700 | 0 |
| 2026-07-07 22:22:33 | cold-first | rocm | mtp-nodetime-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 3.1969 | 203.5700 | 9.3800 | 0 |
| 2026-07-07 22:21:17 | cold-first | rocm | mtp-ubtiming-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 17.9991 | 383.5600 | 26.8400 | 0 |
| 2026-07-07 22:19:16 | cold-first | rocm | mtp-graphtrace-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 21.9363 | 399.0200 | 24.3800 | 0 |
| 2026-07-07 22:13:24 | cold-first | - | mtp2x-n1-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 4096 | 512/128 | q4_0/q4_0 | other | 18.9340 | 616.1100 | 24.1900 | 0 |
