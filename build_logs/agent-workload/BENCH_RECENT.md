# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-05-26 23:40:56 | cold-first | rocm | d013-rocm130k-nommap-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 1.5028 | 793.9100 | 28.6500 | 0 |
| 2026-05-26 23:38:40 | cold-first | rocm | d013-rocm130k-q3k-cublas-threshold0-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0537 | 429.9900 | 1000000.0000 | 0 |
| 2026-05-26 23:34:41 | cold-first | rocm | d013-rocm130k-ub256-paddedfix-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.4138 | 744.4000 | 28.5900 | 0 |
| 2026-05-26 23:33:52 | cold-first | rocm | d013-rocm130k-paddedfix-ub128-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 1.5074 | 796.4800 | 28.6500 | 0 |
| 2026-05-26 23:32:53 | cold-first | rocm | d013-rocm130k-ub256-paddedfix-max1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.0928 | 743.1900 | 1000000.0000 | 0 |
| 2026-05-26 23:25:44 | cold-first | rocm | d013-rocm130k-nodetrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0628 | 502.4400 | 1000000.0000 | 0 |
| 2026-05-26 23:23:37 | cold-first | rocm | d013-rocm130k-mmqtrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 0.0850 | 680.4700 | 1000000.0000 | 0 |
| 2026-05-26 23:22:53 | cold-first | rocm | d013-rocm130k-current-control-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/128 | q4_0/q4_0 | none | 1.5144 | 800.5600 | 28.7600 | 0 |
| 2026-05-26 21:03:31 | cold-first | vulkan | d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-confirm3 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 2.0013 | 1053.1067 | 42.7233 | 0 |
| 2026-05-26 21:02:30 | cold-first | vulkan | d012-vulkan-130k-glu-fast-q3quad-bn256-lowtile3-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 2.0042 | 1055.1900 | 42.4100 | 0 |
| 2026-05-26 20:59:45 | cold-first | vulkan | d011-vulkan-130k-q3quad-vectorret-bn256-lowtile3-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9476 | 1024.4300 | 42.3500 | 0 |
| 2026-05-26 20:57:21 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile3-disablemmvq-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9805 | 1053.4900 | 34.7300 | 0 |
| 2026-05-26 20:56:01 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile3-disablemultiadd-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 2.0013 | 1053.6800 | 42.3700 | 0 |
| 2026-05-26 20:54:26 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile3-fulltrace-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.7817 | 945.3500 | 32.3100 | 0 |
| 2026-05-26 20:53:25 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile2-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1177 | 944.3300 | 1000000.0000 | 0 |
| 2026-05-26 20:52:30 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile2-confirm3 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9926 | 1048.5100 | 42.5433 | 0 |
| 2026-05-26 20:50:35 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile2-nographics-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3679 | 185.3400 | 42.2700 | 0 |
| 2026-05-26 20:49:09 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile2-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 2.0009 | 1053.3800 | 42.3700 | 0 |
| 2026-05-26 20:48:25 | cold-first | vulkan | d010-vulkan-130k-q3quad-bn256-lowtile4-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9877 | 1046.0800 | 42.4600 | 0 |
| 2026-05-26 20:47:28 | cold-first | vulkan | d009-vulkan-130k-q3quad-pipeline-m10240-bn256-lowtile3-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.3783 | 190.6300 | 42.4400 | 0 |
| 2026-05-26 20:46:03 | cold-first | vulkan | d009-vulkan-130k-q3quad-pipeline-m10240-bn256-lowtile3-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1179 | 946.1800 | 1000000.0000 | 0 |
| 2026-05-26 20:43:56 | cold-first | vulkan | d009-vulkan-130k-q3quad-bn256-lowtile3-downsplit6-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9943 | 1049.8600 | 42.2900 | 0 |
| 2026-05-26 20:43:05 | cold-first | vulkan | d009-vulkan-130k-q3quad-pipeline-bn256-lowtile3-confirm3 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9935 | 1048.9800 | 42.5933 | 0 |
| 2026-05-26 20:41:52 | cold-first | vulkan | d009-vulkan-130k-q3quad-pipeline-bn256-lowtile3-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 2.0032 | 1054.7400 | 42.3000 | 0 |
| 2026-05-26 20:40:50 | cold-first | vulkan | d009-vulkan-130k-q3quad-pipeline-bn256-lowtile3-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1176 | 943.5200 | 1000000.0000 | 0 |
| 2026-05-26 20:35:04 | cold-first | vulkan | d009-vulkan-130k-q3quad-gated-bn256-lowtile3-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1101 | 883.2200 | 1000000.0000 | 0 |
| 2026-05-26 20:32:16 | cold-first | vulkan | d009-vulkan-130k-q3quad-bn256-lowtile3-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1122 | 899.8600 | 1000000.0000 | 0 |
| 2026-05-26 20:28:26 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit6-full-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 1.9296 | 1018.6300 | 38.6400 | 0 |
| 2026-05-26 20:27:38 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit8-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1146 | 919.0500 | 1000000.0000 | 0 |
| 2026-05-26 20:26:54 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit7-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1147 | 919.8600 | 1000000.0000 | 0 |
| 2026-05-26 20:26:04 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit6-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1147 | 920.1100 | 1000000.0000 | 0 |
| 2026-05-26 20:25:20 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit5-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1140 | 914.2100 | 1000000.0000 | 0 |
| 2026-05-26 20:24:27 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit4-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1147 | 919.5500 | 1000000.0000 | 0 |
| 2026-05-26 20:23:40 | cold-first | vulkan | d008-vulkan-130k-bn256-lowtile3-downsplit2-point-r1 | Qwen3.6-27B-Q3_K_S.gguf | 131072 | 512/256 | q4_0/q4_0 | none | 0.1142 | 915.6300 | 1000000.0000 | 0 |
