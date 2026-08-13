# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Retention: `2026-07-01` and newer; latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
| 2026-08-13 16:42:50 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf16-close-r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4690 | 1569.5000 | 33.9400 | 0 |
| 2026-08-13 16:41:32 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf8-fixed-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4647 | 1566.5200 | 33.9000 | 0 |
| 2026-08-13 16:34:20 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf8-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.4563 | 1565.6800 | 33.7200 | 0 |
| 2026-08-13 16:33:04 | cold-first | rocm | d096-d6-rocm49k-mainf8-draftf16-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 5.7928 | 1622.6300 | 40.9100 | 0 |
| 2026-08-13 12:51:55 | cold-first | rocm | d099-rocm49k-q8-close-r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 8.7177 | 1739.2200 | 21.9300 | 0 |
| 2026-08-13 12:50:57 | cold-first | rocm | d099-rocm49k-f8-r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 9.0811 | 1809.6200 | 22.9600 | 0 |
| 2026-08-13 12:49:59 | cold-first | rocm | d099-rocm49k-q8-open-r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | none | 8.7257 | 1741.8000 | 21.9500 | 0 |
| 2026-08-13 11:37:38 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113545 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | sweep/sweep | sweep/sweep | mtp | 3.0656 | 1548.33 | 31.26 | 0 |
| 2026-08-13 11:34:55 | autotune | vulkan | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113408 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 16.8156 | 1619.83 | 47.99 | 0 |
| 2026-08-13 11:33:40 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113259 | Qwen3.6-27B-Q4_K_M.gguf | 12288 | sweep/sweep | sweep/sweep | mtp | 16.4508 | 1559.14 | 48.37 | 0 |
| 2026-08-13 11:32:41 | autotune | rocm | gui-autotune-Qwen3.6-27B-Q4_K_M-20260813-113030 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | sweep/sweep | sweep/sweep | mtp | 6.5217 | 1818.18 | 43.21 | 0 |
| 2026-08-13 11:25:56 | cold-first | rocm | d098-g5-default-smoke-12k-s12r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 8192/1024 | f8_e4m3/f8_e4m3 | none | 2.7627 | 1688.8500 | 25.9900 | 0 |
| 2026-08-13 11:19:35 | cold-first | rocm | d098-g5-98k-mtp2-q8-close-s11r3 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.9036 | 1441.9300 | 33.5700 | 0 |
| 2026-08-13 11:18:22 | cold-first | rocm | d098-g5-98k-mtp2-f8-s11r2 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 2.9919 | 1482.8600 | 35.8200 | 0 |
| 2026-08-13 11:17:09 | cold-first | rocm | d098-g5-98k-mtp2-q8-open-s11r1 | Qwen3.6-27B-Q4_K_M.gguf | 98304 | 8192/1024 | q8_0/q8_0 | mtp | 2.9084 | 1445.1400 | 33.4300 | 0 |
| 2026-08-13 11:15:12 | cold-first | rocm | d098-g5-49k-mtp2-q8-close-s10r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9744 | 1678.9000 | 38.2100 | 0 |
| 2026-08-13 11:14:16 | cold-first | rocm | d098-g5-49k-mtp2-f8-s10r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | f8_e4m3/f8_e4m3 | mtp | 6.2860 | 1751.5800 | 41.9800 | 0 |
| 2026-08-13 11:13:25 | cold-first | rocm | d098-g5-49k-mtp2-q8-open-s10r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 8192/1024 | q8_0/q8_0 | mtp | 5.9256 | 1665.1600 | 37.7100 | 0 |
| 2026-08-13 11:09:10 | cold-first | rocm | d098-g4-final-49k-q8-close-s9r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.6167 | 1687.1600 | 21.8100 | 0 |
| 2026-08-13 11:08:13 | cold-first | rocm | d098-g4-final-49k-f8-s9r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0508 | 1769.3700 | 22.9700 | 0 |
| 2026-08-13 11:07:16 | cold-first | rocm | d098-g4-final-49k-q8-open-s9r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.6196 | 1686.3400 | 21.8400 | 0 |
| 2026-08-13 11:04:44 | cold-first | rocm | d098-g4-w8-49k-q8-s8r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.5715 | 1670.8700 | 21.8200 | 0 |
| 2026-08-13 11:01:58 | cold-first | rocm | d098-g4-w8-49k-f16-s8r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 8.8600 | 1691.4300 | 23.3600 | 0 |
| 2026-08-13 11:01:01 | cold-first | rocm | d098-g4-w8-49k-w8-s8r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0191 | 1761.8500 | 22.9200 | 0 |
| 2026-08-13 10:59:58 | cold-first | rocm | d098-g4-w8-49k-w8-s8r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 9.0036 | 1760.5300 | 22.8400 | 0 |
| 2026-08-13 10:59:03 | cold-first | rocm | d098-g4-w8-49k-w4-s8r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.6995 | 1706.4600 | 21.9600 | 0 |
| 2026-08-13 10:57:43 | cold-first | rocm | d098-g4-w8-12k-w8-s8r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.8716 | 1843.6850 | 24.7100 | 0 |
| 2026-08-13 10:56:44 | cold-first | rocm | d098-g4-w8-12k-w4-s8r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.5858 | 1822.2300 | 24.2200 | 0 |
| 2026-08-13 10:47:33 | cold-first | rocm | d098-g4-spillfree-49k-f16-s7r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 8.8413 | 1688.7400 | 23.3300 | 0 |
| 2026-08-13 10:46:34 | cold-first | rocm | d098-g4-spillfree-49k-full-s7r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.6802 | 1699.1900 | 21.9800 | 0 |
| 2026-08-13 10:45:20 | cold-first | rocm | d098-g4-spillfree-49k-full-s7r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.5964 | 1673.2800 | 21.9700 | 0 |
| 2026-08-13 10:44:17 | cold-first | rocm | d098-g4-spillfree-49k-kq-s7r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.3371 | 1720.7200 | 19.6400 | 0 |
| 2026-08-13 10:43:16 | cold-first | rocm | d098-g4-spillfree-12k-full-s7r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.5977 | 1818.5800 | 24.2800 | 0 |
| 2026-08-13 10:42:18 | cold-first | rocm | d098-g4-spillfree-12k-kq-s7r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.2754 | 1817.0450 | 23.6250 | 0 |
| 2026-08-13 09:38:37 | cold-first | rocm | d098-g4-pcvt-49k-full-s6r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 7.4343 | 1468.2400 | 18.5500 | 0 |
| 2026-08-13 09:37:32 | cold-first | rocm | d098-g4-pcvt-49k-kq-s6r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.4408 | 1739.8300 | 19.8400 | 0 |
| 2026-08-13 09:36:35 | cold-first | rocm | d098-g4-pcvt-49k-f16-s6r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 8.8691 | 1694.1600 | 23.3400 | 0 |
| 2026-08-13 09:35:26 | cold-first | rocm | d098-g4-pcvt-12k-full-s6r3 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 15.7483 | 1725.8650 | 23.0000 | 0 |
| 2026-08-13 09:34:26 | cold-first | rocm | d098-g4-pcvt-12k-kq-s6r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 16.4083 | 1831.4850 | 23.8000 | 0 |
| 2026-08-13 09:33:26 | cold-first | rocm | d098-g4-pcvt-12k-f16-s6r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f16/f16 | none | 16.7687 | 1785.6600 | 24.8850 | 0 |
| 2026-08-13 09:07:45 | cold-first | vulkan | d096-d43-49k-f8-direct-trace-s5r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 7.2490 | 1192.9000 | 25.8300 | 0 |
| 2026-08-13 09:05:51 | cold-first | vulkan | d096-d43-49k-f8-default-trace-s5r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 8.2548 | 1445.5800 | 25.5600 | 0 |
| 2026-08-13 09:04:44 | cold-first | vulkan | d096-d43-49k-q8-reboot-s5r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.4095 | 1445.8100 | 27.0800 | 0 |
| 2026-08-13 08:54:33 | cold-first | vulkan | d096-d43-49k-f8-native-decode-s5r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.6580 | 1581.7200 | 13.2800 | 0 |
| 2026-08-13 08:50:43 | cold-first | vulkan | d096-d43-49k-f8-reboot-s5r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.1729 | 1542.8200 | 25.2400 | 0 |
| 2026-08-13 08:49:03 | cold-first | vulkan | d096-d43-49k-f16-reboot-s5r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 9.3770 | 1698.5550 | 27.1450 | 0 |
| 2026-08-13 08:34:54 | cold-first | vulkan | d096-d43-49k-f16-recovery-s5r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 6.1741 | 1623.4350 | 11.2550 | 0 |
| 2026-08-12 22:06:56 | cold-first | vulkan | d096-d43-49k-f16-repro-s4r16 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 0.0000 | - | - | 1 |
| 2026-08-12 21:15:43 | cold-first | vulkan | d096-d43-49k-f16K-f8V-s4r15 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-12 21:08:08 | cold-first | vulkan | d096-d43-49k-f8-dbg-s4r13 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-12 16:21:47 | cold-first | vulkan | d096-d43-49k-f8-perf4-s4r12 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 0.0000 | - | - | 1 |
| 2026-08-12 16:18:07 | cold-first | vulkan | d096-d43-49k-f16-sync-s4r11 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 0.2105 | 1614.3050 | 19.1800 | 0 |
| 2026-08-12 16:14:17 | cold-first | vulkan | d096-d43-49k-f8-sync-s4r10 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 0.1862 | 1429.9150 | 19.1550 | 0 |
| 2026-08-12 16:10:57 | cold-first | vulkan | d096-d43-49k-f16-perf-s4r9 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 0.3646 | 702.2000 | 20.4400 | 0 |
| 2026-08-12 16:07:53 | cold-first | vulkan | d096-d43-49k-f8-ds4-s4r8 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.3762 | 1597.1450 | 12.2100 | 0 |
| 2026-08-12 16:01:36 | cold-first | vulkan | d096-d43-49k-f8-perf-s4r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 0.4532 | 886.9250 | 15.5650 | 0 |
| 2026-08-12 15:46:08 | cold-first | vulkan | d096-d43-49k-f8-branchless-s4r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.7666 | 1618.6150 | 13.4150 | 0 |
| 2026-08-12 15:40:24 | cold-first | vulkan | d096-d43-49k-f8-pcdec-s4r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 5.4404 | 1608.3600 | 26.1800 | 0 |
| 2026-08-12 15:33:38 | cold-first | vulkan | d096-d43-49k-q8-gpuload-s4r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.7352 | 1521.4950 | 27.5300 | 0 |
| 2026-08-12 15:31:10 | cold-first | vulkan | d096-d43-49k-f8-gpuload-s4r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.7639 | 1617.0700 | 13.4100 | 0 |
| 2026-08-12 15:28:09 | cold-first | vulkan | d096-d43-49k-f8-bc128-s4r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.7693 | 1620.3850 | 13.4100 | 0 |
| 2026-08-12 15:23:55 | cold-first | vulkan | d096-d43-49k-f8-16b-s4r1 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.7768 | 1623.8150 | 13.3950 | 0 |
| 2026-08-12 15:14:08 | cold-first | vulkan | d096-d43-49k-f8k-f16v-s3r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f16 | none | 0.0000 | - | - | 1 |
| 2026-08-12 15:09:08 | cold-first | vulkan | d096-d43-49k-f8-staging-s3r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.7792 | 1621.6050 | 13.4300 | 0 |
| 2026-08-12 15:03:49 | cold-first | vulkan | d096-d43-49k-f16-s3r4 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f16/f16 | none | 9.5874 | 1739.2650 | 27.6350 | 0 |
| 2026-08-12 15:01:08 | cold-first | vulkan | d096-d43-49k-q8-s3r3 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.7919 | 1531.8950 | 27.7000 | 0 |
| 2026-08-12 14:58:54 | cold-first | vulkan | d096-d43-49k-f8inline-s3r2 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.7910 | 1623.1050 | 13.4650 | 0 |
| 2026-08-12 14:55:52 | cold-first | vulkan | d096-d43-12k-f8inline-s3r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 15.3162 | 1811.7050 | 21.5600 | 0 |
| 2026-08-12 14:49:54 | cold-first | vulkan | d096-d43-12k-f16-s2r5 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f16/f16 | none | 19.0427 | 1846.9750 | 29.4800 | 0 |
| 2026-08-12 14:46:55 | cold-first | vulkan | d096-d43-12k-f8lut4b-s2r4 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 15.0541 | 1808.3650 | 21.0600 | 0 |
| 2026-08-12 14:35:11 | cold-first | vulkan | d096-d43-12k-q8-nommdq-s2r3 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | q8_0/q8_0 | none | 17.3459 | 1602.6800 | 27.5850 | 0 |
| 2026-08-12 14:32:04 | cold-first | vulkan | d096-d43-12k-q8-s2r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | q8_0/q8_0 | none | 17.4041 | 1636.1350 | 27.6350 | 0 |
| 2026-08-12 14:30:15 | cold-first | vulkan | d096-d43-12k-f8lut-s2r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 13.9870 | 1751.6450 | 19.2500 | 0 |
| 2026-08-12 13:09:00 | cold-first | vulkan | d096-d41-49k-q8-s1r7 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | q8_0/q8_0 | none | 8.6061 | 1496.0100 | 27.2050 | 0 |
| 2026-08-12 13:06:36 | cold-first | vulkan | d096-d3-49k-scalar-s1r6 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.6643 | 1578.0200 | 13.3450 | 0 |
| 2026-08-12 13:03:49 | cold-first | vulkan | d096-d3-49k-coopmat-s1r5 | Qwen3.6-27B-Q4_K_M.gguf | 49152 | 512/512 | f8_e4m3/f8_e4m3 | none | 6.6448 | 1574.2400 | 13.2950 | 0 |
| 2026-08-12 13:00:23 | cold-first | vulkan | d096-d41-12k-q8-nommdq-s1r4 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | q8_0/q8_0 | none | 17.4478 | 1635.4950 | 27.5250 | 0 |
| 2026-08-12 12:56:08 | cold-first | vulkan | d096-d41-12k-q8-s1r3 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | q8_0/q8_0 | none | 17.5757 | 1612.9800 | 28.3050 | 0 |
| 2026-08-12 12:54:15 | cold-first | vulkan | d096-d3-12k-scalar-s1r2 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 14.5610 | 1756.5400 | 20.3350 | 0 |
| 2026-08-12 12:52:15 | cold-first | vulkan | d096-d3-12k-coopmat-s1r1 | Qwen3.6-27B-Q4_K_M.gguf | 16384 | 512/512 | f8_e4m3/f8_e4m3 | none | 14.1733 | 1758.4300 | 19.5950 | 0 |
