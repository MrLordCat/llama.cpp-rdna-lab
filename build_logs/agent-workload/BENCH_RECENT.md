# Canonical Benchmark Recent Runs

Автоматически обновляется `scripts/agent_workload_bench.py`.
Содержит последние прогоны с метриками, полезными для быстрого сравнения.

Limit: latest 80 rows from `BENCH_RUNS.csv`.

| Timestamp | Scope | Backend | Label | Model | Ctx | Batch/UBatch | KV | Spec | TPS | Prompt tok/s | Decode tok/s | Errors |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|
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
| 2026-07-09 16:24:22 | cold-first | rocm | upstream-port-rocm1-dflash-trace-sync-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:22:15 | cold-first | rocm | upstream-port-rocm1-dflash-trace-stage-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 16:19:11 | cold-first | rocm | upstream-port-rocm1-dflash-trace-graphstate-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:26:32 | cold-first | rocm | upstream-port-rocm1-dflash-trace-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:25:31 | cold-first | rocm | upstream-port-rocm1-dflash-clean-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:24:17 | cold-first | rocm | upstream-port-rocm1-dflash-early-env-ub32-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 1.1294 | 161.8200 | - | 0 |
| 2026-07-09 14:21:40 | cold-first | rocm | upstream-port-rocm1-dflash-runtime-disable-ub32-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:12:16 | cold-first | rocm | upstream-port-rocm1-dflash-hipgraphguard-ub32-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:09:50 | cold-first | rocm | upstream-port-rocm1-dflash-encode-trace-graphs-ub32-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:08:48 | cold-first | rocm | upstream-port-rocm1-dflash-encode-trace-nographs-ub32-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 1.1158 | 159.5400 | - | 0 |
| 2026-07-09 14:06:10 | cold-first | rocm | upstream-port-rocm1-dflash-nographs-ub32-trace-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 14:03:36 | cold-first | rocm | upstream-port-rocm1-dflash-presync-ub32-trace-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:59:22 | cold-first | rocm | upstream-port-rocm1-dflash-draftctxub1-ub32-trace-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:58:12 | cold-first | rocm | upstream-port-rocm1-dflash-draftctxub1-ub32-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:56:07 | cold-first | rocm | upstream-port-rocm1-dflash-cap1-ub32-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:53:30 | cold-first | rocm | upstream-port-rocm1-dflash-syncinject-ub2-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/2 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:52:27 | cold-first | rocm | upstream-port-rocm1-dflash-syncinject-ub32-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:50:17 | cold-first | rocm | upstream-port-rocm1-dflash-chunktrace-ub2-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/2 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:49:28 | cold-first | rocm | upstream-port-rocm1-dflash-chunktrace-ub4-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/4 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:48:32 | cold-first | rocm | upstream-port-rocm1-dflash-chunktrace-ub8-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/8 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:47:41 | cold-first | rocm | upstream-port-rocm1-dflash-chunktrace-ub16-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/16 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:46:41 | cold-first | rocm | upstream-port-rocm1-dflash-inputguard-ub32-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:45:42 | cold-first | rocm | upstream-port-rocm1-dflash-chunktrace-ub1-smallprompt-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 2048 | 128/1 | q4_0/q4_0 | other | 0.2108 | 29.7400 | - | 0 |
| 2026-07-09 13:43:02 | cold-first | rocm | upstream-port-rocm1-dflash-upstreamencbatch-ub1-smallprompt-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 2048 | 128/1 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:40:46 | cold-first | rocm | upstream-port-rocm1-dflash-drafttrace-ub1-smallprompt-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 2048 | 128/1 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:38:56 | cold-first | rocm | upstream-port-rocm1-dflash-inputguard-ub1-smallprompt-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 2048 | 128/1 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:34:47 | cold-first | rocm | upstream-port-rocm1-dflash-firstchunk-ub1-smallprompt-mt1-r1 | Qwen3.6-27B-Q3_K_S.gguf | 2048 | 128/1 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:33:37 | cold-first | rocm | upstream-port-rocm1-dflash-firstchunk-ub32-smallprompt-mt4-r1 | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/32 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:32:27 | cold-first | rocm | upstream-port-rocm1-dflash-firstchunk-trace-c8192-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:30:15 | cold-first | rocm | upstream-port-rocm1-dflash-encode-nextn-c8192-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:25:35 | cold-first | rocm | upstream-port-rocm1-dflash-batchenc-devd-c8192-n4-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:23:47 | cold-first | rocm | upstream-port-rocm1-dflash-outputdiag-devd-c8192-n4-mt16-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:21:46 | cold-first | rocm | upstream-port-rocm1-dflash-compat4-devd-c8192-n4-mt64-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:20:07 | cold-first | rocm | upstream-port-rocm1-dflash-compat3-devd-c8192-n4-mt64-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:18:07 | cold-first | rocm | upstream-port-rocm1-dflash-compat2-devd-c8192-n4-mt64-r1 | Qwen3.6-27B-Q3_K_S.gguf | 8192 | 512/128 | q4_0/q4_0 | other | 0.0000 | - | - | 1 |
| 2026-07-09 13:04:56 | cold-first | rocm | upstream-port-dual-mtpgguf-none-c8192-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 18.2087 | 922.2600 | 26.1500 | 0 |
| 2026-07-09 13:04:15 | cold-first | rocm | upstream-port-dual-mtp-c8192-n8-processfix-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 25.6636 | 784.2700 | 51.5400 | 0 |
| 2026-07-09 13:03:21 | cold-first | rocm | upstream-port-dual-mtpgguf-none-c8192-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 7.4970 | 910.4600 | 26.1100 | 0 |
| 2026-07-09 13:02:11 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n2-processfix-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 17.9258 | 707.0200 | 29.3200 | 0 |
| 2026-07-09 13:01:21 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n4-processfix-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 14.8165 | 709.4200 | 21.7900 | 0 |
| 2026-07-09 13:00:18 | cold-first | rocm | upstream-port-rocm1-mtpgguf-none-c8192-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 18.7266 | 767.9700 | 29.8700 | 0 |
| 2026-07-09 12:59:31 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n8-processfix-mt256-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 25.1266 | 713.0000 | 54.6400 | 0 |
| 2026-07-09 12:58:29 | cold-first | rocm | upstream-port-rocm1-mtpgguf-none-c8192-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 6.7949 | 758.0800 | 29.9200 | 0 |
| 2026-07-09 12:57:41 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n8-processfix-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 6.7683 | 704.2000 | 39.4700 | 0 |
| 2026-07-09 12:55:59 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n8-archfix-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 0.0000 | - | - | 1 |
| 2026-07-09 12:52:56 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n8-kvfix-mt64-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 6.5174 | 746.5300 | 26.3500 | 0 |
| 2026-07-09 12:16:17 | cold-first | rocm | upstream-port-rocm1-mtp-c8192-n8-mt32-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | mtp | 3.6547 | 738.2100 | 26.9300 | 0 |
| 2026-07-09 12:15:21 | cold-first | rocm | upstream-port-rocm1-mtpgguf-none-c8192-mt32-r1 | Qwen3.6-27B-Q3_K_S_mtp.gguf | 8192 | 512/128 | q4_0/q4_0 | none | 3.7804 | 753.1200 | 30.1500 | 0 |
| 2026-07-09 10:47:58 | cold-first | - | mtp-s2-dual-128-n1-nextn-rebuilt | Qwen3.6-27B-Q3_K_S.gguf | 4096 | 512/128 | q4_0/q4_0 | mtp | 21.5007 | 436.9200 | 23.0700 | 0 |
