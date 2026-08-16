# D105: Vulkan decode — VRAM bandwidth ceiling measured (synthetic read bench)

Branch: `research/vulkan-decode` · 2026-08-16 · author: coordinator + tools
Status: measurement baseline done; decode-side hypotheses open.

## 1. Question

Decode is bandwidth-bound: every token re-reads all weights
(17.10 GB file, 16.38 GB per token excluding `token_embd` for
Qwen3.6-27B-Q4_K_M). GPU-Z showed only 28-30% memory controller load.
How far from the *actually achievable* bandwidth are we? What is the
theoretical decode ceiling on this rig?

## 2. Tool: `scripts/research/vk_bandwidth.cpp` + `vk_bw_read.comp`

Minimal Vulkan compute that streams a DEVICE_LOCAL buffer linearly as
float4 (16 B) loads, one accumulator per thread, one word written per
workgroup to keep the loads alive. MinGW g++ build, `vulkan-1.dll`
loaded at runtime, device-level entry points resolved via
`vkGetDeviceProcAddr` (the global DLL trampolines misbehave on the
AMD driver: queue functions returned instantly / garbage).

Build:

```powershell
export PATH="/c/VulkanSDK/1.4.350.0/Bin:/c/Strawberry/c/bin:$PATH"
glslc scripts/research/vk_bw_read.comp -o /tmp/vk_bw_read.spv
g++ -O2 -std=c++17 scripts/research/vk_bandwidth.cpp -I /c/VulkanSDK/1.4.350.0/Include -o /tmp/vk_bandwidth.exe
(cd /tmp && ./vk_bandwidth.exe 1 2 3)   # device 1, 2 GiB, 3 passes
```

Driver-safety lessons from writing it:
- do NOT re-record a command buffer for a second pass (AMD driver
  crashes: SIGSEGV/SIGILL on the first API call of the re-record);
  record once and resubmit the same primary cb per pass;
- fresh fence per pass (resubmitting a signaled fence is invalid and
  also made the driver crash);
- `vkResetCommandPool`/`vkResetFences` paths also crashed — not used.

## 3. Measured achievable read bandwidth (stable, 2-3 GiB buffers)

| Device | buffer | bandwidth | % of 644.6 GB/s spec |
|---|---|---|---|
| GPU1 (non-display) | 2 GiB | 619-620 GB/s | 96.0-96.2 % |
| GPU1 | 3 GiB | 626-627 GB/s | 97.1-97.3 % |
| GPU0 (display) | 2 GiB | 599-609 GB/s | 93.0-94.4 % |

**Achievable read ceiling ≈ 620 GB/s per card (≈96% of spec peak)** —
much higher than the 55-70% assumed earlier. Both cards together:
≈1.24 TB/s.

### 3.1 Anomaly (open question)

Buffers >= 4 GiB report super-linear "bandwidth" (4 GiB ~9 TB/s,
12 GiB ~19 TB/s, stable across passes, unaffected by
`vkCmdFillBuffer` pre-fill). Physically impossible — the passes are
not reading the whole buffer. Not explained yet (int32 indexing is
valid up to 8 GiB of floats; sizes are multiples of 262144 threads;
device-local heap confirmed via memoryType flags). Suspects:
driver lazy page commit / hardware zero-page path for large fresh
allocations; needs a unique-pattern write shader to settle. Does NOT
affect the decode conclusion: real decode reads weights that were
written by upload, and the 2-3 GiB numbers are stable and sane.

## 4. Where decode stands vs the ceiling

Measured decode (12K lane, spec=none, Qwen3.6-27B-Q4_K_M):
29.5 t/s. Per token: 16.38 GB weights + 0.54 GB KV + 0.15 GB
recurrent = 17.07 GB total. Limiting card (GPU1) carries ≈8.8 GB/token.

- achieved: 29.5 × 8.8 ≈ **253 GB/s** on the limiting card
  = **41% of the 620 GB/s achievable ceiling**
- GPU-Z MC load 28-30% is consistent (it reads the same counter
  domain); it is NOT a broken metric, it matches the fraction.

**Ceilings for Q4_K_M decode on this rig:**

| case | effective bandwidth | t/s |
|---|---|---|
| measured now (both cards, layer-split, sequential chain) | ~513 GB/s avg | 29.5 |
| ideal per-card kernels (100% of 620 on every MMV) | 620 GB/s | ~33-36 |
| byte reduction (Q4_K16, −8%) | same BW | ~+8% |

Note: layer split (`-sm layer`) splits the model into 2-3 scheduler
splits per decode step (first half of layers on one card, second half
on the other, output last). The layer chain is strictly sequential
for a single token (each layer consumes the previous layer's hidden
state), so the two cards can NOT overlap for spec=none decode — the
"41% of 2x620" number in the original version of this note is the
sequential-card topology, not kernel inefficiency. See §5.

## 5. Decode diagnostics (2026-08-16, same session, adjacent lanes)

Adjacent 12K spec=none probes (Qwen3.6-27B-Q4_K_M, q4_0 KV):

| probe | decode t/s | verdict |
|---|---|---|
| control (canon D104 lane) | 29.69 | baseline |
| `GGML_VK_FORCE_MMVQ=1` | 29.52 | −0.6% noise: MMVQ kernel is NOT the limiter |
| `GGML_VK_DISABLE_MMVQ=1` | 29.06 | −2.1% noise: direct MMV kernel equally fast |
| `LLAMA_OUTPUT_DEVICE=Vulkan1` (output moved GPU0→GPU1, +1004 MiB model on GPU1, verified in log) | 29.54 | −0.5%: total bytes/token unchanged → invariant, consistent with BW-bound |

Per-kernel timings from `GGML_VK_PERF_LOGGER=1` (decode, one token's
kernel set, GPU1):
- FFN `q4_K m=17408 k=5120` MMV: 50 MB in ~95 us = **526 GB/s** (85% of peak)
- output `q6_K m=248320 k=5120` MMV: 1.04 GB in ~1671 us = **622 GB/s** (100% of peak)
- attn projections ~1700-2000 GFLOPS/s each, all in the same 80-90% BW range

Perf logger itself costs ~23% decode (29.7 → 22.9 t/s) — the query
pool per dispatch; numbers above are logger-inclusive per-kernel
rates, so real utilization is at least this good.

### 5.1 Revised conclusion (replaces the 41% interpretation)
1. Decode kernels run at ~85-100% of the achievable 620 GB/s per
   card. The "41% of ceiling" from the original §4 was an artifact:
   bytes-per-token divided by WALL time and compared against the
   SUM of two cards' bandwidth. Code check (ggml-backend.cpp:1711,
   ggml_backend_sched_graph_compute_async): each scheduler split is
   launched async on its backend, but every split that consumes an
   activation produced on the other card waits on that card's event —
   and with `-sm layer` the model is 2-3 splits (layers 0-31, 32-63,
   output), forming a strictly sequential chain for one token.
   GPU-Z "28-30% MC load" is a memory-controller busy-cycle counter,
   not bandwidth utilization, and does not contradict this.
2. Measured budget per token (33.9 ms wall): ≈14.9 ms GPU1 layers
   (logger-adjusted), ≈17.5 ms GPU0 layers, ≈1.5 ms dispatch/sync
   residue. Each card reads its share (~8 GB) at ~85-95% peak while
   it is the active card.
2b. GGML_SCHED_SPLIT_TIMING trace (d105-split-trace-r1, decode 29.80
    t/s, trace is nearly free) confirms the topology exactly — one
    token = 3 scheduler splits:
    - CPU: GET_ROWS embedding, ~0.01 ms;
    - Vulkan1: layers norm-0..l_out-32, 1986 nodes: copy 0.08 ms,
      compute(launch) 2.3-3.5 ms, compute_sync(GPU busy) 13.7-16.3 ms,
      total 16.1-18.7 ms;
    - Vulkan0: layers norm-33..result_output, 1862 nodes: copy
      0.6-0.7 ms, compute 2.3-2.5 ms, compute_sync 13.7-13.9 ms,
      total 16.7-16.9 ms.
    Sum ≈ 33.4 ms = wall. The two cards are strictly sequential
    (no overlap; wall equals the sum of both splits' totals).
    Note the CPU launch cost: ~2.3-3.5 ms per card to dispatch 1862-
    1986 graph nodes — ~7-10% of token time, a secondary lever.
3. Real ceilings (spec=none, one-token chain is strictly sequential):
   - ideal per-kernel bandwidth (FFN 85% → output-class 100%):
     ~33-36 t/s (+12-20% from 29.5);
   - byte reduction Q4_K16 (16.4→15.1 GB/token): ~+8% on top;
   - overlap of the two cards is NOT possible for a single token
     (activation dependency). The only parallelism lever for decode is
     speculative decoding: MTP batch-verify amortizes weight reads
     (already ~1.29-1.42x in Vulkan lanes) and does not reduce
     bytes/token for the main model pass.
4. MMVQ vs direct MMV: no material difference (±2%) — decode is
   bandwidth/topology-bound, NOT kernel-pattern-bound. Further MMVQ
   micro-optimization for Q4_K_M decode is NOT a priority.

### 5.2 Next steps (decode)

- [x] P1: sequential-card model — confirmed by code (scheduler splits
      with event dependencies) + GGML_SCHED_SPLIT_TIMING trace run
      (d105-split-trace-r1): token = CPU-embed → Vulkan1 (layers 0-32,
      ~16.6 ms) → Vulkan0 (layers 33-63+output, ~16.8 ms), wall = sum.
- [~] P2 (diagnosed, kernel fix is fork-owner work): decode matmuls go
      through the MMVQ path (AMD n=1 k≥2048 → MMVQ; verified by
      FORCE/DISABLE_MMVQ being neutral). Per-kernel BW in decode
      (d105-perf-clean-r1): q4_K MMVs 526-602 GB/s, q6_K 594-622 GB/s,
      output q6_K m=248320 = 622 GB/s (100% of synthetic peak). FFN
      q4_K m=17408 (fused gate+up) is the low outlier at 526 GB/s
      (85%); dispatch geometry is NOT a tail issue (blocks_m=272 ×
      k-slabs over 64 CU). The ~15% gap needs either an AMD Vulkan
      kernel profiler or an MMVQ-pattern synthetic — candidates:
      q4_K dequant/scale layout, q8_1 src1 staging sync, L2/row
      swizzle. No env lever exists in this branch (low-tile split_k
      gates are prefill-only, n≥128). Expected gain if FFN reaches
      output-kernel efficiency: ~+4% decode.
- [~] P3 (measured 2026-08-16, d105-p3-k16-r1): Q4_K16.gguf loads on
      this binary (320×q4_K16_M + 18×q4_K16 tensors) but runs on the
      fallback direct MMV path (no Q8_1 MMVQ kernels generated for
      K16 by design). Result vs Q4_K_M (same lane): prefill 212.7 vs
      1472.6 tps (−86%), decode 27.64 vs 29.69 (−6.9%) despite −3.8%
      bytes/token. VERDICT: Q4_K16 is not usable on Vulkan until the
      MMVQ (decode) and batched (prefill) kernels are ported — this is
      q4-k16-quant branch §3.4 work. The −6.9% decode matches the
      DISABLE_MMVQ probe (−2.1%) plus K16 row layout overhead.
- [x] P4 (MTP draft-n sweep, same lane, Q4_K_M): 128-token honest
      results — none 28.66; n=2 50.20 (1.75x); n=3 52.52 (1.83x);
      n=4 52.22 (1.82x). Sweet spot n=3. Table in §5.3.

### 5.3 P4 MTP draft-n sweep (Q4_K_M, 12K, q4_0 KV, dual Vulkan)

| run | max_tokens | prompt t/s | decode t/s | vs spec=none |
|---|---|---|---|---|
| none ctrl (d105-ctrl-r1) | 16 | 1472.6 | 29.69 | 1.00x |
| mtp n=2 (d105-p4-mtp2-r1) | 16 | 1457.5 | 54.02 | 1.82x (unreliable) |
| mtp n=3 (d105-p4-mtp3-r1) | 16 | 1453.3 | 53.44 | 1.80x (unreliable) |
| mtp n=4 (d105-p4-mtp4-r1) | 16 | 1413.8 | 62.30 | 2.10x (unreliable) |
| none ctrl 128 (d105-p4b-none128-r1) | 128 | 1455.7 | 28.66 | 1.00x |
| mtp n=2 128 (d105-p4b-mtp2-128-r1) | 128 | 1404.4 | 50.20 | 1.75x |
| mtp n=3 128 (d105-p4b-mtp3-128-r1) | 128 | 1433.6 | 52.52 | 1.83x |
| mtp n=4 128 (d105-p4b-mtp4-128-r1) | 128 | 1364.5 | 52.22 | 1.82x |

Honest 128-token acceptance (draft_n_accepted / draft_n, per task):
n=2: 66-84%; n=3: 55-86%; n=4: 47-78% (acceptance falls with n, but
per-call accepted count rises until n=3).

P4 VERDICT: MTP n=3 is the decode sweet spot on this lane (~52.5 t/s,
1.83x); n=4 adds cost without speedup (52.2, higher variance). MTP is
the ONLY proven large decode lever — batch verification amortizes the
per-token weight re-read that the sequential two-card chain cannot
overlap otherwise. 16-token numbers above are directional only
(prefill-dominated, harness warning); the 128-token block is the
canonical P4 result.

## 6. Artifacts

- `scripts/research/vk_bandwidth.cpp`, `scripts/research/vk_bw_read.comp`
  (committed with this note)
- bench history: `build_logs/agent-workload/BENCH_RECENT.md` (existing
  12K decode numbers reused; no new server lanes run)
