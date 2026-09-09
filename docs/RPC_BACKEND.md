# Remote GPU (RPC) Backend

Detailed reference moved out of the README.

The fork restores the upstream-removed RPC backend to offload a slice of the
model to a remote Vulkan GPU over TCP. The lab setup uses a local RTX 3080
(10 GB) as the third device alongside the two RX 9070 XT cards.

- **Code layout** — the RPC stack lives in `ggml/src/ggml-rpc/`
  (`rpc_types.h` protocol, `rpc_common.cpp` transport/queue/hash,
  `rpc_client.cpp`, `rpc_server.cpp`, `transport.*`); the standalone server
  binary is `tools/rpc/rpc-server`.
- **Protocol** — version `5.0.1` with a fail-closed check against old servers.
  The client and server must be built from the same tree.
- **Key optimizations** — alloc-size caching (removes hundreds of round-trips
  per ubatch), F16 activation staging at the boundary, async outbound queue and
  run-ahead, opt-in block-Q8_0 activation wire (`GGML_RPC_ACT_Q8_0`), and the
  MTP `graph_recompute` hash fix.
- **Run** — `llama-server --rpc <host>:<port> -dev Vulkan1,Vulkan0,RPC0
  -sm layer -ts <f1>,<f2>,<f3>` with `RPC0` naming the remote device.

## Measured results (Qwen3.8-27B Q4_K_M)

- **12K 3-GPU lane (target reached 2026-08-23)** — **1314 ptps / 23.6 t/s**,
  PPL 4.0148 ≈ baseline, up from 839 ptps after alloc-size caching, a 16 MB
  send buffer, and `-ts 0.9,0.6,1.5` (layers 20/13/32; the 3080 holds
  8.9/10 GB). The target was ≥1277 ptps, i.e. 80% of the solo 1596.8 ptps.
- **MTP through RPC (fixed)** — `graph_recompute` hash mismatch made the
  verify graph read another context's weights (acceptance 1.4%). After the
  fix: acceptance **59.7%** (89/149), decode **37.0 t/s** (12K, n=4,
  loopback; local control 40.15 t/s).
- **94K lane** — `1328.65/25.12` on RPC vs local `1351.18/25.05`
  (−0.3% over the wire).
- **Opt-in Q8_0 activation wire** — 94K `1063.87/39.47`, **+4.63% prefill**
  vs F16 control `1016.82/37.59`, wire traffic 10.49 → 5.57 MiB per ubatch
  (−46.9%). Still gated behind the quality/PPL check.
- **Caveat** — the 27B numbers above were captured before the causal-mask
  (non-causal contour) fix on the RPC path; a re-measure is pending before
  they are treated as final.

The stack was first validated on a small 9B model as a development harness:
F16 activation staging gave +20% prefill, loopback overhead is about
−10% prompt / −12% decode, and over 1 GbE LAN the 3080 loses −38% / −61% —
the network, not the GPU, is the bottleneck at this scale.

- **Practice** — run the server from SYSTEM (`schtasks`) so it survives
  WinRM sessions, keep the NV shader cache warm, stop gracefully (`/exit`,
  never hard-kill a server with a loaded model), and use live logs (`tee`).

Detailed design and iteration notes:
[docs/research/rpc-vulkan/RPC_ARCHITECTURE.md](../docs/research/rpc-vulkan/RPC_ARCHITECTURE.md),
[RPC_CHANGES.md](../docs/research/rpc-vulkan/RPC_CHANGES.md),
[RPC_BASELINE_TABLE.md](../docs/research/rpc-vulkan/RPC_BASELINE_TABLE.md), and
[RPC_PREFILL_RESUME_PLAYBOOK.md](../docs/research/rpc-vulkan/RPC_PREFILL_RESUME_PLAYBOOK.md).
