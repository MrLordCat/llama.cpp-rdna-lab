# Active Branches

Production and daily work stay on `master`; larger experiments run on named
branches and are merged back selectively. Snapshot: 2026-08-26.

| Branch | Status | Latest state |
| --- | --- | --- |
| `master` | Stable baseline | D131 R9 MTP window audit PASS, multi-stream scale-view fix (`276121b7e`). 2026-08-26: merged RPC stack + qwen4exp port (`1d1aa5a6c`). |
| `rpc-vulkan` | **Merged, kept for continuation** | RPC backend restored for Vulkan offload; merged into `master` 2026-08-26. Recovers the RPC backend (removed upstream), fixed quantized `alloc_size` OOB crash (q3_K/q6_K), split `ggml-rpc` into a layered file stack, and reached the 3-GPU 12K RPC lane target **1314 ptps / 23.6 t/s** (target ≥1277 ptps; PPL 4.0148 ≈ baseline), up from 839 ptps after alloc-size caching, a 16 MB send buffer, and `-ts 0.9,0.6,1.5`. See [RPC resume playbook](research/rpc-vulkan/RPC_PREFILL_RESUME_PLAYBOOK.md). |
| `dflash2` | Paused, to be resumed | DFlash2 block-diffusion drafter port for Qwen3.8-27B (local 2-tap depthwise conv + candidate selector, `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` in `models/`), 5 commits ahead of `master`. Paused at a measured ROCm checkpoint (2026-08-21): opt-in multi-scheduler reuse gives 29.59 aggregate / 36.90 decode / 785.58 prompt tok/s on 498+128 vs 29.00/36.33/749.70 for the single-cache control, with the strict n=3 boundary/parity gate bit-exact. Open items include the upstream batched-greedy divergence and the Vulkan long-decode crash. Resume: [ROCm playbook](research/dflash/ROCM_DFLASH2_RESUME_PLAYBOOK.md), [Vulkan playbook](research/dflash/VULKAN_DFLASH2_RESUME_PLAYBOOK.md). |
| `research/vulkan-decode` | Backlog, to be resumed | D104 (Q6_K prefill dispatch) and D105 (decode bandwidth) closed; Q4_K16 port log reached Vulkan PPL parity with bf16 (6.6124 vs 6.6202 on 256 chunks); MTP n=3 measured decode optimum 1.83x. 15 commits ahead, 10 behind `master` (rebase needed before continuation). |
| `research/vulkan-fp8-kv` | Backlog, to be resumed | D131 R9: fp8 K per-block scale (`LLAMA_VK_F8_K_SCALE`), K-scale broadcast fix, MTP window audit PASS, multi-stream scale-view fix. C2 closed: f8_direct prefill lost 23.5% to preconvert. 20 commits ahead, 10 behind `master`. |
