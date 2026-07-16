# E334: ROCm quantized-KV scratch reservation

Date: 2026-07-15

## Goal

Find the code-level reason that ROCm memory grows while a long prompt is being
processed, even when the model and configured KV cache fit in VRAM, and remove
that growth without adding `hipFree` synchronization to the hot path.

## Root cause

Qwen3.6 uses the HIP TILE FlashAttention path for prompt evaluation with head
size 256. For quantized KV, every TILE launch requested `need_f16_K=true` and
`need_f16_V=true`. The old `launch_fattn()` implementation allocated full F16
copies of the currently visible K and V tensors from `ggml_cuda_pool`.

That is particularly expensive on Windows ROCm:

1. The build uses the HIP legacy pool because VMM is unavailable.
2. F16 K/V scratch grows with the used context length.
3. The legacy pool cannot grow one virtual allocation. It allocates a new,
   larger physical block and retains the old size for reuse.
4. Vulkan dequantizes quantized K/V while loading shader tiles and therefore
   does not create this sequence of full-context F16 buffers.

The old E333 traces already exposed the scaling. Per-device HIP pool allocation
was about 317-345 MiB on the short lane, 574-596 MiB at 98K, and 1269-1291 MiB
at 131K. At 131K only about 385-390 MiB was active at once; roughly 0.9 GiB per
device was retained allocation history.

This matches upstream issue #19979 and the maintainer analysis in discussion
#9936. Upstream fixed it in commit `f8f0a47a5` (`#23907`) after this fork had
diverged.

## Implementation

The upstream fix was adapted to the fork's modularized CUDA/HIP runtime:

- `fattn-common.cuh` computes aligned K/V F16 scratch addresses immediately
  after the normal FlashAttention output allocation;
- TILE/MMA launches write to those graph-owned addresses instead of allocating
  `K_f16` and `V_f16` from the legacy pool;
- `fattn.cu` reports the extra size required by the selected FA kernel;
- `runtime_buffers.inc` includes that size when the scheduler allocates a
  `GGML_OP_FLASH_ATTN_EXT` tensor.

This reserves a bounded maximum during graph allocation. It does not call
`hipFree` during prompt evaluation and it preserves the same TILE kernel and
the same F16 arithmetic.

## Single-GPU validation

The requested validation used only the free, non-game GPU (`ROCm0`, PCI
`0000:0e:00.0`). It used Q3_K_S because Q4_K_M model tensors alone do not leave
enough headroom for an informative one-card test.

Common lane:

- model: `Qwen3.6-27B-Q3_K_S.gguf`
- context: 49,152
- actual prompt: 29,561 tokens
- `b8192/ub1024`, q8_0 K/V, FlashAttention, no MTP
- one GPU: `-dev ROCm0 -sm none`
- 16 output tokens, no cache reuse, no warmup

| Build | Prompt TPS | Decode TPS | Pool peak | Pool active peak | Driver allocations | Dedicated peak | Shared peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before reservation | 1072.48 | 30.58 | 344.88 MiB | 334.32 MiB | 7 | 14.140 GiB | 0.356 GiB |
| After reservation | 1048.02 | 30.43 | 294.09 MiB | 283.50 MiB | 5 | 14.302 GiB | 0.356 GiB |

At this 30K prompt the old F16 scratch could reuse larger, fixed Q3 staging
blocks, so only 50.79 MiB and two driver allocations disappear from the pool.
The total WDDM peak does not fall: reservation makes the worst-case graph
workspace explicit up front. The important change is that this allocation is
bounded and old K/V scratch sizes no longer accumulate with prompt length.

That bounded behavior was verified directly at the same `ctx=49152`: the
post-fix 8,301-token prompt and the post-fix 29,561-token prompt both ended at
exactly 294.09 MiB pool allocation, 283.50 MiB peak active allocation, and five
driver allocations. Only cache-hit count increased with work performed
(`7917 -> 24146`). Prompt growth no longer increased the legacy pool.

The three remaining large pool blocks are fixed Q3_K cuBLAS staging buffers,
not prompt growth: 100 MiB (`5120x10240` F16), 10 MiB (`1024x5120` F16), and
170 MiB (`5120x17408` F16). Their actual legacy-pool sizes include the 5%
look-ahead margin and total 294.09 MiB.

The prompt throughput difference is not treated as a proven regression from a
single cold control. The patch changes allocation ownership, not the arithmetic
or the selected TILE kernel. A second post-fix TILE run measured 1043.34 prompt
tok/s and 31.00 decode tok/s, showing the same run-to-run range.

## Rejected direct-VEC probe

A traced probe forced the existing q8 VEC kernel for the D=256 prompt path.
It did eliminate the need for full F16 K/V scratch, but prompt evaluation fell
from 1178.40 to 730.53 tok/s on the matched 8,301-token lane (`-38.0%`). The
probe was removed and is not present in the final source.

E337 tested direct Q8 loaders and found that repeated in-tile dequantization
cost too much prompt throughput. The production follow-up instead converts one
bounded 4096-token K/V chunk and reuses the existing WMMA kernel, then combines
chunk-local softmax outputs online. This removes the context-sized staging
without selecting the slow VEC fallback. See
[E337: bounded ROCm Q8 FlashAttention WMMA](E337_rocm_q8_chunked_wmma.md).

## Reproduction

```powershell
$env:GGML_TRACE_CUDA_POOL = '1'
python scripts\agent_workload_bench.py `
  --label e334-rocm0-q3ks-q8-post-reserve-r1 `
  --server-bin build-rocm-full\bin\llama-server.exe `
  --model models\Qwen3.6-27B-Q3_K_S.gguf `
  --tasks quick --task-ids triage_diff --runs 1 `
  --ctx-size 49152 --batch-size 8192 --ubatch-size 1024 `
  --cache-type-k q8_0 --cache-type-v q8_0 `
  --max-tokens 16 --real-context-mode repo-snapshot `
  --real-context-chars 96000 --no-reuse --no-v2-prime-pass `
  --no-disable-thinking --allow-ctx-above-16k `
  --background-server-policy fail --task-hard-timeout 0 `
  --task-fail-timeout 0 `
  --server-extra '-dev ROCm0 -sm none --spec-type none --cache-ram 0 --ctx-checkpoints 0 -fit off'
```

WDDM artifacts use the `e334-rocm0-q3ks-q8-` prefix under
`build_logs/agent-workload` and were captured with
`scripts/research/windows_gpu_memory_monitor.py`.

## Verification

- `cmake --build build-rocm-full --target llama-server -j 8`: pass
- one-card 49K/30K ROCm request: pass
- graceful server cleanup: pass
- no hard kill and no direct `hipMemGetInfo` query were used

## Dual-GPU Q4 follow-up

The one-GPU Q3 result proved that converted-KV pool history was bounded, while
E335 showed that the context-sized active staging still contributed to Q4
pressure. E337 now removes that active context scaling for the eligible RDNA4
Q8 K/V path and recovers 216 MiB at `ctx=49152` on one card. The production
dual-GPU Q4 lanes still require a fresh rebaseline because WDDM residency also
contains model, KV, recurrent, compute, and split-buffer allocations. See
[E335: ROCm post-reservation rebaseline](E335_rocm_post_reservation_rebaseline.md).

## References

- <https://github.com/ggml-org/llama.cpp/issues/19979>
- <https://github.com/ggml-org/llama.cpp/discussions/9936>
- <https://github.com/ggml-org/llama.cpp/discussions/21526>
- <https://github.com/ggml-org/llama.cpp/commit/f8f0a47a55167bea25199d4d761372cd7cee76b7>
