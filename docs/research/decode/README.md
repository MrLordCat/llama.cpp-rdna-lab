# Decode lane: ROCm/Vulkan without speculation

Date: 2026-08-30

Branch: `decode`

## Objective

Improve non-speculative decode throughput without reducing prompt throughput.
The primary problem is the dual-RX 9070 XT ROCm layer-split lane. Vulkan is the
control backend. MTP, n-gram speculation, unsafe peer-copy overrides, and
quality-changing KV formats are outside this lane.

The admission gate for a keep candidate is:

- at least `+3%` adjacent decode throughput;
- no more than `-1%` adjacent prompt throughput;
- deterministic/correct output and graceful server shutdown;
- no additional active Shared-memory spill;
- repeat at L0 before promotion to longer contexts.

## Locked L0 contract

- Model: `models/Qwen3.8-27B-UD-Q4_K_M.gguf`.
- Prompt: synthetic, 3,995 evaluated tokens.
- Decode: 64 tokens, temperature zero, seed 42.
- Context: 8,192.
- Batch/ubatch: `8192/1024`.
- KV: `f8_e4m3/f8_e4m3`.
- Flash Attention on, `spec=none`, `-fit off`.
- Warm-up: 553 prompt tokens plus 16 decode tokens.
- ROCm production topology: `ROCm1,ROCm0 -sm layer -ts 1,1`.
- Vulkan control topology: `Vulkan1,Vulkan0 -sm layer -ts 1,1`.

## Baseline matrix

Fresh one-shot controls on the branch:

| Backend/topology | Prompt tok/s | Decode tok/s | Interpretation |
| --- | ---: | ---: | --- |
| ROCm0 single, `-sm none` | 143.17 | 29.79 | Model barely exceeds the WDDM budget; prefill actively spills, decode remains a useful diagnostic ceiling |
| ROCm1 single, `-sm none` | 56.87 | 6.81 | Heavier Shared spill; not a usable performance lane |
| ROCm dual `ROCm1,ROCm0` | 1471.83 | 26.37 | Production topology; prompt is healthy, decode loses about 3 tok/s |
| ROCm dual `ROCm0,ROCm1` | 1510.02 | 26.82 | Order changes only about 1.7%; it is not the root cause |
| Vulkan dual `Vulkan1,Vulkan0` | 1181.59 | 27.98 | Decode control is about 5.4% above the adjacent ROCm result |

The exact reproduction of the reported single-card row is
`decode-rocm-single0-l0-r1` at `143.1705 / 29.7854` prompt/decode tok/s.

## Dual-ROCm boundary trace

`GGML_SCHED_SPLIT_TIMING=1` shows three serial splits per decode token:

1. CPU input embedding;
2. ROCm1, layers 0 through 32;
3. ROCm0, layers 33 through 64 plus the output head.

The 64-token forced-sync means are:

| Split | Copy | Host submit | GPU sync/work | Total |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.005 ms | 0.009 ms | 0.000 ms | 0.018 ms |
| ROCm1 | 0.764 ms | 0.958 ms | 15.688 ms | 17.414 ms |
| ROCm0 | 1.547 ms | 0.915 ms | 17.182 ms | 19.648 ms |

The comparable single-ROCm0 decode graph is:

| Split | Copy | Host submit | GPU sync/work | Total |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.006 ms | 0.014 ms | 0.000 ms | 0.024 ms |
| ROCm0 | 0.751 ms | 2.714 ms | 30.303 ms | 33.774 ms |

The dual token is about 3.3 ms slower. About 2.6 ms is in the summed device
work of the two graph halves, and about 0.7 ms is additional copy/submit cost.
Changing only the transfer path therefore cannot recover the full gap.

In the normal non-sync trace, the 20 KiB `l_out-32` copy reports about 16.1 ms:
that number is the required wait for the ROCm1 graph, not PCIe payload time.
The actual host-staged boundary transfer is about 0.7 ms. The second copied
inter-device input is an 8 KiB converted attention mask and costs about
0.25 ms. Layer split is dependency-serial for one sequence.

Artifacts:

- `build_logs/bench/decode-rocm-dual10-splitsync-l0-r1/`;
- `build_logs/bench/decode-rocm-single0-splitsync-l0-r1/`;
- `build_logs/bench/decode-rocm-dual10-splitasync-l0-r1/`.

## No-code topology gate

Adjacent two-shot control: `1489.92 / 26.55` prompt/decode tok/s.

| Candidate | Prompt tok/s | Decode tok/s | Prompt delta | Decode delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `-ts 1,1` | 1489.92 | 26.55 | - | - | control |
| `-ts 17,16` | 1476.16 | 26.35 | -0.92% | -0.76% | reject |
| `-ts 17,15` | 1451.75 | 26.81 | -2.56% | +1.01% | reject: prompt gate |
| `-ts 1,1 --main-gpu 1` | 1498.22 | 26.59 | +0.56% | +0.15% | neutral |

Historical row/tensor split results remain closed: ROCm row split timed out
without a valid decode, while tensor split was unsupported or crashed. Wider
layer ratios also lost substantially. Equal layer split remains the production
topology.

## Routes already closed

Do not repeat these without a material runtime or hardware change:

- Direct peer copy. HIP 7.1 reports `can_access=0` and
  `access_supported=0` in both directions. Forced use previously corrupted
  output and destabilized the driver.
- Event-chained pinned host staging. It improved 30K prompt throughput by
  2.65% but changed decode by -0.45%.
- Batched host inputs/cross-device copies. The measured ceiling was about
  0.3 ms and no wall gain reproduced.
- Local rematerialization of the converted attention mask. Correct but slower.
- Removing the pipeline scheduler path. Decode did not improve.
- More ROCm scheduler copies. One copy is already the best single-request
  policy and avoids large WDDM residency growth.
- Q4_K/Q5_K MMVQ geometry sweeps, Q5_K row batching/nwarps=4, Q6_K forced old
  small-K, one-row WMMA FA, reduced vector-FA block count, and paired-head FA.

See E269, E294-E299, E313, E338, and E345-E356 for the original evidence.

## Qwen3.8 decode route

The selected first-token graphs contain 1,890 nodes on ROCm1 and 1,766 nodes
on ROCm0. A synchronized node trace is intentionally distorted in absolute
time, but its operation counts identify the route:

- 493 `MUL_MAT` nodes;
- 48 fused autoregressive `GATED_DELTA_NET` nodes;
- 16 native-F8 vector `FLASH_ATTN_EXT` nodes.

Decode matvec uses mixed quantization. Frequent shapes include Q4_K
`5120x10240`, `5120x6144`, and `5120x17408`; Q5_K `6144x5120`; and dense
IQ4_XS projections. Q4_K/Q5_K/Q6_K nearby geometry is already covered by the
existing production gates.

IQ4_XS uses `nwarps=8`, 44 registers without fusion or 52 with fusion, eight
blocks per SM, reported 100% occupancy, and 64 waves per SM. There is no
resource-waste signal strong enough to admit an IQ4_XS warp-count patch.

The new Qwen3.8-specific launch-reduction candidate is the pair of narrow
Q8_0 matvecs in every GDN layer:

- `ssm_beta`: `5120 -> 48`;
- `ssm_alpha`: `5120 -> 48`.

There are 96 such narrow matvec nodes per token across 48 GDN layers. A fused
two-output kernel could reuse the input vector and remove one launch per GDN
layer, but its expected whole-token upside is only about 1-2%. It is admitted
only after an isolated microbenchmark shows enough device-time reduction to
clear the 3% whole-lane gate; otherwise it remains deferred.

Artifacts:

- `build_logs/bench/decode-rocm-dual10-kernelroutes-l0-r1/`;
- `build_logs/bench/decode-rocm-dual10-nodeprofile-l0-r1/`;
- `build_logs/bench/decode-rocm-dual10-mmvqresources-l0-r1/`.

## Next work

1. Build a focused ROCm microbenchmark for the paired Q8_0 `5120 -> 48`
   alpha/beta matvec contract. Compare two existing launches with one
   two-output launch, including deterministic output checks.
2. Admit the fusion to the model graph only if the microbenchmark predicts at
   least 3% L0 decode improvement. Keep prefill on the existing large-batch
   route unless a separate prompt A/B proves neutrality.
3. If the fusion ceiling is too small, move to device-side profiling of the
   mixed-quant MMVQ graph and require a concrete memory/coalescing waste signal
   before changing kernels.
4. Re-run the standalone peer capability gate only after a HIP/driver update;
   never bypass `can_access=0`.

No performance claim from instrumentation runs is used as a wall comparator.