# E290: ROCm Q3_K cuBLAS Pipeline Gate

Date: 2026-07-14

## Scope

The existing experimental Q3_K cuBLAS pipeline splits large weight matrices
into row chunks, overlaps Q3_K-to-F16 conversion with GEMM on four auxiliary
streams, and uses two extra hipBLAS handles. This experiment tested whether it
could improve the remaining prompt route without changing batch geometry.

## Results

With the default graph guard, enabling the pipeline produced exactly the same
result as the E289 control because `ctx.any_cuda_graph_enabled()` blocked it
even though prompt eval was not being captured.

Allowing the route to run gave:

| Variant | Cold prompt | Warm prompt median | Decode |
| --- | ---: | ---: | ---: |
| E289 control | 1,598.66 | 1,824.11 | 28.43 |
| chunked pipeline, 6,144 rows | 937.29 | 1,017.36 | 29.19 |

The first auxiliary hipBLAS call paid a `358.9 ms` initialization cost, but
the route remained about `44%` slower after warmup. The pipeline also added
about `364 MiB` of unaccounted runtime memory per GPU.

## Root Cause and Decision

The overlap is real, but it is purchased by breaking a large efficient rocBLAS
GEMM into two or three smaller GEMMs and synchronizing the auxiliary streams
after every llama.cpp matrix operation. Lost GEMM efficiency and synchronization
dominate saved conversion overlap.

Reject the pipeline as a prompt route and keep it default-off. A future prompt
solution must preserve a full-size GEMM or use a new compressed GEMM body; it
must not repeat intra-matrix row chunking with per-op synchronization.

Primary artifacts:

- `e290-rocm-dual-q3k-pipeline6144-12k-none-r1r3.*`;
- `e290-rocm-dual-q3k-pipeline6144-allowgraphs-trace-12k-none-r1.*`;
- `e290-rocm-dual-q3k-pipeline6144-allowgraphs-12k-none-r1r3.*`.
