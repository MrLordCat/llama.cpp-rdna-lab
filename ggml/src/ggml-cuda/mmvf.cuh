#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// G10: fused pair of narrow f32 MMVF matvecs with a shared input vector
// (GDN ssm_alpha + ssm_beta, 5120 -> 48, n=1). Writes both outputs from a
// single launch; identical inner-loop and reduce order to the separate
// mul_mat_vec_f_cuda<float> calls so the results stay bit-for-bit equal.
void ggml_cuda_mul_mat_vec_f_pair(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_a, const ggml_tensor * src1, ggml_tensor * dst_a,
        const ggml_tensor * src0_b, ggml_tensor * dst_b, cudaStream_t stream);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);
