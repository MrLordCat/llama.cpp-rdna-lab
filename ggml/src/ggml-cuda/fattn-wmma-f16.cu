// Old and deprecated WMMA FlashAttention implementation.
// It is still needed for Volta since the memory layout of NVIDIA tensor cores changed with Turing.
// Long-term the WMMA code should be replaced with a dedicated Volta implementation.

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-wmma-f16.cuh"
#include "fp8.cuh"

#ifdef GGML_USE_WMMA_FATTN
#if !defined(GGML_USE_HIP)
#include <mma.h>
#if defined(GGML_USE_MUSA)
namespace wmma = mtmusa::wmma;
#else // GGML_USE_MUSA
namespace wmma = nvcuda::wmma;
#endif // GGML_USE_MUSA
#elif defined(GGML_USE_HIP)
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
#endif // !defined(GGML_USE_HIP)
#endif // GGML_USE_WMMA_FATTN

static int ggml_cuda_wmma_fattn_forced_cols_per_block() {
    static int cached = -1;
    if (cached != -1) {
        return cached;
    }

    const char * env = std::getenv("GGML_FATTN_WMMA_FORCE_COLS_PER_BLOCK");
    if (env == nullptr || env[0] == '\0') {
        cached = 0;
        return cached;
    }

    const int parsed = atoi(env);
    cached = (parsed == 16 || parsed == 32) ? parsed : 0;
    return cached;
}

// D102: template-bounded phase census for the D256 full-native N<=4 body.
// The small-N decode grid is at most (1, 8, 24) = 192 blocks. Each block
// writes its four accumulated clock64() phase deltas into pinned host memory
// through a device pointer installed by a tiny init kernel captured in the
// same graph. The host reads the pinned mirror with plain memory access at
// exit; no device-symbol host lookup and no HIP call after teardown is used.
constexpr int GGML_ROCM_FATTN_PHASE_CENSUS_BLOCKS = 192;
constexpr int GGML_ROCM_FATTN_PHASE_CENSUS_PHASES = 4;
__device__ unsigned long long * gg_rocm_fattn_phase_census_dst = nullptr;

#if defined(GGML_USE_HIP)
static __global__ void ggml_rocm_fattn_phase_census_init_kernel(
        unsigned long long * dst) {
    gg_rocm_fattn_phase_census_dst = dst;
}

static unsigned long long * gg_rocm_fattn_phase_census_host = nullptr;
static unsigned long long * gg_rocm_fattn_phase_census_host_dev = nullptr;
static int gg_rocm_fattn_phase_census_device_count = 0;

// D102: per-device phase shares are mirrored into a pinned host buffer by the
// census kernel itself. The mirror is refreshed on every graph replay, so the
// exit handler only reads plain host memory and never touches HIP.
static void ggml_rocm_fattn_phase_census_report() {
    if (gg_rocm_fattn_phase_census_host == nullptr) {
        return;
    }
    constexpr size_t region = GGML_ROCM_FATTN_PHASE_CENSUS_BLOCKS*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES;
    for (int dev = 0; dev < gg_rocm_fattn_phase_census_device_count; ++dev) {
        const unsigned long long * data = gg_rocm_fattn_phase_census_host + dev*region;

        unsigned long long totals[GGML_ROCM_FATTN_PHASE_CENSUS_PHASES] = {0, 0, 0, 0};
        int used = 0;
        for (int s = 0; s < GGML_ROCM_FATTN_PHASE_CENSUS_BLOCKS; ++s) {
            unsigned long long total = 0;
            for (int p = 0; p < GGML_ROCM_FATTN_PHASE_CENSUS_PHASES; ++p) {
                total += data[s*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES + p];
            }
            if (total == 0) {
                continue;
            }
            ++used;
            for (int p = 0; p < GGML_ROCM_FATTN_PHASE_CENSUS_PHASES; ++p) {
                totals[p] += data[s*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES + p];
            }
        }

        const unsigned long long sum = totals[0] + totals[1] + totals[2] + totals[3];
        if (sum > 0) {
            fprintf(stderr,
                "GGML_TRACE_FATTN_PHASE_CENSUS: dev=%d blocks=%d kq=%.1f%% softmax=%.1f%% pv=%.1f%% merge=%.1f%%\n",
                dev, used,
                100.0*totals[0]/sum, 100.0*totals[1]/sum,
                100.0*totals[2]/sum, 100.0*totals[3]/sum);
        }
    }
}

static void ggml_rocm_fattn_phase_census_prepare_copy(ggml_backend_cuda_context & ctx) {
    constexpr size_t region = GGML_ROCM_FATTN_PHASE_CENSUS_BLOCKS*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES;
    if (gg_rocm_fattn_phase_census_host == nullptr) {
        gg_rocm_fattn_phase_census_device_count = ggml_cuda_info().device_count;
        if (hipHostMalloc((void **) &gg_rocm_fattn_phase_census_host,
                sizeof(unsigned long long)*region*gg_rocm_fattn_phase_census_device_count,
                hipHostMallocDefault) != hipSuccess ||
            hipHostGetDevicePointer((void **) &gg_rocm_fattn_phase_census_host_dev,
                gg_rocm_fattn_phase_census_host, 0) != hipSuccess) {
            gg_rocm_fattn_phase_census_host = nullptr;
            gg_rocm_fattn_phase_census_host_dev = nullptr;
            gg_rocm_fattn_phase_census_device_count = 0;
            return;
        }
        std::atexit(ggml_rocm_fattn_phase_census_report);
    }
    // Captured init node: every replay re-installs this device's mirror
    // pointer before the census kernel writes its phase deltas into it.
    ggml_rocm_fattn_phase_census_init_kernel<<<1, 1, 0, ctx.stream()>>>(
        gg_rocm_fattn_phase_census_host_dev + ctx.device*region);
}
#endif // defined(GGML_USE_HIP)

// D == head size, VKQ_stride == num VKQ rows calculated in parallel:
template<int D, int ncols, int nwarps, int VKQ_stride, typename KQ_acc_t,
    bool use_logit_softcap, bool q8_v_direct = false, bool write_meta_single = false,
    bool native_f8_kq = false, bool native_f8_v = false, bool phase_census = false>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(GGML_HIP_ROCWMMA_FATTN) && defined(GGML_USE_WMMA_FATTN))
#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
    if constexpr (native_f8_kq || native_f8_v) {
        NO_DEVICE_CODE;
        return;
    } else {
#endif
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int ic0 = ncols*blockIdx.x; // Index of the first Q/QKV column to work on.

    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
#if defined(GGML_USE_HIP) && HIP_VERSION >= 60500000
    using kq_input_t = typename std::conditional<native_f8_kq, wmma::float8_t, _Float16>::type;
    using v_input_t  = typename std::conditional<native_f8_v,  wmma::float8_t, _Float16>::type;
    using vkq_acc_t  = typename std::conditional<native_f8_v,  float, _Float16>::type;
    static_assert(!native_f8_kq || std::is_same<KQ_acc_t, float>::value,
        "gfx12 FP8 WMMA requires fp32 KQ accumulators");
    static_assert(!native_f8_v || std::is_same<KQ_acc_t, float>::value,
        "gfx12 FP8 V phase requires fp32 KQ accumulators");
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, kq_input_t, wmma::row_major> frag_a_K;
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, v_input_t,  wmma::col_major> frag_a_V;
    typedef wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, kq_input_t, wmma::col_major> frag_b;
    // P (post-softmax KQ) feeds the V phase; with native FP8 V it must be
    // re-quantized to E4M3 for the fp8 x fp8 MMA.
    typedef wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, v_input_t, wmma::col_major> frag_b_v;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, vkq_acc_t>                         frag_c_VKQ;
#else
    static_assert(!native_f8_kq, "native FP8 KQ is HIP/rocWMMA-only");
    static_assert(!native_f8_v, "native FP8 V is HIP/rocWMMA-only");
    using kq_input_t = half;
    using v_input_t  = half;
    using vkq_acc_t  = half;
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, half, wmma::row_major> frag_a_K;
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, half, wmma::col_major> frag_a_V;
    typedef wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, half, wmma::col_major> frag_b_v;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c_VKQ;
#endif

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t)/sizeof(half);

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f    = (const float *) (Q    + nb03* sequence         + nb02* head              + nb01*ic0);
    const char  * K_data =                    K    + nb13* sequence         + nb12*(head / gqa_ratio);
    const char  * V_data =                    V    + nb23* sequence         + nb22*(head / gqa_ratio);
    const half  * V_h    = (const half  *) V_data;
    const block_q8_0 * V_q8 = (const block_q8_0 *) V_data;
    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33)                           + nb31*ic0);
    const half2 * mask2  = (const half2 *)  maskh;
    const float * sinksf = (const float *) sinks;

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_K  = nb11 / sizeof(kq_input_t);
    const int stride_V  = nb21 / sizeof(v_input_t);
    const int stride_V_q8 = nb21 / sizeof(block_q8_0);

    // D102 phase census accumulators. Unused in the production instantiations
    // and eliminated by the compiler there.
    unsigned long long census_t_kq = 0;
    unsigned long long census_t_sm = 0;
    unsigned long long census_t_pv = 0;
    unsigned long long census_t_mg = 0;
    unsigned long long census_t_prev = 0;

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);
    const half2 slope2 = make_half2(slopef, slopef);

    const half2 logit_softcap_2 = make_half2(logit_softcap, logit_softcap);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int q8_v_tile_rows = 32;
    constexpr int mem_KQ = ncols*kqs_padded*kqar;
    // With native FP8 V the VKQ accumulators are fp32, so the VKQ-part store
    // lives here in fp32 units (the P f8 tile is written in the softmax loop).
    constexpr int vkq_part_elems = VKQ_ratio*ncols*D_padded;
    constexpr int mem_VKQ_parts = native_f8_v ? vkq_part_elems*sizeof(float)/sizeof(half) : vkq_part_elems;
    constexpr int mem_V_q8 = q8_v_direct ? q8_v_tile_rows*D : 1;
    constexpr int mem_KQ_base = mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts;
    constexpr int mem_KQ_or_V = mem_KQ_base >= mem_V_q8 ? mem_KQ_base : mem_V_q8;
    __shared__ half KQ_or_V[mem_KQ_or_V];
    half * KQ = KQ_or_V;
    half * V_q8_f16 = KQ_or_V;
    float * KQ_f = (float *) KQ;
    half2 * KQ2 = (half2 *) KQ;
    float * VKQ_parts_f = (float *) KQ_or_V; // fp32 VKQ store, native FP8 V only
    __shared__ uint8_t P_f8[native_f8_v ? ncols*kqs_padded : 1];

    float    KQ_rowsum_f[ncols/nwarps] = {0.0f};
    float       KQ_max_f[ncols/nwarps];
    float KQ_max_scale_f[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX/2.0f;
    }

    half2    KQ_rowsum_h2[ncols/nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols/nwarps];
    half2 KQ_max_scale_h2[ncols/nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-HALF_MAX_HALF, -HALF_MAX_HALF);
    }

    __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    half2 * VKQ2 = (half2 *) VKQ;
    static_assert(!q8_v_direct || (D == 256 && (ncols == 16 || ncols == 32) && nwarps == 4),
        "RDNA4 direct Q8 V WMMA is specialized for D=256, ncols=16/32, nwarps=4");

#if defined(GGML_USE_HIP) && HIP_VERSION >= 60500000
    const kq_input_t * K_kq   = reinterpret_cast<const kq_input_t *>(K_data);
    const _Float16 * V_h_f16  = reinterpret_cast<const _Float16 *>(V_h);
    _Float16       * V_q8_f16_wmma = reinterpret_cast<_Float16 *>(V_q8_f16);
    kq_input_t     * Q_kq     = reinterpret_cast<kq_input_t *>(KQ);
    _Float16       * KQ_f16   = reinterpret_cast<_Float16 *>(KQ);
    _Float16       * VKQ_f16  = reinterpret_cast<_Float16 *>(VKQ);
#else
    const kq_input_t * K_kq = reinterpret_cast<const kq_input_t *>(K_data);
    const half * V_h_f16  = V_h;
    half       * V_q8_f16_wmma = V_q8_f16;
    kq_input_t * Q_kq     = reinterpret_cast<kq_input_t *>(KQ);
    half       * KQ_f16   = KQ;
    half       * VKQ_f16  = VKQ;
#endif

#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D/2 && i >= D/2) {
                break;
            }
            VKQ2[j*(D_padded/2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D && i >= D) {
                break;
            }
            const float q = ic0 + j < int(ne01.z) ? Q_f[j*stride_Q + i] : 0.0f;
            if constexpr (native_f8_kq) {
                reinterpret_cast<uint8_t *>(Q_kq)[j*D_padded + i] = ggml_cuda_fp32_to_f8_e4m3(q);
            } else {
                KQ[j*D_padded + i] = q * scale;
            }
        }
    }

    __syncthreads();

    // Load Q into tensor core fragments/registers since it will be used frequently:
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            wmma::load_matrix_sync(Q_b[i0/16][j0/frag_n], Q_kq + j0*D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    if constexpr (phase_census) {
        census_t_prev = clock64();
    }
    for (int k_VKQ_0 = blockIdx.y*FATTN_KQ_STRIDE; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                wmma::fill_fragment(KQ_c[j], static_cast<KQ_acc_t>(0.0f));
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                wmma::load_matrix_sync(
                    K_a,
                    K_kq + int64_t(k_VKQ_0 + i_KQ_0 + frag_m * threadIdx.y) * stride_K + k_KQ_0,
                    stride_K);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync(
                    (KQ_acc_t *) KQ + j0*kqs_padded + i_KQ_0 + frag_m*threadIdx.y,
                    KQ_c[j0/frag_n], kqs_padded, wmma::mem_col_major);
            }
        }

        __syncthreads();

        if constexpr (phase_census) {
            const unsigned long long census_t_now = clock64();
            census_t_kq += census_t_now - census_t_prev;
            census_t_prev = census_t_now;
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float KQ_f_tmp[FATTN_KQ_STRIDE / warp_size];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/warp_size] = KQ_f[j*kqs_padded + k];

                    if constexpr (native_f8_kq) {
                        KQ_f_tmp[k0/warp_size] *= scale;
                    }

                    if (use_logit_softcap) {
                        KQ_f_tmp[k0/warp_size] = logit_softcap*tanhf(KQ_f_tmp[k0/warp_size]);
                    }
                }

                float KQ_max_new = KQ_max_f[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    KQ_f_tmp[k0/warp_size] += mask && ic0 + j < int(ne01.z) ?
                        __half2float(slopeh*maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0/warp_size] + FATTN_KQ_MAX_OFFSET);
                }
                KQ_max_new = warp_reduce_max<warp_size>(KQ_max_new);

                const float diff = KQ_max_f[j0/nwarps] - KQ_max_new;
                KQ_max_scale_f[j0/nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0/nwarps] = 0.0f;
                }
                KQ_max_f[j0/nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
                if constexpr (native_f8_v) {
                    // D098 G4: P is pre-scaled into [0, 128], where gfx12 OCP
                    // E4M3 bytes match the persistent cache contract. Pack two
                    // lanes per native conversion instead of running the
                    // scalar software encoder for every softmax value.
                    static_assert(FATTN_KQ_STRIDE % (2*warp_size) == 0,
                        "packed P conversion requires an even number of warp slices");
                    constexpr float p_f8_scale = 128.0f;
#pragma unroll 1
                    for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += 2*warp_size) {
                        const int k = k0 + threadIdx.x;
                        const float diff0 = KQ_f_tmp[k0/warp_size] - KQ_max_f[j0/nwarps];
                        const float diff1 = KQ_f_tmp[k0/warp_size + 1] - KQ_max_f[j0/nwarps];
                        float p0 = diff0 > SOFTMAX_FTZ_THRESHOLD ? expf(diff0) : 0.0f;
                        float p1 = diff1 > SOFTMAX_FTZ_THRESHOLD ? expf(diff1) : 0.0f;
                        KQ_rowsum_add += p0 + p1;
                        const uint16_t packed = ggml_cuda_fp32x2_to_f8_e4m3_p(
                            make_float2(p0*p_f8_scale, p1*p_f8_scale));
                        P_f8[j*kqs_padded + k] = (uint8_t) packed;
                        P_f8[j*kqs_padded + k + warp_size] = (uint8_t) (packed >> 8);
                    }
                } else {
#pragma unroll
                    for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                        const int k = k0 + threadIdx.x;
                        const float diff = KQ_f_tmp[k0/warp_size] - KQ_max_f[j0/nwarps];
                        KQ_f_tmp[k0/warp_size] = expf(diff);
                        if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                            KQ_f_tmp[k0/warp_size] = 0.0f;
                        }
                        KQ_rowsum_add += KQ_f_tmp[k0/warp_size];
                        KQ[j*(kqar*kqs_padded) + k] = KQ_f_tmp[k0/warp_size];
                    }
                }
                KQ_rowsum_add = warp_reduce_sum<warp_size>(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0/nwarps] = KQ_max_scale_f[j0/nwarps]*KQ_rowsum_f[j0/nwarps] + KQ_rowsum_add;
            } else {
                half2 KQ2_tmp[FATTN_KQ_STRIDE/(2*warp_size)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/warp_size] = KQ2[j*(kqs_padded/2) + k];

                    if (use_logit_softcap) {
                        // There is no dedicated tangens hyperbolicus function for half2.
                        KQ2_tmp[k0/warp_size] = h2exp(KQ2_tmp[k0/warp_size]*make_half2(2.0f, 2.0f));
                        KQ2_tmp[k0/warp_size] = (KQ2_tmp[k0/warp_size] - make_half2(1.0f, 1.0f))
                                               /(KQ2_tmp[k0/warp_size] + make_half2(1.0f, 1.0f));

                        KQ2_tmp[k0/warp_size] *= logit_softcap_2;
                    }
                }

                half2 KQ_max_new = KQ_max_h2[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    KQ2_tmp[k0/warp_size] += mask && ic0 + j < int(ne01.z) ?
                        slope2*mask2[(j*(nb31/sizeof(half)) + k_VKQ_0)/2 + k] :
                        make_half2(0.0f, 0.0f);
                    KQ_max_new = ggml_cuda_hmax2(KQ_max_new, KQ2_tmp[k0/warp_size]);
                }
                KQ_max_new = __half2half2(warp_reduce_max<warp_size>(ggml_cuda_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0/nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0/nwarps] = h2exp(diff);
                const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                *((uint32_t *) &KQ_max_scale_h2[j0/nwarps]) &= ftz_mask;
                KQ_max_h2[j0/nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += warp_size) {
                    const int k = k0 + threadIdx.x;

                    const half2 diff = KQ2_tmp[k0/warp_size] - KQ_max_h2[j0/nwarps];
                    KQ2_tmp[k0/warp_size] = h2exp(diff);
                    const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                    *((uint32_t *) &KQ2_tmp[k0/warp_size]) &= ftz_mask;
                    KQ_rowsum_add += KQ2_tmp[k0/warp_size];
                    KQ2[j*(kqs_padded/2) + k] = KQ2_tmp[k0/warp_size];
                }
                KQ_rowsum_add = warp_reduce_sum<warp_size>(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0/nwarps] = KQ_max_scale_h2[j0/nwarps]*KQ_rowsum_h2[j0/nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        if constexpr (phase_census) {
            const unsigned long long census_t_now = clock64();
            census_t_sm += census_t_now - census_t_prev;
            census_t_prev = census_t_now;
        }

        frag_b_v KQ_b[FATTN_KQ_STRIDE/(VKQ_ratio*16)][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (threadIdx.y % VKQ_ratio)*16;
                if constexpr (native_f8_v) {
                    wmma::load_matrix_sync(
                        KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                        reinterpret_cast<const v_input_t *>(P_f8) + j0*kqs_padded + k,
                        kqs_padded);
                } else {
                    wmma::load_matrix_sync(
                        KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                        KQ_f16 + j0*(kqar*kqs_padded) + k,
                        kqar*kqs_padded);
                }
            }
        }

        frag_c_VKQ VKQ_c[D/VKQ_stride][ncols/frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                wmma::fill_fragment(VKQ_c[i_VKQ_0/VKQ_stride][j], static_cast<vkq_acc_t>(0.0f));
            }
        }

        if constexpr (q8_v_direct) {
            static_assert(VKQ_ratio == 1, "direct Q8 V WMMA expects one V accumulator group");
            constexpr int blocks_per_row = D / QK8_0;
            constexpr int thread_count = nwarps * warp_size;
            const int tid = threadIdx.y * warp_size + threadIdx.x;
            __syncthreads();
#pragma unroll
            for (int k_base = 0; k_base < FATTN_KQ_STRIDE; k_base += q8_v_tile_rows) {
                for (int item = tid; item < q8_v_tile_rows * blocks_per_row; item += thread_count) {
                    const int row = item / blocks_per_row;
                    const int ib = item % blocks_per_row;
                    const block_q8_0 * src = V_q8 +
                        int64_t(k_VKQ_0 + k_base + row) * stride_V_q8;
                    half * dst = V_q8_f16 + row * D + ib * QK8_0;
                    dequantize_V_q8_0<half, QK8_0 / 2>(src, dst, ib * QK8_0);
                    dequantize_V_q8_0<half, QK8_0 / 2>(
                        src, dst + QK8_0 / 2, ib * QK8_0 + QK8_0 / 2);
                }
                __syncthreads();

#pragma unroll
                for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
                    for (int k0 = 0; k0 < q8_v_tile_rows; k0 += 16) {
                        frag_a_V v_a;
                        wmma::load_matrix_sync(
                            v_a,
                            V_q8_f16_wmma + int64_t(k0) * D +
                                i_VKQ_0 + frag_m * threadIdx.y,
                            D);
#pragma unroll
                        for (int j = 0; j < ncols/frag_n; ++j) {
                            wmma::mma_sync(
                                VKQ_c[i_VKQ_0/VKQ_stride][j], v_a,
                                KQ_b[(k_base + k0)/16][j],
                                VKQ_c[i_VKQ_0/VKQ_stride][j]);
                        }
                    }
                }
                __syncthreads();
            }
        } else {
            const v_input_t * V_v = reinterpret_cast<const v_input_t *>(V_data);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                    const int k = k0 + (threadIdx.y % VKQ_ratio)*16;

                    frag_a_V v_a;
                    wmma::load_matrix_sync(
                        v_a,
                        V_v + int64_t(k_VKQ_0 + k) * stride_V +
                            i_VKQ_0 + frag_m * (threadIdx.y / VKQ_ratio),
                        stride_V);
#pragma unroll
                    for (int j = 0; j < ncols/frag_n; ++j) {
                        wmma::mma_sync(VKQ_c[i_VKQ_0/VKQ_stride][j], v_a, KQ_b[k0/(VKQ_ratio*16)][j], VKQ_c[i_VKQ_0/VKQ_stride][j]);
                    }
                }
            }
        }

        __syncthreads();

        const int offset_k = (threadIdx.y % VKQ_ratio) * (ncols*D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                if constexpr (native_f8_v) {
                    wmma::store_matrix_sync(
                        VKQ_parts_f + offset_k + j0*D_padded + i_KQ_0 + frag_m*(threadIdx.y/VKQ_ratio),
                        VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                        D_padded, wmma::mem_col_major);
                } else {
                    wmma::store_matrix_sync(
                        KQ_f16 + offset_k + j0*D_padded + i_KQ_0 + frag_m*(threadIdx.y/VKQ_ratio),
                        VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                        D_padded, wmma::mem_col_major);
                }
            }
        }

        __syncthreads();

        if constexpr (phase_census) {
            const unsigned long long census_t_now = clock64();
            census_t_pv += census_t_now - census_t_prev;
            census_t_prev = census_t_now;
        }

        if constexpr (native_f8_v) {
            static_assert(VKQ_ratio == 1, "native FP8 V merge expects one accumulator group");
#pragma unroll 1
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;
                const half2 VKQ_scale =
                    make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]);
#pragma unroll 1
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;
                    const float add0 = VKQ_parts_f[j*D_padded + 2*i]/128.0f;
                    const float add1 = VKQ_parts_f[j*D_padded + 2*i + 1]/128.0f;
                    VKQ2[j*(D_padded/2) + i] =
                        VKQ_scale*VKQ2[j*(D_padded/2) + i] + make_half2(add0, add1);
                }
            }
        } else {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;
                const half2 VKQ_scale = std::is_same<KQ_acc_t, float>::value ?
                    make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]) :
                    KQ_max_scale_h2[j0/nwarps];
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;
                    if (i0 + warp_size > D/2 && i >= D/2) {
                        break;
                    }
                    half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                    for (int l = 0; l < VKQ_ratio; ++l) {
                        VKQ_add += KQ2[l*(ncols*D_padded/2) + j*(D_padded/2) + i];
                    }
                    VKQ2[j*(D_padded/2) + i] =
                        VKQ_scale*VKQ2[j*(D_padded/2) + i] + VKQ_add;
                }
            }
        }

        __syncthreads();

        if constexpr (phase_census) {
            const unsigned long long census_t_now = clock64();
            census_t_mg += census_t_now - census_t_prev;
            const unsigned long long linear =
                ((unsigned long long) blockIdx.z*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;
            if (linear < GGML_ROCM_FATTN_PHASE_CENSUS_BLOCKS && threadIdx.x == 0 && threadIdx.y == 0 &&
                    gg_rocm_fattn_phase_census_dst != nullptr) {
                gg_rocm_fattn_phase_census_dst[linear*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES + 0] = census_t_kq;
                gg_rocm_fattn_phase_census_dst[linear*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES + 1] = census_t_sm;
                gg_rocm_fattn_phase_census_dst[linear*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES + 2] = census_t_pv;
                gg_rocm_fattn_phase_census_dst[linear*GGML_ROCM_FATTN_PHASE_CENSUS_PHASES + 3] = census_t_mg;
                __threadfence_system();
            }
        }
    }

    // Apply attention sinks
    if (sinksf && blockIdx.y == 0) {
        const float sinkf = sinksf[head];
        const half  sinkh = __float2half(sinkf);

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (std::is_same<KQ_acc_t, float>::value) {
                float kqmax_new = fmaxf(KQ_max_f[j0/nwarps], sinkf);

                const float KQ_max_scale = expf(KQ_max_f[j0/nwarps] - kqmax_new);
                KQ_max_f[j0/nwarps] = kqmax_new;

                KQ_rowsum_f[j0/nwarps] = KQ_rowsum_f[j0/nwarps] * KQ_max_scale + expf(sinkf - KQ_max_f[j0/nwarps]);

                const half2 scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;
                    if (i0 + warp_size > D/2 && i >= D/2) break;
                    VKQ2[j*(D_padded/2) + i] *= scale_h2;
                }
            } else {
                half kqmax_old = __low2half(KQ_max_h2[j0/nwarps]);
                half kqmax_new = fmaxf(kqmax_old, sinkh);
                KQ_max_h2[j0/nwarps] = __half2half2(kqmax_new);

                const half  KQ_max_scale_h = hexp(kqmax_old - kqmax_new);
                const half2 KQ_max_scale   = __half2half2(KQ_max_scale_h);

                KQ_rowsum_h2[j0/nwarps] = KQ_rowsum_h2[j0/nwarps] * KQ_max_scale;
                const half val = hexp(sinkh - kqmax_new);
                KQ_rowsum_h2[j0/nwarps].x = __hadd(KQ_rowsum_h2[j0/nwarps].x, val);

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + threadIdx.x;
                    if (i0 + warp_size > D/2 && i >= D/2) break;
                    VKQ2[j*(D_padded/2) + i] *= KQ_max_scale;
                }
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + threadIdx.y;
        if (ic0 + j_VKQ >= int(ne01.z)) {
            return;
        }

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0/nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0/nwarps]) + __high2float(KQ_rowsum_h2[j0/nwarps]);
        }

        const int j_dst_unrolled = ((sequence*int(ne01.z) + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (i0 + warp_size > D && i >= D) {
                break;
            }
            float dst_val = VKQ[j_VKQ*D_padded + i];
            if (gridDim.y == 1) {
                dst_val /= KQ_rowsum_j;
            }
            dst[j_dst_unrolled*D + i] = dst_val;
        }

        if ((gridDim.y == 1 && !write_meta_single) || threadIdx.x != 0) {
            continue;
        }

        float2 dst_meta_val;
        if (std::is_same<KQ_acc_t, float>::value) {
            dst_meta_val.x = KQ_max_f[j0/nwarps];
        } else {
            dst_meta_val.x = __low2float(KQ_max_h2[j0/nwarps]);
        }
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[j_dst_unrolled] = dst_meta_val;
    }
#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX12__)
    }
#endif
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(GGML_HIP_ROCWMMA_FATTN) && defined(GGML_USE_WMMA_FATTN))
}

constexpr int get_max_power_of_2(int x) {
    return x % 2 == 0 ? 2*get_max_power_of_2(x/2) : 1;
}

static_assert(get_max_power_of_2(1) == 1, "Test failed.");
static_assert(get_max_power_of_2(2) == 2, "Test failed.");
static_assert(get_max_power_of_2(4) == 4, "Test failed.");
static_assert(get_max_power_of_2(6) == 2, "Test failed.");

// Number of VKQ rows calculated in parallel:
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    return (get_max_power_of_2(D/frag_m) < nwarps ? get_max_power_of_2(D/frag_m) : nwarps)*frag_m;
}

static_assert(get_VKQ_stride(128, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride(128, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride(128, 4, 32) == 128, "Test failed.");
static_assert(get_VKQ_stride( 64, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride( 64, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 64, 4, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 80, 1, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 2, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 4, 16) ==  16, "Test failed.");

template <int D>
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_chunk(
        const float * __restrict__ partial,
        const float2 * __restrict__ partial_meta,
        float * __restrict__ accum,
        float2 * __restrict__ accum_meta,
        const bool first_chunk) {
    const int row = blockIdx.x;
    const int i = threadIdx.x;

    const float2 next_meta = partial_meta[row];
    if (first_chunk) {
        accum[row*D + i] = next_meta.y > 0.0f ? partial[row*D + i] : 0.0f;
        if (i == 0) {
            accum_meta[row] = next_meta;
        }
        return;
    }

    const float2 prev_meta = accum_meta[row];
    if (next_meta.y <= 0.0f) {
        return;
    }
    if (prev_meta.y <= 0.0f) {
        accum[row*D + i] = partial[row*D + i];
        if (i == 0) {
            accum_meta[row] = next_meta;
        }
        return;
    }

    const float max_new = fmaxf(prev_meta.x, next_meta.x);
    const float prev_weight = prev_meta.y * expf(prev_meta.x - max_new);
    const float next_weight = next_meta.y * expf(next_meta.x - max_new);
    const float sum_new = prev_weight + next_weight;

    accum[row*D + i] =
        (prev_weight * accum[row*D + i] + next_weight * partial[row*D + i]) / sum_new;
    if (i == 0) {
        accum_meta[row] = make_float2(max_new, sum_new);
    }
}

template <int D, int ncols>
static void launch_fattn_chunked_q8_wmma(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel) {
    constexpr int nwarps = 4;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(V->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(Q->ne[0] == D && V->ne[0] == D);
    GGML_ASSERT(K->ne[1] == V->ne[1]);
    GGML_ASSERT(K->ne[2] == V->ne[2]);
    GGML_ASSERT(K->ne[3] == V->ne[3]);

    const ggml_cuda_flash_attn_ext_chunked_extra_data extra =
        ggml_cuda_flash_attn_ext_get_chunked_extra_data(dst);
    half * K_f16 = reinterpret_cast<half *>(extra.K);
    half * V_f16 = reinterpret_cast<half *>(extra.V);
    float * partial = reinterpret_cast<float *>(extra.partial);
    float2 * partial_meta = reinterpret_cast<float2 *>(extra.partial_meta);
    float2 * accum_meta = reinterpret_cast<float2 *>(extra.accum_meta);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> KV_max(ctx.pool());

    const int ntiles_x = (Q->ne[1] + ncols - 1) / ncols;
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0) {
        KV_max.alloc(ntiles_x * Q->ne[3]);
    }

    const to_fp16_nc_cuda_t K_to_f16 = ggml_get_to_fp16_nc_cuda(K->type);
    const to_fp16_nc_cuda_t V_to_f16 = ggml_get_to_fp16_nc_cuda(V->type);
    GGML_ASSERT(K_to_f16 != nullptr && V_to_f16 != nullptr);

    const size_t K_ts = ggml_type_size(K->type);
    const size_t V_ts = ggml_type_size(V->type);
    const int64_t K_s01 = K->nb[1] / K_ts;
    const int64_t K_s02 = K->nb[2] / K_ts;
    const int64_t K_s03 = K->nb[3] / K_ts;
    const int64_t V_s01 = V->nb[1] / V_ts;
    const int64_t V_s02 = V->nb[2] / V_ts;
    const int64_t V_s03 = V->nb[3] / V_ts;

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -max_bias / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    const dim3 block_dim(ggml_cuda_info().devices[ctx.device].warp_size, nwarps, 1);
    const dim3 blocks_num(ntiles_x, 1, Q->ne[2] * Q->ne[3]);
    const dim3 combine_blocks(ggml_nrows(dst), 1, 1);
    const dim3 combine_threads(D, 1, 1);

    bool first_chunk = true;
    for (int64_t k0 = 0; k0 < K->ne[1]; k0 += GGML_ROCM_FATTN_Q8_CHUNK_SIZE) {
        const int64_t nk = std::min<int64_t>(GGML_ROCM_FATTN_Q8_CHUNK_SIZE, K->ne[1] - k0);
        GGML_ASSERT(nk % FATTN_KQ_STRIDE == 0);

        K_to_f16(
            reinterpret_cast<const char *>(K->data) + k0*K->nb[1], K_f16,
            K->ne[0], nk, K->ne[2], K->ne[3], K_s01, K_s02, K_s03, stream);
        V_to_f16(
            reinterpret_cast<const char *>(V->data) + k0*V->nb[1], V_f16,
            V->ne[0], nk, V->ne[2], V->ne[3], V_s01, V_s02, V_s03, stream);

        const char * mask_chunk = mask ?
            reinterpret_cast<const char *>(mask->data) + k0*sizeof(half) : nullptr;
        if (mask && KV_max.ptr != nullptr) {
            const dim3 mask_blocks(ntiles_x, Q->ne[3], 1);
            const dim3 mask_threads(FATTN_KQ_STRIDE/2, 1, 1);
            flash_attn_mask_to_KV_max<ncols><<<mask_blocks, mask_threads, 0, stream>>>(
                reinterpret_cast<const half2 *>(mask_chunk), KV_max.ptr,
                nk / FATTN_KQ_STRIDE,
                mask->nb[1] / sizeof(half2), mask->nb[3] / sizeof(half2));
            CUDA_CHECK(cudaGetLastError());
        }

        const int32_t nb11 = D * sizeof(half);
        const int32_t nb12 = nk * nb11;
        const int64_t nb13 = K->ne[2] * int64_t(nb12);
        const int32_t nb21 = D * sizeof(half);
        const int32_t nb22 = nk * nb21;
        const int64_t nb23 = V->ne[2] * int64_t(nb22);

        fattn_kernel<<<blocks_num, block_dim, 0, stream>>>(
            reinterpret_cast<const char *>(Q->data),
            reinterpret_cast<const char *>(K_f16),
            reinterpret_cast<const char *>(V_f16),
            mask_chunk,
            first_chunk && sinks ? reinterpret_cast<const char *>(sinks->data) : nullptr,
            KV_max.ptr,
            partial, partial_meta,
            scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            Q->ne[0], ne01, Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], nk, K->ne[2], K->ne[3], nb11, nb12, nb13,
            nb21, nb22, nb23,
            mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
            mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0);
        CUDA_CHECK(cudaGetLastError());

        flash_attn_combine_chunk<D><<<combine_blocks, combine_threads, 0, stream>>>(
            partial, partial_meta, reinterpret_cast<float *>(dst->data), accum_meta, first_chunk);
        CUDA_CHECK(cudaGetLastError());
        first_chunk = false;
    }
}

template <int D, int cols_per_block, typename KQ_acc_t, int nwarps = 4, bool f8_native_only = false>
void ggml_cuda_flash_attn_ext_wmma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    constexpr int frag_m = cols_per_block == 8 && D % 32 == 0 ? 32 : 16;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    const bool q8_v_direct = ggml_cuda_flash_attn_ext_use_rdna4_q8_v_direct_wmma(ctx.device, dst);
    const bool q8_chunked = ggml_cuda_flash_attn_ext_use_rdna4_q8_chunked_wmma(ctx.device, dst);
    const bool f8_native_kq = ggml_cuda_flash_attn_ext_use_rdna4_f8_native_kq(ctx.device, dst);
    const bool f8_native_v  = ggml_cuda_flash_attn_ext_use_rdna4_f8_native_v(ctx.device, dst);

    if constexpr (f8_native_only) {
        static_assert(D == 256 && cols_per_block == 16 && nwarps == 8,
            "the native-only body owns the D=256/cols16/warps8 shape");
        GGML_ASSERT(f8_native_v);
        const ggml_tensor * Q = dst->src[0];
        bool phase_census_enabled = false;
#if defined(GGML_USE_HIP)
        // D102: per-phase clock64 shares, decode/verify rows only. Prefill and
        // every other route keep the unchanged production instantiation.
        phase_census_enabled = std::getenv("GGML_ROCM_FATTN_PHASE_CENSUS") != nullptr &&
            Q->ne[1] <= 4;
#endif
        fattn_kernel_t f8_kernel;
        if (logit_softcap == 0.0f) {
            if (f8_native_kq) {
                if (phase_census_enabled) {
#if defined(GGML_USE_HIP)
                    // D102: mirror the census slots into pinned host memory
                    // with an async node inside the same capture. No sync and
                    // no HIP call ever happens after teardown.
                    ggml_rocm_fattn_phase_census_prepare_copy(ctx);
#endif
                    f8_kernel = flash_attn_ext_f16<
                        D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                        float, false, false, false, true, true, true>;
                } else {
                    f8_kernel = flash_attn_ext_f16<
                        D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                        float, false, false, false, true, true>;
                }
            } else {
                f8_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    float, false, false, false, false, true>;
            }
        } else {
            if (f8_native_kq) {
                f8_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    float, true, false, false, true, true>;
            } else {
                f8_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    float, true, false, false, false, true>;
            }
        }
        launch_fattn<D, cols_per_block, 1>(
            ctx, dst, f8_kernel, nwarps, 0, FATTN_KQ_STRIDE,
            !f8_native_kq, false, false, warp_size);
        return;
    }

    if constexpr (!f8_native_only) {
    fattn_kernel_t fattn_kernel;
    if (f8_native_kq || f8_native_v) {
        // D099: four-wave native phases own only the resource-audited D/cols16
        // matrix. Full native D256 is dispatched through the separate eight-
        // wave specialization above; D112 deliberately permits KQ only.
        if constexpr ((D == 64 || D == 80 || D == 96 || D == 112 || D == 128 || D == 256) &&
                cols_per_block == 16) {
            if (logit_softcap == 0.0f) {
                if constexpr (D == 64 || D == 128 || D == 256) {
                    if (f8_native_kq && f8_native_v) {
                        fattn_kernel = flash_attn_ext_f16<
                            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                            float, false, false, false, true, true>;
                    } else if (f8_native_kq) {
                        fattn_kernel = flash_attn_ext_f16<
                            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                            float, false, false, false, true, false>;
                    } else {
                        fattn_kernel = flash_attn_ext_f16<
                            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                            float, false, false, false, false, true>;
                    }
                } else {
                    GGML_ASSERT(f8_native_kq && !f8_native_v);
                    fattn_kernel = flash_attn_ext_f16<
                        D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                        float, false, false, false, true, false>;
                }
            } else {
                if constexpr (D == 64 || D == 128 || D == 256) {
                    if (f8_native_kq && f8_native_v) {
                        fattn_kernel = flash_attn_ext_f16<
                            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                            float, true, false, false, true, true>;
                    } else if (f8_native_kq) {
                        fattn_kernel = flash_attn_ext_f16<
                            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                            float, true, false, false, true, false>;
                    } else {
                        fattn_kernel = flash_attn_ext_f16<
                            D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                            float, true, false, false, false, true>;
                    }
                } else {
                    GGML_ASSERT(f8_native_kq && !f8_native_v);
                    fattn_kernel = flash_attn_ext_f16<
                        D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                        float, true, false, false, true, false>;
                }
            }
            launch_fattn<D, cols_per_block, 1>(
                ctx, dst, fattn_kernel, nwarps, 0, FATTN_KQ_STRIDE,
                !f8_native_kq, !f8_native_v, false, warp_size);
            return;
        } else {
            GGML_ABORT("native FP8 route outside the D099 resource matrix");
        }
    }

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        if constexpr (D == 256 && (cols_per_block == 16 || cols_per_block == 32)) {
            if (q8_chunked) {
                fattn_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    KQ_acc_t, use_logit_softcap, false, true>;
            } else if (q8_v_direct) {
                fattn_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    KQ_acc_t, use_logit_softcap, true>;
            } else {
                fattn_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    KQ_acc_t, use_logit_softcap, false>;
            }
        } else {
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                KQ_acc_t, use_logit_softcap, false>;
        }
    } else {
        constexpr bool use_logit_softcap = true;
        if constexpr (D == 256 && (cols_per_block == 16 || cols_per_block == 32)) {
            if (q8_chunked) {
                fattn_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    KQ_acc_t, use_logit_softcap, false, true>;
            } else if (q8_v_direct) {
                fattn_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    KQ_acc_t, use_logit_softcap, true>;
            } else {
                fattn_kernel = flash_attn_ext_f16<
                    D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                    KQ_acc_t, use_logit_softcap, false>;
            }
        } else {
            fattn_kernel = flash_attn_ext_f16<
                D, cols_per_block, nwarps, get_VKQ_stride(D, nwarps, frag_m),
                KQ_acc_t, use_logit_softcap, false>;
        }
    }
    if constexpr (D == 256 && (cols_per_block == 16 || cols_per_block == 32)) {
        if (q8_chunked) {
            launch_fattn_chunked_q8_wmma<D, cols_per_block>(ctx, dst, fattn_kernel);
            return;
        }
    }
    launch_fattn<D, cols_per_block, 1>(
        ctx, dst, fattn_kernel, nwarps, 0, FATTN_KQ_STRIDE,
        true, !q8_v_direct, false, warp_size);
    }
}

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int forced_cols_per_block = ggml_cuda_wmma_fattn_forced_cols_per_block();
    const bool trace_wmma_cfg = std::getenv("GGML_TRACE_FATTN_WMMA_CONFIG") != nullptr;

    auto trace_dispatch = [&](const int cols_per_block) {
        if (!trace_wmma_cfg) {
            return;
        }
        const uintptr_t q_ptr = (uintptr_t) Q->data;
        const uintptr_t k_ptr = (uintptr_t) dst->src[1]->data;
        const uintptr_t v_ptr = (uintptr_t) dst->src[2]->data;
        fprintf(stderr,
            "GGML_TRACE_FATTN_WMMA_CONFIG: D=%lld q_rows=%lld prec=%d forced_cols=%d selected_cols=%d "
            "q_mod128=%llu k_mod128=%llu v_mod128=%llu q_nb1=%lld q_nb2=%lld\n",
            (long long) Q->ne[0],
            (long long) Q->ne[1],
            (int) prec,
            forced_cols_per_block,
            cols_per_block,
            (unsigned long long) (q_ptr & 127ULL),
            (unsigned long long) (k_ptr & 127ULL),
            (unsigned long long) (v_ptr & 127ULL),
            (long long) Q->nb[1],
            (long long) Q->nb[2]);
    };

    if (ggml_cuda_flash_attn_ext_use_rdna4_f8_native_kq(ctx.device, dst) ||
        ggml_cuda_flash_attn_ext_use_rdna4_f8_native_v(ctx.device, dst)) {
        // D099 native FP8 always uses the fp32-accumulator 16-column body.
        constexpr int cols_per_block = 16;
        // The full native FP8 body uses eight waves to halve each lane's VKQ
        // fragment count. Keeping this as a separate native-only instantiation
        // avoids pulling the four-wave-only q8 variants into the 256-thread
        // specialization. On gfx1201 this is spill-free (154-156 VGPR) and
        // wins both the 12K and 49K full-FP8 lanes.
        if (Q->ne[0] == 256) {
            if (ggml_cuda_flash_attn_ext_use_rdna4_f8_native_v(ctx.device, dst)) {
                trace_dispatch(cols_per_block);
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, float, 8, true>(ctx, dst);
            } else {
                trace_dispatch(cols_per_block);
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
            }
            return;
        }

        trace_dispatch(cols_per_block);

        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                break;
            case 80:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                break;
            case 112:
                ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                break;
            default:
                GGML_ABORT("native FP8 dispatch outside the D099 head-dimension matrix");
        }
        return;
    }

    if (prec != GGML_PREC_DEFAULT) {
        if (forced_cols_per_block == 16 ||
                (forced_cols_per_block == 0 && (Q->ne[1] <= 32 || Q->ne[0] > 128))) {
            constexpr int cols_per_block = 16;
            trace_dispatch(cols_per_block);
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                    break;
                case 256:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
                    break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            trace_dispatch(cols_per_block);
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                    break;
                // case 256:
                //     ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
                //     break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        }
        return;
    }

#if !defined(GGML_USE_HIP)
    if (Q->ne[1] <= 8 && Q->ne[0] % warp_size == 0) {
        constexpr int cols_per_block = 8;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, half>(ctx, dst);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }
#endif // !defined(GGML_USE_HIP)

    if (forced_cols_per_block == 16 || (forced_cols_per_block == 0 && Q->ne[1] <= 32)) {
        constexpr int cols_per_block = 16;
        trace_dispatch(cols_per_block);
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, half>(ctx, dst);
                break;
            case 80:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, half>(ctx, dst);
                break;
            case 112:
                ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, half>(ctx, dst);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    constexpr int cols_per_block = 32;
    trace_dispatch(cols_per_block);
    switch (Q->ne[0]) {
        case 64:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 64, cols_per_block, half>(ctx, dst);
            break;
        case 80:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 80, cols_per_block, half>(ctx, dst);
            break;
        case 96:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 96, cols_per_block, half>(ctx, dst);
            break;
        case 112:
            ggml_cuda_flash_attn_ext_wmma_f16_case<112, cols_per_block, half>(ctx, dst);
            break;
        case 128:
            ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, half>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_wmma_f16_case<256, cols_per_block, half>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
