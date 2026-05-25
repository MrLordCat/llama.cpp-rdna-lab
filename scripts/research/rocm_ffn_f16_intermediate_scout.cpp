// Standalone ROCm/rocBLAS FFN intermediate precision scout.
// Compares a current-style f32 up/gate -> f32 GLU -> f32-to-f16 -> down GEMM
// against a hypothetical f16 up/gate -> f16 GLU -> down GEMM route.
// Diagnostic utility only; not part of normal builds.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#define HIP_CHECK(expr)                                                                         \
    do {                                                                                        \
        hipError_t status__ = (expr);                                                           \
        if (status__ != hipSuccess) {                                                           \
            std::cerr << "HIP error: " << hipGetErrorString(status__) << " at " << __LINE__     \
                      << std::endl;                                                             \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

#define ROCBLAS_CHECK(expr)                                                                     \
    do {                                                                                        \
        rocblas_status status__ = (expr);                                                       \
        if (status__ != rocblas_status_success) {                                               \
            std::cerr << "rocBLAS error: " << rocblas_status_to_string(status__) << " at "     \
                      << __LINE__ << std::endl;                                                 \
            return 1;                                                                           \
        }                                                                                       \
    } while (0)

struct args_t {
    int n = 2048;
    int k_in = 5120;
    int k_ff = 17408;
    int m_out = 5120;
    int warmup = 4;
    int iters = 12;
    int device = 0;
};

static __device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

static __global__ void glu_f32_kernel(const float * up, const float * gate, float * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = silu_f(gate[i]) * up[i];
    }
}

static __global__ void glu_f16_kernel(const __half * up, const __half * gate, __half * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float u = __half2float(up[i]);
        const float g = __half2float(gate[i]);
        dst[i] = __float2half(silu_f(g) * u);
    }
}

static __global__ void f32_to_f16_kernel(const float * src, __half * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

static void usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " [--n N --k-in K --k-ff K --m-out M] [--warmup N] [--iters N] [--device N]\n";
}

static bool parse_int(const char * text, int & out) {
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_args(int argc, char ** argv, args_t & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](int & value) -> bool {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return false;
            }
            return parse_int(argv[++i], value);
        };

        if (key == "--n") {
            if (!need_value(args.n)) return false;
        } else if (key == "--k-in") {
            if (!need_value(args.k_in)) return false;
        } else if (key == "--k-ff") {
            if (!need_value(args.k_ff)) return false;
        } else if (key == "--m-out") {
            if (!need_value(args.m_out)) return false;
        } else if (key == "--warmup") {
            if (!need_value(args.warmup)) return false;
        } else if (key == "--iters") {
            if (!need_value(args.iters)) return false;
        } else if (key == "--device") {
            if (!need_value(args.device)) return false;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            return false;
        }
    }

    return args.n > 0 && args.k_in > 0 && args.k_ff > 0 && args.m_out > 0 && args.warmup >= 0 && args.iters > 0;
}

static rocblas_status gemm_f32_out(
        rocblas_handle handle, int m, int n, int k,
        const void * a, const void * b, float * d) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        m, n, k,
        &alpha, a, rocblas_datatype_f16_r, k,
        b, rocblas_datatype_f16_r, k,
        &beta, d, rocblas_datatype_f32_r, m,
        d, rocblas_datatype_f32_r, m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

static rocblas_status gemm_f16_out(
        rocblas_handle handle, int m, int n, int k,
        const void * a, const void * b, __half * d) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        m, n, k,
        &alpha, a, rocblas_datatype_f16_r, k,
        b, rocblas_datatype_f16_r, k,
        &beta, d, rocblas_datatype_f16_r, m,
        d, rocblas_datatype_f16_r, m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

static int run_current(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a_up,
        const void * a_gate,
        const void * a_down,
        const void * b_in,
        float * up_f32,
        float * gate_f32,
        float * glu_f32,
        __half * glu_f16,
        float * out_f32,
        double & avg_ms) {
    const size_t ff_elems = (size_t) args.k_ff * (size_t) args.n;
    const int block = 256;
    const dim3 grid((unsigned int) ((ff_elems + block - 1) / block));
    auto once = [&]() -> int {
        ROCBLAS_CHECK(gemm_f32_out(handle, args.k_ff, args.n, args.k_in, a_up, b_in, up_f32));
        ROCBLAS_CHECK(gemm_f32_out(handle, args.k_ff, args.n, args.k_in, a_gate, b_in, gate_f32));
        glu_f32_kernel<<<grid, block, 0, stream>>>(up_f32, gate_f32, glu_f32, ff_elems);
        f32_to_f16_kernel<<<grid, block, 0, stream>>>(glu_f32, glu_f16, ff_elems);
        ROCBLAS_CHECK(gemm_f32_out(handle, args.m_out, args.n, args.k_ff, a_down, glu_f16, out_f32));
        return 0;
    };
    for (int i = 0; i < args.warmup; ++i) {
        if (once() != 0) return 1;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        if (once() != 0) return 1;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = (double) elapsed_ms / (double) args.iters;
    return 0;
}

static int run_candidate(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a_up,
        const void * a_gate,
        const void * a_down,
        const void * b_in,
        __half * up_f16,
        __half * gate_f16,
        __half * glu_f16,
        float * out_f32,
        double & avg_ms) {
    const size_t ff_elems = (size_t) args.k_ff * (size_t) args.n;
    const int block = 256;
    const dim3 grid((unsigned int) ((ff_elems + block - 1) / block));
    auto once = [&]() -> int {
        ROCBLAS_CHECK(gemm_f16_out(handle, args.k_ff, args.n, args.k_in, a_up, b_in, up_f16));
        ROCBLAS_CHECK(gemm_f16_out(handle, args.k_ff, args.n, args.k_in, a_gate, b_in, gate_f16));
        glu_f16_kernel<<<grid, block, 0, stream>>>(up_f16, gate_f16, glu_f16, ff_elems);
        ROCBLAS_CHECK(gemm_f32_out(handle, args.m_out, args.n, args.k_ff, a_down, glu_f16, out_f32));
        return 0;
    };
    for (int i = 0; i < args.warmup; ++i) {
        if (once() != 0) return 1;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        if (once() != 0) return 1;
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = (double) elapsed_ms / (double) args.iters;
    return 0;
}

int main(int argc, char ** argv) {
    args_t args;
    if (!parse_args(argc, argv, args)) {
        return 2;
    }

    HIP_CHECK(hipSetDevice(args.device));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    const size_t a_up_elems = (size_t) args.k_in * (size_t) args.k_ff;
    const size_t a_down_elems = (size_t) args.k_ff * (size_t) args.m_out;
    const size_t b_elems = (size_t) args.k_in * (size_t) args.n;
    const size_t ff_elems = (size_t) args.k_ff * (size_t) args.n;
    const size_t out_elems = (size_t) args.m_out * (size_t) args.n;

    void * a_up = nullptr;
    void * a_gate = nullptr;
    void * a_down = nullptr;
    void * b_in = nullptr;
    float * up_f32 = nullptr;
    float * gate_f32 = nullptr;
    float * glu_f32 = nullptr;
    __half * up_f16 = nullptr;
    __half * gate_f16 = nullptr;
    __half * glu_f16 = nullptr;
    float * out_f32 = nullptr;

    HIP_CHECK(hipMalloc(&a_up, a_up_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&a_gate, a_up_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&a_down, a_down_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b_in, b_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&up_f32), ff_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&gate_f32), ff_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&glu_f32), ff_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&up_f16), ff_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&gate_f16), ff_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&glu_f16), ff_elems * sizeof(__half)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&out_f32), out_elems * sizeof(float)));

    HIP_CHECK(hipMemsetAsync(a_up, 0x3c, a_up_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(a_gate, 0x2a, a_up_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(a_down, 0x1d, a_down_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(b_in, 0x13, b_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(up_f32, 0, ff_elems * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(gate_f32, 0, ff_elems * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(glu_f32, 0, ff_elems * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(up_f16, 0, ff_elems * sizeof(__half), stream));
    HIP_CHECK(hipMemsetAsync(gate_f16, 0, ff_elems * sizeof(__half), stream));
    HIP_CHECK(hipMemsetAsync(glu_f16, 0, ff_elems * sizeof(__half), stream));
    HIP_CHECK(hipMemsetAsync(out_f32, 0, out_elems * sizeof(float), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double current_ms = 0.0;
    double candidate_ms = 0.0;
    int status = run_current(handle, stream, args, a_up, a_gate, a_down, b_in, up_f32, gate_f32, glu_f32, glu_f16, out_f32, current_ms);
    if (status != 0) return status;
    status = run_candidate(handle, stream, args, a_up, a_gate, a_down, b_in, up_f16, gate_f16, glu_f16, out_f32, candidate_ms);
    if (status != 0) return status;

    std::cout << "n,k_in,k_ff,m_out,warmup,iters,route,avg_ms,relative_to_current\n";
    std::cout << args.n << "," << args.k_in << "," << args.k_ff << "," << args.m_out << ","
              << args.warmup << "," << args.iters << ",current_f32_intermediate,"
              << std::fixed << std::setprecision(4) << current_ms << ",1.0000\n";
    std::cout << args.n << "," << args.k_in << "," << args.k_ff << "," << args.m_out << ","
              << args.warmup << "," << args.iters << ",candidate_f16_intermediate,"
              << std::fixed << std::setprecision(4) << candidate_ms << ","
              << std::fixed << std::setprecision(4) << (candidate_ms / current_ms) << "\n";

    HIP_CHECK(hipFree(a_up));
    HIP_CHECK(hipFree(a_gate));
    HIP_CHECK(hipFree(a_down));
    HIP_CHECK(hipFree(b_in));
    HIP_CHECK(hipFree(up_f32));
    HIP_CHECK(hipFree(gate_f32));
    HIP_CHECK(hipFree(glu_f32));
    HIP_CHECK(hipFree(up_f16));
    HIP_CHECK(hipFree(gate_f16));
    HIP_CHECK(hipFree(glu_f16));
    HIP_CHECK(hipFree(out_f32));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
