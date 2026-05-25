// Minimal ROCm/rocBLAS tall-GEMM scout for llama.cpp ROCm route-body research.
// This is a standalone diagnostic utility, not part of normal builds.

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
    int m = 17408;
    int n = 2048;
    int k = 5120;
    int warmup = 8;
    int iters = 20;
    int device = 0;
};

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " [--m M --n N --k K] [--warmup N] [--iters N] [--device N]\n";
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

        if (key == "--m") {
            if (!need_value(args.m)) return false;
        } else if (key == "--n") {
            if (!need_value(args.n)) return false;
        } else if (key == "--k") {
            if (!need_value(args.k)) return false;
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

    return args.m > 0 && args.n > 0 && args.k > 0 && args.iters > 0 && args.warmup >= 0;
}

static rocblas_status gemm(
        rocblas_handle handle,
        const int m,
        const int n,
        const int k,
        const void * a,
        const void * b,
        float * d) {
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

static int time_separate(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a_pair,
        const void * b,
        float * d_pair,
        double & avg_pair_ms) {
    const size_t a_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.m);
    const size_t d_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);
    const void * a0 = a_pair;
    const void * a1 = static_cast<const uint16_t *>(a_pair) + a_elems;
    float * d0 = d_pair;
    float * d1 = d_pair + d_elems;

    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm(handle, args.m, args.n, args.k, a0, b, d0));
        ROCBLAS_CHECK(gemm(handle, args.m, args.n, args.k, a1, b, d1));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm(handle, args.m, args.n, args.k, a0, b, d0));
        ROCBLAS_CHECK(gemm(handle, args.m, args.n, args.k, a1, b, d1));
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    avg_pair_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
    return 0;
}

static int time_tall(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a_pair,
        const void * b,
        float * d_pair,
        double & avg_pair_ms) {
    const int tall_m = args.m * 2;

    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm(handle, tall_m, args.n, args.k, a_pair, b, d_pair));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm(handle, tall_m, args.n, args.k, a_pair, b, d_pair));
    }

    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    avg_pair_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
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

    const size_t a_pair_elems = 2ull * static_cast<size_t>(args.k) * static_cast<size_t>(args.m);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t d_pair_elems = 2ull * static_cast<size_t>(args.m) * static_cast<size_t>(args.n);
    const size_t a_pair_bytes = a_pair_elems * sizeof(uint16_t);
    const size_t b_bytes = b_elems * sizeof(uint16_t);
    const size_t d_pair_bytes = d_pair_elems * sizeof(float);

    void * a_pair = nullptr;
    void * b = nullptr;
    float * d_pair = nullptr;
    HIP_CHECK(hipMalloc(&a_pair, a_pair_bytes));
    HIP_CHECK(hipMalloc(&b, b_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_pair), d_pair_bytes));
    HIP_CHECK(hipMemsetAsync(a_pair, 0x3c, a_pair_bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0x13, b_bytes, stream));
    HIP_CHECK(hipMemsetAsync(d_pair, 0, d_pair_bytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double separate_ms = 0.0;
    double tall_ms = 0.0;
    int status = time_separate(handle, stream, args, a_pair, b, d_pair, separate_ms);
    if (status != 0) {
        return status;
    }
    status = time_tall(handle, stream, args, a_pair, b, d_pair, tall_ms);
    if (status != 0) {
        return status;
    }

    std::cout << "m,n,k,warmup,iters,kind,avg_pair_ms,avg_single_equiv_ms,relative_to_separate\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",separate," << std::fixed << std::setprecision(4) << separate_ms << ","
              << std::fixed << std::setprecision(4) << (separate_ms / 2.0) << ",1.0000\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",tall_m2," << std::fixed << std::setprecision(4) << tall_ms << ","
              << std::fixed << std::setprecision(4) << (tall_ms / 2.0) << ","
              << std::fixed << std::setprecision(4) << (tall_ms / separate_ms) << "\n";

    HIP_CHECK(hipFree(a_pair));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(d_pair));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
