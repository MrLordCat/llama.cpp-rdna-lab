// Minimal ROCm/rocBLAS pair-GEMM scout for llama.cpp ROCm route-body research.
// This is a standalone diagnostic utility, not part of normal builds.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    int warmup = 4;
    int iters = 12;
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

static int run_separate(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a0,
        const void * a1,
        const void * b,
        float * d0,
        float * d1,
        double & avg_pair_ms) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = args.k;
    const int ldb = args.k;
    const int ldc = args.m;
    const int ldd = args.m;

    auto gemm = [&](const void * a, float * d) -> rocblas_status {
        return rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb,
            &beta, d, rocblas_datatype_f32_r, ldc,
            d, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    };

    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm(a0, d0));
        ROCBLAS_CHECK(gemm(a1, d1));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm(a0, d0));
        ROCBLAS_CHECK(gemm(a1, d1));
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

static int run_batched(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * const * a_array_dev,
        const void * const * b_array_dev,
        float * const * d_array_dev,
        double & avg_pair_ms) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = args.k;
    const int ldb = args.k;
    const int ldc = args.m;
    const int ldd = args.m;
    constexpr int batch_count = 2;

    auto gemm = [&]() -> rocblas_status {
        return rocblas_gemm_batched_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a_array_dev, rocblas_datatype_f16_r, lda,
            b_array_dev, rocblas_datatype_f16_r, ldb,
            &beta, d_array_dev, rocblas_datatype_f32_r, ldc,
            reinterpret_cast<void *>(const_cast<float **>(d_array_dev)), rocblas_datatype_f32_r, ldd,
            batch_count, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    };

    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm());
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm());
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

static int run_concurrent(
        rocblas_handle handle0,
        rocblas_handle handle1,
        hipStream_t stream0,
        hipStream_t stream1,
        const args_t & args,
        const void * a0,
        const void * a1,
        const void * b,
        float * d0,
        float * d1,
        double & avg_pair_ms) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = args.k;
    const int ldb = args.k;
    const int ldc = args.m;
    const int ldd = args.m;

    auto gemm = [&](rocblas_handle handle, const void * a, float * d) -> rocblas_status {
        return rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb,
            &beta, d, rocblas_datatype_f32_r, ldc,
            d, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    };

    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm(handle0, a0, d0));
        ROCBLAS_CHECK(gemm(handle1, a1, d1));
    }
    HIP_CHECK(hipStreamSynchronize(stream0));
    HIP_CHECK(hipStreamSynchronize(stream1));

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm(handle0, a0, d0));
        ROCBLAS_CHECK(gemm(handle1, a1, d1));
    }
    HIP_CHECK(hipStreamSynchronize(stream0));
    HIP_CHECK(hipStreamSynchronize(stream1));
    const auto stop = std::chrono::high_resolution_clock::now();

    avg_pair_ms = std::chrono::duration<double, std::milli>(stop - start).count() /
        static_cast<double>(args.iters);
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
    hipStream_t stream1 = nullptr;
    HIP_CHECK(hipStreamCreate(&stream1));

    rocblas_handle handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
    rocblas_handle handle1 = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&handle1));
    ROCBLAS_CHECK(rocblas_set_stream(handle1, stream1));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle1, rocblas_pointer_mode_host));

    const size_t a_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.m);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t d_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);
    const size_t a_bytes = a_elems * sizeof(uint16_t);
    const size_t b_bytes = b_elems * sizeof(uint16_t);
    const size_t d_bytes = d_elems * sizeof(float);

    void * a0 = nullptr;
    void * a1 = nullptr;
    void * b = nullptr;
    float * d0 = nullptr;
    float * d1 = nullptr;
    HIP_CHECK(hipMalloc(&a0, a_bytes));
    HIP_CHECK(hipMalloc(&a1, a_bytes));
    HIP_CHECK(hipMalloc(&b, b_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d0), d_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d1), d_bytes));
    HIP_CHECK(hipMemsetAsync(a0, 0x3c, a_bytes, stream));
    HIP_CHECK(hipMemsetAsync(a1, 0x2a, a_bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0x13, b_bytes, stream));
    HIP_CHECK(hipMemsetAsync(d0, 0, d_bytes, stream));
    HIP_CHECK(hipMemsetAsync(d1, 0, d_bytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    const void * a_array_host[2] = { a0, a1 };
    const void * b_array_host[2] = { b, b };
    float * d_array_host[2] = { d0, d1 };
    const void ** a_array_dev = nullptr;
    const void ** b_array_dev = nullptr;
    float ** d_array_dev = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&a_array_dev), sizeof(a_array_host)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&b_array_dev), sizeof(b_array_host)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_array_dev), sizeof(d_array_host)));
    HIP_CHECK(hipMemcpyAsync(a_array_dev, a_array_host, sizeof(a_array_host), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(b_array_dev, b_array_host, sizeof(b_array_host), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_array_dev, d_array_host, sizeof(d_array_host), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double separate_ms = 0.0;
    double batched_ms = 0.0;
    double concurrent_ms = 0.0;
    int status = run_separate(handle, stream, args, a0, a1, b, d0, d1, separate_ms);
    if (status != 0) {
        return status;
    }
    status = run_concurrent(handle, handle1, stream, stream1, args, a0, a1, b, d0, d1, concurrent_ms);
    if (status != 0) {
        return status;
    }
    status = run_batched(handle, stream, args, a_array_dev, b_array_dev, d_array_dev, batched_ms);
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
              << ",concurrent_streams," << std::fixed << std::setprecision(4) << concurrent_ms << ","
              << std::fixed << std::setprecision(4) << (concurrent_ms / 2.0) << ","
              << std::fixed << std::setprecision(4) << (concurrent_ms / separate_ms) << "\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",batched," << std::fixed << std::setprecision(4) << batched_ms << ","
              << std::fixed << std::setprecision(4) << (batched_ms / 2.0) << ","
              << std::fixed << std::setprecision(4) << (batched_ms / separate_ms) << "\n";

    HIP_CHECK(hipFree(a_array_dev));
    HIP_CHECK(hipFree(b_array_dev));
    HIP_CHECK(hipFree(d_array_dev));
    HIP_CHECK(hipFree(a0));
    HIP_CHECK(hipFree(a1));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(d0));
    HIP_CHECK(hipFree(d1));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle1));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
