// Minimal ROCm/rocBLAS GEMM layout scout for llama.cpp ROCm H42 research.
// Compares the current Q3_K cublas fallback contract A^T*B against a hypothetical
// pre-transposed fp16 staging contract A*B. Diagnostic utility only.

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
    int warmup = 6;
    int iters = 20;
    int device = 0;
};

static void usage(const char * argv0) {
    std::cerr << "usage: " << argv0
              << " [--m M --n N --k K] [--warmup N] [--iters N] [--device N]\n";
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

    return args.m > 0 && args.n > 0 && args.k > 0 && args.warmup >= 0 && args.iters > 0;
}

static int run_gemm(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a,
        const void * b,
        float * d,
        bool a_pretransposed,
        double & avg_ms) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const rocblas_operation trans_a = a_pretransposed ? rocblas_operation_none : rocblas_operation_transpose;
    const int lda = a_pretransposed ? args.m : args.k;
    const int ldb = args.k;
    const int ldc = args.m;
    const int ldd = args.m;

    auto gemm = [&]() -> rocblas_status {
        return rocblas_gemm_ex(
            handle, trans_a, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb,
            &beta, d, rocblas_datatype_f32_r, ldc,
            d, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
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

    avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(args.iters);
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

    const size_t a_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.k);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t d_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);

    void * a_current = nullptr;
    void * a_pretransposed = nullptr;
    void * b = nullptr;
    float * d = nullptr;
    HIP_CHECK(hipMalloc(&a_current, a_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&a_pretransposed, a_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b, b_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d), d_elems * sizeof(float)));
    HIP_CHECK(hipMemsetAsync(a_current, 0x3c, a_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(a_pretransposed, 0x2a, a_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(b, 0x13, b_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(d, 0, d_elems * sizeof(float), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double current_ms = 0.0;
    double pretransposed_ms = 0.0;
    int status = run_gemm(handle, stream, args, a_current, b, d, false, current_ms);
    if (status != 0) {
        return status;
    }
    status = run_gemm(handle, stream, args, a_pretransposed, b, d, true, pretransposed_ms);
    if (status != 0) {
        return status;
    }

    std::cout << "m,n,k,warmup,iters,layout,transA,lda,avg_ms,relative_to_current\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",current_A_colmajor_kxm,T," << args.k << ","
              << std::fixed << std::setprecision(4) << current_ms << ",1.0000\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",pretransposed_A_colmajor_mxk,N," << args.m << ","
              << std::fixed << std::setprecision(4) << pretransposed_ms << ","
              << std::fixed << std::setprecision(4) << (pretransposed_ms / current_ms) << "\n";

    HIP_CHECK(hipFree(a_current));
    HIP_CHECK(hipFree(a_pretransposed));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(d));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
