// Minimal ROCm/rocBLAS output-type scout for llama.cpp ROCm H42 research.
// Compares current f32 GEMM output against f16 GEMM output plus f16->f32 convert.
// Diagnostic utility only.

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
    int m = 17408;
    int n = 2048;
    int k = 5120;
    int warmup = 6;
    int iters = 20;
    int device = 0;
};

static __global__ void f16_to_f32_kernel(const __half * src, float * dst, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

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

static rocblas_status gemm_f32_out(
        rocblas_handle handle,
        const args_t & args,
        const void * a,
        const void * b,
        float * d) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        args.m, args.n, args.k,
        &alpha, a, rocblas_datatype_f16_r, args.k,
        b, rocblas_datatype_f16_r, args.k,
        &beta, d, rocblas_datatype_f32_r, args.m,
        d, rocblas_datatype_f32_r, args.m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

static rocblas_status gemm_f16_out(
        rocblas_handle handle,
        const args_t & args,
        const void * a,
        const void * b,
        __half * d) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        args.m, args.n, args.k,
        &alpha, a, rocblas_datatype_f16_r, args.k,
        b, rocblas_datatype_f16_r, args.k,
        &beta, d, rocblas_datatype_f16_r, args.m,
        d, rocblas_datatype_f16_r, args.m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

static int run_f32(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a,
        const void * b,
        float * d,
        double & avg_ms) {
    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm_f32_out(handle, args, a, b, d));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm_f32_out(handle, args, a, b, d));
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = (double) elapsed_ms / (double) args.iters;
    return 0;
}

static int run_f16_plus_convert(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a,
        const void * b,
        __half * d16,
        float * d32,
        double & avg_ms) {
    const size_t n_elem = (size_t) args.m * (size_t) args.n;
    const int block = 256;
    const dim3 grid((unsigned int) ((n_elem + block - 1) / block));
    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm_f16_out(handle, args, a, b, d16));
        f16_to_f32_kernel<<<grid, block, 0, stream>>>(d16, d32, n_elem);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm_f16_out(handle, args, a, b, d16));
        f16_to_f32_kernel<<<grid, block, 0, stream>>>(d16, d32, n_elem);
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

static int run_f16_only(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a,
        const void * b,
        __half * d16,
        double & avg_ms) {
    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(gemm_f16_out(handle, args, a, b, d16));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(gemm_f16_out(handle, args, a, b, d16));
    }
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    avg_ms = (double) elapsed_ms / (double) args.iters;
    return 0;
}

static int run_convert_only(
        hipStream_t stream,
        const args_t & args,
        const __half * d16,
        float * d32,
        double & avg_ms) {
    const size_t n_elem = (size_t) args.m * (size_t) args.n;
    const int block = 256;
    const dim3 grid((unsigned int) ((n_elem + block - 1) / block));
    for (int i = 0; i < args.warmup; ++i) {
        f16_to_f32_kernel<<<grid, block, 0, stream>>>(d16, d32, n_elem);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < args.iters; ++i) {
        f16_to_f32_kernel<<<grid, block, 0, stream>>>(d16, d32, n_elem);
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

    const size_t a_elems = (size_t) args.k * (size_t) args.m;
    const size_t b_elems = (size_t) args.k * (size_t) args.n;
    const size_t d_elems = (size_t) args.m * (size_t) args.n;

    void * a = nullptr;
    void * b = nullptr;
    float * d32 = nullptr;
    __half * d16 = nullptr;
    HIP_CHECK(hipMalloc(&a, a_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b, b_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d32), d_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d16), d_elems * sizeof(__half)));
    HIP_CHECK(hipMemsetAsync(a, 0x3c, a_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(b, 0x13, b_elems * sizeof(uint16_t), stream));
    HIP_CHECK(hipMemsetAsync(d32, 0, d_elems * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(d16, 0, d_elems * sizeof(__half), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double f32_ms = 0.0;
    double f16_only_ms = 0.0;
    double convert_only_ms = 0.0;
    double f16_convert_ms = 0.0;
    int status = run_f32(handle, stream, args, a, b, d32, f32_ms);
    if (status != 0) {
        return status;
    }
    status = run_f16_only(handle, stream, args, a, b, d16, f16_only_ms);
    if (status != 0) {
        return status;
    }
    status = run_convert_only(stream, args, d16, d32, convert_only_ms);
    if (status != 0) {
        return status;
    }
    status = run_f16_plus_convert(handle, stream, args, a, b, d16, d32, f16_convert_ms);
    if (status != 0) {
        return status;
    }

    std::cout << "m,n,k,warmup,iters,route,avg_ms,relative_to_f32\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",f32_output," << std::fixed << std::setprecision(4) << f32_ms
              << ",1.0000\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",f16_output_only," << std::fixed << std::setprecision(4) << f16_only_ms
              << "," << std::fixed << std::setprecision(4) << (f16_only_ms / f32_ms) << "\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",convert_only," << std::fixed << std::setprecision(4) << convert_only_ms
              << "," << std::fixed << std::setprecision(4) << (convert_only_ms / f32_ms) << "\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << ",f16_output_plus_convert," << std::fixed << std::setprecision(4) << f16_convert_ms
              << "," << std::fixed << std::setprecision(4) << (f16_convert_ms / f32_ms) << "\n";

    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(d32));
    HIP_CHECK(hipFree(d16));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
