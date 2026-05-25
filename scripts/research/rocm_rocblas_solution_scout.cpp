// Minimal ROCm/rocBLAS GEMM_EX solution-index scout for llama.cpp ROCm perf work.
// Build manually with ROCm hipcc/clang++ on Windows; this is not part of normal builds.

#define ROCBLAS_BETA_FEATURES_API

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
    int warmup = 2;
    int iters = 6;
    int max_solutions = 48;
    int device = 0;
    bool skip_default = false;
    bool check_solution_index = false;
    rocblas_performance_metric metric = rocblas_default_performance_metric;
    uint32_t gemm_flags = 0;
    std::vector<int> explicit_solutions;
};

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--m M --n N --k K] [--warmup N] [--iters N] [--max-solutions N]\n"
        << "       [--solutions comma,list] [--device N] [--skip-default]\n"
        << "       [--check-solution-index] [--metric default|device|cu]\n"
        << "       [--gemm-flags none|cu-efficiency|fp16-alt|fp16-rnz]\n";
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

static bool parse_solution_list(const char * text, std::vector<int> & out) {
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int value = 0;
        if (!parse_int(item.c_str(), value)) {
            return false;
        }
        out.push_back(value);
    }
    return true;
}

static bool parse_metric(const char * text, rocblas_performance_metric & out) {
    const std::string value = text;
    if (value == "default") {
        out = rocblas_default_performance_metric;
    } else if (value == "device") {
        out = rocblas_device_efficiency_performance_metric;
    } else if (value == "cu") {
        out = rocblas_cu_efficiency_performance_metric;
    } else {
        return false;
    }
    return true;
}

static bool parse_gemm_flags(const char * text, uint32_t & out) {
    const std::string value = text;
    if (value == "none") {
        out = 0u;
    } else if (value == "cu-efficiency") {
        out = static_cast<uint32_t>(rocblas_gemm_flags_use_cu_efficiency);
    } else if (value == "fp16-alt") {
        out = static_cast<uint32_t>(rocblas_gemm_flags_fp16_alt_impl);
    } else if (value == "fp16-rnz") {
        out = static_cast<uint32_t>(rocblas_gemm_flags_fp16_alt_impl_rnz);
    } else {
        return false;
    }
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
        } else if (key == "--max-solutions") {
            if (!need_value(args.max_solutions)) return false;
        } else if (key == "--device") {
            if (!need_value(args.device)) return false;
        } else if (key == "--solutions") {
            if (i + 1 >= argc || !parse_solution_list(argv[++i], args.explicit_solutions)) {
                usage(argv[0]);
                return false;
            }
        } else if (key == "--skip-default") {
            args.skip_default = true;
        } else if (key == "--check-solution-index") {
            args.check_solution_index = true;
        } else if (key == "--metric") {
            if (i + 1 >= argc || !parse_metric(argv[++i], args.metric)) {
                usage(argv[0]);
                return false;
            }
        } else if (key == "--gemm-flags") {
            if (i + 1 >= argc || !parse_gemm_flags(argv[++i], args.gemm_flags)) {
                usage(argv[0]);
                return false;
            }
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

static int run_gemm(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a,
        const void * b,
        float * d,
        int solution_index,
        bool use_solution_index,
        uint32_t flags,
        double & avg_ms) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = args.k;
    const int ldb = args.k;
    const int ldc = args.m;
    const int ldd = args.m;

    const rocblas_gemm_algo algo = use_solution_index ?
        rocblas_gemm_algo_solution_index :
        rocblas_gemm_algo_standard;

    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb,
            &beta, d, rocblas_datatype_f32_r, ldc,
            d, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, algo, solution_index, flags));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb,
            &beta, d, rocblas_datatype_f32_r, ldc,
            d, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, algo, solution_index, flags));
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
    ROCBLAS_CHECK(rocblas_set_performance_metric(handle, args.metric));

    const size_t a_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.m);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t d_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);
    const size_t a_bytes = a_elems * sizeof(uint16_t);
    const size_t b_bytes = b_elems * sizeof(uint16_t);
    const size_t d_bytes = d_elems * sizeof(float);

    void * a = nullptr;
    void * b = nullptr;
    float * d = nullptr;
    HIP_CHECK(hipMalloc(&a, a_bytes));
    HIP_CHECK(hipMalloc(&b, b_bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d), d_bytes));
    HIP_CHECK(hipMemsetAsync(a, 0x3c, a_bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0x13, b_bytes, stream));
    HIP_CHECK(hipMemsetAsync(d, 0, d_bytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int lda = args.k;
    const int ldb = args.k;
    const int ldc = args.m;
    const int ldd = args.m;

    std::vector<int> solutions;
    if (!args.explicit_solutions.empty()) {
        solutions = args.explicit_solutions;
    } else {
        rocblas_int solution_count = 0;
        ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            args.m, args.n, args.k,
            &alpha, a, rocblas_datatype_f16_r, lda,
            b, rocblas_datatype_f16_r, ldb,
            &beta, d, rocblas_datatype_f32_r, ldc,
            d, rocblas_datatype_f32_r, ldd,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0,
            nullptr, &solution_count));

        solutions.resize(std::min<int>(solution_count, args.max_solutions));
        if (!solutions.empty()) {
            rocblas_int list_size = static_cast<rocblas_int>(solutions.size());
            ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions(
                handle, rocblas_operation_transpose, rocblas_operation_none,
                args.m, args.n, args.k,
                &alpha, a, rocblas_datatype_f16_r, lda,
                b, rocblas_datatype_f16_r, ldb,
                &beta, d, rocblas_datatype_f32_r, ldc,
                d, rocblas_datatype_f32_r, ldd,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0,
                solutions.data(), &list_size));
            solutions.resize(list_size);
        }
    }

    std::sort(solutions.begin(), solutions.end());
    solutions.erase(std::unique(solutions.begin(), solutions.end()), solutions.end());

    const uint32_t solution_flags = args.gemm_flags |
        (args.check_solution_index ? static_cast<uint32_t>(rocblas_gemm_flags_check_solution_index) : 0u);

    std::cout << "m,n,k,warmup,iters,kind,solution_index,avg_ms,relative_to_default\n";
    double default_ms = 0.0;
    if (!args.skip_default) {
        const int status = run_gemm(handle, stream, args, a, b, d, 0, false, args.gemm_flags, default_ms);
        if (status != 0) {
            return status;
        }
        std::cout << args.m << "," << args.n << "," << args.k << ","
                  << args.warmup << "," << args.iters
                  << ",default,0," << std::fixed << std::setprecision(4) << default_ms
                  << ",1.0000\n";
    }

    for (const int solution_index : solutions) {
        if (solution_index == 0) {
            continue;
        }
        double avg_ms = 0.0;
        const int status = run_gemm(
            handle, stream, args, a, b, d, solution_index, true, solution_flags, avg_ms);
        if (status != 0) {
            std::cerr << "failed solution_index=" << solution_index << std::endl;
            continue;
        }
        const double rel = default_ms > 0.0 ? avg_ms / default_ms : 0.0;
        std::cout << args.m << "," << args.n << "," << args.k << ","
                  << args.warmup << "," << args.iters
                  << ",solution," << solution_index << ","
                  << std::fixed << std::setprecision(4) << avg_ms << ","
                  << std::fixed << std::setprecision(4) << rel << "\n";
    }

    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(d));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
