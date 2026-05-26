// Standalone ROCm hipBLASLt grouped-GEMM scout for llama.cpp Q3_K route research.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <rocblas/rocblas.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

#define HIPBLASLT_CHECK(expr)                                                                   \
    do {                                                                                        \
        hipblasStatus_t status__ = (expr);                                                      \
        if (status__ != HIPBLAS_STATUS_SUCCESS) {                                               \
            std::cerr << "hipBLASLt error: " << static_cast<int>(status__) << " at "           \
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
    int algos = 16;
    size_t workspace_mb = 256;
    bool output_f16 = false;
    bool compute_fast16 = false;
};

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--m M --n N --k K] [--warmup N] [--iters N] [--device N]"
        << " [--algos N] [--workspace-mb N] [--output-f16] [--compute-fast16]\n";
}

static bool parse_int(const char * text, int & out) {
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

static bool parse_size(const char * text, size_t & out) {
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        return false;
    }
    out = static_cast<size_t>(value);
    return true;
}

static bool parse_args(int argc, char ** argv, args_t & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_int = [&](int & value) -> bool {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return false;
            }
            return parse_int(argv[++i], value);
        };
        auto need_size = [&](size_t & value) -> bool {
            if (i + 1 >= argc) {
                usage(argv[0]);
                return false;
            }
            return parse_size(argv[++i], value);
        };

        if (key == "--m") {
            if (!need_int(args.m)) return false;
        } else if (key == "--n") {
            if (!need_int(args.n)) return false;
        } else if (key == "--k") {
            if (!need_int(args.k)) return false;
        } else if (key == "--warmup") {
            if (!need_int(args.warmup)) return false;
        } else if (key == "--iters") {
            if (!need_int(args.iters)) return false;
        } else if (key == "--device") {
            if (!need_int(args.device)) return false;
        } else if (key == "--algos") {
            if (!need_int(args.algos)) return false;
        } else if (key == "--workspace-mb") {
            if (!need_size(args.workspace_mb)) return false;
        } else if (key == "--output-f16") {
            args.output_f16 = true;
        } else if (key == "--compute-fast16") {
            args.compute_fast16 = true;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            return false;
        }
    }

    return args.m > 0 && args.n > 0 && args.k > 0 && args.iters > 0 &&
           args.warmup >= 0 && args.algos > 0;
}

static rocblas_status rocblas_gemm(
        rocblas_handle handle,
        int m,
        int n,
        int k,
        const void * a,
        const void * b,
    void * d,
    rocblas_datatype output_type) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        m, n, k,
        &alpha, a, rocblas_datatype_f16_r, k,
        b, rocblas_datatype_f16_r, k,
        &beta, d, output_type, m,
        d, output_type, m,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
}

static int time_rocblas_separate(
        rocblas_handle handle,
        hipStream_t stream,
        const args_t & args,
        const void * a0,
        const void * a1,
        const void * b,
        void * d0,
        void * d1,
        double & avg_pair_ms) {
    const rocblas_datatype output_type = args.output_f16 ? rocblas_datatype_f16_r : rocblas_datatype_f32_r;
    for (int i = 0; i < args.warmup; ++i) {
        ROCBLAS_CHECK(rocblas_gemm(handle, args.m, args.n, args.k, a0, b, d0, output_type));
        ROCBLAS_CHECK(rocblas_gemm(handle, args.m, args.n, args.k, a1, b, d1, output_type));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        ROCBLAS_CHECK(rocblas_gemm(handle, args.m, args.n, args.k, a0, b, d0, output_type));
        ROCBLAS_CHECK(rocblas_gemm(handle, args.m, args.n, args.k, a1, b, d1, output_type));
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

static int time_hipblaslt_grouped(
        hipblasLtHandle_t lt_handle,
        hipStream_t stream,
        const args_t & args,
        const void * a0,
        const void * a1,
        const void * b,
        void * d0,
        void * d1,
        void * workspace,
        size_t workspace_bytes,
        double & avg_pair_ms,
        int & algo_index,
        std::string & kernel_name) {
    using namespace hipblaslt_ext;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipDataType output_type = args.output_f16 ? HIP_R_16F : HIP_R_32F;
    const hipblasComputeType_t compute_type = args.compute_fast16 ? HIPBLAS_COMPUTE_32F_FAST_16F : HIPBLAS_COMPUTE_32F;

    GroupedGemm grouped(
        lt_handle,
        HIPBLAS_OP_T,
        HIPBLAS_OP_N,
        HIP_R_16F,
        HIP_R_16F,
        output_type,
        output_type,
        compute_type);

    std::vector<int64_t> m = { args.m, args.m };
    std::vector<int64_t> n = { args.n, args.n };
    std::vector<int64_t> k = { args.k, args.k };
    std::vector<int64_t> batch_count = { 1, 1 };
    std::vector<int64_t> lda = { args.k, args.k };
    std::vector<int64_t> ldb = { args.k, args.k };
    std::vector<int64_t> ldc = { args.m, args.m };
    std::vector<int64_t> ldd = { args.m, args.m };
    std::vector<int64_t> stride_a = { 0, 0 };
    std::vector<int64_t> stride_b = { 0, 0 };
    std::vector<int64_t> stride_c = { 0, 0 };
    std::vector<int64_t> stride_d = { 0, 0 };
    std::vector<GemmEpilogue> epilogue(2);
    std::vector<GemmInputs> inputs(2);
    inputs[0].setA(a0);
    inputs[0].setB(b);
    inputs[0].setC(d0);
    inputs[0].setD(d0);
    inputs[0].setAlpha(&alpha);
    inputs[0].setBeta(&beta);
    inputs[1].setA(a1);
    inputs[1].setB(b);
    inputs[1].setC(d1);
    inputs[1].setD(d1);
    inputs[1].setAlpha(&alpha);
    inputs[1].setBeta(&beta);

    GemmProblemType problem_type(
        HIPBLAS_OP_T,
        HIPBLAS_OP_N,
        HIP_R_16F,
        HIP_R_16F,
        output_type,
        output_type,
        compute_type);

    HIPBLASLT_CHECK(grouped.setProblem(
        m, n, k, batch_count,
        lda, ldb, ldc, ldd,
        stride_a, stride_b, stride_c, stride_d,
        epilogue, inputs, problem_type));

    GemmPreference preference;
    preference.setMaxWorkspaceBytes(workspace_bytes);
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    HIPBLASLT_CHECK(grouped.algoGetHeuristic(args.algos, preference, results));
    if (results.empty()) {
        const hipblasStatus_t all_status = getAllAlgos(
            lt_handle,
            GemmType::HIPBLASLT_GROUPED_GEMM,
            HIPBLAS_OP_T,
            HIPBLAS_OP_N,
            HIP_R_16F,
            HIP_R_16F,
            output_type,
            output_type,
            compute_type,
            results);
        if (all_status != HIPBLAS_STATUS_SUCCESS || results.empty()) {
            std::cerr << "hipBLASLt grouped returned no algorithms\n";
            return 1;
        }
    }

    hipblasLtMatmulAlgo_t selected_algo{};
    size_t selected_workspace_bytes = 0;
    bool found = false;
    for (hipblasLtMatmulHeuristicResult_t & result : results) {
        if (result.state != HIPBLAS_STATUS_SUCCESS) {
            continue;
        }
        size_t required_workspace = 0;
        const hipblasStatus_t supported = grouped.isAlgoSupported(result.algo, required_workspace);
        if (supported == HIPBLAS_STATUS_SUCCESS && required_workspace <= workspace_bytes) {
            selected_algo = result.algo;
            selected_workspace_bytes = required_workspace;
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "hipBLASLt grouped found no supported algorithm within workspace\n";
        return 1;
    }

    grouped.setMaxWorkspaceBytes(selected_workspace_bytes);
    HIPBLASLT_CHECK(grouped.initialize(selected_algo, workspace, true, stream));
    algo_index = getIndexFromAlgo(selected_algo);
    kernel_name = grouped.getKernelName();

    for (int i = 0; i < args.warmup; ++i) {
        HIPBLASLT_CHECK(grouped.run(stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));

    for (int i = 0; i < args.iters; ++i) {
        HIPBLASLT_CHECK(grouped.run(stream));
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

    rocblas_handle rb_handle = nullptr;
    ROCBLAS_CHECK(rocblas_create_handle(&rb_handle));
    ROCBLAS_CHECK(rocblas_set_stream(rb_handle, stream));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(rb_handle, rocblas_pointer_mode_host));

    hipblasLtHandle_t lt_handle = nullptr;
    HIPBLASLT_CHECK(hipblasLtCreate(&lt_handle));

    const size_t a_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.m);
    const size_t b_elems = static_cast<size_t>(args.k) * static_cast<size_t>(args.n);
    const size_t d_elems = static_cast<size_t>(args.m) * static_cast<size_t>(args.n);
    const size_t a_bytes = a_elems * sizeof(uint16_t);
    const size_t b_bytes = b_elems * sizeof(uint16_t);
    const size_t d_bytes = d_elems * (args.output_f16 ? sizeof(uint16_t) : sizeof(float));
    const size_t workspace_bytes = args.workspace_mb * 1024ull * 1024ull;

    void * a0 = nullptr;
    void * a1 = nullptr;
    void * b = nullptr;
    void * d0 = nullptr;
    void * d1 = nullptr;
    void * workspace = nullptr;
    HIP_CHECK(hipMalloc(&a0, a_bytes));
    HIP_CHECK(hipMalloc(&a1, a_bytes));
    HIP_CHECK(hipMalloc(&b, b_bytes));
    HIP_CHECK(hipMalloc(&d0, d_bytes));
    HIP_CHECK(hipMalloc(&d1, d_bytes));
    if (workspace_bytes > 0) {
        HIP_CHECK(hipMalloc(&workspace, workspace_bytes));
    }

    HIP_CHECK(hipMemsetAsync(a0, 0x3c, a_bytes, stream));
    HIP_CHECK(hipMemsetAsync(a1, 0x2a, a_bytes, stream));
    HIP_CHECK(hipMemsetAsync(b, 0x13, b_bytes, stream));
    HIP_CHECK(hipMemsetAsync(d0, 0, d_bytes, stream));
    HIP_CHECK(hipMemsetAsync(d1, 0, d_bytes, stream));
    if (workspace != nullptr) {
        HIP_CHECK(hipMemsetAsync(workspace, 0, workspace_bytes, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    double rocblas_ms = 0.0;
    double grouped_ms = 0.0;
    int grouped_algo = -1;
    std::string grouped_kernel;

    int status = time_rocblas_separate(rb_handle, stream, args, a0, a1, b, d0, d1, rocblas_ms);
    if (status != 0) {
        return status;
    }
    status = time_hipblaslt_grouped(
        lt_handle, stream, args, a0, a1, b, d0, d1,
        workspace, workspace_bytes, grouped_ms, grouped_algo, grouped_kernel);
    if (status != 0) {
        return status;
    }

    std::cout
        << "m,n,k,warmup,iters,output,compute,kind,avg_pair_ms,avg_single_equiv_ms,relative_to_rocblas,algo,kernel\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << "," << (args.output_f16 ? "f16" : "f32")
              << "," << (args.compute_fast16 ? "fast16" : "f32")
              << ",rocblas_separate," << std::fixed << std::setprecision(4) << rocblas_ms << ","
              << std::fixed << std::setprecision(4) << (rocblas_ms / 2.0) << ",1.0000,,\n";
    std::cout << args.m << "," << args.n << "," << args.k << ","
              << args.warmup << "," << args.iters
              << "," << (args.output_f16 ? "f16" : "f32")
              << "," << (args.compute_fast16 ? "fast16" : "f32")
              << ",hipblaslt_grouped," << std::fixed << std::setprecision(4) << grouped_ms << ","
              << std::fixed << std::setprecision(4) << (grouped_ms / 2.0) << ","
              << std::fixed << std::setprecision(4) << (grouped_ms / rocblas_ms) << ","
              << grouped_algo << "," << grouped_kernel << "\n";

    if (workspace != nullptr) {
        HIP_CHECK(hipFree(workspace));
    }
    HIP_CHECK(hipFree(a0));
    HIP_CHECK(hipFree(a1));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(d0));
    HIP_CHECK(hipFree(d1));
    HIPBLASLT_CHECK(hipblasLtDestroy(lt_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(rb_handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
