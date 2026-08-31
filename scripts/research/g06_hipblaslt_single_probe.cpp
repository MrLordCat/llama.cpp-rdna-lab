// Standalone hipBLASLt single-GEMM probe for ROCm 7.2 prefill shapes.
// G06: exact-shape offline tuning. The grouped scout (E249) returns "no
// algorithms"; this probe checks the plain hipblasLtMatmul path (GemmType
// HIPBLASLT_GEMM) for the same f16 x f16 -> f32/f16 dense prefill contract
// and reports algorithm count, supported algo, kernel name and time.
//
// Build (Windows, ROCm 7.2):
//   hipcc -std=c++17 -O2 -IC:/PROGRA~1/AMD/ROCm/7.2/include \
//     scripts/research/g06_hipblaslt_single_probe.cpp \
//     -LC:/PROGRA~1/AMD/ROCm/7.2/lib -lhipblaslt \
//     -o build-rocm72/bin/g06-hipblaslt-single-probe.exe
//
// Usage: --m M --n N --k K [--warmup N --iters N --algos N --workspace-mb N
//        --output-f16 --compute-fast16]

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                                           \
    do {                                                                                          \
        hipError_t s__ = (expr);                                                                  \
        if (s__ != hipSuccess) {                                                                  \
            std::fprintf(stderr, "%s:%d HIP error: %s\n", __FILE__, __LINE__,                   \
                         hipGetErrorString(s__));                                                  \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

#define LT_CHECK(expr)                                                                            \
    do {                                                                                          \
        hipblasStatus_t s__ = (expr);                                                             \
        if (s__ != HIPBLAS_STATUS_SUCCESS) {                                                      \
            std::fprintf(stderr, "%s:%d hipBLASLt error: %d\n", __FILE__, __LINE__,              \
                         (int) s__);                                                              \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

struct opts_t {
    int64_t m = 0, n = 0, k = 0;
    int warmup = 4, iters = 10, algos = 1024;
    int max_algos = 64;          // 0 = all supported
    int rebench_top = 3;         // precise re-bench of top-N scan winners
    size_t workspace_mb = 512;
    bool output_f16 = false;
    bool compute_fast16 = false;
    bool bias = false;
    bool tune_splitk = false;
    bool tune_wgm = false;
    bool setup_bench = false;
    int device = 0;
};

static bool parse_int(const char * text, int64_t & out) {
    char * end = nullptr;
    const long long v = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || v <= 0) {
        return false;
    }
    out = v;
    return true;
}

static bool parse_int(const char * text, int & out) {
    char * end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v <= 0) {
        return false;
    }
    out = (int) v;
    return true;
}

static void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --m M --n N --k K [--warmup N --iters N --algos N "
                         "--workspace-mb N --device N --output-f16 --compute-fast16 "
                         "--setup-bench]\n", argv0);
}

int main(int argc, char ** argv) {
    opts_t o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--m") {
            if (i + 1 < argc) { ++i; if (!parse_int(argv[i], o.m)) { usage(argv[0]); return 1; } }
        } else if (key == "--n") {
            if (i + 1 < argc) { ++i; if (!parse_int(argv[i], o.n)) { usage(argv[0]); return 1; } }
        } else if (key == "--k") {
            if (i + 1 < argc) { ++i; if (!parse_int(argv[i], o.k)) { usage(argv[0]); return 1; } }
        } else if (key == "--warmup") {
            if (i + 1 < argc) { ++i; if (!parse_int(argv[i], o.warmup)) { usage(argv[0]); return 1; } }
        } else if (key == "--iters") {
            if (i + 1 < argc) { ++i; if (!parse_int(argv[i], o.iters)) { usage(argv[0]); return 1; } }
        } else if (key == "--algos") {
            if (i + 1 < argc) { ++i; if (!parse_int(argv[i], o.algos)) { usage(argv[0]); return 1; } }
        } else if (key == "--workspace-mb") {
            if (i + 1 < argc) { ++i; o.workspace_mb = (size_t) std::strtoull(argv[i], nullptr, 10) * 1024 * 1024; }
        } else if (key == "--max-algos") {
            if (i + 1 < argc) { ++i; o.max_algos = atoi(argv[i]); }
        } else if (key == "--rebench") {
            if (i + 1 < argc) { ++i; o.rebench_top = atoi(argv[i]); }
        } else if (key == "--device") {
            if (i + 1 < argc) { ++i; o.device = atoi(argv[i]); }
        } else if (key == "--output-f16") {
            o.output_f16 = true;
        } else if (key == "--compute-fast16") {
            o.compute_fast16 = true;
        } else if (key == "--bias") {
            o.bias = true;
        } else if (key == "--tune-splitk") {
            o.tune_splitk = true;
        } else if (key == "--tune-wgm") {
            o.tune_wgm = true;
        } else if (key == "--setup-bench") {
            o.setup_bench = true;
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (o.m <= 0 || o.n <= 0 || o.k <= 0) {
        usage(argv[0]);
        return 1;
    }

    HIP_CHECK(hipSetDevice(o.device));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    hipblasLtHandle_t lt_handle = nullptr;
    LT_CHECK(hipblasLtCreate(&lt_handle));

    using namespace hipblaslt_ext;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipDataType out = o.output_f16 ? HIP_R_16F : HIP_R_32F;
    const hipblasComputeType_t compute = o.compute_fast16 ? HIPBLAS_COMPUTE_32F_FAST_16F : HIPBLAS_COMPUTE_32F;

    // Column-major contract matching ggml's cublasGemmEx(T, N): A is stored
    // as [k,m] and transposed, B is stored as [k,n], D is [m,n].
    const size_t a_bytes = (size_t) o.k * o.m * 2;          // f16
    const size_t b_bytes = (size_t) o.k * o.n * 2;          // f16
    const size_t d_bytes = (size_t) o.m * o.n * (o.output_f16 ? 2 : 4);

    void * A = nullptr, * B = nullptr, * D = nullptr, * workspace = nullptr, * bias = nullptr;
    HIP_CHECK(hipMalloc(&A, a_bytes));
    HIP_CHECK(hipMalloc(&B, b_bytes));
    HIP_CHECK(hipMalloc(&D, d_bytes));
    if (o.bias) {
        HIP_CHECK(hipMalloc(&bias, (size_t) o.n * 4));
    }
    if (o.workspace_mb > 0) {
        HIP_CHECK(hipMalloc(&workspace, o.workspace_mb));
    }
    HIP_CHECK(hipMemsetAsync(A, 0x3c, a_bytes, stream));
    HIP_CHECK(hipMemsetAsync(B, 0x1f, b_bytes, stream));
    HIP_CHECK(hipMemsetAsync(D, 0, d_bytes, stream));
    if (bias) {
        HIP_CHECK(hipMemsetAsync(bias, 0x2e, (size_t) o.n * 4, stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    Gemm gemm(lt_handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
              HIP_R_16F, HIP_R_16F, out, out, compute);
    GemmEpilogue epilogue;
    if (o.bias) {
        epilogue.setMode(HIPBLASLT_EPILOGUE_BIAS);
        epilogue.setBiasDataType(HIP_R_32F);
    }
    GemmInputs inputs;
    inputs.setA(A);
    inputs.setB(B);
    inputs.setC(D);
    inputs.setD(D);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    if (bias) {
        inputs.setBias(bias);
    }
    GemmProblemType problem_type(HIPBLAS_OP_T, HIPBLAS_OP_N,
                                 HIP_R_16F, HIP_R_16F, out, out, compute);
    LT_CHECK(gemm.setProblem(o.m, o.n, o.k, 1,
                             o.k, o.k, o.m, o.m,
                             0, 0, 0, 0,
                             epilogue, inputs, problem_type));

    GemmPreference pref;
    pref.setMaxWorkspaceBytes(o.workspace_mb);
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    int heur_status = (int) gemm.algoGetHeuristic(o.algos, pref, results);
    std::string list_source = "heuristic";
    if (results.empty()) {
        heur_status = (int) getAllAlgos(lt_handle, GemmType::HIPBLASLT_GEMM,
                                        HIPBLAS_OP_T, HIPBLAS_OP_N,
                                        HIP_R_16F, HIP_R_16F, out, out, compute, results);
        list_source = "getAllAlgos";
    }

    if (results.empty()) {
        std::printf("m=%lld n=%lld k=%lld output=%s compute=%s algos=0 source=%s "
                    "status=%d result=no_algorithms\n",
                    (long long) o.m, (long long) o.n, (long long) o.k,
                    o.output_f16 ? "f16" : "f32",
                    o.compute_fast16 ? "fast16" : "f32",
                    list_source.c_str(), heur_status);
    } else {
        // Offline tuning: bench every supported algorithm from the heuristic
        // list (bounded to 64) with a short warmup/measure cycle, keep the
        // best; report the winner plus the rocblas-comparable number.
        struct alg_run_t {
            hipblasLtMatmulAlgo_t algo{};
            size_t ws = 0;
            std::string kernel;
            double ms = 0.0;
        };
        std::vector<alg_run_t> supported_list;
        for (auto & r : results) {
            if (r.state != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            size_t req_ws = 0;
            if (gemm.isAlgoSupported(r.algo, req_ws) == HIPBLAS_STATUS_SUCCESS &&
                req_ws <= o.workspace_mb) {
                alg_run_t rec;
                rec.algo = r.algo;
                rec.ws = req_ws;
                supported_list.push_back(rec);
                if (o.max_algos > 0 && (int) supported_list.size() >= o.max_algos) {
                    break;
                }
            }
        }

        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0));
        HIP_CHECK(hipEventCreate(&e1));
        // Phase 1: coarse scan of every supported algorithm.
        int scan_iters = std::max(4, o.iters);
        for (auto & rec : supported_list) {
            gemm.setMaxWorkspaceBytes(rec.ws);
            if (gemm.initialize(rec.algo, workspace, true, stream) != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            rec.kernel = gemm.getKernelName();
            for (int i = 0; i < std::min(o.warmup, 2); ++i) {
                gemm.run(stream);
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            float ms_f = 0.0f;
            HIP_CHECK(hipEventRecord(e0, stream));
            for (int i = 0; i < scan_iters; ++i) {
                gemm.run(stream);
            }
            HIP_CHECK(hipEventRecord(e1, stream));
            HIP_CHECK(hipEventSynchronize(e1));
            HIP_CHECK(hipEventElapsedTime(&ms_f, e0, e1));
            rec.ms = ms_f / scan_iters;
        }

        // Phase 2: precise re-bench (warmup 4, iters 32) of the top-N scan
        // winners. This is the offline-tuning result that would be cached in
        // a runtime backend.
        std::vector<size_t> order(supported_list.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return supported_list[a].ms < supported_list[b].ms;
        });
        const size_t reb_count = std::min((size_t) std::max(o.rebench_top, 1), order.size());
        for (size_t j = 0; j < reb_count; ++j) {
            alg_run_t & rec = supported_list[order[j]];
            if (rec.ms <= 0.0) {
                continue;
            }
            gemm.setMaxWorkspaceBytes(rec.ws);
            if (gemm.initialize(rec.algo, workspace, true, stream) != HIPBLAS_STATUS_SUCCESS) {
                continue;
            }
            for (int i = 0; i < 4; ++i) {
                gemm.run(stream);
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            float ms_f = 0.0f;
            HIP_CHECK(hipEventRecord(e0, stream));
            for (int i = 0; i < 32; ++i) {
                gemm.run(stream);
            }
            HIP_CHECK(hipEventRecord(e1, stream));
            HIP_CHECK(hipEventSynchronize(e1));
            HIP_CHECK(hipEventElapsedTime(&ms_f, e0, e1));
            rec.ms = ms_f / 32.0;
        }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return supported_list[a].ms < supported_list[b].ms;
        });

        // Mirror the first runtime proxy: rebuild the problem, re-check the
        // selected algorithm and re-initialize it for every GEMM. The normal
        // scan above measures only hot gemm.run(), so this separates kernel
        // speed from extension-wrapper setup cost.
        if (o.setup_bench && !order.empty()) {
            alg_run_t & rec = supported_list[order[0]];
            float setup_device_ms = 0.0f;
            HIP_CHECK(hipStreamSynchronize(stream));
            const auto setup_wall_start = std::chrono::steady_clock::now();
            HIP_CHECK(hipEventRecord(e0, stream));
            for (int i = 0; i < o.iters; ++i) {
                LT_CHECK(gemm.setProblem(o.m, o.n, o.k, 1,
                                         o.k, o.k, o.m, o.m,
                                         0, 0, 0, 0,
                                         epilogue, inputs, problem_type));
                size_t setup_ws = 0;
                LT_CHECK(gemm.isAlgoSupported(rec.algo, setup_ws));
                if (setup_ws > o.workspace_mb) {
                    std::fprintf(stderr, "setup bench workspace exceeds allocation\n");
                    return 1;
                }
                gemm.setMaxWorkspaceBytes(setup_ws);
                LT_CHECK(gemm.initialize(rec.algo, workspace, true, stream));
                LT_CHECK(gemm.run(stream));
            }
            HIP_CHECK(hipEventRecord(e1, stream));
            HIP_CHECK(hipEventSynchronize(e1));
            const auto setup_wall_end = std::chrono::steady_clock::now();
            HIP_CHECK(hipEventElapsedTime(&setup_device_ms, e0, e1));
            const double setup_wall_ms = std::chrono::duration<double, std::milli>(
                setup_wall_end - setup_wall_start).count() / o.iters;
            const double setup_gpu_ms = setup_device_ms / o.iters;
            std::printf("setup_bench hot_run_ms=%.4f setup_each_device_ms=%.4f "
                        "setup_each_wall_ms=%.4f wall_over_hot_ms=%.4f\n",
                        rec.ms, setup_gpu_ms, setup_wall_ms, setup_wall_ms - rec.ms);

            // Build the raw C API descriptors once, then vary only pointers at
            // dispatch time. This is the intended runtime cache design.
            hipblasLtMatmulDesc_t raw_op = nullptr;
            hipblasLtMatrixLayout_t raw_a = nullptr;
            hipblasLtMatrixLayout_t raw_b = nullptr;
            hipblasLtMatrixLayout_t raw_c = nullptr;
            hipblasLtMatrixLayout_t raw_d = nullptr;
            LT_CHECK(hipblasLtMatmulDescCreate(&raw_op, compute, HIP_R_32F));
            const hipblasOperation_t raw_trans_a = HIPBLAS_OP_T;
            const hipblasOperation_t raw_trans_b = HIPBLAS_OP_N;
            LT_CHECK(hipblasLtMatmulDescSetAttribute(
                raw_op, HIPBLASLT_MATMUL_DESC_TRANSA, &raw_trans_a, sizeof(raw_trans_a)));
            LT_CHECK(hipblasLtMatmulDescSetAttribute(
                raw_op, HIPBLASLT_MATMUL_DESC_TRANSB, &raw_trans_b, sizeof(raw_trans_b)));
            LT_CHECK(hipblasLtMatrixLayoutCreate(
                &raw_a, HIP_R_16F, o.k, o.m, o.k));
            LT_CHECK(hipblasLtMatrixLayoutCreate(
                &raw_b, HIP_R_16F, o.k, o.n, o.k));
            LT_CHECK(hipblasLtMatrixLayoutCreate(
                &raw_c, out, o.m, o.n, o.m));
            LT_CHECK(hipblasLtMatrixLayoutCreate(
                &raw_d, out, o.m, o.n, o.m));

            hipblasLtMatmulAlgo_t raw_algo = rec.algo;
            size_t raw_ws = 0;
            LT_CHECK(matmulIsAlgoSupported(
                lt_handle, raw_op, &alpha, raw_a, raw_b, &beta,
                raw_c, raw_d, raw_algo, raw_ws));
            if (raw_ws > o.workspace_mb) {
                std::fprintf(stderr, "raw cached workspace exceeds allocation\n");
                return 1;
            }
            for (int i = 0; i < 4; ++i) {
                LT_CHECK(hipblasLtMatmul(
                    lt_handle, raw_op, &alpha, A, raw_a, B, raw_b,
                    &beta, D, raw_c, D, raw_d, &raw_algo,
                    workspace, raw_ws, stream));
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            float raw_device_ms = 0.0f;
            const auto raw_wall_start = std::chrono::steady_clock::now();
            HIP_CHECK(hipEventRecord(e0, stream));
            for (int i = 0; i < o.iters; ++i) {
                LT_CHECK(hipblasLtMatmul(
                    lt_handle, raw_op, &alpha, A, raw_a, B, raw_b,
                    &beta, D, raw_c, D, raw_d, &raw_algo,
                    workspace, raw_ws, stream));
            }
            HIP_CHECK(hipEventRecord(e1, stream));
            HIP_CHECK(hipEventSynchronize(e1));
            const auto raw_wall_end = std::chrono::steady_clock::now();
            HIP_CHECK(hipEventElapsedTime(&raw_device_ms, e0, e1));
            const double raw_wall_ms = std::chrono::duration<double, std::milli>(
                raw_wall_end - raw_wall_start).count() / o.iters;
            std::printf("raw_cached device_ms=%.4f wall_ms=%.4f workspace=%zu\n",
                        raw_device_ms / o.iters, raw_wall_ms, raw_ws);

            LT_CHECK(hipblasLtMatrixLayoutDestroy(raw_d));
            LT_CHECK(hipblasLtMatrixLayoutDestroy(raw_c));
            LT_CHECK(hipblasLtMatrixLayoutDestroy(raw_b));
            LT_CHECK(hipblasLtMatrixLayoutDestroy(raw_a));
            LT_CHECK(hipblasLtMatmulDescDestroy(raw_op));
        }

        // Phase 3: user tuning (splitK / WGM) forced on the scan winner.
        auto bench_tuned = [&](alg_run_t & rec, GemmTuning & tun) -> double {
            size_t ws2 = 0;
            if (gemm.isAlgoSupported(rec.algo, tun, ws2) != HIPBLAS_STATUS_SUCCESS ||
                ws2 > o.workspace_mb) {
                return -1.0;
            }
            gemm.setMaxWorkspaceBytes(ws2);
            if (gemm.initialize(rec.algo, tun, workspace, true, stream) != HIPBLAS_STATUS_SUCCESS) {
                return -1.0;
            }
            for (int i = 0; i < 2; ++i) {
                gemm.run(stream);
            }
            HIP_CHECK(hipStreamSynchronize(stream));
            float ms_f = 0.0f;
            HIP_CHECK(hipEventRecord(e0, stream));
            for (int i = 0; i < 16; ++i) {
                gemm.run(stream);
            }
            HIP_CHECK(hipEventRecord(e1, stream));
            HIP_CHECK(hipEventSynchronize(e1));
            HIP_CHECK(hipEventElapsedTime(&ms_f, e0, e1));
            return ms_f / 16.0;
        };
        if ((o.tune_splitk || o.tune_wgm) && !order.empty()) {
            alg_run_t & rec = supported_list[order[0]];
            if (o.tune_splitk) {
                for (int sk : {1, 2, 4, 8}) {
                    GemmTuning tun;
                    tun.setSplitK((uint16_t) sk);
                    double tt = bench_tuned(rec, tun);
                    std::printf("tune_splitk_%d_ms=%s\n", sk,
                                tt > 0 ? (std::to_string(tt)).c_str() : "not_supported");
                }
            }
            if (o.tune_wgm) {
                for (int wgm : {1, 2, 4, 8}) {
                    GemmTuning tun;
                    tun.setWgm((int16_t) wgm);
                    double tt = bench_tuned(rec, tun);
                    std::printf("tune_wgm_%d_ms=%s\n", wgm,
                                tt > 0 ? (std::to_string(tt)).c_str() : "not_supported");
                }
            }
        }

        auto print_algo = [&](const char * tag, const alg_run_t & rec) {
            hipblasLtMatmulAlgo_t algo = rec.algo;   // getIndexFromAlgo takes non-const ref
            std::printf("%s_ms=%.4f %s_algo=%d %s_kernel=%s\n", tag, rec.ms, tag,
                        getIndexFromAlgo(algo), tag, rec.kernel.c_str());
        };
        std::printf("m=%lld n=%lld k=%lld output=%s compute=%s bias=%d "
                    "algos_heuristic=%zu supported_tested=%zu reb_count=%zu\n",
                    (long long) o.m, (long long) o.n, (long long) o.k,
                    o.output_f16 ? "f16" : "f32",
                    o.compute_fast16 ? "fast16" : "f32",
                    o.bias ? 1 : 0,
                    results.size(), supported_list.size(), reb_count);
        for (size_t i = 0; i < std::min((size_t) 3, order.size()); ++i) {
            std::printf("rank%zu ", i + 1);
            print_algo((i == 0) ? "best" : (i == 1) ? "second" : "third", supported_list[order[i]]);
        }
        HIP_CHECK(hipEventDestroy(e0));
        HIP_CHECK(hipEventDestroy(e1));
    }

    if (bias) HIP_CHECK(hipFree(bias));
    if (workspace) HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipFree(D));
    HIP_CHECK(hipFree(B));
    HIP_CHECK(hipFree(A));
    LT_CHECK(hipblasLtDestroy(lt_handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
