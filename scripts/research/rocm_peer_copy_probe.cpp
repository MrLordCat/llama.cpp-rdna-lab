#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

enum class probe_mode {
    none,
    capabilities,
    copy_sync,
    copy_event,
};

struct options {
    probe_mode mode = probe_mode::none;
    int src = -1;
    int dst = -1;
    size_t bytes = 0;
    int iterations = 1;
    bool acknowledge_driver_risk = false;
};

volatile std::sig_atomic_t g_stop_requested = 0;

void on_interrupt(int) {
    g_stop_requested = 1;
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "ROCm peer-copy correctness probe. No HIP calls are made without an explicit mode.\n"
        "\n"
        "Capability-only mode:\n"
        "  %s --capabilities\n"
        "\n"
        "Copy modes (one direction and one size per process):\n"
        "  %s --copy-sync  --acknowledge-driver-risk --src 0 --dst 1 --bytes 20480 [--iterations 1]\n"
        "  %s --copy-event --acknowledge-driver-risk --src 0 --dst 1 --bytes 20480 [--iterations 1]\n"
        "\n"
        "--copy-sync waits for source-side completion before reading the destination.\n"
        "--copy-event reproduces llama.cpp's source-event/destination-wait ordering.\n"
        "Ctrl+C requests a soft stop after the in-flight HIP operation completes.\n",
        argv0, argv0, argv0);
}

bool parse_int(const char * text, int & value) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_size(const char * text, size_t & value) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 || parsed > 64ull * 1024 * 1024) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool select_mode(options & opts, probe_mode mode) {
    if (opts.mode != probe_mode::none) {
        return false;
    }
    opts.mode = mode;
    return true;
}

bool parse_options(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--capabilities") == 0) {
            if (!select_mode(opts, probe_mode::capabilities)) {
                return false;
            }
        } else if (std::strcmp(arg, "--copy-sync") == 0) {
            if (!select_mode(opts, probe_mode::copy_sync)) {
                return false;
            }
        } else if (std::strcmp(arg, "--copy-event") == 0) {
            if (!select_mode(opts, probe_mode::copy_event)) {
                return false;
            }
        } else if (std::strcmp(arg, "--acknowledge-driver-risk") == 0) {
            opts.acknowledge_driver_risk = true;
        } else if (std::strcmp(arg, "--src") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], opts.src)) {
                return false;
            }
        } else if (std::strcmp(arg, "--dst") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], opts.dst)) {
                return false;
            }
        } else if (std::strcmp(arg, "--bytes") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], opts.bytes)) {
                return false;
            }
        } else if (std::strcmp(arg, "--iterations") == 0 && i + 1 < argc) {
            if (!parse_int(argv[++i], opts.iterations) || opts.iterations < 1 || opts.iterations > 10000) {
                return false;
            }
        } else {
            return false;
        }
    }

    if (opts.mode == probe_mode::capabilities) {
        return opts.src == -1 && opts.dst == -1 && opts.bytes == 0 && opts.iterations == 1 &&
            !opts.acknowledge_driver_risk;
    }
    if (opts.mode == probe_mode::copy_sync || opts.mode == probe_mode::copy_event) {
        return opts.acknowledge_driver_risk && opts.src >= 0 && opts.dst >= 0 && opts.src != opts.dst &&
            opts.bytes > 0;
    }
    return false;
}

bool hip_call(hipError_t status, const char * operation) {
    if (status == hipSuccess) {
        return true;
    }
    std::fprintf(stderr, "HIP failure: operation=%s code=%d message=%s\n",
        operation, static_cast<int>(status), hipGetErrorString(status));
    return false;
}

void print_p2p_attribute(int src, int dst, hipDeviceP2PAttr attr, const char * name) {
    int value = 0;
    const hipError_t status = hipDeviceGetP2PAttribute(&value, attr, src, dst);
    if (status == hipSuccess) {
        std::printf("p2p src=%d dst=%d attr=%s value=%d\n", src, dst, name, value);
    } else {
        std::printf("p2p src=%d dst=%d attr=%s unavailable code=%d message=%s\n",
            src, dst, name, static_cast<int>(status), hipGetErrorString(status));
    }
}

int run_capabilities() {
    int device_count = 0;
    if (!hip_call(hipGetDeviceCount(&device_count), "hipGetDeviceCount")) {
        return 1;
    }
    std::printf("device_count=%d\n", device_count);

    for (int device = 0; device < device_count; ++device) {
        hipDeviceProp_t props = {};
        char pci_bus_id[32] = {};
        if (!hip_call(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties")) {
            return 1;
        }
        const hipError_t pci_status = hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device);
        std::printf("device=%d name=%s arch=%s pci=%s\n",
            device,
            props.name,
            props.gcnArchName,
            pci_status == hipSuccess ? pci_bus_id : "unavailable");
    }

    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) {
                continue;
            }
            int can_access = 0;
            const hipError_t status = hipDeviceCanAccessPeer(&can_access, src, dst);
            if (status != hipSuccess) {
                std::printf("p2p src=%d dst=%d can_access=unavailable code=%d message=%s\n",
                    src, dst, static_cast<int>(status), hipGetErrorString(status));
                continue;
            }
            std::printf("p2p src=%d dst=%d can_access=%d\n", src, dst, can_access);
            print_p2p_attribute(src, dst, hipDevP2PAttrAccessSupported, "access_supported");
            print_p2p_attribute(src, dst, hipDevP2PAttrPerformanceRank, "performance_rank");
            print_p2p_attribute(src, dst, hipDevP2PAttrNativeAtomicSupported, "native_atomic_supported");
            print_p2p_attribute(src, dst, hipDevP2PAttrHipArrayAccessSupported, "array_access_supported");
        }
    }
    return 0;
}

uint8_t pattern_byte(size_t index, int iteration, uint64_t salt) {
    uint64_t value = index + salt + static_cast<uint64_t>(iteration + 1) * 0x9e3779b97f4a7c15ull;
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return static_cast<uint8_t>(value);
}

uint64_t fnv1a64(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

bool verify_bytes(const std::vector<uint8_t> & expected, const uint8_t * actual, size_t total_size) {
    for (size_t i = 0; i < total_size; ++i) {
        if (expected[i] != actual[i]) {
            std::fprintf(stderr,
                "CORRUPTION: first_mismatch=%zu expected=%u actual=%u expected_hash=%016llx actual_hash=%016llx\n",
                i,
                static_cast<unsigned>(expected[i]),
                static_cast<unsigned>(actual[i]),
                static_cast<unsigned long long>(fnv1a64(expected.data(), total_size)),
                static_cast<unsigned long long>(fnv1a64(actual, total_size)));
            return false;
        }
    }
    return true;
}

bool wait_for_event(hipEvent_t event, const char * label) {
    bool stop_reported = false;
    while (true) {
        const hipError_t status = hipEventQuery(event);
        if (status == hipSuccess) {
            return true;
        }
        if (status != hipErrorNotReady) {
            return hip_call(status, label);
        }
        if (g_stop_requested && !stop_reported) {
            std::fprintf(stderr, "Soft stop requested; waiting for the in-flight HIP operation before cleanup.\n");
            stop_reported = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

struct peer_resources {
    int src = -1;
    int dst = -1;
    void * src_data = nullptr;
    void * dst_data = nullptr;
    void * host_result = nullptr;
    hipStream_t src_stream = nullptr;
    hipStream_t dst_stream = nullptr;
    hipEvent_t src_done = nullptr;
    hipEvent_t dst_done = nullptr;
    bool peer_enabled_here = false;
};

void cleanup_call(hipError_t status, const char * operation) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "Cleanup warning: operation=%s code=%d message=%s\n",
            operation, static_cast<int>(status), hipGetErrorString(status));
    }
}

void cleanup(peer_resources & resources) {
    if (resources.src >= 0) {
        cleanup_call(hipSetDevice(resources.src), "hipSetDevice(src cleanup)");
        if (resources.src_done != nullptr) {
            cleanup_call(hipEventDestroy(resources.src_done), "hipEventDestroy(src_done)");
        }
        if (resources.src_stream != nullptr) {
            cleanup_call(hipStreamDestroy(resources.src_stream), "hipStreamDestroy(src)");
        }
        if (resources.src_data != nullptr) {
            cleanup_call(hipFree(resources.src_data), "hipFree(src)");
        }
    }
    if (resources.dst >= 0) {
        cleanup_call(hipSetDevice(resources.dst), "hipSetDevice(dst cleanup)");
        if (resources.dst_done != nullptr) {
            cleanup_call(hipEventDestroy(resources.dst_done), "hipEventDestroy(dst_done)");
        }
        if (resources.dst_stream != nullptr) {
            cleanup_call(hipStreamDestroy(resources.dst_stream), "hipStreamDestroy(dst)");
        }
        if (resources.dst_data != nullptr) {
            cleanup_call(hipFree(resources.dst_data), "hipFree(dst)");
        }
    }
    if (resources.host_result != nullptr) {
        cleanup_call(hipHostFree(resources.host_result), "hipHostFree(result)");
    }
    if (resources.peer_enabled_here) {
        cleanup_call(hipSetDevice(resources.src), "hipSetDevice(peer disable)");
        const hipError_t status = hipDeviceDisablePeerAccess(resources.dst);
        if (status != hipSuccess && status != hipErrorPeerAccessNotEnabled) {
            std::fprintf(stderr, "Warning: hipDeviceDisablePeerAccess failed: %s\n", hipGetErrorString(status));
        }
    }
}

bool prepare_peer_access(int src, int dst, bool & enabled_here) {
    int can_access = 0;
    if (!hip_call(hipDeviceCanAccessPeer(&can_access, src, dst), "hipDeviceCanAccessPeer")) {
        return false;
    }
    if (!can_access) {
        std::fprintf(stderr, "Peer route rejected: src=%d cannot access dst=%d.\n", src, dst);
        return false;
    }

    int attr_access = 0;
    const hipError_t attr_status = hipDeviceGetP2PAttribute(
        &attr_access, hipDevP2PAttrAccessSupported, src, dst);
    if (attr_status == hipSuccess && !attr_access) {
        std::fprintf(stderr, "Peer route rejected: access_supported=0 for src=%d dst=%d.\n", src, dst);
        return false;
    }
    if (attr_status != hipSuccess) {
        std::fprintf(stderr, "Warning: access_supported attribute unavailable: %s\n",
            hipGetErrorString(attr_status));
    }

    if (!hip_call(hipSetDevice(src), "hipSetDevice(src)")) {
        return false;
    }
    const hipError_t enable_status = hipDeviceEnablePeerAccess(dst, 0);
    if (enable_status == hipSuccess) {
        enabled_here = true;
        return true;
    }
    if (enable_status == hipErrorPeerAccessAlreadyEnabled) {
        enabled_here = false;
        return true;
    }
    return hip_call(enable_status, "hipDeviceEnablePeerAccess");
}

int run_copy_probe(const options & opts) {
    int device_count = 0;
    if (!hip_call(hipGetDeviceCount(&device_count), "hipGetDeviceCount")) {
        return 1;
    }
    if (opts.src >= device_count || opts.dst >= device_count) {
        std::fprintf(stderr, "Invalid device pair: src=%d dst=%d device_count=%d.\n",
            opts.src, opts.dst, device_count);
        return 2;
    }

    const bool cross_device_event = opts.mode == probe_mode::copy_event;
    constexpr size_t guard_size = 4096;
    const size_t total_size = opts.bytes + 2 * guard_size;
    peer_resources resources;
    resources.src = opts.src;
    resources.dst = opts.dst;

    if (!prepare_peer_access(opts.src, opts.dst, resources.peer_enabled_here)) {
        return 1;
    }

    bool setup_ok = true;
    setup_ok = setup_ok && hip_call(hipSetDevice(opts.src), "hipSetDevice(src setup)");
    setup_ok = setup_ok && hip_call(hipStreamCreateWithFlags(&resources.src_stream, hipStreamNonBlocking),
        "hipStreamCreateWithFlags(src)");
    setup_ok = setup_ok && hip_call(hipEventCreateWithFlags(&resources.src_done, hipEventDisableTiming),
        "hipEventCreateWithFlags(src_done)");
    setup_ok = setup_ok && hip_call(hipMalloc(&resources.src_data, total_size), "hipMalloc(src)");

    setup_ok = setup_ok && hip_call(hipSetDevice(opts.dst), "hipSetDevice(dst setup)");
    setup_ok = setup_ok && hip_call(hipMalloc(&resources.dst_data, total_size), "hipMalloc(dst)");
    setup_ok = setup_ok && hip_call(hipHostMalloc(&resources.host_result, total_size, hipHostMallocDefault),
        "hipHostMalloc(result)");
    if (cross_device_event) {
        setup_ok = setup_ok && hip_call(hipStreamCreateWithFlags(&resources.dst_stream, hipStreamNonBlocking),
            "hipStreamCreateWithFlags(dst)");
        setup_ok = setup_ok && hip_call(hipEventCreateWithFlags(&resources.dst_done, hipEventDisableTiming),
            "hipEventCreateWithFlags(dst_done)");
    }
    if (!setup_ok) {
        cleanup(resources);
        return 1;
    }

    std::vector<uint8_t> host_src(total_size);
    std::vector<uint8_t> host_dst(total_size);
    std::vector<uint8_t> expected(total_size);
    auto * host_result = static_cast<uint8_t *>(resources.host_result);
    auto * src_payload = static_cast<uint8_t *>(resources.src_data) + guard_size;
    auto * dst_payload = static_cast<uint8_t *>(resources.dst_data) + guard_size;

    std::printf("copy_test mode=%s src=%d dst=%d bytes=%zu iterations=%d guard=%zu\n",
        cross_device_event ? "event" : "sync",
        opts.src, opts.dst, opts.bytes, opts.iterations, guard_size);

    for (int iteration = 0; iteration < opts.iterations; ++iteration) {
        if (g_stop_requested) {
            std::fprintf(stderr, "Soft stop before iteration %d.\n", iteration + 1);
            cleanup(resources);
            return 130;
        }

        for (size_t i = 0; i < total_size; ++i) {
            host_src[i] = pattern_byte(i, iteration, 0x13579bdf2468ace0ull);
            host_dst[i] = pattern_byte(i, iteration, 0xfdb97531eca86420ull);
        }
        expected = host_dst;
        std::copy_n(host_src.data() + guard_size, opts.bytes, expected.data() + guard_size);

        bool iteration_ok = true;
        iteration_ok = iteration_ok && hip_call(hipSetDevice(opts.src), "hipSetDevice(src upload)");
        iteration_ok = iteration_ok && hip_call(
            hipMemcpy(resources.src_data, host_src.data(), total_size, hipMemcpyHostToDevice),
            "hipMemcpy(src upload)");
        iteration_ok = iteration_ok && hip_call(hipSetDevice(opts.dst), "hipSetDevice(dst upload)");
        iteration_ok = iteration_ok && hip_call(
            hipMemcpy(resources.dst_data, host_dst.data(), total_size, hipMemcpyHostToDevice),
            "hipMemcpy(dst upload)");
        iteration_ok = iteration_ok && hip_call(hipSetDevice(opts.src), "hipSetDevice(src copy)");
        if (!iteration_ok) {
            cleanup(resources);
            return 1;
        }

        const auto start = std::chrono::steady_clock::now();
        const hipError_t copy_status = hipMemcpyPeerAsync(
            dst_payload, opts.dst, src_payload, opts.src, opts.bytes, resources.src_stream);
        if (!hip_call(copy_status, "hipMemcpyPeerAsync")) {
            std::fprintf(stderr, "Peer path is tainted; exiting without attempting a host fallback.\n");
            return 1;
        }
        if (!hip_call(hipEventRecord(resources.src_done, resources.src_stream), "hipEventRecord(src_done)")) {
            std::fprintf(stderr, "Peer path is tainted; exiting without attempting a host fallback.\n");
            return 1;
        }

        bool completion_ok = false;
        if (!cross_device_event) {
            completion_ok = wait_for_event(resources.src_done, "hipEventQuery(src_done)");
            if (completion_ok) {
                completion_ok = hip_call(hipSetDevice(opts.dst), "hipSetDevice(dst download)") &&
                    hip_call(hipMemcpy(host_result, resources.dst_data, total_size, hipMemcpyDeviceToHost),
                        "hipMemcpy(dst download)");
            }
        } else {
            completion_ok = hip_call(hipSetDevice(opts.dst), "hipSetDevice(dst wait)") &&
                hip_call(hipStreamWaitEvent(resources.dst_stream, resources.src_done, 0),
                    "hipStreamWaitEvent(dst <- src_done)") &&
                hip_call(hipMemcpyAsync(host_result, resources.dst_data, total_size,
                    hipMemcpyDeviceToHost, resources.dst_stream), "hipMemcpyAsync(dst download)") &&
                hip_call(hipEventRecord(resources.dst_done, resources.dst_stream), "hipEventRecord(dst_done)") &&
                wait_for_event(resources.dst_done, "hipEventQuery(dst_done)");
        }
        if (!completion_ok) {
            std::fprintf(stderr, "Peer path is tainted; exiting without attempting a host fallback.\n");
            return 1;
        }

        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        if (!verify_bytes(expected, host_result, total_size)) {
            std::fprintf(stderr, "Peer route rejected after silent data corruption.\n");
            cleanup(resources);
            return 1;
        }

        std::printf("PASS iteration=%d/%d elapsed_ms=%.3f hash=%016llx\n",
            iteration + 1,
            opts.iterations,
            elapsed_ms,
            static_cast<unsigned long long>(fnv1a64(host_result, total_size)));
    }

    cleanup(resources);
    std::printf("RESULT: PASS mode=%s src=%d dst=%d bytes=%zu iterations=%d\n",
        cross_device_event ? "event" : "sync",
        opts.src, opts.dst, opts.bytes, opts.iterations);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        print_usage(argv[0]);
        return 2;
    }

    std::signal(SIGINT, on_interrupt);
    if (opts.mode == probe_mode::capabilities) {
        return run_capabilities();
    }
    return run_copy_probe(opts);
}
