// D096-A: Vulkan probe for fp8 (E4M3) cooperative matrix driver acceptance.
// Usage: probe.exe <kernel.spv>
//   - creates instance/device with VK_KHR_cooperative_matrix (+ feature bit)
//   - prints VkCooperativeMatrixPropertiesKHR (component types)
//   - creates a compute pipeline from the given SPIR-V module
//   - dispatches C[16x16 f32] = A[16x16 fp8] * B[16x16 fp8] (row-major)
//   - verifies the result against host-side fp32 math
// Exit code: 0 = full PASS (pipeline created + math correct), 1 = FAIL, 2 = usage
//
// Build (MinGW, Vulkan SDK):
//   g++ -std=c++17 -O2 -I"${VULKAN_SDK}/Include" probe.cpp \
//       -L"${VULKAN_SDK}/Lib" -lvulkan-1 -o probe.exe
// Run: ./probe.exe fp8_mul.spv

#include <vulkan/vulkan.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<uint32_t> read_spv(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::vector<uint32_t> words;
    char buf[4];
    while (f.read(buf, 4)) {
        uint32_t w;
        std::memcpy(&w, buf, 4);
        words.push_back(w);
    }
    return words;
}

// fp32 -> E4M3 (bias 7, 3-bit mantissa, clamp 448; round half away from zero)
static uint8_t f32_to_e4m3(float v) {
    if (v != v) return 0x7F; // NaN -> qNaN-ish
    uint32_t b;
    std::memcpy(&b, &v, 4);
    uint32_t sign = (b >> 31) & 1;
    int32_t e = (int32_t)((b >> 23) & 0xFF) - 127 + 7; // fp8 exponent bias 7
    uint32_t man = (b >> 20) & 0x7;                    // top 3 mantissa bits
    // round: look at bit 19
    if (((b >> 19) & 1) && man < 7) man += 1;
    if (e < -7) return (uint8_t)(sign << 7);           // underflow -> 0
    if (e > 15) {                                      // overflow -> max
        return (uint8_t)((sign << 7) | 0x7E);
    }
    if (e <= 0) { // subnormal: value = man * 2^-7 ... keep it simple
        int32_t s = 1 - e; // shift
        uint32_t m = (b >> 19) & 0xFF; // 8 bits for subnormal approximation
        if (s >= 8) return (uint8_t)(sign << 7);
        m >>= s;
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | m);
    }
    return (uint8_t)((sign << 7) | ((uint32_t)e << 3) | man);
}

static float e4m3_to_f32(uint8_t v) {
    uint32_t sign = (v >> 7) & 1;
    uint32_t e = (v >> 3) & 0xF;
    uint32_t m = v & 0x7;
    float f;
    if (e == 0) {
        f = (float)m * 0.0078125f; // 2^-7
    } else if (e == 15) {
        f = m ? 0.0f / 0.0f : 1.0f / 0.0f; // NaN/Inf
    } else {
        f = (float)((1 << 3) | m) * exp2f((int)e - 7 - 3);
    }
    return sign ? -f : f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: probe.exe <kernel.spv>\n");
        return 2;
    }
    const std::vector<uint32_t> spv = read_spv(argv[1]);

    try {
        vk::Instance inst = vk::createInstance({ {}, nullptr, 0, nullptr, 0, nullptr });
        auto devices = inst.enumeratePhysicalDevices();

        vk::PhysicalDevice phys;
        for (auto& d : devices) {
            auto props = d.getProperties();
            std::printf("device: %s (vendor 0x%X)\n", props.deviceName, props.vendorID);
            if (phys) continue;
            auto exts = d.enumerateDeviceExtensionProperties();
            for (auto& e : exts) {
                if (std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
                    phys = d;
                    std::printf("  -> has VK_KHR_cooperative_matrix\n");
                    break;
                }
            }
            if (!phys) phys = d; // fallback: first device
        }
        if (!phys) { std::fprintf(stderr, "no physical device\n"); return 1; }

        // ---- property query (C API: extension function not statically linked) ----
        auto pfn_cm = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(static_cast<VkInstance>(inst),
                                  "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (!pfn_cm) { std::fprintf(stderr, "no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); return 1; }
        uint32_t count = 0;
        pfn_cm(static_cast<VkPhysicalDevice>(phys), &count, nullptr);
        std::vector<VkCooperativeMatrixPropertiesKHR> props(count);
        pfn_cm(static_cast<VkPhysicalDevice>(phys), &count, props.data());
        bool saw_fp8 = false;
        for (auto& p : props) {
            std::printf("  coopmat prop: scope=%u M=%u N=%u K=%u A=0x%X B=0x%X C=0x%X acc=0x%X\n",
                p.scope, p.MSize, p.NSize, p.KSize, p.AType, p.BType, p.CType, p.ResultType);
            if ((uint32_t)p.AType == 1000491002u) saw_fp8 = true; // VK_COMPONENT_TYPE_FLOAT8_E4M3_NV
        }
        std::printf("  fp8 E4M3 coopmat advertised: %s\n", saw_fp8 ? "YES" : "no");

        // ---- device with cooperative matrix ----
        auto qfam_props = phys.getQueueFamilyProperties();
        uint32_t qfam = UINT32_MAX;
        for (uint32_t i = 0; i < (uint32_t)qfam_props.size(); i++) {
            if ((qfam_props[i].queueFlags & vk::QueueFlagBits::eCompute) && qfam_props[i].queueCount > 0) {
                qfam = i;
                break;
            }
        }
        if (qfam == UINT32_MAX) { std::fprintf(stderr, "no compute queue family\n"); return 1; }
        std::printf("  using queue family %u (flags 0x%X)\n", qfam, (uint32_t)qfam_props[qfam].queueFlags);

        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR cm_feat;
        cm_feat.cooperativeMatrix = VK_TRUE;
        float prio = 1.0f;
        vk::DeviceQueueCreateInfo qci({}, qfam, 1, &prio);
        const char* ext = VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        vk::DeviceCreateInfo dci({}, 1, &qci, 0, nullptr, 1, &ext, nullptr);
        dci.pNext = &cm_feat;
        vk::Device dev = phys.createDevice(dci);

        // ---- shader module + pipeline ----
        vk::ShaderModule sm = dev.createShaderModule({ {}, spv.size() * 4, spv.data() });

        vk::DescriptorSetLayoutBinding binds[3] = {
            { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr },
            { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr },
            { 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr },
        };
        vk::DescriptorSetLayout dsl = dev.createDescriptorSetLayout({ {}, 3, binds });
        vk::PipelineLayout pl = dev.createPipelineLayout({ {}, 1, &dsl });
        vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, sm, "main");
        vk::ComputePipelineCreateInfo cpi({}, stage, pl);
        vk::Pipeline pipe;
        try {
            auto pv = dev.createComputePipeline(nullptr, cpi);
            if (pv.result != vk::Result::eSuccess) {
                std::printf("PIPELINE CREATE FAILED: %s\n", vk::to_string(pv.result).c_str());
                return 1;
            }
            pipe = pv.value;
        } catch (vk::SystemError& e) {
            std::printf("PIPELINE CREATE FAILED: %s\n", e.what());
            return 1;
        }
        std::printf("pipeline created OK\n");

        // ---- buffers: A, B (fp8, 256 B), C (f32, 1024 B) ----
        const vk::DeviceSize sizes[3] = { 256, 256, 1024 };
        vk::Buffer bufs[3];
        vk::DeviceMemory mems[3];
        uint8_t* maps[3];
        for (int i = 0; i < 3; i++) {
            bufs[i] = dev.createBuffer({ {}, sizes[i], vk::BufferUsageFlagBits::eStorageBuffer,
                                         vk::SharingMode::eExclusive });
            auto req = dev.getBufferMemoryRequirements(bufs[i]);
            vk::MemoryAllocateInfo ai(req.size, 0);
            // find host-visible memory type
            auto mprops = phys.getMemoryProperties();
            for (uint32_t t = 0; t < mprops.memoryTypeCount; t++) {
                if (req.memoryTypeBits & (1u << t)) {
                    auto flags = mprops.memoryTypes[t].propertyFlags;
                    if (flags & vk::MemoryPropertyFlagBits::eHostVisible) { ai.memoryTypeIndex = t; break; }
                }
            }
            mems[i] = dev.allocateMemory(ai);
            dev.bindBufferMemory(bufs[i], mems[i], 0);
            maps[i] = (uint8_t*)dev.mapMemory(mems[i], 0, sizes[i]);
        }

        // fill A and B with exact E4M3 values (powers of two)
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                maps[0][r * 16 + c] = f32_to_e4m3((r == c) ? 1.0f : 0.0f);          // A = I
                maps[1][r * 16 + c] = f32_to_e4m3((float)((r + c) % 4) - 1.0f);       // B: -1..2
            }
        }
        std::memset(maps[2], 0, 1024);

        // descriptor set
        std::array<vk::DescriptorPoolSize, 1> ps = { vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3) };
        vk::DescriptorPool pool = dev.createDescriptorPool({ {}, 1, ps });
        vk::DescriptorSet ds = dev.allocateDescriptorSets({ pool, 1, &dsl })[0];
        vk::DescriptorBufferInfo binfo[3] = {
            { bufs[0], 0, sizes[0] }, { bufs[1], 0, sizes[1] }, { bufs[2], 0, sizes[2] },
        };
        vk::WriteDescriptorSet writes[3] = {
            { ds, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &binfo[0] },
            { ds, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &binfo[1] },
            { ds, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &binfo[2] },
        };
        dev.updateDescriptorSets(3, writes, 0, nullptr);

        // command buffer
        vk::CommandPool cpool = dev.createCommandPool({ {}, qfam });
        vk::CommandBuffer cb = dev.allocateCommandBuffers({ cpool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        cb.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
        cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pl, 0, 1, &ds, 0, nullptr);
        cb.dispatch(1, 1, 1);
        cb.end();
        vk::Fence fence = dev.createFence({});
        vk::Queue queue = dev.getQueue(qfam, 0);
        try {
            queue.submit({ vk::SubmitInfo(0, nullptr, nullptr, 1, &cb) }, fence);
        } catch (vk::SystemError& e) {
            std::printf("SUBMIT FAILED: %s\n", e.what());
            return 1;
        }
        vk::Result wr = dev.waitForFences(1, &fence, VK_TRUE, UINT64_MAX);
        std::printf("dispatch+wait: %s\n", vk::to_string(wr).c_str());
        if (wr != vk::Result::eSuccess) {
            std::printf("  -> fence wait not success; device lost? %s\n",
                wr == vk::Result::eErrorDeviceLost ? "YES" : "no");
        }

        // verify C
        float* C = (float*)maps[2];
        float max_err = 0.0f;
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 16; c++) {
                float expect = 0.0f;
                for (int k = 0; k < 16; k++) {
                    expect += e4m3_to_f32(maps[0][r * 16 + k]) * e4m3_to_f32(maps[1][k * 16 + c]);
                }
                float err = fabsf(C[r * 16 + c] - expect);
                if (err > max_err) max_err = err;
            }
        }
        std::printf("max |C_gpu - C_cpu| = %g  -> %s\n", max_err, max_err < 1e-3f ? "PASS" : "FAIL");
        std::printf("C[0][0..7]  = %g %g %g %g %g %g %g %g\n",
            C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);
        std::printf("C[1][0..3]  = %g %g %g %g\n", C[16], C[17], C[18], C[19]);
        return max_err < 1e-3f ? 0 : 1;
    } catch (vk::SystemError& e) {
        std::printf("VULKAN ERROR: %s\n", e.what());
        return 1;
    }
}
