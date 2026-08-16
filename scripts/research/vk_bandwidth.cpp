// vk_bandwidth.cpp — synthetic Vulkan device-local READ bandwidth benchmark
// for the llama.cpp-with-GUI decode research (branch research/vulkan-decode).
//
// Build (MinGW, Vulkan SDK headers only; loads vulkan-1.dll at runtime):
//   export PATH="/c/VulkanSDK/1.4.350.0/Bin:/c/Strawberry/c/bin:$PATH"
//   glslc scripts/research/vk_bw_read.comp -o /tmp/vk_bw_read.spv
//   g++ -O2 -std=c++17 scripts/research/vk_bandwidth.cpp \
//       -I /c/VulkanSDK/1.4.350.0/Include -o /tmp/vk_bandwidth.exe
//
// Usage:
//   vk_bandwidth.exe [device_index] [size_gb] [iters]
//   device_index: 0/1 (default 1 — the non-display GPU)
//   size_gb: buffer size in GiB (default 8)
//   iters: timed passes over the buffer (default 4, first pass is warmup)
//
// The shader streams the whole buffer linearly as float4 (16 B) loads,
// one accumulator per thread, one word written per workgroup to keep the
// loads alive. Reported GB/s = size_gb * 1GiB * iters / elapsed.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <chrono>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#define VK_DLL LoadLibraryA("vulkan-1.dll")
#define VK_GET_PROC GetProcAddress
#else
#include <dlfcn.h>
#define VK_DLL dlopen("libvulkan.so.1", RTLD_NOW)
#define VK_GET_PROC dlsym
#endif

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "vk error %d at %s:%d (%s)\n", (int)_r, __FILE__, __LINE__, #x); \
    return 1; } } while (0)

#define DECL_VK(name) PFN_##name name = nullptr
#define LOAD_VK(name) name = (PFN_##name)VK_GET_PROC(VK_DLL, #name); \
    if (!name) { std::fprintf(stderr, "missing export: %s\n", #name); return 1; }

DECL_VK(vkCreateInstance);
DECL_VK(vkEnumeratePhysicalDevices);
DECL_VK(vkGetPhysicalDeviceProperties);
DECL_VK(vkGetPhysicalDeviceMemoryProperties);
DECL_VK(vkCreateDevice);
DECL_VK(vkGetDeviceQueue);
DECL_VK(vkGetDeviceProcAddr);
DECL_VK(vkCreateBuffer);
DECL_VK(vkGetBufferMemoryRequirements);
DECL_VK(vkAllocateMemory);
DECL_VK(vkBindBufferMemory);
DECL_VK(vkMapMemory);
DECL_VK(vkUnmapMemory);
DECL_VK(vkCreateShaderModule);
DECL_VK(vkCreateDescriptorSetLayout);
DECL_VK(vkCreatePipelineLayout);
DECL_VK(vkCreateComputePipelines);
DECL_VK(vkCreateDescriptorPool);
DECL_VK(vkAllocateDescriptorSets);
DECL_VK(vkUpdateDescriptorSets);
DECL_VK(vkCreateCommandPool);
DECL_VK(vkResetCommandPool);
DECL_VK(vkAllocateCommandBuffers);
DECL_VK(vkBeginCommandBuffer);
DECL_VK(vkCmdBindPipeline);
DECL_VK(vkCmdBindDescriptorSets);
DECL_VK(vkCmdPushConstants);
DECL_VK(vkCmdFillBuffer);
DECL_VK(vkCmdDispatch);
DECL_VK(vkEndCommandBuffer);
DECL_VK(vkQueueSubmit);
DECL_VK(vkQueueWaitIdle);
DECL_VK(vkCreateFence);
DECL_VK(vkWaitForFences);
DECL_VK(vkResetFences);
DECL_VK(vkDestroyFence);
DECL_VK(vkDestroyShaderModule);
DECL_VK(vkDestroyCommandPool);
DECL_VK(vkFreeCommandBuffers);
DECL_VK(vkDestroyDescriptorPool);
DECL_VK(vkDestroyDescriptorSetLayout);
DECL_VK(vkDestroyPipeline);
DECL_VK(vkDestroyPipelineLayout);
DECL_VK(vkFreeMemory);
DECL_VK(vkDestroyBuffer);
DECL_VK(vkDestroyDevice);
DECL_VK(vkDestroyInstance);

static VkInstance instance = VK_NULL_HANDLE;
static VkDevice device = VK_NULL_HANDLE;
static VkQueue queue = VK_NULL_HANDLE;

static std::vector<char> read_file(const char* path) {
    std::vector<char> data;
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    data.resize(n);
    if (n > 0 && std::fread(data.data(), 1, n, f) != (size_t)n) {
        std::fprintf(stderr, "short read %s\n", path);
        std::exit(1);
    }
    std::fclose(f);
    return data;
}

int main(int argc, char** argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    const uint32_t dev_index = argc > 1 ? (uint32_t)std::atoi(argv[1]) : 1;
    const uint64_t size_gb   = argc > 2 ? (uint64_t)std::atoll(argv[2]) : 8;
    const uint32_t iters     = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 4;

    if (!VK_DLL) { std::fprintf(stderr, "vulkan-1.dll not found\n"); return 1; }

    LOAD_VK(vkCreateInstance);
    LOAD_VK(vkEnumeratePhysicalDevices);
    LOAD_VK(vkGetPhysicalDeviceProperties);
    LOAD_VK(vkGetPhysicalDeviceMemoryProperties);
    LOAD_VK(vkCreateDevice);
    LOAD_VK(vkGetDeviceQueue);
    LOAD_VK(vkGetDeviceProcAddr);
    LOAD_VK(vkCreateBuffer);
    LOAD_VK(vkGetBufferMemoryRequirements);
    LOAD_VK(vkAllocateMemory);
    LOAD_VK(vkBindBufferMemory);
    LOAD_VK(vkMapMemory);
    LOAD_VK(vkUnmapMemory);
    LOAD_VK(vkCreateShaderModule);
    LOAD_VK(vkCreateDescriptorSetLayout);
    LOAD_VK(vkCreatePipelineLayout);
    LOAD_VK(vkCreateComputePipelines);
    LOAD_VK(vkCreateDescriptorPool);
    LOAD_VK(vkAllocateDescriptorSets);
    LOAD_VK(vkUpdateDescriptorSets);
    LOAD_VK(vkCreateCommandPool);
    LOAD_VK(vkAllocateCommandBuffers);
    LOAD_VK(vkBeginCommandBuffer);
    LOAD_VK(vkCmdBindPipeline);
    LOAD_VK(vkCmdBindDescriptorSets);
    LOAD_VK(vkCmdPushConstants);
    LOAD_VK(vkCmdDispatch);
    LOAD_VK(vkEndCommandBuffer);
    LOAD_VK(vkQueueSubmit);
    LOAD_VK(vkQueueWaitIdle);
    LOAD_VK(vkCreateFence);
    LOAD_VK(vkWaitForFences);
    LOAD_VK(vkResetFences);
    LOAD_VK(vkDestroyFence);
    LOAD_VK(vkDestroyShaderModule);
    LOAD_VK(vkDestroyCommandPool);
    LOAD_VK(vkFreeCommandBuffers);
    LOAD_VK(vkDestroyDescriptorPool);
    LOAD_VK(vkDestroyDescriptorSetLayout);
    LOAD_VK(vkDestroyPipeline);
    LOAD_VK(vkDestroyPipelineLayout);
    LOAD_VK(vkFreeMemory);
    LOAD_VK(vkDestroyBuffer);
    LOAD_VK(vkDestroyDevice);
    LOAD_VK(vkDestroyInstance);

    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "vk_bandwidth";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));

    uint32_t dev_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &dev_count, nullptr));
    std::vector<VkPhysicalDevice> devs(dev_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &dev_count, devs.data()));
    if (dev_index >= dev_count) {
        std::fprintf(stderr, "device %u not found (%u devices)\n", dev_index, dev_count);
        return 1;
    }
    VkPhysicalDevice pdev = devs[dev_index];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pdev, &props);
    VkPhysicalDeviceMemoryProperties memprops{};
    vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);
    std::printf("device %u: %s\n", dev_index, props.deviceName);
    for (uint32_t i = 0; i < memprops.memoryHeapCount; ++i) {
        std::printf("  heap %u: %.2f GiB%s\n", i,
                    memprops.memoryHeaps[i].size / 1073741824.0,
                    (memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " [device-local]" : "");
    }

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qprio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(pdev, &dci, nullptr, &device));
    vkGetDeviceQueue(device, 0, 0, &queue);

    // Resolve device-level entry points through the device (driver trampolines).
    auto get_dev_proc = [&](const char* name) {
        PFN_vkVoidFunction fn = vkGetDeviceProcAddr(device, name);
        if (!fn) {
            std::fprintf(stderr, "missing device export: %s\n", name);
            std::exit(1);
        }
        return fn;
    };
    vkCreateBuffer = (PFN_vkCreateBuffer)get_dev_proc("vkCreateBuffer");
    vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)get_dev_proc("vkGetBufferMemoryRequirements");
    vkAllocateMemory = (PFN_vkAllocateMemory)get_dev_proc("vkAllocateMemory");
    vkBindBufferMemory = (PFN_vkBindBufferMemory)get_dev_proc("vkBindBufferMemory");
    vkMapMemory = (PFN_vkMapMemory)get_dev_proc("vkMapMemory");
    vkUnmapMemory = (PFN_vkUnmapMemory)get_dev_proc("vkUnmapMemory");
    vkCreateShaderModule = (PFN_vkCreateShaderModule)get_dev_proc("vkCreateShaderModule");
    vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)get_dev_proc("vkCreateDescriptorSetLayout");
    vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout)get_dev_proc("vkCreatePipelineLayout");
    vkCreateComputePipelines = (PFN_vkCreateComputePipelines)get_dev_proc("vkCreateComputePipelines");
    vkCreateDescriptorPool = (PFN_vkCreateDescriptorPool)get_dev_proc("vkCreateDescriptorPool");
    vkAllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)get_dev_proc("vkAllocateDescriptorSets");
    vkUpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)get_dev_proc("vkUpdateDescriptorSets");
    vkCreateCommandPool = (PFN_vkCreateCommandPool)get_dev_proc("vkCreateCommandPool");
    vkResetCommandPool = (PFN_vkResetCommandPool)get_dev_proc("vkResetCommandPool");
    vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)get_dev_proc("vkAllocateCommandBuffers");
    vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)get_dev_proc("vkBeginCommandBuffer");
    vkCmdBindPipeline = (PFN_vkCmdBindPipeline)get_dev_proc("vkCmdBindPipeline");
    vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)get_dev_proc("vkCmdBindDescriptorSets");
    vkCmdPushConstants = (PFN_vkCmdPushConstants)get_dev_proc("vkCmdPushConstants");
    vkCmdFillBuffer = (PFN_vkCmdFillBuffer)get_dev_proc("vkCmdFillBuffer");
    vkCmdDispatch = (PFN_vkCmdDispatch)get_dev_proc("vkCmdDispatch");
    vkEndCommandBuffer = (PFN_vkEndCommandBuffer)get_dev_proc("vkEndCommandBuffer");
    vkQueueSubmit = (PFN_vkQueueSubmit)get_dev_proc("vkQueueSubmit");
    vkQueueWaitIdle = (PFN_vkQueueWaitIdle)get_dev_proc("vkQueueWaitIdle");
    vkCreateFence = (PFN_vkCreateFence)get_dev_proc("vkCreateFence");
    vkWaitForFences = (PFN_vkWaitForFences)get_dev_proc("vkWaitForFences");
    vkResetFences = (PFN_vkResetFences)get_dev_proc("vkResetFences");
    vkDestroyFence = (PFN_vkDestroyFence)get_dev_proc("vkDestroyFence");
    vkDestroyShaderModule = (PFN_vkDestroyShaderModule)get_dev_proc("vkDestroyShaderModule");
    vkDestroyCommandPool = (PFN_vkDestroyCommandPool)get_dev_proc("vkDestroyCommandPool");
    vkFreeCommandBuffers = (PFN_vkFreeCommandBuffers)get_dev_proc("vkFreeCommandBuffers");
    vkDestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)get_dev_proc("vkDestroyDescriptorPool");
    vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)get_dev_proc("vkDestroyDescriptorSetLayout");
    vkDestroyPipeline = (PFN_vkDestroyPipeline)get_dev_proc("vkDestroyPipeline");
    vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)get_dev_proc("vkDestroyPipelineLayout");
    vkFreeMemory = (PFN_vkFreeMemory)get_dev_proc("vkFreeMemory");
    vkDestroyBuffer = (PFN_vkDestroyBuffer)get_dev_proc("vkDestroyBuffer");
    vkDestroyDevice = (PFN_vkDestroyDevice)get_dev_proc("vkDestroyDevice");

    const uint64_t bytes = size_gb * 1073741824ULL;
    const uint64_t n_vec4 = bytes / 16;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buf = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buf));
    VkMemoryRequirements mreq{};
    vkGetBufferMemoryRequirements(device, buf, &mreq);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mreq.size;
    for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i) {
        if ((mreq.memoryTypeBits & (1u << i)) &&
            (memprops.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &mem));
    VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
    std::printf("main buffer: heap=%u type=%u flags=%u (req=%llu GiB)\n",
                memprops.memoryTypes[mai.memoryTypeIndex].heapIndex, mai.memoryTypeIndex,
                memprops.memoryTypes[mai.memoryTypeIndex].propertyFlags,
                (unsigned long long)(mreq.size / 1073741824ULL));

    // Output buffer: one word per workgroup (keeps loads alive).
    const uint32_t out_words = 16384;
    VkBufferCreateInfo obci{};
    obci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    obci.size = out_words * sizeof(float);
    obci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer obuf = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(device, &obci, nullptr, &obuf));
    VkMemoryRequirements omreq{};
    vkGetBufferMemoryRequirements(device, obuf, &omreq);
    VkMemoryAllocateInfo omai{};
    omai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    omai.allocationSize = omreq.size;
    for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i) {
        if ((omreq.memoryTypeBits & (1u << i)) &&
            (memprops.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            omai.memoryTypeIndex = i;
            break;
        }
    }
    VkDeviceMemory omem = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(device, &omai, nullptr, &omem));
    VK_CHECK(vkBindBufferMemory(device, obuf, omem, 0));

    std::vector<char> spv = read_file("vk_bw_read.spv"); // expects CWD
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule smod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &smod));

    VkDescriptorSetLayoutBinding binds[2]{};
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 2;
    dlci.pBindings = binds;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dlci, nullptr, &dsl));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 16;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pl));

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = smod;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkDescriptorPoolSize dps{};
    dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    dps.descriptorCount = 2;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &dset));

    VkDescriptorBufferInfo binfos[2]{};
    binfos[0].buffer = buf;
    binfos[0].offset = 0;
    binfos[0].range = VK_WHOLE_SIZE;
    binfos[1].buffer = obuf;
    binfos[1].offset = 0;
    binfos[1].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds[2]{};
    wds[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[0].dstSet = dset;
    wds[0].dstBinding = 0;
    wds[0].descriptorCount = 1;
    wds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds[0].pBufferInfo = &binfos[0];
    wds[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds[1].dstSet = dset;
    wds[1].dstBinding = 1;
    wds[1].descriptorCount = 1;
    wds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds[1].pBufferInfo = &binfos[1];
    vkUpdateDescriptorSets(device, 2, wds, 0, nullptr);

    VkCommandPoolCreateInfo cpoolci{};
    cpoolci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpoolci.queueFamilyIndex = 0;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(device, &cpoolci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cb));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

    const uint32_t wgs = 1024; // 1024 workgroups x 256 threads = 262144 lanes
    const uint64_t total_vec4 = n_vec4;

    // Record the command buffer once; resubmit the same recorded buffer per
    // pass (valid Vulkan: a completed primary cb may be resubmitted). This
    // sidesteps a driver crash on re-recording observed with the AMD driver.
    //
    // Optional fill first: commits physical pages so reads really hit DRAM
    // (fresh allocations may read through a driver zero-page fast path).
    const bool do_fill = argc > 4 && std::atoi(argv[4]) == 1;
    if (do_fill) {
        VkCommandBufferBeginInfo fbi{};
        fbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cb, &fbi));
        vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0x3f800000u); // 1.0f pattern
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo fsi{};
        fsi.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        fsi.commandBufferCount = 1;
        fsi.pCommandBuffers = &cb;
        VkFence ff = VK_NULL_HANDLE;
        VkFenceCreateInfo ffci{};
        ffci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(device, &ffci, nullptr, &ff));
        VK_CHECK(vkQueueSubmit(queue, 1, &fsi, ff));
        VK_CHECK(vkWaitForFences(device, 1, &ff, VK_TRUE, UINT64_MAX));
        vkDestroyFence(device, ff, nullptr);
        std::printf("buffer filled (pages committed)\n");
    }

    VkCommandBufferBeginInfo bbi{};
    bbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &bbi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
    struct PC { uint32_t total_vec4; uint32_t iters; uint32_t pad0; uint32_t pad1; } pc{
        (uint32_t)total_vec4, 1, 0, 0};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cb, wgs, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    // Fresh fence per pass: avoids both re-recording crashes and
    // resubmitting a signaled fence.
    std::printf("buffer: %llu GiB (%llu vec4), warmup + %u timed passes\n",
                (unsigned long long)size_gb, (unsigned long long)n_vec4, iters);

    // One recorded command buffer, resubmitted per pass; fresh fence per pass.
    auto run = [&]() -> int {
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence f = VK_NULL_HANDLE;
        VkResult rf = vkCreateFence(device, &fci, nullptr, &f);
        if ((int)rf != 0) { std::fprintf(stderr, "createFence rc=%d\n", (int)rf); return 1; }
        VkResult rs = vkQueueSubmit(queue, 1, &si, f);
        if ((int)rs != 0) { std::fprintf(stderr, "submit rc=%d\n", (int)rs); return 1; }
        VkResult rw = vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX);
        if ((int)rw != 0) { std::fprintf(stderr, "wait rc=%d\n", (int)rw); return 1; }
        vkDestroyFence(device, f, nullptr);
        return 0;
    };

    if (run() != 0) { std::fprintf(stderr, "warmup failed\n"); return 1; } // warmup

    double best = 0.0;
    for (uint32_t it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        if (run() != 0) { std::fprintf(stderr, "pass %u failed\n", it); return 1; }
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gbs = (double)bytes / sec / 1e9;
        if (gbs > best) best = gbs;
        std::printf("  pass %u: %.3f s -> %7.2f GB/s (%5.2f%% of 644.6 GB/s spec)\n",
                    it, sec, gbs, gbs / 644.6 * 100.0);
    }
    std::printf("BEST: %.2f GB/s = %.2f%% of 644.6 GB/s spec peak\n", best, best / 644.6 * 100.0);

    vkFreeMemory(device, omem, nullptr);
    vkDestroyBuffer(device, obuf, nullptr);
    vkDestroyCommandPool(device, cpool, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyPipeline(device, pipe, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyShaderModule(device, smod, nullptr);
    vkFreeMemory(device, mem, nullptr);
    vkDestroyBuffer(device, buf, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
