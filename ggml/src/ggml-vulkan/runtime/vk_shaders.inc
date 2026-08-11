// Runtime shader registry and device-specific pipeline specialization.

// D096-C: kernel-side dump buffer for the fp8 native FA kernel. It must be
// created OUTSIDE graph recording (in ggml_vk_load_shaders) with its own
// DeviceMemory (not the VMA pool): kernel dump writes crash the device when
// the buffer is allocated mid-graph (verified v23-v34; the probe, which
// allocates before command recording, never crashes).
static vk_buffer & ggml_vk_fa8_dbg_buf_ref() {
    static vk_buffer buf;
    return buf;
}

static void ggml_vk_fa8_dbg_buf_init(vk_device & device) {
    vk_buffer & buf = ggml_vk_fa8_dbg_buf_ref();
    if (buf) {
        return;
    }
    vk_buffer_struct * bs = new vk_buffer_struct();
    bs->size = 512 * 1024;
    bs->device = device;
    bs->buffer = device->device.createBuffer(
        { vk::BufferCreateFlags(), bs->size,
          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
          vk::SharingMode::eExclusive, 0, nullptr });
    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(bs->buffer);
    uint32_t mt = UINT32_MAX;
    vk::PhysicalDeviceMemoryProperties mprops = device->physical_device.getMemoryProperties();
    for (uint32_t t = 0; t < mprops.memoryTypeCount; t++) {
        if ((mem_req.memoryTypeBits & (1u << t)) &&
            (mprops.memoryTypes[t].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            mt = t;
            break;
        }
    }
    if (mt == UINT32_MAX) {
        GGML_ABORT("ggml_vulkan: fa_native dump: no device-local memory type");
    }
    bs->device_memory = device->device.allocateMemory({ mem_req.size, mt });
    device->device.bindBufferMemory(bs->buffer, bs->device_memory, 0);
    bs->memory_property_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    buf = vk_buffer(bs);
}

static void ggml_vk_load_shaders(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    std::lock_guard<std::recursive_mutex> guard(device->mutex);
    // some shaders have a minimum subgroup size
    const uint32_t subgroup_size_8 = std::max(device->subgroup_size, 8u);
    const uint32_t subgroup_size_16 = std::max(device->subgroup_size, 16u);
    const uint32_t subgroup_size_32 = std::max(device->subgroup_size, 32u);

    const uint32_t mul_mat_subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
    const uint32_t mul_mat_subgroup_size_8 = std::max(mul_mat_subgroup_size, 8u);
    const uint32_t mul_mat_subgroup_size_16 = std::max(mul_mat_subgroup_size, 16u);
    const uint32_t mul_mat_subgroup_size_32 = std::max(mul_mat_subgroup_size, 32u);

    const bool subgroup_min_size_16 = (!device->subgroup_size_control && device->subgroup_size >= 16) ||
                                      (device->subgroup_size_control && device->subgroup_max_size >= 16);

    // mulmat
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;
    if (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        l_warptile_mmqid = { 256, 128, 128, 32, 1, device->subgroup_size };
        m_warptile_mmqid = { 256, 128, 64, 32, 0, device->subgroup_size };
        s_warptile_mmqid = { 256, 128, 64, 32, 0, device->subgroup_size };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } else {
        // Matrix cores require different warp group sizes
        const uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        const uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        const uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
        m_warptile = { 128,              64,  64, 16, subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        s_warptile = { subgroup_size_32, 32,  32, 16, s_warptile_wm,       32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        l_warptile_mmq = { 128,             128, 128, 32, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
        m_warptile_mmq = { 128,              64,  64, 32, subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        s_warptile_mmq = { subgroup_size_32, 32,  32, 32, s_warptile_wm,       32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, subgroup_size_8,     32, 2, 2, 2, 1, subgroup_size_8 };
        s_warptile_mmq_int = { subgroup_size_32, 32,  32, 32, s_warptile_wm,       32, 2, 2, 1, 1, subgroup_size_8 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, subgroup_size_8 * 2, 64, 1, 4, 4, 1, subgroup_size_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, subgroup_size_8,     32, 1, 2, 2, 1, subgroup_size_8 };
        s_warptile_mmq_int_k = { subgroup_size_32,   32,  32, 32, s_warptile_wm,       32, 1, 2, 1, 1, subgroup_size_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_subgroup_size_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_subgroup_size_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_16 };
        s_warptile_id = { mul_mat_subgroup_size_16,  32,  32, 16, s_warptile_wm,                32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_8 };
        s_warptile_mmqid = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,               32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, 4, 4, 1, mul_mat_subgroup_size_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_subgroup_size_8,     32, 2, 2, 2, 1, mul_mat_subgroup_size_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,               32, 2, 2, 1, 1, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_subgroup_size_16 * 2, 64, 1, 4, 4, 1, mul_mat_subgroup_size_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_subgroup_size_16,     32, 1, 2, 2, 1, mul_mat_subgroup_size_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_size_32, 32,  32, 32, s_warptile_wm,                32, 1, 2, 1, 1, mul_mat_subgroup_size_16 };

        const bool force_amd_large_matmul_env = getenv("GGML_VK_FORCE_AMD_LARGE_MATMUL") != nullptr;
        const bool disable_amd_large_matmul = getenv("GGML_VK_DISABLE_AMD_LARGE_MATMUL") != nullptr;
        const bool auto_amd_large_matmul = device->vendor_id == VK_VENDOR_ID_AMD &&
                           device->driver_id == vk::DriverId::eAmdProprietary &&
                           device->architecture == AMD_RDNA3 &&
                           device->coopmat_support &&
                           !device->uma;
        const bool force_amd_large_matmul = force_amd_large_matmul_env || (auto_amd_large_matmul && !disable_amd_large_matmul);
        const char * amd_large_matmul_variant = getenv("GGML_VK_AMD_LARGE_MATMUL_VARIANT");
        const bool disable_amd_tuned_default =
            getenv("GGML_VK_DISABLE_AMD_WN32_DEFAULT") != nullptr ||
            // Retain the old diagnostic switch as an alias so existing A/B
            // commands still disable whichever tuned default is current.
            getenv("GGML_VK_DISABLE_AMD_BN256_DEFAULT") != nullptr;
        if (amd_large_matmul_variant == nullptr && auto_amd_large_matmul && !disable_amd_large_matmul &&
            !disable_amd_tuned_default) {
            // RX 9070 XT + recent AMD proprietary drivers regress heavily with
            // the previous bn256 shape. wn32 restores the large Q4_K prompt
            // route while preserving decode performance.
            amd_large_matmul_variant = "wn32";
        }
        bool amd_large_matmul_variant_active = false;

        // chip specific tuning
        if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && (device->driver_id != vk::DriverId::eAmdProprietary || force_amd_large_matmul)) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, subgroup_size_16, 64, 1, 4, 2, 1, subgroup_size_16 };
            const std::vector<uint32_t> l_warptile_mmq_base = l_warptile_mmq;
            const std::vector<uint32_t> l_warptile_mmq_int_base = l_warptile_mmq_int;

            if (amd_large_matmul_variant != nullptr) {
                amd_large_matmul_variant_active = true;
                const std::string variant(amd_large_matmul_variant);
                if (variant == "wmiter1") {
                    l_warptile_mmq[6] = 1;
                    l_warptile_mmq_int[6] = 1;
                } else if (variant == "block128") {
                    l_warptile_mmq[0] = 128;
                    l_warptile_mmq_int[0] = 128;
                } else if (variant == "block512") {
                    l_warptile_mmq[0] = 512;
                    l_warptile_mmq_int[0] = 512;
                } else if (variant == "block64") {
                    l_warptile_mmq[0] = 64;
                    l_warptile_mmq_int[0] = 64;
                } else if (variant == "wm128") {
                    l_warptile_mmq[4] = 128;
                    l_warptile_mmq_int[4] = 128;
                } else if (variant == "wn32") {
                    l_warptile_mmq[5] = 32;
                    l_warptile_mmq_int[5] = 32;
                } else if (variant == "wn16") {
                    l_warptile_mmq[5] = 16;
                    l_warptile_mmq_int[5] = 16;
                } else if (variant == "wn48") {
                    l_warptile_mmq[5] = 48;
                    l_warptile_mmq_int[5] = 48;
                } else if (variant == "wn96") {
                    l_warptile_mmq[5] = 96;
                    l_warptile_mmq_int[5] = 96;
                } else if (variant == "block128-wn32") {
                    l_warptile_mmq[0] = 128;
                    l_warptile_mmq[5] = 32;
                    l_warptile_mmq_int[0] = 128;
                    l_warptile_mmq_int[5] = 32;
                } else if (variant == "wn32-wmiter1") {
                    l_warptile_mmq[5] = 32;
                    l_warptile_mmq[6] = 1;
                    l_warptile_mmq_int[5] = 32;
                    l_warptile_mmq_int[6] = 1;
                } else if (variant == "block128-wn32-wmiter1") {
                    l_warptile_mmq[0] = 128;
                    l_warptile_mmq[5] = 32;
                    l_warptile_mmq[6] = 1;
                    l_warptile_mmq_int[0] = 128;
                    l_warptile_mmq_int[5] = 32;
                    l_warptile_mmq_int[6] = 1;
                } else if (variant == "wm32-wn32") {
                    l_warptile_mmq[1] = 64;
                    l_warptile_mmq[2] = 64;
                    l_warptile_mmq[4] = 32;
                    l_warptile_mmq[5] = 32;
                    l_warptile_mmq_int[1] = 64;
                    l_warptile_mmq_int[2] = 64;
                    l_warptile_mmq_int[4] = 32;
                    l_warptile_mmq_int[5] = 32;
                } else if (variant == "wm128-wn32") {
                    l_warptile_mmq[4] = 128;
                    l_warptile_mmq[5] = 32;
                    l_warptile_mmq_int[4] = 128;
                    l_warptile_mmq_int[5] = 32;
                } else if (variant == "bm64") {
                    l_warptile_mmq[1] = 64;
                    l_warptile_mmq_int[1] = 64;
                } else if (variant == "bn64") {
                    l_warptile_mmq[2] = 64;
                    l_warptile_mmq_int[2] = 64;
                } else if (variant == "bn256") {
                    l_warptile_mmq[2] = 256;
                    l_warptile_mmq_int[2] = 256;
                } else if (variant == "bm64-bn64") {
                    l_warptile_mmq[1] = 64;
                    l_warptile_mmq[2] = 64;
                    l_warptile_mmq_int[1] = 64;
                    l_warptile_mmq_int[2] = 64;
                } else if (variant == "block128-bn64") {
                    l_warptile_mmq[0] = 128;
                    l_warptile_mmq[2] = 64;
                    l_warptile_mmq_int[0] = 128;
                    l_warptile_mmq_int[2] = 64;
                } else if (variant == "block128-wm128") {
                    l_warptile_mmq[0] = 128;
                    l_warptile_mmq[4] = 128;
                    l_warptile_mmq_int[0] = 128;
                    l_warptile_mmq_int[4] = 128;
                }

                if (!ggml_vk_matmul_prepare_variant_warptile(device, l_warptile_mmq) ||
                    !ggml_vk_matmul_prepare_variant_warptile(device, l_warptile_mmq_int)) {
                    GGML_LOG_WARN(
                        "ggml_vulkan: AMD large matmul variant '%s' has invalid MMQ tile layout "
                        "(tile=[%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u]), using base large matmul profile\n",
                        variant.c_str(),
                        l_warptile_mmq[0], l_warptile_mmq[1], l_warptile_mmq[2], l_warptile_mmq[3],
                        l_warptile_mmq[4], l_warptile_mmq[5], l_warptile_mmq[6], l_warptile_mmq[7],
                        l_warptile_mmq[8], l_warptile_mmq[9], l_warptile_mmq[10]);
                    l_warptile_mmq = l_warptile_mmq_base;
                    l_warptile_mmq_int = l_warptile_mmq_int_base;
                    amd_large_matmul_variant_active = false;
                }
            }

        } else if (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support && device->architecture == INTEL_XE2) {
            // Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
            l_warptile_mmq = { 512, 128, 128, 32, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        if (amd_large_matmul_variant_active) {
            const std::string variant(amd_large_matmul_variant);
            if (variant == "bm64") {
                l_mmq_wg_denoms = { 64, 128, 1 };
            } else if (variant == "bn64" || variant == "block128-bn64") {
                l_mmq_wg_denoms = { 128, 64, 1 };
            } else if (variant == "bn256") {
                l_mmq_wg_denoms = { 128, 256, 1 };
            } else if (variant == "bm64-bn64" || variant == "wm32-wn32") {
                l_mmq_wg_denoms = { 64, 64, 1 };
            }
        }
        l_align = 128;
        m_align =  64;
        s_align =  32;

        for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, false, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::endl;
                throw std::runtime_error("Shared memory size too small for matrix multiplication.");
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, false, t)) {
                device->mul_mat_m[i] = false;
                device->mul_mat_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, false, t)) {
                device->mul_mat_l[i] = false;
            }

            // Disable mul_mat_id if not enough shared memory is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, true, t)) {
                device->mul_mat_id_s[i] = false;
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, true, t)) {
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, true, t)) {
                device->mul_mat_id_l[i] = false;
            }
        }
    }

    if (!device->pipeline_matmul_f32) {
        device->pipeline_matmul_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_f32_f16) {
        device->pipeline_matmul_f32_f16 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_id_f32) {
        device->pipeline_matmul_id_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_bf16) {
        device->pipeline_matmul_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_id_bf16) {
        device->pipeline_matmul_id_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
    }

    std::vector<std::future<void>> compiles;
    auto const &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& base_pipeline, const char *name, size_t spv_size, const void* spv_data, const char *entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {

        if (!require_full_subgroups && required_subgroup_size == 0) {
            required_subgroup_size = get_subgroup_size(name, device->architecture);
        }

        vk_pipeline *ptr = &base_pipeline;

        int num_pipelines = 1;
#if defined(VK_EXT_shader_64bit_indexing)
        if (device->shader_64b_indexing) {
            num_pipelines = 2;
        }
#endif
        for (int i = 0; i < num_pipelines; ++i, ptr = &(*ptr)->next) {
            vk_pipeline &pipeline = *ptr;
            if (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            if (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_count = parameter_count;
                pipeline->push_constant_size = push_constant_size;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = true;
#if defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#endif
            }

            if (!pipeline->needed || pipeline->compiled) {
                continue;
            }

            // TODO: We're no longer benefitting from the async compiles (shaders are
            // compiled individually, as needed) and this complexity can be removed.
            {
                // wait until fewer than N compiles are in progress
                uint32_t N = std::max(1u, std::thread::hardware_concurrency());
                std::unique_lock<std::mutex> guard(compile_count_mutex);
                while (compile_count >= N) {
                    compile_count_cond.wait(guard);
                }
                compile_count++;
            }

            compiles.push_back(std::async(ggml_vk_create_pipeline_func, std::ref(device), std::ref(pipeline), spv_size, spv_data, entrypoint,
                                          parameter_count, wg_denoms, specialization_constants, disable_robustness, require_full_subgroups, required_subgroup_size));
        }
    };

    auto const &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, const std::string &name, size_t spv_size, const void* spv_data, const char *entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {
        return ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_size, spv_data, entrypoint,
                                       parameter_count, push_constant_size, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_full_subgroups, required_subgroup_size);
    };

#define CREATE_FA(TYPE, NAMELC, FAPATH, SUFFIX) \
        for (auto &fa : device->pipeline_flash_attn_f32_f16[TYPE]) { \
            FaCodePath path = fa.first.path; \
            uint32_t Br = fa.first.Br; \
            uint32_t Bc = fa.first.Bc; \
            bool aligned = fa.first.aligned; \
            bool f32acc = fa.first.f32acc; \
            uint32_t fa_sgs = fa.first.subgroup_size; \
            bool fa_ds = fa.first.subgroup_size == 0; \
            if (path == FAPATH) { \
                if (aligned) { \
                    if (f32acc) { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_aligned_f32acc" #NAMELC, flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _data,  "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), Bc, true, (!fa_ds && (FAPATH!=FA_COOPMAT2)), ((!fa_ds && (FAPATH!=FA_COOPMAT2)) ? fa_sgs : 0));     \
                    } else { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_aligned_f16acc" #NAMELC, flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), Bc, true, (!fa_ds && (FAPATH!=FA_COOPMAT2)), ((!fa_ds && (FAPATH!=FA_COOPMAT2)) ? fa_sgs : 0));     \
                    } \
                } else { \
                    if (f32acc) { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f32acc"         #NAMELC, flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ##            SUFFIX ## _data,  "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1,  true, (!fa_ds && (FAPATH!=FA_COOPMAT2)), ((!fa_ds && (FAPATH!=FA_COOPMAT2)) ? fa_sgs : 0));     \
                    } else { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f16acc"         #NAMELC, flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _len,  flash_attn_f32_f16_ ## NAMELC ## _f16acc ## SUFFIX ## _data,  "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1,  true, (!fa_ds && (FAPATH!=FA_COOPMAT2)), ((!fa_ds && (FAPATH!=FA_COOPMAT2)) ? fa_sgs : 0));     \
                    } \
                } \
            } \
        }

    if (device->fp16) {
        CREATE_FA(GGML_TYPE_F32, f32, FA_SCALAR, )
        CREATE_FA(GGML_TYPE_F16, f16, FA_SCALAR, )

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product && device->subgroup_clustered) {
            CREATE_FA(GGML_TYPE_Q4_0,     q4_0, FA_SCALAR, _int8)
            CREATE_FA(GGML_TYPE_Q8_0,     q8_0, FA_SCALAR, _int8)
            CREATE_FA(GGML_TYPE_Q4_1,     q4_1, FA_SCALAR, _int8)
            CREATE_FA(GGML_TYPE_Q5_0,     q5_0, FA_SCALAR, _int8)
            CREATE_FA(GGML_TYPE_Q5_1,     q5_1, FA_SCALAR, _int8)
            CREATE_FA(GGML_TYPE_IQ4_NL, iq4_nl, FA_SCALAR, _int8)
            CREATE_FA(GGML_TYPE_F8_E4M3, f8_e4m3, FA_SCALAR, )
        } else
#endif
        {
            CREATE_FA(GGML_TYPE_Q4_0,     q4_0, FA_SCALAR, )
            CREATE_FA(GGML_TYPE_Q8_0,     q8_0, FA_SCALAR, )
            CREATE_FA(GGML_TYPE_Q4_1,     q4_1, FA_SCALAR, )
            CREATE_FA(GGML_TYPE_Q5_0,     q5_0, FA_SCALAR, )
            CREATE_FA(GGML_TYPE_Q5_1,     q5_1, FA_SCALAR, )
            CREATE_FA(GGML_TYPE_IQ4_NL, iq4_nl, FA_SCALAR, )
            CREATE_FA(GGML_TYPE_F8_E4M3, f8_e4m3, FA_SCALAR, )
        }
    } else {
        CREATE_FA(GGML_TYPE_F32, f32, FA_SCALAR, _fp32)
        CREATE_FA(GGML_TYPE_F16, f16, FA_SCALAR, _fp32)

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product && device->subgroup_clustered) {
            CREATE_FA(GGML_TYPE_Q4_0,     q4_0, FA_SCALAR, _fp32_int8)
            CREATE_FA(GGML_TYPE_Q8_0,     q8_0, FA_SCALAR, _fp32_int8)
            CREATE_FA(GGML_TYPE_Q4_1,     q4_1, FA_SCALAR, _fp32_int8)
            CREATE_FA(GGML_TYPE_Q5_0,     q5_0, FA_SCALAR, _fp32_int8)
            CREATE_FA(GGML_TYPE_Q5_1,     q5_1, FA_SCALAR, _fp32_int8)
            CREATE_FA(GGML_TYPE_IQ4_NL, iq4_nl, FA_SCALAR, _fp32_int8)
            CREATE_FA(GGML_TYPE_F8_E4M3, f8_e4m3, FA_SCALAR, _fp32)
        } else
#endif
        {
            CREATE_FA(GGML_TYPE_Q4_0,     q4_0, FA_SCALAR, _fp32)
            CREATE_FA(GGML_TYPE_Q8_0,     q8_0, FA_SCALAR, _fp32)
            CREATE_FA(GGML_TYPE_Q4_1,     q4_1, FA_SCALAR, _fp32)
            CREATE_FA(GGML_TYPE_Q5_0,     q5_0, FA_SCALAR, _fp32)
            CREATE_FA(GGML_TYPE_Q5_1,     q5_1, FA_SCALAR, _fp32)
            CREATE_FA(GGML_TYPE_IQ4_NL, iq4_nl, FA_SCALAR, _fp32)
            CREATE_FA(GGML_TYPE_F8_E4M3, f8_e4m3, FA_SCALAR, _fp32)
        }
    }
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat1_fa_support) {
        CREATE_FA(GGML_TYPE_F32, f32, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_F16, f16, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q4_0, q4_0, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q8_0, q8_0, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q4_1, q4_1, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q5_0, q5_0, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_Q5_1, q5_1, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_IQ4_NL, iq4_nl, FA_COOPMAT1, _cm1)
        CREATE_FA(GGML_TYPE_F8_E4M3, f8_e4m3, FA_COOPMAT1, _cm1)
    }

    // D096 P2: build-generated module whose S stage alone uses E4M3 WMMA.
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16[GGML_TYPE_F8_E4M3]) {
            if (fa.first.path == FA_F8_P2 && fa.first.f32acc) {
                const uint32_t Br = fa.first.Br;
                const uint32_t fa_sgs = fa.first.subgroup_size;
                const bool fa_ds = fa.first.subgroup_size == 0;
                ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f8_p2", flash_attn_f32_f16_f8_p2_len, flash_attn_f32_f16_f8_p2_data, "main", 9, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, !fa_ds, fa_ds ? 0 : fa_sgs);
            }
        }
    }

    // D096 P5: same fused-Q transform as P4, but V is read raw f8 from the
    // source cache (fp8 P*V with f32 accumulator, no f16 preconvert pass).
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16[GGML_TYPE_F8_E4M3]) {
            if (fa.first.path == FA_F8_P5 && fa.first.f32acc) {
                const uint32_t Br = fa.first.Br;
                const uint32_t fa_sgs = fa.first.subgroup_size;
                const bool fa_ds = fa.first.subgroup_size == 0;
                ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f8_p5", flash_attn_f32_f16_f8_p5_len, flash_attn_f32_f16_f8_p5_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, !fa_ds, fa_ds ? 0 : fa_sgs);
            }
        }
    }

    // D096 P4: fused-Q transform with V read from a dense f16 preconvert buffer.
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16[GGML_TYPE_F8_E4M3]) {
            if (fa.first.path == FA_F8_P4 && fa.first.f32acc) {
                const uint32_t Br = fa.first.Br;
                const uint32_t fa_sgs = fa.first.subgroup_size;
                const bool fa_ds = fa.first.subgroup_size == 0;
                ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f8_p4", flash_attn_f32_f16_f8_p4_len, flash_attn_f32_f16_f8_p4_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, !fa_ds, fa_ds ? 0 : fa_sgs);
            }
        }
    }

    // D096 P3: P2's generated S-only transform with Q8 quantization fused into
    // the canonical FA workgroup. It has only the seven canonical descriptors.
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16[GGML_TYPE_F8_E4M3]) {
            if (fa.first.path == FA_F8_P3 && fa.first.f32acc) {
                const uint32_t Br = fa.first.Br;
                const uint32_t fa_sgs = fa.first.subgroup_size;
                const bool fa_ds = fa.first.subgroup_size == 0;
                ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f8_p3", flash_attn_f32_f16_f8_p3_len, flash_attn_f32_f16_f8_p3_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, !fa_ds, fa_ds ? 0 : fa_sgs);
            }
        }
    }

    // D096-A2: fp8 native FA kernel - hand-written SPIR-V assembly.
    // f32acc only (f32 accumulation hardcoded); single variant (no aligned
    // split: the coopmat stride parameter handles arbitrary strides), so a
    // dedicated registration is used instead of CREATE_FA.
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16[GGML_TYPE_F8_E4M3]) {
            if (fa.first.path == FA_F8_NATIVE && fa.first.f32acc) {
                uint32_t Br = fa.first.Br;
                uint32_t fa_sgs = fa.first.subgroup_size;
                bool fa_ds = fa.first.subgroup_size == 0;
                ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_f8_native", flash_attn_f32_f16_f8_native_len, flash_attn_f32_f16_f8_native_data, "main", 8, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, !fa_ds, fa_ds ? 0 : fa_sgs);
            }
        }
    }
    // D096-C: the fp8 native FA dump buffer must be allocated OUTSIDE graph
    // recording (kernel dump writes crash the device if the buffer is created
    // mid-graph; verified v23-v34).
    if (getenv("GGML_VK_FA_F8_DUMP")) {
        ggml_vk_fa8_dbg_buf_init(device);
    }
#endif
#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#define CREATE_FA_CM2_MIXED() \
        for (int fa_k_ty = 0; fa_k_ty < (int)GGML_TYPE_COUNT; ++fa_k_ty) { \
        for (auto &fa : device->pipeline_flash_attn_f32_f16[fa_k_ty]) { \
            FaCodePath path = fa.first.path; \
            uint32_t Br = fa.first.Br; \
            uint32_t Bc = fa.first.Bc; \
            bool aligned = fa.first.aligned; \
            bool f32acc = fa.first.f32acc; \
            if (path == FA_COOPMAT2) { \
                if (aligned) { \
                    if (f32acc) { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_mixed_aligned_f32acc_cm2", flash_attn_f32_f16_mixed_cm2_len, flash_attn_f32_f16_mixed_cm2_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), Bc, true, false, 0); \
                    } else { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_mixed_aligned_f16acc_cm2", flash_attn_f32_f16_mixed_f16acc_cm2_len, flash_attn_f32_f16_mixed_f16acc_cm2_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), Bc, true, false, 0); \
                    } \
                } else { \
                    if (f32acc) { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_mixed_f32acc_cm2", flash_attn_f32_f16_mixed_cm2_len, flash_attn_f32_f16_mixed_cm2_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, false, 0); \
                    } else { \
                        ggml_vk_create_pipeline(device, fa.second, "flash_attn_f32_f16_mixed_f16acc_cm2", flash_attn_f32_f16_mixed_f16acc_cm2_len, flash_attn_f32_f16_mixed_f16acc_cm2_data, "main", 7, sizeof(vk_flash_attn_push_constants), {Br, 1, 1}, get_fa_spec_constants(fa.first), 1, true, false, 0); \
                    } \
                } \
            } \
        } \
        }
    if (device->coopmat2) {
        CREATE_FA_CM2_MIXED();
    }
#undef CREATE_FA_CM2_MIXED
#endif
#undef CREATE_FA

    const int mul_mat_id_param_count = 5;

#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {

        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, true);   \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        CREATE_MM(PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \
        CREATE_MM(PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \

        CREATE_MM2(pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3)
        }
#endif
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q1_0], matmul_q1_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_0], matmul_q4_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_1], matmul_q4_1_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_0], matmul_q5_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_1], matmul_q5_1_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q8_0], matmul_q8_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q2_K], matmul_q2_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q3_K], matmul_q3_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_K], matmul_q4_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_K], matmul_q5_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q6_K], matmul_q6_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ1_S],   matmul_iq1_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ1_M],   matmul_iq1_m_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_S],   matmul_iq2_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ3_S],   matmul_iq3_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_MXFP4],   matmul_mxfp4_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_NVFP4],   matmul_nvfp4_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)

        GGML_ASSERT(device->subgroup_ballot);

        CREATE_MM2(pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, 5)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, 5)
        }
#endif
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_subgroup_q1_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_subgroup_iq1_s_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_subgroup_iq1_m_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_subgroup_iq2_xxs_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_subgroup_iq2_xs_f16,  mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_subgroup_iq2_s_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_subgroup_iq3_xxs_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_subgroup_iq3_s_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_subgroup_iq4_xs_f16,  mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_subgroup_iq4_nl_f16,  mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_subgroup_mxfp4_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_subgroup_nvfp4_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
#undef CREATE_MM
#undef CREATE_MM2
    } else
#endif  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat_support) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, true);   \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->coopmat_acc_f16_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \
        if (device->coopmat_acc_f32_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->l, #NAMELC        "_l", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \
        if (device->mul_mat ## ID ## _m[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->m, #NAMELC        "_m", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \
        if (device->mul_mat ## ID ## _s[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->s, #NAMELC        "_s", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16_f32, matmul_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, )
        }
#endif

        if (device->coopmat_acc_f16_support) {
            CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0], matmul_q1_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0], matmul_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1], matmul_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0], matmul_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1], matmul_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0], matmul_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K], matmul_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K], matmul_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q3_k_quad, matmul_q3_k_f32_quad, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K], matmul_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K], matmul_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K], matmul_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S],   matmul_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M],   matmul_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S],   matmul_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S],   matmul_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4],   matmul_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4],   matmul_nvfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

        } else {
            CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0].f32acc, matmul_q1_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q3_k_quad.f32acc, matmul_q3_k_f32_quad, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S].f32acc,   matmul_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M].f32acc,   matmul_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS].f32acc, matmul_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS].f32acc,  matmul_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S].f32acc,   matmul_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS].f32acc, matmul_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S].f32acc,   matmul_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS].f32acc,  matmul_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL].f32acc,  matmul_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4].f32acc,   matmul_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4].f32acc,   matmul_nvfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        }

        GGML_ASSERT(device->subgroup_ballot);

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_subgroup_f32_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_subgroup_f16_f32, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        }
#endif

        CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_subgroup_q1_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_subgroup_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_subgroup_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_subgroup_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_subgroup_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_subgroup_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_subgroup_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_subgroup_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_subgroup_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_subgroup_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_subgroup_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_subgroup_nvfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);

#undef CREATE_MM2
#undef CREATE_MMQ
#undef CREATE_MM
    } else
#endif  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->fp16) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->l, #NAMELC        "_l", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \
        if (device->mul_mat ## ID ## _m[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->m, #NAMELC        "_m", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \
        if (device->mul_mat ## ID ## _s[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->s, #NAMELC        "_s", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16_f32, matmul_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0], matmul_q1_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0], matmul_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1], matmul_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0], matmul_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1], matmul_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0], matmul_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K], matmul_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K], matmul_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q3_k_quad, matmul_q3_k_f32_quad, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K], matmul_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K], matmul_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K], matmul_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S],   matmul_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M],   matmul_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S],   matmul_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S],   matmul_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4],   matmul_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4],   matmul_nvfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_0], matmul_q4_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_1], matmul_q4_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_0], matmul_q5_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_1], matmul_q5_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q8_0], matmul_q8_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);

            CREATE_MMQ(GGML_TYPE_MXFP4, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_MXFP4], matmul_mxfp4_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);

            CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q2_K], matmul_q2_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q3_K], matmul_q3_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_K], matmul_q4_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_K], matmul_q5_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q6_K], matmul_q6_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
        }
#endif

        if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_subgroup_f32_f32, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_subgroup_f16_f32, wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);

            CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_subgroup_q1_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_subgroup_iq1_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_subgroup_iq1_m_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_subgroup_iq2_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_subgroup_iq2_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_subgroup_iq2_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_subgroup_iq3_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_subgroup_iq3_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_subgroup_iq4_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_subgroup_iq4_nl_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_subgroup_mxfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_subgroup_nvfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);

                CREATE_MMQ(GGML_TYPE_MXFP4, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_MXFP4], matmul_id_subgroup_mxfp4_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);

                CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            }
#endif
        } else {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_f16_f32, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

            CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_q1_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_q4_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_q4_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_q5_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_q5_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_q8_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_q2_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_q3_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_q4_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_q5_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_q6_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_iq1_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_iq1_m_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_iq2_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_iq2_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_iq2_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_iq3_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_iq3_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_iq4_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_iq4_nl_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_mxfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_nvfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_0], matmul_id_q4_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_1], matmul_id_q4_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_0], matmul_id_q5_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_1], matmul_id_q5_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q8_0], matmul_id_q8_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

                CREATE_MMQ(GGML_TYPE_MXFP4, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_MXFP4], matmul_id_mxfp4_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

                CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q2_K], matmul_id_q2_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q3_K], matmul_id_q3_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_K], matmul_id_q4_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_K], matmul_id_q5_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q6_K], matmul_id_q6_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            }
#endif
        }
#undef CREATE_MM2
#undef CREATE_MMQ
#undef CREATE_MM
    } else {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC "_l", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC "_m", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC "_s", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_f16.f32acc, matmul_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_f16_f32.f32acc, matmul_f16_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0].f32acc, matmul_q1_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q3_k_quad.f32acc, matmul_q3_k_f32_quad, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S].f32acc,   matmul_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M].f32acc,   matmul_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS].f32acc, matmul_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS].f32acc,  matmul_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S].f32acc,   matmul_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS].f32acc, matmul_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S].f32acc,   matmul_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS].f32acc,  matmul_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL].f32acc,  matmul_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4].f32acc,   matmul_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4].f32acc,   matmul_nvfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );

            CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
        }
#endif

        if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_subgroup_f32_f32, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16.f32acc, matmul_id_subgroup_f16, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16_f32.f32acc, matmul_id_subgroup_f16_f32, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);

            CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0].f32acc, matmul_id_subgroup_q1_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f32acc, matmul_id_subgroup_q4_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f32acc, matmul_id_subgroup_q4_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f32acc, matmul_id_subgroup_q5_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f32acc, matmul_id_subgroup_q5_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f32acc, matmul_id_subgroup_q8_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f32acc, matmul_id_subgroup_q2_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f32acc, matmul_id_subgroup_q3_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f32acc, matmul_id_subgroup_q4_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f32acc, matmul_id_subgroup_q5_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f32acc, matmul_id_subgroup_q6_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f32acc,   matmul_id_subgroup_iq1_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f32acc,   matmul_id_subgroup_iq1_m_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f32acc, matmul_id_subgroup_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f32acc,  matmul_id_subgroup_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f32acc,   matmul_id_subgroup_iq2_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f32acc, matmul_id_subgroup_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f32acc,   matmul_id_subgroup_iq3_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f32acc,  matmul_id_subgroup_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f32acc,  matmul_id_subgroup_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f32acc,   matmul_id_subgroup_mxfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4].f32acc,   matmul_id_subgroup_nvfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
        } else {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16.f32acc, matmul_id_f16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16_f32.f32acc, matmul_id_f16_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

            CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0].f32acc, matmul_id_q1_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f32acc, matmul_id_q4_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f32acc, matmul_id_q4_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f32acc, matmul_id_q5_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f32acc, matmul_id_q5_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f32acc, matmul_id_q8_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f32acc, matmul_id_q2_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f32acc, matmul_id_q3_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f32acc, matmul_id_q4_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f32acc, matmul_id_q5_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f32acc, matmul_id_q6_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f32acc,   matmul_id_iq1_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f32acc,   matmul_id_iq1_m_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f32acc, matmul_id_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f32acc,  matmul_id_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f32acc,   matmul_id_iq2_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f32acc, matmul_id_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f32acc,   matmul_id_iq3_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f32acc,  matmul_id_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f32acc,  matmul_id_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f32acc,   matmul_id_mxfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4].f32acc,   matmul_id_nvfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
        }
    }
    // reusing CREATE_MM from the fp32 path
    if ((device->coopmat2 || device->coopmat_support)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#endif
        ) {
        const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;

        // use scalar tile sizes
        l_warptile = { 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        m_warptile = { 128,  64,  64, 16, subgroup_size_8, 32, 2, 4, 2, 1, subgroup_size_8 };
        s_warptile = { subgroup_size_32, 32, 32, 16, s_warptile_wm, 32, 2, 2, 2, 1, subgroup_size_8 };

        l_wg_denoms = {128, 128, 1 };
        m_wg_denoms = { 64,  64, 1 };
        s_wg_denoms = { 32,  32, 1 };

        if (device->vendor_id == VK_VENDOR_ID_INTEL && device->architecture == INTEL_XE2) {
            // Xe2/Xe3 - bf16 warptile performance tuning
            l_warptile = { 512, 128, 128, 16, subgroup_size_8, 32, 2, 4, 4, 1, subgroup_size_8 };
        }

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
    }
#undef CREATE_MM

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    auto const &rm_iq_int = [](uint32_t i) { return i == 0 ? 8u : 4u; };
    if (device->vendor_id == VK_VENDOR_ID_AMD) {
        if (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } else if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    uint32_t rm_iq = 2 * rm_kq;

    const bool use_subgroups = device->subgroup_arithmetic && device->architecture != vk_device_architecture::AMD_GCN;
    // Ensure a subgroup size >= 16 is available
    const bool use_subgroups16 = use_subgroups && subgroup_min_size_16;

    const uint32_t subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control && device->subgroup_min_size <= 16 && device->subgroup_max_size >= 16) ? 16 : device->subgroup_size;
    const uint32_t subgroup_size16 = std::max(subgroup_size, 16u);

    const uint32_t force_subgroup_size = use_subgroups ? subgroup_size : 0;
    const uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    static constexpr uint32_t mul_mat_vec_num_bindings = 5;
    static constexpr uint32_t mul_mat_vec_id_num_bindings = 6;

    for (uint32_t w = 0; w < DMMV_WG_SIZE_COUNT; ++w) {
        const uint32_t wg_size_subgroup   = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size : (subgroup_size * 4);
        const uint32_t wg_size_subgroup16 = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        const shader_reduction_mode reduc = (use_subgroups && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                            SHADER_REDUCTION_MODE_SHMEM;

        const shader_reduction_mode reduc16 = (use_subgroups16 && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                              SHADER_REDUCTION_MODE_SHMEM;

        for (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   arr_dmmv_mxfp4_f32_f32_len[reduc16],   arr_dmmv_mxfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   arr_dmmv_nvfp4_f32_f32_len[reduc16],   arr_dmmv_nvfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   arr_dmmv_mxfp4_f16_f32_len[reduc16],   arr_dmmv_mxfp4_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   arr_dmmv_nvfp4_f16_f32_len[reduc16],   arr_dmmv_nvfp4_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
                const uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_size_int);

            }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_data[reduc],     "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, {wg_size_subgroup, 1}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_data[reduc],     "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   arr_dmmv_id_mxfp4_f32_f32_len[reduc16],   arr_dmmv_id_mxfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   arr_dmmv_id_nvfp4_f32_f32_len[reduc16],   arr_dmmv_id_nvfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
            const uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_size_int);
        }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#if !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_UNUSED(rm_stdq_int);
    GGML_UNUSED(rm_kq_int);
    GGML_UNUSED(rm_iq_int);
#endif

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F8_E4M3], "dequant_f8_e4m3", dequant_f8_e4m3_len, dequant_f8_e4m3_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_BF16], "get_rows_bf16", get_rows_bf16_len, get_rows_bf16_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q1_0], "get_rows_q1_0", get_rows_q1_0_len, get_rows_q1_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K], "get_rows_q2_k", get_rows_q2_k_len, get_rows_q2_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K], "get_rows_q3_k", get_rows_q3_k_len, get_rows_q3_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_K], "get_rows_q4_k", get_rows_q4_k_len, get_rows_q4_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_K], "get_rows_q5_k", get_rows_q5_k_len, get_rows_q5_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q6_K], "get_rows_q6_k", get_rows_q6_k_len, get_rows_q6_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_rows_iq1_s_len,   get_rows_iq1_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_rows_iq1_m_len,   get_rows_iq1_m_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_rows_iq2_xxs_len, get_rows_iq2_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_rows_iq2_xs_len,  get_rows_iq2_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_rows_iq2_s_len,   get_rows_iq2_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_rows_iq3_xxs_len, get_rows_iq3_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_rows_iq3_s_len,   get_rows_iq3_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_rows_iq4_xs_len,  get_rows_iq4_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_rows_iq4_nl_len,  get_rows_iq4_nl_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_rows_mxfp4_len,   get_rows_mxfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_rows_nvfp4_len,   get_rows_nvfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_I32],     "get_rows_i32",     get_rows_i32_len,     get_rows_i32_data,     "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_rows_bf16_f32_len, get_rows_bf16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_rows_q1_0_f32_len, get_rows_q1_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_rows_q2_k_f32_len, get_rows_q2_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_rows_q3_k_f32_len, get_rows_q3_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_rows_q4_k_f32_len, get_rows_q4_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_rows_q5_k_f32_len, get_rows_q5_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_rows_q6_k_f32_len, get_rows_q6_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_rows_iq1_s_f32_len,   get_rows_iq1_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_rows_iq1_m_f32_len,   get_rows_iq1_m_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_rows_iq2_xxs_f32_len, get_rows_iq2_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_rows_iq2_xs_f32_len,  get_rows_iq2_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_rows_iq2_s_f32_len,   get_rows_iq2_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_rows_iq3_xxs_f32_len, get_rows_iq3_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_rows_iq3_s_f32_len,   get_rows_iq3_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_rows_iq4_xs_f32_len,  get_rows_iq4_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_rows_iq4_nl_f32_len,  get_rows_iq4_nl_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_rows_mxfp4_f32_len,   get_rows_mxfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_rows_nvfp4_f32_len,   get_rows_nvfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_data, "main", 3, sizeof(vk_op_flash_attn_split_k_reduce_push_constants), {1, device->subgroup_size, 1}, {device->subgroup_size}, 1, true);

    for (auto &it : device->pipeline_fa_mask_opt) {
        auto BrBc = it.first;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_mask_opt_len, fa_mask_opt_data, "main", 2, sizeof(vk_op_flash_attn_mask_opt_push_constants), {1, 1, 1}, {128, 128 / device->subgroup_size, BrBc.first, BrBc.second}, 1, true, true, device->subgroup_size);
    }

    if (device->subgroup_clustered && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_data, "main", 2, sizeof(vk_quantize_q8_1_push_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_data, "main", 2, sizeof(vk_quantize_q8_1_push_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1);
    }

    for (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
        if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_add_len, mul_mat_vec_p021_f16_f32_subgroup_add_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_p021_push_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true, true);
        } else {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_data,              "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_p021_push_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_nc_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1, true);

    if (sizeof(vk_op_rms_norm_mul_rope_push_constants) <= device->properties.limits.maxPushConstantsSize) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_data, "main", 7, sizeof(vk_op_rms_norm_mul_rope_push_constants), {1, 1, 1}, {0, 1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_data, "main", 7, sizeof(vk_op_rms_norm_mul_rope_push_constants), {1, 1, 1}, {0, 1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_back_f32, "rms_norm_back_f32", rms_norm_back_f32_len, rms_norm_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f32, "cpy_f32_f32", cpy_f32_f32_len, cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f16, "cpy_f16_f16", cpy_f16_f16_len, cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f32, "contig_cpy_f32_f32", contig_cpy_f32_f32_len, contig_cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f16, "contig_cpy_f16_f16", contig_cpy_f16_f16_len, contig_cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_F8_E4M3], "cpy_f32_f8_e4m3", cpy_f32_f8_e4m3_len, cpy_f32_f8_e4m3_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fa_q_f32_f8, "fa_q_f32_f8", fa_q_f32_f8_len, fa_q_f32_f8_data, "main", 3, sizeof(vk_fa_q_f32_f8_push_constants), {1, 1, 1}, {}, 1);

#define SET_ROWS(itype) \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_F32],  "set_rows_f32" #itype,  set_rows_f32 ## itype ## _len,  set_rows_f32 ## itype ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_F16],  "set_rows_f16" #itype,  set_rows_f16 ## itype ## _len,  set_rows_f16 ## itype ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_BF16], "set_rows_bf16" #itype, set_rows_bf16 ## itype ## _len, set_rows_bf16 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q1_0], "set_rows_q1_0" #itype, set_rows_q1_0 ## itype ## _len, set_rows_q1_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q4_0], "set_rows_q4_0" #itype, set_rows_q4_0 ## itype ## _len, set_rows_q4_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q4_1], "set_rows_q4_1" #itype, set_rows_q4_1 ## itype ## _len, set_rows_q4_1 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q5_0], "set_rows_q5_0" #itype, set_rows_q5_0 ## itype ## _len, set_rows_q5_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q5_1], "set_rows_q5_1" #itype, set_rows_q5_1 ## itype ## _len, set_rows_q5_1 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q8_0], "set_rows_q8_0" #itype, set_rows_q8_0 ## itype ## _len, set_rows_q8_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_IQ4_NL], "set_rows_iq4_nl" #itype, set_rows_iq4_nl ## itype ## _len, set_rows_iq4_nl ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_F8_E4M3], "set_rows_f8_e4m3" #itype, set_rows_f8_e4m3 ## itype ## _len, set_rows_f8_e4m3 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);

    SET_ROWS(_i32)
    SET_ROWS(_i64)
#undef SET_ROWS


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        return s;
    };

#define CREATE_BINARY(name, namemod, spec, bindings) \
    for (int s0 : {0,1}) for (int s1 : {0,1}) for (int d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## _data[s0][s1][d], \
                                "main", (bindings), sizeof(vk_op_binary_push_constants), {512, 1, 1}, spec, 1);

    CREATE_BINARY(add, , {0}, 4)
    CREATE_BINARY(add, _norepeat, {1}, 4)
    CREATE_BINARY(sub, , {0}, 3)
    CREATE_BINARY(sub, _norepeat, {1}, 3)
    CREATE_BINARY(mul, , {0}, 3)
    CREATE_BINARY(mul, _norepeat, {1}, 3)
    CREATE_BINARY(div, , {0}, 3)
    CREATE_BINARY(div, _norepeat, {1}, 3)
    CREATE_BINARY(add_rms, , {0}, 4)
    CREATE_BINARY(add_rms, _norepeat, {1}, 4)
#undef CREATE_BINARY

    if (device->multi_add) {
        for (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add[i],     "multi_add_f32_"     + std::to_string(i+1), multi_add_f32_len,     multi_add_f32_data,     "main", MAX_PARAMETER_COUNT, sizeof(vk_op_multi_add_push_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_add_rms_f32_len, multi_add_rms_f32_data, "main", MAX_PARAMETER_COUNT, sizeof(vk_op_multi_add_push_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_add_id_f32, "add_id_f32", add_id_f32_len, add_id_f32_data, "main", 4, sizeof(vk_op_add_id_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_set_f32, "set_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_f32, "concat_f32", concat_f32_len, concat_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_f16, "concat_f16", concat_f16_len, concat_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    if (getenv("GGML_VK_CONCAT_FAST")) {
        ggml_vk_create_pipeline(device, device->pipeline_concat_fast_f32, "concat_fast_f32", concat_fast_f32_len, concat_fast_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_concat_fast_f16, "concat_fast_f16", concat_fast_f16_len, concat_fast_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
        ggml_vk_create_pipeline(device, device->pipeline_concat_fast_i32, "concat_fast_i32", concat_fast_i32_len, concat_fast_i32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    }

    ggml_vk_create_pipeline(device, device->pipeline_upscale_nearest_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_NEAREST}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sqr_f32, "sqr_f32", sqr_f32_len, sqr_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sqrt_f32, "sqrt_f32", sqrt_f32_len, sqrt_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sin_f32, "sin_f32", sin_f32_len, sin_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cos_f32, "cos_f32", cos_f32_len, cos_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_clamp_f32, "clamp_f32", clamp_f32_len, clamp_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_data, "main", 2, sizeof(vk_op_pad_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_f32, "repeat_f32", repeat_f32_len, repeat_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_back_f32, "repeat_back_f32", repeat_back_f32_len, repeat_back_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_quick)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(abs)
    CREATE_UNARY(softplus)
    CREATE_UNARY(step)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(exp)
#undef CREATE_UNARY

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_quick)
#undef CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_leaky_relu_f32, "leaky_relu_f32", leaky_relu_f32_len, leaky_relu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_silu_back_f32, "silu_back_f32", silu_back_f32_len, silu_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_data, "main", 2, sizeof(vk_op_diag_mask_push_constants), {1, 512, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_back_f32, "soft_max_back_f32", soft_max_back_f32_len, soft_max_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    for (uint32_t i = 0; i < num_argsort_pipelines; ++i) {
        uint32_t BLOCK_SIZE = 1u << std::min(i, device->max_workgroup_size_log2);
        if (i <= device->max_workgroup_size_log2 &&
            2 * sizeof(int) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
            const uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_data, "main", 3, sizeof(vk_op_argsort_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, true);
        }
        const uint32_t WG_UNROLL_FACTOR = BLOCK_SIZE > 1 ? 2 : 1;
        BLOCK_SIZE /= WG_UNROLL_FACTOR;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_data, "main", 3, sizeof(vk_op_argsort_push_constants), {BLOCK_SIZE * WG_UNROLL_FACTOR, 1, 1}, {BLOCK_SIZE, WG_UNROLL_FACTOR}, 1, true);
    }

    for (uint32_t i = 0; i < num_topk_pipelines; ++i) {
        const uint32_t BLOCK_SIZE = 1u << i;
        const uint32_t NCOLS_PADDED_LOG2 = i;
        if (i <= device->max_workgroup_size_log2) {
            uint32_t nary_shmem = 2 * sizeof(int) * BLOCK_SIZE +
                                  sizeof(int) * device->subgroup_size +
                                  2 * sizeof(int) +
                                  2 * (BLOCK_SIZE / device->subgroup_size) * sizeof(int);
            if (device->subgroup_arithmetic && device->subgroup_require_full_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_search_f32_len, topk_nary_search_f32_data, "main", 2, sizeof(vk_op_topk_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, device->subgroup_size, device->subgroup_size_log2}, 1, true, true, device->subgroup_size);
            } else if (2 * sizeof(int) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_data, "main", 2, sizeof(vk_op_topk_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, true);
            }
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sum_rows_f32, "sum_rows_f32", sum_rows_f32_len, sum_rows_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    const uint32_t cumsum_elem_per_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { 256, device->subgroup_size, cumsum_elem_per_thread }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { 128, device->subgroup_size, 1 }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_data, "main", 3, sizeof(vk_op_sum_rows_push_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_data, "main", 3, sizeof(vk_op_sum_rows_push_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, true, true, device->subgroup_size);

    ggml_vk_create_pipeline(device, device->pipeline_count_equal_i32, "count_equal_i32", count_equal_i32_len, count_equal_i32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_count_experts, "count_experts", count_experts_len, count_experts_data, "main", 2, sizeof(vk_op_count_experts_push_constants), {1, 1, 1}, {}, 1, true);

    for (auto &s : device->pipeline_solve_tri_f32) {
        const vk_solve_tri_pipeline_state &state = s.first;

        // Max number of rows to load at a time, limited by shared memory
        const uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((state.N + state.K) * sizeof(float));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        const uint32_t block_size = std::max(128u, 1u << (uint32_t)ceilf(log2f(float(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_data, "main", 3,
            sizeof(vk_op_binary_push_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, block_size }, 1, true);
    }

#define IM2COL(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);
    if (device->shader_int64 && device->buffer_device_address) {
        IM2COL(_bda)
    } else {
        IM2COL()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_data, "main", 2, sizeof(vk_op_timestep_embedding_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_data, "main", 3, sizeof(vk_op_conv_transpose_1d_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_data, "main", 2, sizeof(vk_op_pool2d_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_data, "main", 7, sizeof(vk_op_rwkv_wkv6_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_data, "main", 8, sizeof(vk_op_rwkv_wkv7_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    {
        const uint32_t gdn_sizes[] = {32, 64, 128};
        const char * gdn_names[][2] = {
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        const bool use_subgroup_reduce = device->subgroup_arithmetic;
        for (uint32_t si = 0; si < 3; si++) {
            const uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            uint32_t lanes_per_column;
            if (S_V >= 128u && device->subgroup_clustered) {
                lanes_per_column = 8u;
            } else {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_per_column = std::min(S_V, device->subgroup_size);
            }

            const bool need_clustered_shader = lanes_per_column != 1 && (lanes_per_column < device->subgroup_size);
            size_t gdn_len;
            const void * gdn_data;
            if (use_subgroup_reduce && need_clustered_shader) {
                gdn_len = gated_delta_net_f32_len;
                gdn_data = (const void *)gated_delta_net_f32_data;
            } else if (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_data = (const void *)gated_delta_net_f32_nocluster_data;
            } else {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_data = (const void *)gated_delta_net_f32_shmem_data;
            }

            const uint32_t cols_per_wg = device->subgroup_size / lanes_per_column;
            const std::array<uint32_t, 3> wg_denoms = {1u, 1u, cols_per_wg};

            for (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_names[si][kda], gdn_len, gdn_data, "main", 7, sizeof(vk_op_gated_delta_net_push_constants),
                    wg_denoms, {S_V, kda, device->subgroup_size, lanes_per_column}, 1, true, use_subgroup_reduce, device->subgroup_size);
            }
        }
    }

    if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {128, device->subgroup_size}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {256, device->subgroup_size}, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {128, device->subgroup_size, 16}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {256, device->subgroup_size, 16}, 1, true, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32, "ssm_conv_f32", ssm_conv_f32_len, ssm_conv_f32_data, "main", 3, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", opt_step_adamw_f32_len, opt_step_adamw_f32_data, "main", 5, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_sgd_f32, "opt_step_sgd_f32", opt_step_sgd_f32_len, opt_step_sgd_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d
    for (uint32_t s = 0; s < CONV_SHAPE_COUNT; ++s) {
        uint32_t conv2d_WG_SIZE  = 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_block_size conv2d_BS = vk_conv_block_sizes[s];
        bool conv2d_UNROLL = true;

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#endif

        if (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = false;
        } else if (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            if (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = false;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        if (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_size,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        uint32_t conv2d_shmem_req =
            (conv2d_BS.K * (conv2d_BS.CRS + conv2d_SHMEM_PAD) + conv2d_BS.CRS * (conv2d_BS.NPQ + conv2d_SHMEM_PAD)) * sizeof(float);
        if (device->properties.limits.maxComputeSharedMemorySize < conv2d_shmem_req) {
            conv2d_BS.CRS = 8;
            if (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_size, conv2d_BS.CRS);
            }
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_SIZE, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

#define CREATE_CONV(name, type_suffix, spv_suffix) \
        for (auto &c : device->pipeline_##name##type_suffix[s]) { \
            const vk_conv2d_pipeline_state &state = c.first;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants_cpy, 1, true, use_collectives);    \
        }
#define CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } else
#endif
        if (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } else {
            CREATE_CONVS( )
        }
#undef CREATE_CONV
#undef CREATE_CONVS
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);

    for (uint32_t use_push = 0; use_push < 2; ++use_push) {
        for (uint32_t i = 0; i < num_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_push], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_data, "main", 4, sizeof(vk_op_topk_moe_push_constants), {1, 1, 1}, {device->subgroup_size, 1u<<i, use_push}, 1, true, true, device->subgroup_size);
        }
    }

    for (auto &c : compiles) {
        c.wait();
    }
}
