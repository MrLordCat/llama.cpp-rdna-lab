// Late-bound instance and vendor capability helpers used during device discovery.
// Extension availability
static bool ggml_vk_instance_layer_settings_available() {
#ifdef GGML_VULKAN_VALIDATE
    // Check if validation layer provides the extension
    const std::string layer_name = "VK_LAYER_KHRONOS_validation";
    for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
        if (layer_name == layer.layerName.data()) {
            for (const auto& ext : vk::enumerateInstanceExtensionProperties(layer_name)) {
                if (strcmp("VK_EXT_layer_settings", ext.extensionName.data()) == 0) {
                    return true;
                }
            }
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Validation layer or layer extension VK_EXT_layer_settings not found." << std::endl;
#endif
    return false;
}
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef __APPLE__
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
#endif
    return false;

    UNUSED(instance_extensions);
}

// Extension availability
static bool ggml_vk_instance_debug_utils_ext_available(
    const std::vector<vk::ExtensionProperties> & instance_extensions) {
    // Check for portability enumeration extension for MoltenVK support
    for (const auto & properties : instance_extensions) {
        if (strcmp("VK_EXT_debug_utils", properties.extensionName) == 0) {
            return true;
        }
    }

    std::cerr << "ggml_vulkan: WARNING: Instance extension VK_EXT_debug_utils not found." << std::endl;
    return false;

    UNUSED(instance_extensions);
}

static bool ggml_vk_device_is_supported(const vk::PhysicalDevice & vkdev) {
    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    vkGetPhysicalDeviceFeatures2(vkdev, &device_features2);

    return vk11_features.storageBuffer16BitAccess;
}

static bool ggml_vk_khr_cooperative_matrix_support(const vk::PhysicalDeviceProperties& props, const vk::PhysicalDeviceDriverProperties& driver_props, vk_device_architecture arch) {
    switch (props.vendorID) {
    case VK_VENDOR_ID_INTEL:
        // Only allowing Xe2 GPU at the moment since Xe2 GPU can gain significant performance boost,
        // while some older hardware (ex. Arc A770) has performance regressions
        return arch == vk_device_architecture::INTEL_XE2;
    case VK_VENDOR_ID_AMD:
        if (driver_props.driverID == vk::DriverId::eAmdProprietary || driver_props.driverID == vk::DriverId::eAmdOpenSource) {
            // Workaround for AMD proprietary driver reporting support on all GPUs
            return arch == vk_device_architecture::AMD_RDNA3;
        }
        return true;
    default:
        return true;
    }
}

static uint32_t ggml_vk_intel_shader_core_count(const vk::PhysicalDevice& vkdev) {
    VkPhysicalDeviceProperties2 props = vkdev.getProperties2();

    if (props.properties.vendorID != VK_VENDOR_ID_INTEL) {
        return 0;
    }

    const uint32_t device_id = props.properties.deviceID;

    switch (device_id) {
    case 0x56A6:  // A310
        return 6;
    case 0x5693:  // A370M
    case 0x56A5:  // A380
    case 0x56B1:  // Pro A40/A50
        return 8;
    case 0x5697:  // A530M
        return 12;
    case 0x5692:  // A550M
    case 0x56B3:  // Pro A60
        return 16;
    case 0x56A2:  // A580
        return 24;
    case 0x5691:  // A730M
    case 0x56A1:  // A750
        return 28;
    case 0x56A0:  // A770
    case 0x5690:  // A770M
        return 32;
    case 0xE212:  // Pro B50
        return 16;
    case 0xE20C:  // B570
        return 18;
    case 0xE20B:  // B580
    case 0xE211:  // Pro B60
        return 20;
    default:
        return 0;
    }
}
