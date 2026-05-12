if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)

    # Fail fast on Windows if HIP is configured with a non-ROCm C++ compiler.
    get_filename_component(_ggml_hip_cxx_name "${CMAKE_CXX_COMPILER}" NAME_WE)
    string(TOLOWER "${_ggml_hip_cxx_name}" _ggml_hip_cxx_name_lc)
    if (NOT _ggml_hip_cxx_name_lc MATCHES "clang" AND NOT _ggml_hip_cxx_name_lc MATCHES "hipcc")
        message(FATAL_ERROR
            "GGML_HIP on Windows requires ROCm clang++ or hipcc as CMAKE_CXX_COMPILER. "
            "Current compiler: ${CMAKE_CXX_COMPILER}")
    endif()
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

include(${CMAKE_CURRENT_LIST_DIR}/hip-source-bundles.cmake)

ggml_hip_collect_sources(
    GGML_HEADERS_ROCM
    GGML_SOURCES_ROCM
    GGML_HOST_SOURCES_ROCM
    GGML_HIP_SOURCE_PROFILE
    GGML_HIP_PROFILE_REDUCED_FATTN
    GGML_HIP_PROFILE_NO_FA)

if (GGML_HIP_PROFILE_REDUCED_FATTN)
    if (GGML_HIP_PROFILE_NO_FA)
        message(WARNING
            "GGML_HIP_EXPERIMENT_PROFILE=${GGML_HIP_SOURCE_PROFILE}: building local MMVQ-focused HIP profile "
            "with FlashAttention kernels disabled. Use only for MMVQ/decode-path A/B and build-pressure research.")
    else()
        message(WARNING
            "GGML_HIP_EXPERIMENT_PROFILE=${GGML_HIP_SOURCE_PROFILE}: building local reduced FlashAttention HIP profile "
            "for Qwen/RDNA4 A/B runs.")
    endif()
endif()

if (GGML_CUDA_FA_ALL_QUANTS)
    if (GGML_HIP_PROFILE_NO_FA)
        message(WARNING
            "GGML_CUDA_FA_ALL_QUANTS=ON has no effect when GGML_HIP_EXPERIMENT_PROFILE=${GGML_HIP_SOURCE_PROFILE} "
            "disables FlashAttention kernels.")
    else()
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    endif()
endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

if (GGML_HOST_SOURCES_ROCM)
    set_source_files_properties(${GGML_HOST_SOURCES_ROCM} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
    target_sources(ggml-hip PRIVATE ${GGML_HOST_SOURCES_ROCM})
endif()

set(GGML_HIP_ROCWMMA_ENABLED OFF)
if (GGML_HIP_ROCWMMA_FATTN)
    set(_rocwmma_candidates "")
    if (GGML_HIP_ROCWMMA_INCLUDE_DIR)
        list(APPEND _rocwmma_candidates "${GGML_HIP_ROCWMMA_INCLUDE_DIR}")
    endif()
    list(APPEND _rocwmma_candidates "${ROCM_PATH}/include")

    foreach(_candidate IN LISTS _rocwmma_candidates)
        if (EXISTS "${_candidate}/rocwmma/rocwmma.hpp" AND EXISTS "${_candidate}/rocwmma/rocwmma-version.hpp")
            set(GGML_HIP_ROCWMMA_ENABLED ON)
            set(GGML_HIP_ROCWMMA_INCLUDE_DIR_RESOLVED "${_candidate}")
            break()
        endif()
    endforeach()

    if (GGML_HIP_ROCWMMA_ENABLED)
        message(STATUS "rocWMMA FlashAttention enabled: ${GGML_HIP_ROCWMMA_INCLUDE_DIR_RESOLVED}")
        target_include_directories(ggml-hip PRIVATE "${GGML_HIP_ROCWMMA_INCLUDE_DIR_RESOLVED}")
    else()
        message(WARNING
            "GGML_HIP_ROCWMMA_FATTN=ON but rocWMMA headers were not found; falling back to non-rocWMMA path. "
            "Set GGML_HIP_ROCWMMA_INCLUDE_DIR to the directory containing rocwmma/ to enable it.")
    endif()
endif()

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_ENABLED)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA OR GGML_HIP_PROFILE_NO_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (GGML_HIP_PROFILE_NO_FA)
    add_compile_definitions(GGML_HIP_MMVQ_FOCUSED_PROFILE)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)

if (WIN32)
    target_link_libraries(ggml-hip PRIVATE dxgi pdh)
endif()
