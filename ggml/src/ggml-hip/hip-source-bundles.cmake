if (COMMAND ggml_hip_collect_sources)
    return()
endif()

function(ggml_hip_collect_sources out_headers out_sources out_host_sources out_profile out_reduced out_no_fa)
    string(TOLOWER "${GGML_HIP_EXPERIMENT_PROFILE}" _profile)
    if (_profile STREQUAL "")
        set(_profile "default")
    endif()

    if (_profile STREQUAL "mmvq-isolated")
        set(_profile "mmvq-focused")
    endif()

    if (GGML_HIP_QWEN_FA_REDUCED)
        if (_profile STREQUAL "default")
            set(_profile "qwen-fa-reduced")
        elseif (NOT _profile STREQUAL "qwen-fa-reduced")
            message(WARNING
                "GGML_HIP_QWEN_FA_REDUCED=ON is ignored because "
                "GGML_HIP_EXPERIMENT_PROFILE=${_profile} was selected.")
        endif()
    endif()

    set(_supported_profiles default qwen-fa-reduced mmvq-focused)
    list(FIND _supported_profiles "${_profile}" _profile_idx)
    if (_profile_idx EQUAL -1)
        message(FATAL_ERROR
            "Unsupported GGML_HIP_EXPERIMENT_PROFILE='${GGML_HIP_EXPERIMENT_PROFILE}'. "
            "Supported values: default, qwen-fa-reduced, mmvq-focused (or alias mmvq-isolated).")
    endif()

    set(_reduced OFF)
    set(_no_fa OFF)

    if (_profile STREQUAL "qwen-fa-reduced")
        set(_reduced ON)
    elseif (_profile STREQUAL "mmvq-focused")
        set(_reduced ON)
        set(_no_fa ON)
    endif()

    file(GLOB _headers "../ggml-cuda/*.cuh")
    list(APPEND _headers "../../include/ggml-cuda.h")

    file(GLOB _sources "../ggml-cuda/*.cu")
    set(_host_sources "")

    if (_reduced)
        list(FILTER _sources EXCLUDE REGEX ".*/fattn\\.cu$")
        list(FILTER _sources EXCLUDE REGEX ".*/fattn-qwen-reduced\\.cu$")
        list(FILTER _sources EXCLUDE REGEX ".*/fattn-tile\\.cu$")
        list(APPEND _host_sources ../ggml-cuda/fattn-qwen-reduced.cpp)

        if (_no_fa)
            list(FILTER _sources EXCLUDE REGEX ".*/fattn-wmma-f16\\.cu$")
        endif()
    else()
        file(GLOB _sr "../ggml-cuda/template-instances/fattn-tile*.cu")
        list(APPEND _sources ${_sr})

        file(GLOB _sr "../ggml-cuda/template-instances/fattn-mma*.cu")
        list(APPEND _sources ${_sr})
    endif()

    file(GLOB _sr "../ggml-cuda/template-instances/mmq*.cu")
    list(APPEND _sources ${_sr})

    file(GLOB _sr "../ggml-cuda/template-instances/mmf*.cu")
    list(APPEND _sources ${_sr})

    if (NOT _no_fa)
        if (GGML_HIP_FA_ALL_QUANTS)
            file(GLOB _sr "../ggml-cuda/template-instances/fattn-vec*.cu")
            list(APPEND _sources ${_sr})
        else()
            list(APPEND _sources
                ../ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu
                ../ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu
                ../ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu
                ../ggml-cuda/template-instances/fattn-vec-instance-bf16-bf16.cu
                ../ggml-cuda/template-instances/fattn-vec-instance-tkv2_0-tkv2_0.cu
                ../ggml-cuda/template-instances/fattn-vec-instance-tkv3_0-tkv3_0.cu
                ../ggml-cuda/template-instances/fattn-vec-instance-tkv4_0-tkv4_0.cu)
        endif()
    endif()

    list(SORT _headers)
    list(SORT _sources)
    list(SORT _host_sources)

    set(${out_headers} "${_headers}" PARENT_SCOPE)
    set(${out_sources} "${_sources}" PARENT_SCOPE)
    set(${out_host_sources} "${_host_sources}" PARENT_SCOPE)
    set(${out_profile} "${_profile}" PARENT_SCOPE)
    set(${out_reduced} "${_reduced}" PARENT_SCOPE)
    set(${out_no_fa} "${_no_fa}" PARENT_SCOPE)
endfunction()
