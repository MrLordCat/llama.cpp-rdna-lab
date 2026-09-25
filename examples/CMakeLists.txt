# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# Project example whitelist. These are the small diagnostics used by the
# Qwen/BitNet, long-context and speculative-decoding workflows.
set(LLAMA_EXAMPLES_WHITELIST
    "attn-sparsity-scout;kv-precision-scout;parallel;passkey;simple;simple-chat;dump_nextn;speculative;speculative-simple"
    CACHE STRING "Semicolon-separated example directories built by this fork")

if (NOT EMSCRIPTEN)
    foreach(example IN LISTS LLAMA_EXAMPLES_WHITELIST)
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${example}/CMakeLists.txt")
            message(FATAL_ERROR "Unknown entry in LLAMA_EXAMPLES_WHITELIST: ${example}")
        endif()
        add_subdirectory(${example})
    endforeach()
endif()
