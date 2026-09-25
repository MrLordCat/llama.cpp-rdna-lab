# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# Project tool whitelist. Source directories outside this list stay available
# for reference, but are not configured or built by the supported profiles.
set(LLAMA_TOOLS_WHITELIST
    "batched-bench;gguf-split;imatrix;llama-bench;perplexity;quantize;mtmd;server;cli;tokenize;parser;rpc"
    CACHE STRING "Semicolon-separated tool directories built by this fork")

if (NOT EMSCRIPTEN)
    foreach(tool IN LISTS LLAMA_TOOLS_WHITELIST)
        if ((tool STREQUAL "server" OR tool STREQUAL "cli") AND NOT LLAMA_BUILD_SERVER)
            continue()
        endif()
        if (tool STREQUAL "rpc" AND NOT GGML_RPC)
            continue()
        endif()
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${tool}/CMakeLists.txt")
            message(FATAL_ERROR "Unknown entry in LLAMA_TOOLS_WHITELIST: ${tool}")
        endif()
        add_subdirectory(${tool})
    endforeach()
endif()
