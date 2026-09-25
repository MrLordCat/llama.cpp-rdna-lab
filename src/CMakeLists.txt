llama_add_compile_flags()

#
# libraries
#

# llama

# This fork intentionally carries a narrow model surface. Keep this list
# explicit: adding a new architecture is a reviewed project decision, not an
# automatic side effect of dropping another upstream file into models/.
set(LLAMA_MODELS_SOURCES
    models/bitnet.cpp
    models/delta-net-base.cpp
    models/qwen.cpp
    models/qwen2.cpp
    models/qwen2moe.cpp
    models/qwen2vl.cpp
    models/qwen3.cpp
    models/qwen3moe.cpp
    models/qwen3next.cpp
    models/qwen3vl.cpp
    models/qwen3vlmoe.cpp
    models/qwen35.cpp
    models/qwen35moe.cpp
    models/qwen4exp.cpp
)

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-hybrid-idx.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
