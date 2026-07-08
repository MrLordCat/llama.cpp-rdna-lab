#pragma once

#include "llama.h"

#include <vector>

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    llama_batch     hook_batch = {};      // sized to n_ubatch
    std::vector<float> hidden_rows;        // scratch [n_tokens, n_embd] for bulk hidden copies

    // Windowed-prefill gate. When false, handle_mtp_for_ubatch returns before the
    // pipeline-draining synchronize()/readback/MTP-decode. The server disables the
    // hook for the bulk of a long prompt and re-enables it for the tail window, so
    // the MTP KV covers only the recent context (cheap, minor acceptance cost)
    // instead of running the whole prompt through the MTP context (+~40s at 56k).
    bool hook_active = true;

    struct pending_state {
        // Cross-ubatch shift state: pair (h_p, x_{p+1}) at MTP pos p+1. The last
        // h-row of one ubatch needs the first token of the NEXT ubatch to pair
        // with, so it's stashed here until that next ubatch fires. Resets when
        // pos_start of the new ubatch != pending_pos+1 (new prompt or seq_rm gap).
        std::vector<float> h;
        llama_pos          pos = -1;
    };

    std::vector<pending_state> pending;
};
