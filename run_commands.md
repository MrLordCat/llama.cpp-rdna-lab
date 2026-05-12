cd "C:/Users/Chris/Documents/GitHub/llama.cpp-with-GUI" && \
unset HSA_OVERRIDE_GFX_VERSION && \
TS=$(date +%Y%m%d-%H%M%S) && \
PY="C:/Users/Chris/AppData/Local/Programs/Python/Python313/python.exe" && \
EXTRA="--spec-type none" && \
UB=300 && \
LLAMA_UBATCH_SPLIT_POLICY=shape-score \
LLAMA_UBATCH_SHAPE_PREFERRED=192 \
LLAMA_UBATCH_SHAPE_MIN_TAIL=144 \
LLAMA_UBATCH_SHAPE_CHUNK_HINT=96 \
LLAMA_UBATCH_SHAPE_MIN_CHUNK_TAIL=32 \
"$PY" scripts/agent_workload_bench.py \
  --label "p1-manual-${TS}-shape-ub${UB}-r1" \
  --server-bin build-rocm-vec/bin/llama-server.exe \
  --model models/Qwen3.6-27B-Q3_K_S.gguf \
  --tasks v2-review \
  --runs 1 \
  --ctx-size 12288 \
  --batch-size 6144 \
  --ubatch-size "$UB" \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --server-extra "$EXTRA" \
  --real-context-mode repo-snapshot \
  --real-context-chars 21872 \
  --no-reuse \
  --no-v2-prime-pass \
  --no-disable-thinking \
  --max-tokens 120


# Build/import upstream stock ROCm binary into this fork (separate folder)
cd "C:/Users/Chris/Documents/GitHub/llama.cpp-with-GUI" && \
bash scripts/build_upstream_rocm_stock.sh \
  --upstream-dir "C:/Users/Chris/Documents/GitHub/llama.cpp-upstream-stock" \
  --import-dir "C:/Users/Chris/Documents/GitHub/llama.cpp-with-GUI/build-rocm-upstream-stock" \
  --rocm-root "C:/Program Files/AMD/ROCm/7.1" \
  --ggml-openmp OFF \
  --amdgpu-targets gfx1201 \
  --jobs 16