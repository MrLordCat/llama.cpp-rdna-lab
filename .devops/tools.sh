#!/usr/bin/env bash
set -e

# Read the first argument into a variable
arg1="$1"

# Shift the arguments to remove the first one
shift

if [[ "$arg1" == '--quantize' || "$arg1" == '-q' ]]; then
    exec ./llama-quantize "$@"
elif [[ "$arg1" == '--run' || "$arg1" == '-r' ]]; then
    exec ./llama-cli "$@"
elif [[ "$arg1" == '--run-legacy' || "$arg1" == '-l' ]]; then
    exec ./llama-completion "$@"
elif [[ "$arg1" == '--bench' || "$arg1" == '-b' ]]; then
    exec ./llama-bench "$@"
elif [[ "$arg1" == '--perplexity' || "$arg1" == '-p' ]]; then
    exec ./llama-perplexity "$@"
elif [[ "$arg1" == '--server' || "$arg1" == '-s' ]]; then
    exec ./llama-server "$@"
else
    echo "Unknown command: $arg1"
    echo "Available commands: "
    echo "  --run (-r): Run a model (chat) previously converted into ggml"
    echo "              ex: -m /models/7B/ggml-model-q4_0.bin"
    echo "  --run-legacy (-l): Run a model (legacy completion) previously converted into ggml"
    echo "              ex: -m /models/7B/ggml-model-q4_0.bin -no-cnv -p \"Building a website can be done in 10 simple steps:\" -n 512"
    echo "  --bench (-b): Benchmark the performance of the inference for various parameters."
    echo "              ex: -m model.gguf"
    echo "  --perplexity (-p): Measure the perplexity of a model over a given text."
    echo "              ex: -m model.gguf -f file.txt"
    echo "  --quantize (-q): Optimize with quantization process ggml"
    echo "              ex: \"/models/7B/ggml-model-f16.bin\" \"/models/7B/ggml-model-q4_0.bin\" 2"
    echo "  --server (-s): Run a model on the server"
    echo "              ex: -m /models/7B/ggml-model-q4_0.bin -c 2048 -ngl 43 -mg 1 --port 8080"
fi
