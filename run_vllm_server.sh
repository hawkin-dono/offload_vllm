#!/bin/bash

# Configuration
MODEL_PATH="/dev/shm/Qwen3-30B-A3B"
OFFLOAD_GB=0
MAX_SEQS=4
GPU_UTIL="0.9"
PORT=8000

PYTHON_EXEC="/home/hieuvt/vllm-hpclab/.venv/bin/python"

# Run the server
$PYTHON_EXEC -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --cpu-offload-gb "$OFFLOAD_GB" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --override-generation-config '{"temperature": 0.0}' \
    --trust-remote-code \
    --enforce-eager
