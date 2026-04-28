#!/bin/bash
# Run ALL models simultaneously. Each model uses ~1-2GB GPU memory.
# GPU 0 and GPU 2 each host all models. GPU 1 is blocked by Ollama.
# Models take turns using GPU for encoding (bursty), so they rarely conflict.
set -e

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"
SCRIPT="src/train_test_split_experiment.py"
LOG_DIR="data/experiment_results/logs"
mkdir -p "$LOG_DIR"

# 8 ready models + bge-m3 (may need download)
MODELS=(
    "/home/linkco/exa/models/gte-large-en-v1.5"
    "/home/linkco/exa/models/stella_en_400M_v5"
    "/home/linkco/exa/models/inbedder-roberta-large"
    "/home/linkco/exa/models/instructor-large"
    "/home/linkco/exa/models/bart-base"
    "/home/linkco/exa/models/mxbai-embed-large-v1"
    "/home/linkco/exa/models/Qwen3-Embedding-0.6B"
    "/home/linkco/exa/models/roberta-large"
    "/home/linkco/exa/models/bge-m3"
)

echo "=== $(date) Launching ALL models simultaneously ==="

PIDS=""
for model_path in "${MODELS[@]}"; do
    model_name=$(basename "$model_path")

    if [ ! -f "$model_path/config.json" ]; then
        echo "  SKIP: $model_name (not ready)"
        continue
    fi

    echo "  $model_name: launching..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$model_path" --task-range 0:12 > "$LOG_DIR/${model_name}_gpu0.log" 2>&1 &
    PIDS="$PIDS $!"
    CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$model_path" --task-range 12:24 > "$LOG_DIR/${model_name}_gpu2.log" 2>&1 &
    PIDS="$PIDS $!"
done

echo ""
echo "All workers launched. PIDs: $PIDS"
echo "Waiting for all to complete..."
echo ""

# Wait for each
for pid in $PIDS; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "=== ALL DONE at $(date) ==="
