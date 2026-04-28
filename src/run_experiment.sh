#!/bin/bash
# Run train/test split experiment for all models using GPU 0 and GPU 2 only
set -e

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"
SCRIPT="src/train_test_split_experiment.py"
LOG_DIR="data/experiment_results/logs"
mkdir -p "$LOG_DIR"

MODELS=(
    "/home/linkco/exa/models/gte-large-en-v1.5"
    "/home/linkco/exa/models/stella_en_400M_v5"
    "/home/linkco/exa/models/inbedder-roberta-large"
    "/home/linkco/exa/models/instructor-large"
    "/home/linkco/exa/models/bart-base"
    "/home/linkco/exa/models/bge-m3"
)

# Models that may need downloading
PENDING=("/home/linkco/exa/models/mxbai-embed-large-v1" "/home/linkco/exa/models/Qwen3-Embedding-0.6B" "/home/linkco/exa/models/roberta-large")

echo "=== $(date) Starting train/test split experiments (2 GPUs) ==="

for model_path in "${MODELS[@]}"; do
    model_name=$(basename "$model_path")
    if [ ! -f "$model_path/config.json" ]; then
        echo "SKIP: $model_name (not found)"
        continue
    fi

    echo ""
    echo ">>> Starting: $model_name at $(date)"

    # GPU 0: tasks 0-11 (12 tasks)
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$model_path" --task-range 0:12 > "$LOG_DIR/${model_name}_gpu0.log" 2>&1 &
    PID0=$!
    # GPU 2: tasks 12-24 (12 tasks)
    CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$model_path" --task-range 12:24 > "$LOG_DIR/${model_name}_gpu2.log" 2>&1 &
    PID2=$!

    echo "    GPU 0 (0:12): PID $PID0 | GPU 2 (12:24): PID $PID2"
    wait $PID0 $PID2 2>/dev/null || true
    echo "    Finished: $model_name at $(date)"
done

# Wait for downloads then run
echo ""
echo "=== Checking downloaded models ==="
for model_path in "${PENDING[@]}"; do
    model_name=$(basename "$model_path")

    # Wait up to 3 hours for download
    for i in $(seq 1 180); do
        if [ -f "$model_path/config.json" ]; then
            break
        fi
        if [ $i -eq 180 ]; then
            echo "SKIP: $model_name (download timeout after 3h)"
            continue 2
        fi
        sleep 60
    done

    echo ">>> Starting: $model_name at $(date)"
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$model_path" --task-range 0:12 > "$LOG_DIR/${model_name}_gpu0.log" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$model_path" --task-range 12:24 > "$LOG_DIR/${model_name}_gpu2.log" 2>&1 &
    PID2=$!

    echo "    GPU 0 (0:12): PID $PID0 | GPU 2 (12:24): PID $PID2"
    wait $PID0 $PID2 2>/dev/null || true
    echo "    Finished: $model_name at $(date)"
done

echo ""
echo "=== ALL DONE at $(date) ==="
