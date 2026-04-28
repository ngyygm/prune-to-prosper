#!/bin/bash
# Run train/test split experiment: 2 models in parallel, each using 2 GPUs.
# Each model only uses ~3GB, so 2 models on the same GPU (~6GB) fits in 24GB.
set -e

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"
SCRIPT="src/train_test_split_experiment.py"
LOG_DIR="data/experiment_results/logs"
mkdir -p "$LOG_DIR"

ALL_MODELS=(
    "/home/linkco/exa/models/gte-large-en-v1.5"
    "/home/linkco/exa/models/stella_en_400M_v5"
    "/home/linkco/exa/models/inbedder-roberta-large"
    "/home/linkco/exa/models/instructor-large"
    "/home/linkco/exa/models/bart-base"
    "/home/linkco/exa/models/bge-m3"
    "/home/linkco/exa/models/mxbai-embed-large-v1"
    "/home/linkco/exa/models/Qwen3-Embedding-0.6B"
    "/home/linkco/exa/models/roberta-large"
)

echo "=== $(date) Parallel experiment: 2 models at a time ==="

# Process models in pairs
i=0
while [ $i -lt ${#ALL_MODELS[@]} ]; do
    model_a="${ALL_MODELS[$i]}"
    model_b=""
    [ $((i + 1)) -lt ${#ALL_MODELS[@]} ] && model_b="${ALL_MODELS[$((i + 1))]}"

    name_a=$(basename "$model_a")
    name_b=$(basename "$model_b")

    echo ""
    echo ">>> Pair $((i/2 + 1)): $name_a + $name_b at $(date)"

    # Check availability
    [ ! -f "$model_a/config.json" ] && echo "  SKIP: $name_a" && i=$((i+2)) && continue

    # Launch model A on both GPUs
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$model_a" --task-range 0:12 > "$LOG_DIR/${name_a}_gpu0.log" 2>&1 &
    PID_A0=$!
    CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$model_a" --task-range 12:24 > "$LOG_DIR/${name_a}_gpu2.log" 2>&1 &
    PID_A2=$!
    echo "  $name_a: GPU0=$PID_A0 GPU2=$PID_A2"

    # Launch model B on both GPUs (if exists)
    PID_B0="" PID_B2=""
    if [ -n "$model_b" ] && [ -f "$model_b/config.json" ]; then
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$model_b" --task-range 0:12 > "$LOG_DIR/${name_b}_gpu0.log" 2>&1 &
        PID_B0=$!
        CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$model_b" --task-range 12:24 > "$LOG_DIR/${name_b}_gpu2.log" 2>&1 &
        PID_B2=$!
        echo "  $name_b: GPU0=$PID_B0 GPU2=$PID_B2"
    fi

    # Wait for all workers
    echo "  Waiting..."
    for pid in $PID_A0 $PID_A2 $PID_B0 $PID_B2; do
        [ -z "$pid" ] && continue
        wait "$pid" 2>/dev/null || true
    done
    echo "  Pair done at $(date)"

    i=$((i + 2))
done

echo ""
echo "=== ALL DONE at $(date) ==="
