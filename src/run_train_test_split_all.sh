#!/bin/bash
# Master script: run available models + download missing ones in parallel
set -e

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"
SCRIPT="src/train_test_split_experiment.py"
MODELS_DIR="/home/linkco/exa/models"
LOG_DIR="data/experiment_results/logs"
mkdir -p "$LOG_DIR"

# ============================================================
# Models with existing data that need to run
# ============================================================
AVAILABLE_MODELS=(
    "$MODELS_DIR/gte-large-en-v1.5"
    "$MODELS_DIR/stella_en_400M_v5"
    "$MODELS_DIR/inbedder-roberta-large"
    "$MODELS_DIR/instructor-large"
    "$MODELS_DIR/bart-base"
)

# Models that need downloading (HF repo id -> local dir name)
declare -A DOWNLOAD_MODELS
DOWNLOAD_MODELS["BAAI/bge-m3"]="bge-m3"
DOWNLOAD_MODELS["mixedbread-ai/mxbai-embed-large-v1"]="mxbai-embed-large-v1"
DOWNLOAD_MODELS["Qwen/Qwen3-Embedding-0.6B"]="Qwen3-Embedding-0.6B"

# roberta-large is a special case - it's just the base model used with mean pooling
DOWNLOAD_MODELS["FacebookAI/roberta-large"]="roberta-large"

# ============================================================
# Step 1: Download missing models in background
# ============================================================
echo "=== Downloading missing models in background ==="
for hf_id in "${!DOWNLOAD_MODELS[@]}"; do
    local_name="${DOWNLOAD_MODELS[$hf_id]}"
    local_path="$MODELS_DIR/$local_name"

    if [ -d "$local_path" ] && [ -f "$local_path/config.json" ]; then
        echo "  $local_name already exists, skipping download"
        continue
    fi

    echo "  Downloading $hf_id -> $local_path"
    nohup $PYTHON -c "
from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download('$hf_id', local_dir='$local_path', 
print('DONE: $hf_id')
" > "$LOG_DIR/download_${local_name}.log" 2>&1 &
    # PID not tracked — downloads checked by config.json existence
done

# Downloads run in background, checked later
# Downloads run in background, checked later
echo ""

# ============================================================
# Step 2: Run available models sequentially (3 GPUs per model)
# ============================================================
echo "=== Running available models ==="
for model_path in "${AVAILABLE_MODELS[@]}"; do
    model_name=$(basename "$model_path")
    echo ""
    echo ">>> Starting: $model_name"
    echo "    Time: $(date)"

    # Launch 3 GPU workers
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$model_path" --task-range 0:8 > "$LOG_DIR/${model_name}_gpu0.log" 2>&1 &
    PID0=$!
    CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT "$model_path" --task-range 8:16 > "$LOG_DIR/${model_name}_gpu1.log" 2>&1 &
    PID1=$!
    CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$model_path" --task-range 16:24 > "$LOG_DIR/${model_name}_gpu2.log" 2>&1 &
    PID2=$!

    echo "    GPU workers: $PID0 $PID1 $PID2"

    # Wait for all 3 workers
    wait $PID0 $PID1 $PID2 2>/dev/null || true
    echo "    Finished: $model_name at $(date)"
done

# ============================================================
# Step 3: Run newly downloaded models
# ============================================================
echo ""
echo "=== Checking downloaded models ==="
for hf_id in "${!DOWNLOAD_MODELS[@]}"; do
    local_name="${DOWNLOAD_MODELS[$hf_id]}"
    local_path="$MODELS_DIR/$local_name"

    if [ -f "$local_path/config.json" ]; then
        echo ">>> Running: $local_name"
        echo "    Time: $(date)"

        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$local_path" --task-range 0:8 > "$LOG_DIR/${local_name}_gpu0.log" 2>&1 &
        PID0=$!
        CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT "$local_path" --task-range 8:16 > "$LOG_DIR/${local_name}_gpu1.log" 2>&1 &
        PID1=$!
        CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$local_path" --task-range 16:24 > "$LOG_DIR/${local_name}_gpu2.log" 2>&1 &
        PID2=$!

        wait $PID0 $PID1 $PID2 2>/dev/null || true
        echo "    Finished: $local_name at $(date)"
    else
        echo "    SKIP: $local_name (download may have failed)"
    fi
done

echo ""
echo "=== ALL DONE ==="
echo "Results in: data/experiment_results/train_test_split_*.json"
