#!/bin/bash
# Launch train/test split experiment across multiple GPUs in parallel.
#
# Usage:
#   bash src/run_train_test_split.sh <model_path>
#   bash src/run_train_test_split.sh /home/linkco/exa/models/gte-large-en-v1.5
#
# Each GPU runs a subset of tasks. Results are saved per-model with resume support.

set -e

MODEL_PATH="$1"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash src/run_train_test_split.sh <model_path>"
    exit 1
fi

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"
SCRIPT="src/train_test_split_experiment.py"

# Count total tasks (25 non-Classification)
TOTAL_TASKS=25

# Split into 3 groups: 0:9, 9:17, 17:25
echo "Launching experiment for: $MODEL_PATH"
echo "Splitting $TOTAL_TASKS tasks across 3 GPUs..."

# GPU 0: tasks 0-8 (9 tasks)
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT "$MODEL_PATH" --task-range 0:9 &
PID0=$!

# GPU 1: tasks 9-16 (8 tasks)
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT "$MODEL_PATH" --task-range 9:17 &
PID1=$!

# GPU 2: tasks 17-24 (8 tasks)
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT "$MODEL_PATH" --task-range 17:25 &
PID2=$!

echo "Started 3 workers:"
echo "  GPU 0 (tasks 0:9):  PID $PID0"
echo "  GPU 1 (tasks 9:17): PID $PID1"
echo "  GPU 2 (tasks 17:25): PID $PID2"
echo ""
echo "Waiting for all workers to complete..."

# Wait for all
wait $PID0
echo "GPU 0 done (exit: $?)"
wait $PID1
echo "GPU 1 done (exit: $?)"
wait $PID2
echo "GPU 2 done (exit: $?)"

# All results are saved to the same file (resume-safe: each worker skips existing tasks)
echo ""
echo "All done! Results saved per-model in data/experiment_results/"
