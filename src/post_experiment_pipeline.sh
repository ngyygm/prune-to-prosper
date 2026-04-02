#!/bin/bash
# Master post-processing pipeline
# Run this AFTER all rank_chunk_mteb experiments complete
# Checks for completion, then runs all analyses and figure generation

set -e
cd /home/linkco/exa/llm-usefulEeb

PYTHON="/home/linkco/anaconda3/envs/llama/bin/python"

echo "============================================"
echo "Post-Experiment Analysis Pipeline"
echo "============================================"

# Step 0: Check that all model data exists
echo ""
echo "[Step 0] Checking data availability..."
EXPECTED_MODELS=(
    "bge-m3"
    "bart-base"
    "instructor-large"
    "mxbai-embed-large-v1"
    "roberta-large"
    "gtr-t5-large"
    "gte-base"
    "Qwen3-Embedding-0.6B"
    "gte-Qwen2-1.5B-instruct"
)

MISSING=0
for model in "${EXPECTED_MODELS[@]}"; do
    if [ ! -f "data/analyze/${model}.json" ]; then
        echo "  MISSING: data/analyze/${model}.json"
        MISSING=$((MISSING + 1))
    else
        # Check if file has actual chunk data
        HAS_CHUNKS=$($PYTHON -c "
import json
with open('data/analyze/${model}.json') as f:
    d = json.load(f)
tasks = d.get('task_name', {})
has_chunks = any('split_win_size' in t and '2' in t.get('split_win_size', {}) for t in tasks.values())
print('yes' if has_chunks else 'no')
")
        if [ "$HAS_CHUNKS" = "no" ]; then
            echo "  INCOMPLETE: ${model} (no chunk data)"
            MISSING=$((MISSING + 1))
        else
            echo "  OK: ${model}"
        fi
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING models missing or incomplete. Some analyses may fail."
    echo "Press Ctrl+C to abort, or wait 10 seconds to continue..."
    sleep 10
fi

# Step 1: Compute entropy and uniformity metrics for ALL models
echo ""
echo "[Step 1] Computing entropy/uniformity metrics for all models..."
$PYTHON src/compute_all_model_entropy.py

# Step 2: Run magnitude pruning analysis (Part 1 - no MTEB, fast)
echo ""
echo "[Step 2] Running magnitude pruning analysis..."
$PYTHON src/magnitude_pruning.py \
    --analyze_dir data/analyze \
    --output_dir data/experiment_results \
    --models_dir /home/linkco/exa/models

# Step 3: Run basis sensitivity for 3 representative additional models
echo ""
echo "[Step 3] Running basis sensitivity for additional models..."
for model in bge-m3 instructor-large bart-base; do
    echo "  Running basis sensitivity for $model..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON src/basis_sensitivity.py --model $model --gpu 0
done

# Step 4: Generate updated figures
echo ""
echo "[Step 4] Generating updated paper figures..."
$PYTHON src/redesign_figures.py

# Step 5: Compute per-task results for all models
echo ""
echo "[Step 5] Computing per-task results tables..."
$PYTHON -c "
import json, os, numpy as np
from collections import defaultdict

analyze_dir = 'data/analyze'
results = {}

for fname in sorted(os.listdir(analyze_dir)):
    if not fname.endswith('.json'):
        continue
    model = fname.replace('.json', '')
    with open(os.path.join(analyze_dir, fname)) as f:
        d = json.load(f)

    model_results = {}
    for task, task_data in d.get('task_name', {}).items():
        default = task_data.get('defult_score', 0)
        if default <= 0:
            continue
        # Get retention at dim=256
        random_scores = task_data.get('random_score', {}).get('256', [])
        sort_score = task_data.get('sort_score', {}).get('256', 0)
        if random_scores:
            random_ret = np.mean(random_scores) / default * 100
            sort_ret = sort_score / default * 100
            model_results[task] = {
                'default': default,
                'random_ret': random_ret,
                'sort_ret': sort_ret,
            }
    if model_results:
        results[model] = model_results

with open('data/experiment_results/all_models_per_task.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Saved per-task results for {len(results)} models')
for model, tasks in results.items():
    mean_ret = np.mean([t['random_ret'] for t in tasks.values()])
    print(f'  {model}: {len(tasks)} tasks, mean random retention = {mean_ret:.2f}%')
"

echo ""
echo "============================================"
echo "Post-processing pipeline COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Run: /auto-review-loop for Round 2 review"
echo "2. Check data/experiment_results/all_models_entropy.json"
echo "3. Check data/experiment_results/magnitude_analysis.json"
echo "4. Regenerate LaTeX tables from updated data"
