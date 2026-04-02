#!/bin/bash
# Run magnitude pruning for all models (Part 1: no MTEB, just analysis)
# This should run AFTER rank_chunk_mteb.py completes for all models
cd /home/linkco/exa/llm-usefulEeb

echo "=== Running magnitude pruning analysis for all models ==="
/home/linkco/anaconda3/envs/llama/bin/python src/magnitude_pruning.py \
    --analyze_dir data/analyze \
    --output_dir data/experiment_results \
    --models_dir /home/linkco/exa/models

echo "=== Running entropy computation for all models ==="
/home/linkco/anaconda3/envs/llama/bin/python src/compute_all_model_entropy.py

echo "=== Running basis sensitivity for additional models ==="
# Run for representative models spanning different families
for model in bge-m3 instructor-large bart-base; do
    echo "  Running basis sensitivity for $model"
    CUDA_VISIBLE_DEVICES=0 /home/linkco/anaconda3/envs/llama/bin/python src/basis_sensitivity.py --model $model --gpu 0
done

echo "=== All post-experiment analysis complete ==="
