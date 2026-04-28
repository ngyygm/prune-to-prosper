#!/bin/bash
# Master launch script for extended revision experiments.
# Distributes across 3 GPUs for maximum parallelism.
# All scripts reuse EmbeddingCache — encode once per model+task.

set -e
cd /home/linkco/exa/llm-usefulEeb

PYTHON=/home/linkco/anaconda3/envs/llama/bin/python
LOGDIR=logs
mkdir -p $LOGDIR

echo "=== Launching extended revision experiments ==="
echo "=== $(date) ==="

# ─── GPU 0: Chunk size sweep w=1,16,32 + LOO extended ───
echo "[GPU0] Chunk size sweep w=1,16,32 + LOO extended"

tmux new-session -d -s sweep_ext_gpu0 "bash -c '
    $PYTHON src/chunk_size_sweep_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 roberta-large roberta-large-InBedder bge-m3 \
        --win-sizes 1 16 32 \
        --budgets 128 256 \
        --n-random 10 \
        --device cuda:0 \
        2>&1 | tee $LOGDIR/sweep_ext_gpu0.log

    echo \"--- sweep done, starting LOO extended ---\"

    $PYTHON src/loo_extended_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 \
        --n-marginal 50 --n-shapley 64 \
        --device cuda:0 \
        2>&1 | tee -a $LOGDIR/sweep_ext_gpu0.log
'"
echo "  tmux attach -t sweep_ext_gpu0"

# ─── GPU 1: Learnable mask (supervised) + LOO extended (remaining models) ───
echo "[GPU1] Learnable mask + LOO extended"

tmux new-session -d -s mask_gpu1 "bash -c '
    $PYTHON src/learnable_mask_supervised.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 roberta-large roberta-large-InBedder \
        --budgets 64 128 256 \
        --device cuda:1 \
        2>&1 | tee $LOGDIR/mask_gpu1.log

    echo \"--- mask done, starting LOO extended ---\"

    $PYTHON src/loo_extended_fast.py \
        --models roberta-large roberta-large-InBedder \
        --n-marginal 50 --n-shapley 64 \
        --device cuda:1 \
        2>&1 | tee -a $LOGDIR/mask_gpu1.log
'"
echo "  tmux attach -t mask_gpu1"

# ─── GPU 2: ANN benchmark + Learnable projection ───
echo "[GPU2] ANN benchmark + Learnable projection"

tmux new-session -d -s ann_gpu2 "bash -c '
    $PYTHON src/retrieval_ann_benchmark.py \
        --models gte-large-en-v1.5 bge-m3 stella_en_400M_v5 \
        --target-dims 64 128 256 \
        --device cuda:2 \
        2>&1 | tee $LOGDIR/ann_gpu2.log

    echo \"--- ANN done, starting learnable projection ---\"

    $PYTHON src/learnable_projection_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 roberta-large \
        --target-dims 64 128 256 \
        --device cuda:2 \
        2>&1 | tee -a $LOGDIR/ann_gpu2.log
'"
echo "  tmux attach -t ann_gpu2"

echo ""
echo "=== All experiments launched ==="
echo "GPU 0: chunk_sweep(w=1,16,32) + loo_extended(GTE,Stella)"
echo "GPU 1: learnable_mask(all 4) + loo_extended(Roberta,InBedder)"
echo "GPU 2: ann_benchmark(3 models) + learnable_projection(3 models)"
echo ""
echo "Monitor: tail -f $LOGDIR/sweep_ext_gpu0.log"
echo "         tail -f $LOGDIR/mask_gpu1.log"
echo "         tail -f $LOGDIR/ann_gpu2.log"
