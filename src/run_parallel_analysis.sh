#!/bin/bash
# Parallel analysis: 4 models across 3 GPUs
# gte-base: 4 workers on GPU 0 (~2.9GB each, total ~11.6GB)
# gtr-t5-large: 4 workers on GPU 1 (~3.7GB each, total ~14.8GB)
# jina-v3: 1 worker on GPU 2 (~10GB, only 14GB free)
# gte-Qwen2: 2 workers on GPU 0 after gte-base finishes

PYTHON=/home/linkco/anaconda3/envs/llama/bin/python
SCRIPT=src/fast_chunk_analysis.py
MODELS=/home/linkco/exa/models
LOGDIR=/tmp/parallel_analyze

mkdir -p $LOGDIR

TOTAL=36

echo "=== Starting parallel analysis ==="
date

# --- GPU 0: gte-base (4 workers) ---
echo "Launching gte-base on GPU 0 (4 workers)..."
N=4
PER=$((TOTAL / N))
for i in $(seq 0 $((N-1))); do
    s=$((i * PER))
    e=$(((i+1) * PER))
    CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT $MODELS/gte-base --device cuda:0 \
        --task-range ${s}:${e} --worker-id ${i} > $LOGDIR/gte-base_w${i}.log 2>&1 &
    echo "  Worker $i: tasks $s-$e (PID $!)"
done

# --- GPU 1: gtr-t5-large (4 workers) ---
echo "Launching gtr-t5-large on GPU 1 (4 workers)..."
for i in $(seq 0 $((N-1))); do
    s=$((i * PER))
    e=$(((i+1) * PER))
    CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT $MODELS/gtr-t5-large --device cuda:0 \
        --task-range ${s}:${e} --worker-id ${i} > $LOGDIR/gtr-t5-large_w${i}.log 2>&1 &
    echo "  Worker $i: tasks $s-$e (PID $!)"
done

# --- GPU 2: jina-v3 (1 worker, not enough VRAM for 2) ---
echo "Launching jina-embeddings-v3 on GPU 2 (1 worker, sequential)..."
CUDA_VISIBLE_DEVICES=2 $PYTHON $SCRIPT $MODELS/jina-embeddings-v3 --device cuda:0 \
    --task-range 0:${TOTAL} > $LOGDIR/jina-v3.log 2>&1 &
echo "  Worker: all 36 tasks (PID $!)"

# --- GPU 0 (after gte-base): gte-Qwen2 (2 workers) ---
echo "Will launch gte-Qwen2-1.5B-instruct on GPU 0 after gte-base finishes..."
(
    while pgrep -f "gte-base.*worker" > /dev/null 2>&1; do sleep 30; done
    echo "$(date): gte-base done, starting gte-Qwen2 on GPU 0..."
    N_QWEN=2
    PER_QWEN=$((TOTAL / N_QWEN))
    for i in $(seq 0 $((N_QWEN-1))); do
        s=$((i * PER_QWEN))
        e=$(((i+1) * PER_QWEN))
        [ $i -eq $((N_QWEN-1)) ] && e=$TOTAL
        CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT $MODELS/gte-Qwen2-1.5B-instruct --device cuda:0 \
            --task-range ${s}:${e} --worker-id ${i} > $LOGDIR/gte-Qwen2_w${i}.log 2>&1 &
        echo "  Qwen2 Worker $i: tasks $s-$e (PID $!)"
    done
    wait
    echo "$(date): gte-Qwen2 done!"
) &

echo
echo "All workers launched. Logs: $LOGDIR/"
echo

# Wait for everything
wait

echo
echo "=== All models done! Merging results... ==="
date
$PYTHON src/merge_analysis.py
echo "=== Done! ==="
