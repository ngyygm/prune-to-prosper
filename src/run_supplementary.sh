#!/bin/bash
# Supplementary runs: fill missing tasks for 4 existing models
# These are small models that fit alongside the main runs

PYTHON=/home/linkco/anaconda3/envs/llama/bin/python
SCRIPT=src/fast_chunk_analysis.py
MODELS=/home/linkco/exa/models
LOGDIR=/tmp/parallel_analyze

echo "=== Starting supplementary runs ==="
date

# bart-base (1 task) → GPU 0 (16GB free, model ~2GB)
echo "bart-base: MindSmallReranking"
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT $MODELS/bart-base --device cuda:0 \
    --tasks MindSmallReranking --worker-id bart \
    > $LOGDIR/bart-base_supp.log 2>&1 &
PIDS="$!"

# roberta-large (4 tasks) → GPU 0
echo "roberta-large: 4 Retrieval tasks"
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT $MODELS/roberta-large --device cuda:0 \
    --tasks MindSmallReranking,NFCorpus,SCIDOCS,SciFact --worker-id roberta \
    > $LOGDIR/roberta-large_supp.log 2>&1 &
PIDS="$PIDS $!"

# Qwen3-Embedding-0.6B (5 tasks) → GPU 1 (11GB free, model ~1.5GB)
echo "Qwen3-Embedding-0.6B: 5 Retrieval tasks"
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT $MODELS/Qwen3-Embedding-0.6B --device cuda:0 \
    --tasks MindSmallReranking,CQADupstackEnglishRetrieval,NFCorpus,SCIDOCS,SciFact --worker-id qwen3 \
    > $LOGDIR/Qwen3-Embedding-0.6B_supp.log 2>&1 &
PIDS="$PIDS $!"

# mxbai-embed-large-v1 (8 tasks) → GPU 1
echo "mxbai-embed-large-v1: 8 tasks"
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT $MODELS/mxbai-embed-large-v1 --device cuda:0 \
    --tasks SummEval,MindSmallReranking,STSBenchmark,ArguAna,CQADupstackEnglishRetrieval,NFCorpus,SCIDOCS,SciFact --worker-id mxbai \
    > $LOGDIR/mxbai-embed-large-v1_supp.log 2>&1 &
PIDS="$PIDS $!"

echo "PIDs: $PIDS"
echo "Logs: $LOGDIR/*_supp.log"
echo

wait $PIDS

echo
echo "=== Supplementary runs done! Merging... ==="
date

# Merge supplementary results into main files
$PYTHON -c "
import json, os, sys
sys.path.insert(0, '.')
from merge_analysis import merge_model
for m in ['bart-base', 'roberta-large', 'Qwen3-Embedding-0.6B', 'mxbai-embed-large-v1']:
    merge_model(m)
"
echo "=== Done! ==="
