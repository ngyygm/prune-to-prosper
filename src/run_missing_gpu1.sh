#!/bin/bash
cd /home/linkco/exa/llm-usefulEeb
export PYTHONUNBUFFERED=1
/home/linkco/anaconda3/envs/llama/bin/python -u src/run_missing_tasks.py --device cuda:1 --model Qwen3-Embedding-0.6B
echo "=== Qwen3 done ==="
echo "=== GPU 1 ALL DONE ==="
