#!/bin/bash
cd /home/linkco/exa/llm-usefulEeb
export PYTHONUNBUFFERED=1
/home/linkco/anaconda3/envs/llama/bin/python -u src/run_missing_tasks.py --device cuda:0 --model roberta-large
echo "=== roberta-large done ==="
echo "=== GPU 0 ALL DONE ==="
