#!/bin/bash
cd /home/linkco/exa/llm-usefulEeb
export PYTHONUNBUFFERED=1
/home/linkco/anaconda3/envs/llama/bin/python -u src/run_missing_tasks.py --device cuda:2 --model mxbai-embed-large-v1
echo "=== mxbai done ==="
echo "=== GPU 2 ALL DONE ==="
