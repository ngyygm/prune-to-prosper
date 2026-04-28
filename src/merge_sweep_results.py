#!/usr/bin/env python3
"""Merge chunk_size_sweep temp results with existing results."""
import json, os, sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/experiment_results'
temp_dir = os.path.join(output_dir, 'temp_sweep')

models = ['gte-large-en-v1.5', 'bge-m3', 'stella_en_400M_v5',
          'roberta-large-InBedder', 'roberta-large']

for model in models:
    orig_path = os.path.join(output_dir, f'chunk_size_sweep_{model}.json')
    temp_path = os.path.join(temp_dir, f'chunk_size_sweep_{model}.json')

    if not os.path.exists(temp_path):
        print(f"  No temp for {model}, skip")
        continue

    with open(temp_path) as f:
        temp = json.load(f)

    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig = json.load(f)
        # Merge win_sizes into orig
        for task_name, task_data in temp.get('tasks', {}).items():
            if task_name not in orig.get('tasks', {}):
                orig['tasks'][task_name] = task_data
            else:
                for w_key, w_data in task_data.get('win_sizes', {}).items():
                    orig['tasks'][task_name]['win_sizes'][w_key] = w_data
        # Update metadata
        orig_ws = set(orig.get('win_sizes', []))
        new_ws = set(temp.get('win_sizes', []))
        orig['win_sizes'] = sorted(orig_ws | new_ws)
        merged = orig
    else:
        merged = temp

    with open(orig_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"  Merged {model}: win_sizes={merged.get('win_sizes')}")

# Cleanup temp
import shutil
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    print(f"  Cleaned up {temp_dir}")
