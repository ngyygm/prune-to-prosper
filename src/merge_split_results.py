#!/usr/bin/env python3
"""
Merge partial train/test split experiment results from multiple GPU workers.

Usage:
    python src/merge_split_results.py data/experiment_results/train_test_split_<model>.json
    # Or to merge multiple partial files:
    python src/merge_split_results.py partial_0.json partial_1.json partial_2.json -o merged.json
"""

import json
import sys
import argparse
from collections import OrderedDict


def merge_results(files, output_path):
    """Merge multiple result files by combining task results."""
    merged = None
    total_tasks = 0

    for fpath in files:
        print(f"Reading {fpath}...")
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if merged is None:
            merged = {
                "config": data.get("config", {}),
                "model_name": data.get("model_name", ""),
                "model_dim": data.get("model_dim", 0),
                "tasks": {},
            }
            # Take tasks from first file
            merged["tasks"].update(data.get("tasks", {}))
            total_tasks += len(data.get("tasks", {}))
        else:
            # Merge tasks from subsequent files
            for task_name, task_data in data.get("tasks", {}).items():
                if task_name not in merged["tasks"]:
                    merged["tasks"][task_name] = task_data
                    total_tasks += 1
                else:
                    # Task already exists — check if it has more data
                    existing = merged["tasks"][task_name]
                    if "error" in existing and "error" not in task_data:
                        merged["tasks"][task_name] = task_data
                    elif len(task_data.get("budgets", {})) > len(existing.get("budgets", {})):
                        merged["tasks"][task_name] = task_data

    if merged is None:
        print("No files to merge!")
        return

    # Compute summary
    from train_test_split_experiment import compute_summary
    merged["summary"] = compute_summary(merged["tasks"])

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Merged {total_tasks} tasks → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge partial results')
    parser.add_argument('files', nargs='+', help='JSON result files to merge')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    args = parser.parse_args()
    merge_results(args.files, args.output)
