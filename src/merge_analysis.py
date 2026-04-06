"""Merge parallel worker results into single model JSON files."""

import json
import sys
import os

analyze_path = "/home/linkco/exa/llm-usefulEeb/data/analyze"

def merge_model(model_name):
    """Merge all worker files for a model into one."""
    final_file = os.path.join(analyze_path, f"{model_name}.json")
    merged = {"model_name": model_name, "model_dim": 0, "task_name": {}}

    # Find all worker files for this model
    worker_files = sorted([
        f for f in os.listdir(analyze_path)
        if f.startswith(model_name + "_") and f.endswith(".json")
    ])

    if not worker_files:
        print(f"No worker files found for {model_name}")
        return

    for wf in worker_files:
        path = os.path.join(analyze_path, wf)
        with open(path) as f:
            data = json.load(f)
        merged["model_dim"] = data.get("model_dim", merged["model_dim"])
        merged["task_name"].update(data.get("task_name", {}))
        print(f"  {wf}: {len(data.get('task_name', {}))} tasks")

    with open(final_file, 'w') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"  => {final_file}: {len(merged['task_name'])} tasks total")

    # Remove worker files
    for wf in worker_files:
        os.remove(os.path.join(analyze_path, wf))
        print(f"  Removed {wf}")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else [
        "gte-base", "gtr-t5-large", "jina-embeddings-v3", "gte-Qwen2-1.5B-instruct"
    ]
    for m in models:
        print(f"\nMerging {m}:")
        merge_model(m)
