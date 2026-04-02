"""
Compute entropy and uniformity metrics for ALL models from chunk importance data.
This script reads data/analyze/*.json files and computes:
- Normalized Shannon entropy
- Gini coefficient
- Chunk importance coefficient of variation (CV)
- Top-50% concentration

Run AFTER rank_chunk_mteb.py completes for all models.
"""

import json
import os
import numpy as np
# Gini is not in scipy, compute inline


def compute_entropy(chunk_scores):
    """Compute normalized Shannon entropy from chunk importance scores.
    Handles negative scores by shifting to positive range first.
    """
    arr = np.array(chunk_scores, dtype=float)
    # Shift to positive range if needed
    if np.min(arr) < 0:
        arr = arr - np.min(arr) + 1e-10
    total = np.sum(arr)
    if total <= 0:
        return 1.0
    probs = arr / total
    # Filter out zero/near-zero probabilities
    probs = probs[probs > 1e-15]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(arr))
    return float(entropy / max_entropy) if max_entropy > 0 else 1.0


def compute_gini(chunk_scores):
    """Compute Gini coefficient. Handles negative scores by shifting."""
    arr = np.array(chunk_scores, dtype=float)
    if np.min(arr) < 0:
        arr = arr - np.min(arr) + 1e-10
    arr = np.sort(arr)
    n = len(arr)
    total = np.sum(arr)
    if total <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * total))


def compute_cv(chunk_scores):
    """Compute coefficient of variation."""
    arr = np.array(chunk_scores, dtype=float)
    return float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0.0


def compute_top_k_concentration(chunk_scores, k_fraction=0.5):
    """Compute fraction of total importance captured by top-k% of chunks."""
    sorted_scores = sorted(chunk_scores, reverse=True)
    n = len(sorted_scores)
    k = max(1, int(n * k_fraction))
    return sum(sorted_scores[:k]) / sum(sorted_scores)


def analyze_model(model_data):
    """Analyze a single model's chunk importance data."""
    results = {
        "model_name": model_data.get("model_name", "unknown"),
        "model_dim": model_data.get("model_dim", 0),
        "tasks": {},
        "summary": {}
    }

    entropies = []
    ginis = []
    cvs = []
    top50s = []

    task_data = model_data.get("task_name", {})
    for task_name, task_info in task_data.items():
        # Get chunk results from win_size=2 (most granular)
        split_ws = task_info.get("split_win_size", {})
        if "2" in split_ws:
            chunk_scores = split_ws["2"].get("chunk_result", [])
        elif split_ws:
            # Fall back to first available win_size
            first_ws = list(split_ws.keys())[0]
            chunk_scores = split_ws[first_ws].get("chunk_result", [])
        else:
            continue

        if not chunk_scores or len(chunk_scores) < 2:
            continue

        # Compute metrics
        ent = compute_entropy(chunk_scores)
        gin = compute_gini(chunk_scores)
        cv = compute_cv(chunk_scores)
        top50 = compute_top_k_concentration(chunk_scores)

        results["tasks"][task_name] = {
            "normalized_entropy": ent,
            "gini_coefficient": gin,
            "chunk_importance_cv": cv,
            "top_50pct_concentration": top50,
            "n_chunks": len(chunk_scores)
        }

        entropies.append(ent)
        ginis.append(gin)
        cvs.append(cv)
        top50s.append(top50)

    if entropies:
        results["summary"] = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "min_entropy": float(np.min(entropies)),
            "max_entropy": float(np.max(entropies)),
            "mean_gini": float(np.mean(ginis)),
            "mean_cv": float(np.mean(cvs)),
            "mean_top50": float(np.mean(top50s)),
            "n_tasks": len(entropies)
        }

    return results


def main():
    analyze_dir = "/home/linkco/exa/llm-usefulEeb/data/analyze"
    output_path = "/home/linkco/exa/llm-usefulEeb/data/experiment_results/all_models_entropy.json"

    all_results = {}

    # Load all model data
    model_files = sorted([f for f in os.listdir(analyze_dir) if f.endswith('.json')])
    print(f"Found {len(model_files)} model files in {analyze_dir}")

    for fname in model_files:
        model_name = fname.replace('.json', '')
        with open(os.path.join(analyze_dir, fname)) as f:
            model_data = json.load(f)

        result = analyze_model(model_data)
        all_results[model_name] = result

        if result["summary"]:
            s = result["summary"]
            print(f"  {model_name}: entropy={s['mean_entropy']:.4f} ± {s['std_entropy']:.4f}, "
                  f"gini={s['mean_gini']:.4f}, cv={s['mean_cv']:.4f}, "
                  f"top50={s['mean_top50']:.4f}, n_tasks={s['n_tasks']}")
        else:
            print(f"  {model_name}: NO DATA (no chunk results found)")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Model':<25} {'Entropy':>10} {'Gini':>10} {'CV':>10} {'Top50%':>10} {'Tasks':>6}")
    print("-"*80)
    for model_name, r in all_results.items():
        if r["summary"]:
            s = r["summary"]
            print(f"{model_name:<25} {s['mean_entropy']:>10.4f} {s['mean_gini']:>10.4f} "
                  f"{s['mean_cv']:>10.4f} {s['mean_top50']:>10.4f} {s['n_tasks']:>6}")


if __name__ == "__main__":
    main()
