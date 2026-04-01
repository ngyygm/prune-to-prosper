#!/usr/bin/env python3
"""
Round 3 Deep Analysis: Roberta-InBedder mechanism + Magnitude failure diagnosis.

This script produces:
1. Roberta-InBedder boundary case analysis:
   - Compare entropy distribution across all 3 models
   - Analyze chunk importance variance and concentration
   - Identify which tasks show the largest optimized-random gap
   - Compare magnitude CV (coefficient of variation) across models

2. Magnitude failure diagnosis:
   - For GTE-Large, identify dimensions that magnitude removes but are task-important
   - Compute overlap between top-k magnitude chunks and top-k task-important chunks
   - Analyze per-category magnitude failure patterns
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats

DATA_DIR = "/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/analyze"
OUTPUT_DIR = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output"

MODELS = ["gte-large-en-v1.5", "stella_en_400M_v5", "roberta-large-InBedder"]
MODEL_NAMES = {
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella",
    "roberta-large-InBedder": "Roberta-InBedder",
}

def load_analyze_data(model):
    with open(f"{DATA_DIR}/{model}.json") as f:
        raw = json.load(f)
    # Extract the task dict (key that is a dict and not 'model_name')
    for k, v in raw.items():
        if isinstance(v, dict) and len(v) > 5:
            return v
    return raw

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_chunk_importance(data, win_size=2):
    """Compute per-chunk importance scores averaged across all tasks."""
    model_dim = 1024
    n_chunks = model_dim // win_size  # 512 for win_size=2

    # Get all task names
    task_names = [t for t in data.keys() if isinstance(data[t], dict)]

    # Filter to only task dicts
    task_names = [t for t in data.keys() if isinstance(data[t], dict)]

    chunk_scores_all_tasks = np.zeros((len(task_names), n_chunks))

    for i, task in enumerate(task_names):
        task_data = data[task]
        if not isinstance(task_data, dict) or "split_win_size" not in task_data:
            continue
        ws_key = str(win_size)
        if ws_key not in task_data["split_win_size"]:
            continue
        chunk_data = task_data["split_win_size"][ws_key]
        if "chunk_result" in chunk_data:
            cr = chunk_data["chunk_result"]
            if isinstance(cr, list):
                for chunk_id, score in enumerate(cr):
                    chunk_scores_all_tasks[i, chunk_id] = score
            elif isinstance(cr, dict):
                for chunk_id, score in cr.items():
                    chunk_scores_all_tasks[i, int(chunk_id)] = score

    return chunk_scores_all_tasks, task_names


def compute_entropy(chunk_scores):
    """Compute normalized Shannon entropy of importance distribution."""
    scores = np.abs(chunk_scores) + 1e-10
    probs = scores / scores.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))
    return entropy / max_entropy


def analyze_inbedder_mechanism():
    """Deep analysis of why Roberta-InBedder is a boundary case."""
    print("=" * 60)
    print("ANALYSIS 1: Roberta-InBedder Boundary Case Mechanism")
    print("=" * 60)

    results = {}

    for model in MODELS:
        data = load_analyze_data(model)
        chunk_scores_all, task_names = compute_chunk_importance(data, win_size=2)

        # Average chunk importance across tasks
        mean_chunk_scores = np.mean(chunk_scores_all, axis=0)
        std_chunk_scores = np.std(chunk_scores_all, axis=0)

        # Entropy per task
        entropies = []
        for i in range(len(task_names)):
            e = compute_entropy(chunk_scores_all[i])
            entropies.append(e)

        # Cross-task agreement: how much do tasks agree on chunk importance?
        # Use std of chunk scores across tasks (normalized by mean)
        cv_per_chunk = std_chunk_scores / (np.abs(mean_chunk_scores) + 1e-10)

        # Top-k concentration: what fraction of total score is in top-k chunks?
        sorted_scores = np.sort(mean_chunk_scores)[::-1]
        total = np.sum(np.abs(mean_chunk_scores))
        top_10pct = np.sum(sorted_scores[:51]) / total  # top 10% of 512
        top_25pct = np.sum(sorted_scores[:128]) / total
        top_50pct = np.sum(sorted_scores[:256]) / total

        # Gini coefficient (inequality measure)
        abs_scores = np.sort(np.abs(mean_chunk_scores))
        n = len(abs_scores)
        gini = 2 * np.sum((np.arange(1, n+1) - (n+1)/2) * abs_scores) / (n * np.sum(abs_scores))

        # Coefficient of variation of chunk scores
        chunk_cv = np.std(mean_chunk_scores) / (np.abs(np.mean(mean_chunk_scores)) + 1e-10)

        # Optimized-random gap per task
        opt_random_gaps = []
        for i, task in enumerate(task_names):
            task_data = data[task]
            if not isinstance(task_data, dict) or "split_win_size" not in task_data:
                continue
            ws = task_data["split_win_size"].get("2", {})
            if not ws:
                continue

            # Best (optimized) retention
            best_chunks = np.argsort(chunk_scores_all[i])[::-1][:128]  # top 128 chunks for dim=256
            cr = ws.get("chunk_result", [])
            if isinstance(cr, list):
                best_score = np.mean([cr[c] for c in best_chunks if c < len(cr)])
            else:
                best_score = np.mean([cr.get(str(c), 0) for c in best_chunks])

            # Random retention (mean of 10 trials)
            random_scores = task_data.get("random_score", {}).get("256", [])
            if random_scores:
                random_mean = np.mean(random_scores)
                baseline = task_data.get("defult_score", 1.0)
                if baseline > 0:
                    opt_ret = best_score / baseline
                    rnd_ret = random_mean / baseline
                    gap = opt_ret - rnd_ret
                    opt_random_gaps.append((task, gap, opt_ret, rnd_ret))

        results[model] = {
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies),
            "min_entropy": np.min(entropies),
            "max_entropy": np.max(entropies),
            "gini_coefficient": gini,
            "chunk_cv": chunk_cv,
            "top_10pct_concentration": top_10pct,
            "top_25pct_concentration": top_25pct,
            "top_50pct_concentration": top_50pct,
            "cross_task_cv_mean": np.mean(cv_per_chunk),
            "cross_task_cv_std": np.std(cv_per_chunk),
            "opt_random_gaps": sorted(opt_random_gaps, key=lambda x: x[1], reverse=True),
            "n_tasks_with_positive_gap": sum(1 for _, g, _, _ in opt_random_gaps if g > 0),
            "mean_opt_random_gap": np.mean([g for _, g, _, _ in opt_random_gaps]) if opt_random_gaps else 0,
        }

        print(f"\n--- {MODEL_NAMES[model]} ---")
        print(f"  Mean entropy: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
        print(f"  Entropy range: [{np.min(entropies):.4f}, {np.max(entropies):.4f}]")
        print(f"  Gini coefficient: {gini:.4f}")
        print(f"  Chunk score CV: {chunk_cv:.4f}")
        print(f"  Top 10% concentration: {top_10pct:.4f}")
        print(f"  Top 25% concentration: {top_25pct:.4f}")
        print(f"  Top 50% concentration: {top_50pct:.4f}")
        print(f"  Cross-task CV (mean): {np.mean(cv_per_chunk):.4f}")
        print(f"  Optimized-random gap: {np.mean([g for _, g, _, _ in opt_random_gaps]):.4f}")
        print(f"  Tasks with positive gap: {sum(1 for _, g, _, _ in opt_random_gaps if g > 0)}/{len(opt_random_gaps)}")

        # Top 5 tasks with largest optimized-random gap
        print(f"  Top 5 tasks by optimized-random gap:")
        for task, gap, opt_ret, rnd_ret in opt_random_gaps[:5]:
            print(f"    {task}: gap={gap:.4f}, opt={opt_ret:.4f}, rnd={rnd_ret:.4f}")

    # Comparative analysis
    print("\n\n--- Comparative Analysis ---")
    for metric in ["mean_entropy", "gini_coefficient", "chunk_cv", "top_50pct_concentration",
                   "mean_opt_random_gap", "cross_task_cv_mean"]:
        print(f"\n  {metric}:")
        for model in MODELS:
            print(f"    {MODEL_NAMES[model]}: {results[model][metric]:.4f}")

    # What distinguishes InBedder?
    print("\n\n--- What distinguishes Roberta-InBedder? ---")
    ib = results["roberta-large-InBedder"]
    gt = results["gte-large-en-v1.5"]
    st = results["stella_en_400M_v5"]

    # Compare entropy
    avg_other_entropy = (gt["mean_entropy"] + st["mean_entropy"]) / 2
    print(f"  Entropy: InBedder ({ib['mean_entropy']:.4f}) vs avg others ({avg_other_entropy:.4f}), diff={ib['mean_entropy']-avg_other_entropy:.4f}")
    print(f"  Gini: InBedder ({ib['gini_coefficient']:.4f}) vs avg others ({(gt['gini_coefficient']+st['gini_coefficient'])/2:.4f})")
    print(f"  Chunk CV: InBedder ({ib['chunk_cv']:.4f}) vs avg others ({(gt['chunk_cv']+st['chunk_cv'])/2:.4f})")
    print(f"  Cross-task CV: InBedder ({ib['cross_task_cv_mean']:.4f}) vs avg others ({(gt['cross_task_cv_mean']+st['cross_task_cv_mean'])/2:.4f})")
    print(f"  Top-50% concentration: InBedder ({ib['top_50pct_concentration']:.4f}) vs avg others ({(gt['top_50pct_concentration']+st['top_50pct_concentration'])/2:.4f})")

    return results


def analyze_magnitude_failure():
    """Diagnose why magnitude-based pruning fails for GTE-Large."""
    print("\n\n" + "=" * 60)
    print("ANALYSIS 2: Magnitude Failure Diagnosis")
    print("=" * 60)

    mag_data = load_json(f"{OUTPUT_DIR}/magnitude_analysis.json")

    for model in MODELS:
        print(f"\n--- {MODEL_NAMES[model]} ---")

        mag_info = mag_data["magnitude_rankings"][model]
        mag_ranking = mag_info["chunk_ranking"]  # chunks sorted by magnitude (highest first)
        norm_cv = mag_info["norm_cv"]

        # Load task importance ranking
        data = load_analyze_data(model)
        chunk_scores_all, task_names = compute_chunk_importance(data, win_size=2)

        # Average task importance across all tasks
        mean_chunk_scores = np.mean(chunk_scores_all, axis=0)
        task_ranking = np.argsort(mean_chunk_scores)[::-1].tolist()  # highest importance first

        # Overlap analysis: how many of top-k magnitude chunks are also top-k task-important?
        for k in [64, 128, 256]:
            mag_top_k = set(mag_ranking[:k])
            task_top_k = set(task_ranking[:k])
            overlap = len(mag_top_k & task_top_k)
            jaccard = overlap / len(mag_top_k | task_top_k)
            print(f"  Top-{k} overlap: {overlap}/{k} chunks ({100*overlap/k:.1f}%), Jaccard={jaccard:.4f}")

        # Rank correlation between magnitude and task importance
        # Create full ranking arrays
        mag_rank_array = np.zeros(512)
        task_rank_array = np.zeros(512)
        for rank, chunk_id in enumerate(mag_ranking):
            mag_rank_array[chunk_id] = rank
        for rank, chunk_id in enumerate(task_ranking):
            task_rank_array[chunk_id] = rank

        rho, p_val = stats.spearmanr(mag_rank_array, task_rank_array)
        print(f"  Magnitude vs Task-Importance correlation: rho={rho:.4f}, p={p_val:.4f}")

        # Per-category analysis: which categories does magnitude hurt most?
        categories = {
            "Classification": [],
            "Clustering": [],
            "Retrieval": [],
            "Reranking": [],
            "STS": [],
        }

        for i, task in enumerate(task_names):
            task_data = data[task]
            if not isinstance(task_data, dict) or "split_win_size" not in task_data:
                continue
            ws = task_data["split_win_size"].get("2", {})
            if not ws:
                continue

            # Task-specific importance ranking
            task_importance = chunk_scores_all[i]
            task_ranking_i = np.argsort(task_importance)[::-1].tolist()

            # Magnitude retention: score using top-128 magnitude chunks
            mag_top_128 = set(mag_ranking[:128])
            mag_score = np.mean([task_importance[c] for c in mag_top_128 if c < len(task_importance)])

            # Random baseline
            random_scores = task_data.get("random_score", {}).get("256", [])
            if random_scores:
                random_mean = np.mean(random_scores)
                baseline = task_data.get("defult_score", 1.0)
                if baseline > 0:
                    mag_ret = mag_score / baseline
                    rnd_ret = random_mean / baseline
                    diff = mag_ret - rnd_ret

                    # Categorize task
                    for cat in categories:
                        if cat.lower() in task.lower():
                            categories[cat].append(diff)
                            break

        print(f"\n  Per-category magnitude-random difference:")
        for cat, diffs in categories.items():
            if diffs:
                mean_diff = np.mean(diffs)
                print(f"    {cat}: mean={mean_diff:.4f}, n={len(diffs)}")

        # Spatial analysis: where are the high-magnitude chunks?
        print(f"\n  Spatial distribution of top-128 magnitude chunks:")
        chunk_positions = np.array(mag_ranking[:128])
        print(f"    Mean position: {np.mean(chunk_positions):.1f} (of 0-511)")
        print(f"    Std position: {np.std(chunk_positions):.1f}")
        print(f"    In first half: {np.sum(chunk_positions < 256)}/128")
        print(f"    In last half: {np.sum(chunk_positions >= 256)}/128")

    return mag_data


def analyze_pruning_ratio_synthesis():
    """Synthesize findings across all pruning ratios."""
    print("\n\n" + "=" * 60)
    print("ANALYSIS 3: Pruning Ratio Synthesis")
    print("=" * 60)

    reviewer_data = load_json(f"{OUTPUT_DIR}/reviewer_response_analysis.json")

    for model in MODELS:
        print(f"\n--- {MODEL_NAMES[model]} ---")

        model_data = reviewer_data.get("pruning_ratio_sweep", {}).get(model, {})
        if not model_data:
            print("  No pruning ratio data found.")
            continue

        ratios = model_data.get("ratios", {})
        for ratio_str in sorted(ratios.keys(), key=float):
            r = ratios[ratio_str]
            sort_ret = r.get("avg_sort", 0)
            rnd_ret = r.get("avg_random", 0)
            dims_val = r.get("dims", [0])
            dims = dims_val[0] if isinstance(dims_val, list) else dims_val
            gap = sort_ret - rnd_ret
            ratio_pct = float(ratio_str) * 100
            if sort_ret > 0:
                print(f"  dims={dims:>4} ({ratio_pct:>5.1f}% prune): sort={sort_ret:.4f}, random={rnd_ret:.4f}, gap={gap:+.4f}")

    return reviewer_data


if __name__ == "__main__":
    inbedder_results = analyze_inbedder_mechanism()
    mag_results = analyze_magnitude_failure()
    ratio_results = analyze_pruning_ratio_synthesis()

    # Save combined results
    output = {
        "inbedder_mechanism": {},
        "magnitude_diagnosis": {},
    }

    # Convert numpy types for JSON serialization
    for model in MODELS:
        name = MODEL_NAMES[model]
        r = inbedder_results[model]
        output["inbedder_mechanism"][name] = {
            "mean_entropy": float(r["mean_entropy"]),
            "std_entropy": float(r["std_entropy"]),
            "min_entropy": float(r["min_entropy"]),
            "max_entropy": float(r["max_entropy"]),
            "gini_coefficient": float(r["gini_coefficient"]),
            "chunk_cv": float(r["chunk_cv"]),
            "top_10pct_concentration": float(r["top_10pct_concentration"]),
            "top_25pct_concentration": float(r["top_25pct_concentration"]),
            "top_50pct_concentration": float(r["top_50pct_concentration"]),
            "cross_task_cv_mean": float(r["cross_task_cv_mean"]),
            "mean_opt_random_gap": float(r["mean_opt_random_gap"]),
            "n_tasks_with_positive_gap": int(r["n_tasks_with_positive_gap"]),
            "top_gap_tasks": [(t, float(g), float(o), float(r_)) for t, g, o, r_ in r["opt_random_gaps"][:10]],
        }

    with open(f"{OUTPUT_DIR}/round3_deep_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n\nResults saved to round3_deep_analysis.json")
