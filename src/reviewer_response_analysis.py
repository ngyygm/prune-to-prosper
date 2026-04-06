"""
Additional analyses addressing reviewer feedback for "Prune to Prosper" paper.

Addresses:
1. Mechanistic explanation: why high transfer despite zero ranking correlation
2. Redundancy analysis: per-dimension variance and entropy
3. Pruning ratio sweep analysis (from existing data)
4. Top-K vs Bottom-K analysis at different pruning levels
5. Donor ranking uniformity analysis (why weak donors sometimes work)
"""

import os
import json
import numpy as np
from collections import defaultdict
from itertools import combinations


def load_analyze_data(analyze_dir):
    """Load all analyze JSON files."""
    data = {}
    for fname in os.listdir(analyze_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(analyze_dir, fname), "r") as f:
                data[model_name] = json.load(f)
    return data


def load_task_similar_data(task_similar_dir):
    """Load all task_similar JSON files."""
    data = {}
    for fname in os.listdir(task_similar_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(task_similar_dir, fname), "r") as f:
                data[model_name] = json.load(f)
    return data


def analyze_redundancy_mechanism(analyze_data):
    """
    Mechanistic explanation for the paradox: zero ranking correlation + high transfer.

    Hypothesis: Embeddings are highly redundant, so ANY subset of dimensions
    preserves most information. The key metric is not "which dimensions are important"
    (which varies by task) but "how many dimensions are needed" (which is universal).

    We test this by:
    1. Computing per-dimension variance across embeddings (redundancy proxy)
    2. Computing effective rank of the embedding matrix
    3. Analyzing chunk score variance (flat scores = high redundancy)
    4. Testing if "top-K vs bottom-K" gap narrows as K increases
    """
    results = {}

    for model_name, model_data in analyze_data.items():
        model_dim = model_data["model_dim"]
        results[model_name] = {"model_dim": model_dim, "tasks": {}}

        for task_name, task_data in model_data["task_name"].items():
            if "2" not in task_data.get("split_win_size", {}):
                continue

            chunk_scores = task_data["split_win_size"]["2"]["chunk_result"]
            chunk_scores = np.array(chunk_scores)
            n_chunks = len(chunk_scores)

            # 1. Chunk score statistics (flat distribution = high redundancy)
            score_mean = np.mean(chunk_scores)
            score_std = np.std(chunk_scores)
            score_cv = score_std / score_mean if score_mean != 0 else 0  # coefficient of variation
            score_range = np.max(chunk_scores) - np.min(chunk_scores)
            score_iqr = np.percentile(chunk_scores, 75) - np.percentile(chunk_scores, 25)

            # 2. Entropy of chunk importance distribution (higher = more uniform = more redundant)
            # Normalize scores to probabilities
            probs = np.abs(chunk_scores)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_chunks) / n_chunks
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(n_chunks)
            normalized_entropy = entropy / max_entropy  # 1 = perfectly uniform

            # 3. Concentration: what fraction of total score is in top-K chunks?
            sorted_scores = np.sort(chunk_scores)[::-1]
            cumsum = np.cumsum(sorted_scores)
            total = cumsum[-1]
            top_10_pct = cumsum[max(1, n_chunks // 10)] / total if total > 0 else 0
            top_25_pct = cumsum[max(1, n_chunks // 4)] / total if total > 0 else 0
            top_50_pct = cumsum[max(1, n_chunks // 2)] / total if total > 0 else 0

            # 4. Best vs Poor gap at different chunk counts (from split_win_size)
            gap_analysis = {}
            for ws_str, ws_data in task_data.get("split_win_size", {}).items():
                for target_dim_str, td_data in ws_data.get("chunk_win_size", {}).items():
                    head = td_data.get("head_score", {}).get("main_score", 0)
                    end = td_data.get("end_score", {}).get("main_score", 0)
                    default = task_data.get("defult_score", 1)
                    if default > 0 and head > 0 and end > 0:
                        gap_analysis[target_dim_str] = {
                            "best_retention": head / default,
                            "poor_retention": end / default,
                            "gap": (head - end) / default,
                            "n_chunks": int(target_dim_str) // int(ws_str) if int(ws_str) > 0 else 0,
                        }

            results[model_name]["tasks"][task_name] = {
                "chunk_score_cv": float(score_cv),
                "chunk_score_range": float(score_range),
                "chunk_score_iqr": float(score_iqr),
                "normalized_entropy": float(normalized_entropy),
                "top_10pct_concentration": float(top_10_pct),
                "top_25pct_concentration": float(top_25_pct),
                "top_50pct_concentration": float(top_50_pct),
                "n_chunks": n_chunks,
                "best_vs_poor_gaps": gap_analysis,
            }

        # Aggregate across tasks for this model
        task_results = results[model_name]["tasks"]
        if task_results:
            results[model_name]["model_summary"] = {
                "avg_normalized_entropy": float(np.mean([v["normalized_entropy"] for v in task_results.values()])),
                "avg_chunk_score_cv": float(np.mean([v["chunk_score_cv"] for v in task_results.values()])),
                "avg_top_10pct_concentration": float(np.mean([v["top_10pct_concentration"] for v in task_results.values()])),
                "avg_top_25pct_concentration": float(np.mean([v["top_25pct_concentration"] for v in task_results.values()])),
                "avg_top_50pct_concentration": float(np.mean([v["top_50pct_concentration"] for v in task_results.values()])),
            }

    return results


def analyze_pruning_ratio_sweep(analyze_data):
    """
    Analyze performance vs pruning ratio for different methods.
    Uses existing data to show how performance degrades as we prune more.
    """
    results = {}

    for model_name, model_data in analyze_data.items():
        model_dim = model_data["model_dim"]
        results[model_name] = {"model_dim": model_dim, "ratios": {}}

        for task_name, task_data in model_data["task_name"].items():
            default_score = task_data["defult_score"]
            if default_score <= 0:
                continue

            # Get sort scores at different dimensions
            sort_scores = task_data.get("sort_score", {})
            random_scores = task_data.get("random_score", {})

            for dim_str, sort_val in sort_scores.items():
                dim = int(dim_str)
                ratio = dim / model_dim
                retention = sort_val / default_score

                # Random baseline
                rand_vals = random_scores.get(dim_str, [])
                rand_mean = np.mean(rand_vals) if rand_vals else 0
                rand_retention = rand_mean / default_score if default_score > 0 else 0

                if ratio not in results[model_name]["ratios"]:
                    results[model_name]["ratios"][ratio] = {
                        "sort_retentions": [], "random_retentions": [],
                        "dims": [],
                    }
                results[model_name]["ratios"][ratio]["sort_retentions"].append(retention)
                results[model_name]["ratios"][ratio]["random_retentions"].append(rand_retention)
                results[model_name]["ratios"][ratio]["dims"].append(dim)

        # Compute summary stats per ratio
        for ratio, data in results[model_name]["ratios"].items():
            data["avg_sort"] = float(np.mean(data["sort_retentions"]))
            data["avg_random"] = float(np.mean(data["random_retentions"]))
            data["sort_std"] = float(np.std(data["sort_retentions"]))
            data["random_std"] = float(np.std(data["random_retentions"]))
            data["n_tasks"] = len(data["sort_retentions"])
            data["gap"] = data["avg_sort"] - data["avg_random"]

    return results


def analyze_donor_ranking_uniformity(analyze_data, task_similar_data):
    """
    Analyze whether weak task rankings are more uniform (less peaked).
    Hypothesis: weak tasks produce more uniform dimension rankings,
    which happen to transfer better because they don't overfit to any specific pattern.
    """
    results = {}

    for model_name, model_data in analyze_data.items():
        if model_name not in task_similar_data:
            continue

        # Get chunk score distributions for each task
        task_entropy = {}
        task_cv = {}
        for task_name, task_data in model_data["task_name"].items():
            if "2" not in task_data.get("split_win_size", {}):
                continue
            chunk_scores = np.array(task_data["split_win_size"]["2"]["chunk_result"])
            probs = np.abs(chunk_scores)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(chunk_scores)) / len(chunk_scores)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(chunk_scores))
            task_entropy[task_name] = entropy / max_entropy

            score_mean = np.mean(chunk_scores)
            score_std = np.std(chunk_scores)
            task_cv[task_name] = score_std / score_mean if score_mean != 0 else 0

        # Get self-transfer scores (proxy for task difficulty)
        ts_data = task_similar_data[model_name]
        self_scores = {}
        for donor, targets in ts_data.items():
            if donor in targets and isinstance(targets[donor], (int, float)):
                self_scores[donor] = targets[donor]

        # Compute correlation between task difficulty and ranking uniformity
        common_tasks = set(task_entropy.keys()) & set(self_scores.keys())
        if len(common_tasks) < 5:
            continue

        from scipy.stats import spearmanr
        difficulties = [self_scores[t] for t in sorted(common_tasks)]
        entropies = [task_entropy[t] for t in sorted(common_tasks)]
        cvs = [task_cv[t] for t in sorted(common_tasks)]

        r_entropy, p_entropy = spearmanr(difficulties, entropies) if len(common_tasks) >= 3 else (0, 1)
        r_cv, p_cv = spearmanr(difficulties, cvs) if len(common_tasks) >= 3 else (0, 1)

        results[model_name] = {
            "n_tasks": len(common_tasks),
            "difficulty_vs_entropy": {"rho": float(r_entropy), "p": float(p_entropy)},
            "difficulty_vs_cv": {"rho": float(r_cv), "p": float(p_cv)},
            "interpretation": (
                "Positive rho means harder tasks have more peaked rankings (lower entropy)"
                if r_entropy < 0 else
                "Positive rho means harder tasks have more uniform rankings (higher entropy)"
            ),
            "per_task": {
                t: {"entropy": task_entropy[t], "cv": task_cv[t], "self_score": self_scores[t]}
                for t in sorted(common_tasks)
            }
        }

    return results


def analyze_interchangeability_evidence(analyze_data, task_similar_data):
    """
    Compile evidence for the dimension interchangeability hypothesis.
    This is the key analysis for the paper's main claim.
    """
    results = {"models": {}}

    for model_name, model_data in analyze_data.items():
        model_dim = model_data["model_dim"]

        # Evidence 1: Optimized-random gap
        gaps = []
        for task_name, task_data in model_data["task_name"].items():
            random_scores = task_data.get("random_score", {})
            sort_scores = task_data.get("sort_score", {})
            for dim_str in random_scores:
                if dim_str in sort_scores:
                    rand_mean = np.mean(random_scores[dim_str])
                    gaps.append(sort_scores[dim_str] - rand_mean)

        # Evidence 2: Best-Poor gap narrows with more dimensions
        best_poor_gaps = {}
        for task_name, task_data in model_data["task_name"].items():
            for ws_str, ws_data in task_data.get("split_win_size", {}).items():
                for td_str, td_data in ws_data.get("chunk_win_size", {}).items():
                    head = td_data.get("head_score", {}).get("main_score", 0)
                    end = td_data.get("end_score", {}).get("main_score", 0)
                    default = task_data.get("defult_score", 1)
                    if default > 0 and head > 0 and end > 0:
                        n_chunks = int(td_str) // int(ws_str) if int(ws_str) > 0 else 0
                        if n_chunks not in best_poor_gaps:
                            best_poor_gaps[n_chunks] = []
                        best_poor_gaps[n_chunks].append((head - end) / default)

        # Evidence 3: Cross-task transfer (from task_similar)
        transfer_retention = None
        if model_name in task_similar_data:
            ts_data = task_similar_data[model_name]
            retentions = []
            for donor, targets in ts_data.items():
                for target, score in targets.items():
                    if target in ts_data.get(target, {}) and isinstance(score, (int, float)):
                        self_score = ts_data[target].get(target, 0)
                        if isinstance(self_score, (int, float)) and self_score > 0:
                            retentions.append(score / self_score)
            if retentions:
                transfer_retention = {
                    "mean": float(np.mean(retentions)),
                    "std": float(np.std(retentions)),
                    "n": len(retentions),
                }

        # Evidence 4: Chunk score flatness
        chunk_entropies = []
        for task_name, task_data in model_data["task_name"].items():
            if "2" in task_data.get("split_win_size", {}):
                chunk_scores = np.array(task_data["split_win_size"]["2"]["chunk_result"])
                probs = np.abs(chunk_scores)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(chunk_scores)) / len(chunk_scores)
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                max_entropy = np.log(len(chunk_scores))
                chunk_entropies.append(entropy / max_entropy)

        results["models"][model_name] = {
            "model_dim": model_dim,
            "optimized_random_gap": {
                "mean": float(np.mean(gaps)) if gaps else None,
                "std": float(np.std(gaps)) if gaps else None,
                "n": len(gaps),
            },
            "best_poor_gap_vs_n_chunks": {
                str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in sorted(best_poor_gaps.items())
            },
            "cross_task_transfer_retention": transfer_retention,
            "chunk_entropy": {
                "mean": float(np.mean(chunk_entropies)) if chunk_entropies else None,
                "std": float(np.std(chunk_entropies)) if chunk_entropies else None,
                "interpretation": (
                    "Close to 1.0 means dimension importance is nearly uniform (high interchangeability)"
                ),
            },
        }

    return results


def main():
    analyze_dir = "/home/linkco/exa/llm-usefulEeb/data/analyze"
    task_similar_dir = "/home/linkco/exa/llm-usefulEeb/data/task_similar"
    output_dir = "/home/linkco/exa/llm-usefulEeb/data/experiment_results"

    print("Loading data...")
    analyze_data = load_analyze_data(analyze_dir)
    task_similar_data = load_task_similar_data(task_similar_dir)
    print(f"  Analyze data: {len(analyze_data)} models")
    print(f"  Task similar data: {len(task_similar_data)} models")

    all_results = {}

    # 1. Redundancy mechanism analysis
    print("\n[1] Analyzing redundancy mechanism (explaining the paradox)...")
    redundancy = analyze_redundancy_mechanism(analyze_data)
    all_results["redundancy_mechanism"] = redundancy

    for model_name, model_data in redundancy.items():
        summary = model_data.get("model_summary")
        if summary:
            print(f"  {model_name}:")
            print(f"    Normalized entropy: {summary['avg_normalized_entropy']:.3f} (1.0 = uniform)")
            print(f"    Score CV: {summary['avg_chunk_score_cv']:.3f}")
            print(f"    Top-10% concentration: {summary['avg_top_10pct_concentration']:.3f}")
            print(f"    Top-25% concentration: {summary['avg_top_25pct_concentration']:.3f}")
            print(f"    Top-50% concentration: {summary['avg_top_50pct_concentration']:.3f}")

    # 2. Pruning ratio sweep
    print("\n[2] Analyzing pruning ratio sweep...")
    ratio_sweep = analyze_pruning_ratio_sweep(analyze_data)
    all_results["pruning_ratio_sweep"] = ratio_sweep

    for model_name, model_data in ratio_sweep.items():
        print(f"  {model_name}:")
        for ratio in sorted(model_data["ratios"].keys()):
            r = model_data["ratios"][ratio]
            print(f"    Ratio {ratio:.3f} (dim={int(ratio*model_data['model_dim']):>4d}): "
                  f"sort={r['avg_sort']:.3f}±{r['sort_std']:.3f}, "
                  f"random={r['avg_random']:.3f}±{r['random_std']:.3f}, "
                  f"gap={r['gap']:.3f}")

    # 3. Donor ranking uniformity
    print("\n[3] Analyzing donor ranking uniformity...")
    donor_uniformity = analyze_donor_ranking_uniformity(analyze_data, task_similar_data)
    all_results["donor_ranking_uniformity"] = donor_uniformity

    for model_name, model_data in donor_uniformity.items():
        print(f"  {model_name}:")
        print(f"    Difficulty vs entropy: rho={model_data['difficulty_vs_entropy']['rho']:.3f}, p={model_data['difficulty_vs_entropy']['p']:.4f}")
        print(f"    Difficulty vs CV: rho={model_data['difficulty_vs_cv']['rho']:.3f}, p={model_data['difficulty_vs_cv']['p']:.4f}")
        print(f"    Interpretation: {model_data['interpretation']}")

    # 4. Interchangeability evidence compilation
    print("\n[4] Compiling interchangeability evidence...")
    interchangeability = analyze_interchangeability_evidence(analyze_data, task_similar_data)
    all_results["interchangeability_evidence"] = interchangeability

    for model_name, model_data in interchangeability["models"].items():
        gap = model_data["optimized_random_gap"]
        transfer = model_data["cross_task_transfer_retention"]
        entropy = model_data["chunk_entropy"]
        print(f"  {model_name}:")
        if gap and gap["mean"] is not None:
            print(f"    Optimized-random gap: {gap['mean']:.3f}% ± {gap['std']:.3f}")
        if transfer:
            print(f"    Cross-task retention: {transfer['mean']:.3f} ± {transfer['std']:.3f}")
        if entropy and entropy["mean"] is not None:
            print(f"    Chunk entropy: {entropy['mean']:.3f} ({entropy['interpretation']})")

    # Save
    output_path = os.path.join(output_dir, "reviewer_response_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
