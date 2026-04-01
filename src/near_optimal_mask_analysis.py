"""
Near-optimal mask degeneracy analysis for "Prune to Prosper" paper.

Addresses reviewer question: "How many distinct masks achieve within 0.5% or 1%
of oracle performance?" If there are many disjoint good masks, that is STRONGER
evidence for dimension interchangeability than entropy alone.

Two-pronged approach:

A) ACTUAL PERFORMANCE ANALYSIS (from existing evaluation data):
   - Use the 10 random evaluations at each dim from random_score
   - Compare actual random performance vs oracle (head_score) vs default
   - Estimate P(random within X% of oracle) from empirical distribution

B) IMPORTANCE WEIGHT MONTE CARLO:
   - Use chunk importance scores (S_i^T) as additive importance weights
   - Monte Carlo sample N=10000 random subsets of k chunks
   - Compute importance retention = sum(selected) / sum(top-k)
   - This estimates how many distinct near-optimal subsets exist
   - High P(near-optimal) = strong evidence for mask degeneracy

Output: /experiments/analysis_output/near_optimal_mask_analysis.json
"""

import os
import json
import numpy as np
from collections import defaultdict


def load_analyze_data(analyze_dir):
    """Load all analyze JSON files."""
    data = {}
    for fname in os.listdir(analyze_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(analyze_dir, fname), "r") as f:
                data[model_name] = json.load(f)
    return data


def get_task_data(model_data):
    """Extract all per-task data needed for analysis.

    Returns dict: task_name -> {
        'chunk_scores': np.array of 512 importance scores,
        'default_score': float,
        'random_scores': {dim_str: [list of 10 actual random evals]},
        'head_score': {dim_str: float},  # oracle per dim
        'end_score': {dim_str: float},   # worst per dim
        'sort_score': {dim_str: float},  # greedy sorted per dim
    }
    """
    task_data_all = {}
    for task_name, task_data in model_data["task_name"].items():
        default_score = task_data.get("defult_score", 0)
        if default_score <= 0:
            continue

        # Chunk importance scores
        chunk_scores = None
        if "2" in task_data.get("split_win_size", {}):
            chunk_scores = np.array(
                task_data["split_win_size"]["2"]["chunk_result"]
            )

        # Actual random evaluations at each dimension
        random_scores = task_data.get("random_score", {})

        # Oracle/best and worst from split_win_size
        head_scores = {}
        end_scores = {}
        sort_scores = {}
        for ws_str, ws_data in task_data.get("split_win_size", {}).items():
            for dim_str, dim_data in ws_data.get("chunk_win_size", {}).items():
                if "head_score" in dim_data:
                    head_scores[dim_str] = dim_data["head_score"]["main_score"]
                if "end_score" in dim_data:
                    end_scores[dim_str] = dim_data["end_score"]["main_score"]
                if "sort_score" in dim_data:
                    sort_scores[dim_str] = dim_data["sort_score"]["main_score"]

        task_data_all[task_name] = {
            "chunk_scores": chunk_scores,
            "default_score": default_score,
            "random_scores": random_scores,
            "head_scores": head_scores,
            "end_scores": end_scores,
            "sort_scores": sort_scores,
        }

    return task_data_all


def compute_importance_mc_analysis(chunk_scores, k=128, n_samples=10000, seed=42):
    """
    Monte Carlo analysis using chunk importance scores as additive weights.

    The metric: importance_retention(S) = sum(S_i for i in S) / sum(top-k S_i)
    - = 1.0 for oracle subset
    - captures how much of the oracle importance a random subset achieves

    This is a proxy for actual performance retention.
    """
    rng = np.random.default_rng(seed)
    n = len(chunk_scores)

    # Oracle: top-k importance
    sorted_scores = np.sort(chunk_scores)[::-1]
    oracle_sum = np.sum(sorted_scores[:k])
    poor_sum = np.sum(sorted_scores[-k:])
    total_sum = np.sum(chunk_scores)

    # Monte Carlo
    random_importance_retentions = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.choice(n, size=k, replace=False)
        subset_sum = np.sum(chunk_scores[idx])
        random_importance_retentions[i] = subset_sum / oracle_sum  # fraction of oracle

    # Statistics
    stats = {
        "n_chunks": n,
        "k_selected": k,
        "n_samples": n_samples,
        "oracle_importance_sum": float(oracle_sum),
        "poor_importance_sum": float(poor_sum),
        "total_importance_sum": float(total_sum),
        "oracle_frac_of_total": float(oracle_sum / total_sum),
        "poor_frac_of_total": float(poor_sum / total_sum),
        "random_frac_of_total": float(k / n),
        "importance_retention_mean": float(np.mean(random_importance_retentions)),
        "importance_retention_std": float(np.std(random_importance_retentions)),
        "importance_retention_min": float(np.min(random_importance_retentions)),
        "importance_retention_max": float(np.max(random_importance_retentions)),
        "importance_retention_median": float(np.median(random_importance_retentions)),
    }

    # Near-oracle fractions (importance_retention >= 1 - threshold)
    for thresh_pct in [0.5, 1.0, 2.0, 5.0]:
        thresh = thresh_pct / 100
        threshold_val = 1.0 - thresh
        n_within = np.sum(random_importance_retentions >= threshold_val)
        stats[f"within_{thresh_pct:.1f}pct_of_oracle"] = {
            "threshold": float(threshold_val),
            "n_within": int(n_within),
            "fraction": float(n_within / n_samples),
            "percentage": float(n_within / n_samples * 100),
        }

    # Percentiles
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        stats[f"retention_p{p}"] = float(
            np.percentile(random_importance_retentions, p)
        )

    # Estimate number of distinct near-optimal masks
    # P(random mask is within 1% of oracle) = fraction_within_1pct
    # Number of such masks ≈ P * C(n, k)
    from math import comb
    total_subsets = comb(n, k)
    for thresh_pct in [1.0, 2.0]:
        key = f"within_{thresh_pct:.1f}pct_of_oracle"
        p = stats[key]["fraction"]
        estimated_masks = p * total_subsets
        stats[key]["total_subsets_C(n,k)"] = total_subsets
        stats[key]["estimated_near_optimal_masks"] = float(estimated_masks)
        stats[key]["estimated_near_optimal_masks_log10"] = float(
            np.log10(max(1, estimated_masks))
        )

    return stats


def compute_actual_performance_analysis(task_data, dim_str="128"):
    """
    Analyze actual MTEB performance from existing evaluation data.

    Uses the 10 random evaluations + oracle (head_score) + default.
    """
    default = task_data["default_score"]
    random_evals = task_data["random_scores"].get(dim_str, [])
    head_score = task_data["head_scores"].get(dim_str, None)
    end_score = task_data["end_scores"].get(dim_str, None)
    sort_score = task_data["sort_scores"].get(dim_str, None)

    if not random_evals or head_score is None or default <= 0:
        return None

    random_evals = np.array(random_evals)

    # Retention: performance / default
    oracle_retention = head_score / default
    random_retentions = random_evals / default
    end_retention = end_score / default if end_score else None

    # Gap analysis (in MTEB points)
    gap_oracle_random_mean = head_score - np.mean(random_evals)
    gap_oracle_random_max = head_score - np.min(random_evals)
    gap_oracle_random_min = head_score - np.max(random_evals)

    # Fraction of random within X% of oracle (in absolute performance)
    results = {
        "dim": int(dim_str),
        "default_score": float(default),
        "oracle_score": float(head_score),
        "oracle_retention": float(oracle_retention),
        "sort_score": float(sort_score) if sort_score else None,
        "end_score": float(end_score) if end_score else None,
        "end_retention": float(end_retention) if end_retention else None,
        "random_mean": float(np.mean(random_evals)),
        "random_std": float(np.std(random_evals)),
        "random_min": float(np.min(random_evals)),
        "random_max": float(np.max(random_evals)),
        "random_mean_retention": float(np.mean(random_retentions)),
        "gap_oracle_random_mean": float(gap_oracle_random_mean),
        "gap_oracle_random_mean_pct": float(gap_oracle_random_mean / head_score * 100),
        "n_random_samples": len(random_evals),
    }

    # Fraction of actual random evals within X% of oracle
    for thresh_pct in [0.5, 1.0, 2.0, 5.0]:
        threshold_score = head_score * (1 - thresh_pct / 100)
        n_within = np.sum(random_evals >= threshold_score)
        results[f"actual_within_{thresh_pct:.1f}pct"] = {
            "threshold_score": float(threshold_score),
            "n_within": int(n_within),
            "fraction": float(n_within / len(random_evals)),
        }

    return results


def estimate_probability_from_distribution(random_mean, random_std, oracle_score,
                                             n_random=10):
    """
    Use the empirical random distribution to estimate P(random >= oracle * (1-t)).

    Fits a normal distribution to the random evaluations and computes the
    probability analytically. This gives us an estimate even with only 10 samples.
    """
    if random_std <= 0:
        random_std = 0.01  # Avoid division by zero

    from scipy.stats import norm

    results = {}
    for thresh_pct in [0.5, 1.0, 2.0, 5.0]:
        threshold = oracle_score * (1 - thresh_pct / 100)
        # P(random >= threshold) under Normal(random_mean, random_std)
        z = (threshold - random_mean) / random_std
        prob = 1 - norm.cdf(z)
        results[f"{thresh_pct:.1f}%"] = {
            "threshold_score": float(threshold),
            "z_score": float(z),
            "probability": float(prob),
            "interpretation": (
                f"P(random mask within {thresh_pct}% of oracle) = {prob*100:.1f}%"
            ),
        }

    return results


def main():
    analyze_dir = "/home/linkco/exa/llm-usefulEeb/Useful-Embedding/data/analyze"
    output_dir = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output"
    output_path = os.path.join(output_dir, "near_optimal_mask_analysis.json")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading analyze data...")
    analyze_data = load_analyze_data(analyze_dir)
    print(f"  Found {len(analyze_data)} models: {list(analyze_data.keys())}")

    all_results = {
        "analysis_description": (
            "Near-optimal mask degeneracy analysis. Two approaches: "
            "(A) Actual performance analysis using empirical random evaluations "
            "to estimate P(random mask within X% of oracle). "
            "(B) Importance-weight Monte Carlo (10000 samples) to estimate "
            "the fraction of random subsets achieving near-oracle importance retention."
        ),
        "models": {},
    }

    target_dims = ["128", "64", "256"]  # Dimensions to analyze
    n_mc_samples = 10000

    for model_name in sorted(analyze_data.keys()):
        model_data = analyze_data[model_name]
        model_dim = model_data.get("model_dim", "unknown")

        print(f"\n{'='*70}")
        print(f"Model: {model_name} (dim={model_dim})")
        print(f"{'='*70}")

        tasks = get_task_data(model_data)
        n_tasks = len(tasks)
        print(f"  Tasks: {n_tasks}")

        model_results = {
            "model_dim": model_dim,
            "n_tasks": n_tasks,
            "dimensions": {},
        }

        for dim_str in target_dims:
            dim = int(dim_str)
            k = dim // 2  # chunks selected (each chunk = 2 dims)
            print(f"\n  --- dim={dim} (k={k} chunks from 512) ---")

            task_analyses = {}
            for task_name, td in sorted(tasks.items()):
                # A) Actual performance analysis
                actual_analysis = compute_actual_performance_analysis(td, dim_str)

                # B) Importance MC analysis
                mc_analysis = None
                if td["chunk_scores"] is not None and k <= len(td["chunk_scores"]):
                    mc_analysis = compute_importance_mc_analysis(
                        td["chunk_scores"], k=k, n_samples=n_mc_samples
                    )

                # C) Estimated probability from fitted distribution
                prob_analysis = None
                if actual_analysis and actual_analysis["random_std"] > 0:
                    prob_analysis = estimate_probability_from_distribution(
                        actual_analysis["random_mean"],
                        actual_analysis["random_std"],
                        actual_analysis["oracle_score"],
                    )

                task_analyses[task_name] = {
                    "actual_performance": actual_analysis,
                    "importance_mc": mc_analysis,
                    "probability_estimate": prob_analysis,
                }

            # Aggregate: actual performance
            actual_gaps = [
                t["actual_performance"]["gap_oracle_random_mean_pct"]
                for t in task_analyses.values()
                if t["actual_performance"]
            ]
            actual_random_retentions = [
                t["actual_performance"]["random_mean_retention"]
                for t in task_analyses.values()
                if t["actual_performance"]
            ]
            actual_oracle_retentions = [
                t["actual_performance"]["oracle_retention"]
                for t in task_analyses.values()
                if t["actual_performance"]
            ]

            # Aggregate: probability estimates
            prob_within_1pct = [
                t["probability_estimate"]["1.0%"]["probability"]
                for t in task_analyses.values()
                if t["probability_estimate"]
            ]
            prob_within_05pct = [
                t["probability_estimate"]["0.5%"]["probability"]
                for t in task_analyses.values()
                if t["probability_estimate"]
            ]
            prob_within_2pct = [
                t["probability_estimate"]["2.0%"]["probability"]
                for t in task_analyses.values()
                if t["probability_estimate"]
            ]

            # Count tasks where P(within 1%) > 50%
            n_tasks_majority_1pct = sum(1 for p in prob_within_1pct if p > 0.5)
            n_tasks_majority_05pct = sum(1 for p in prob_within_05pct if p > 0.5)
            n_tasks_majority_2pct = sum(1 for p in prob_within_2pct if p > 0.5)

            # Aggregate: importance MC
            mc_within_1pct = [
                t["importance_mc"]["within_1.0pct_of_oracle"]["fraction"]
                for t in task_analyses.values()
                if t["importance_mc"]
            ]
            mc_within_2pct = [
                t["importance_mc"]["within_2.0pct_of_oracle"]["fraction"]
                for t in task_analyses.values()
                if t["importance_mc"]
            ]

            # Log10 estimated masks
            mc_log10_masks = [
                t["importance_mc"]["within_1.0pct_of_oracle"]["estimated_near_optimal_masks_log10"]
                for t in task_analyses.values()
                if t["importance_mc"]
            ]

            dim_summary = {
                "dim": dim,
                "k_chunks": k,
                "n_tasks_analyzed": len([t for t in task_analyses.values() if t["actual_performance"]]),
                "actual_performance_summary": {
                    "avg_oracle_retention": float(np.mean(actual_oracle_retentions)) if actual_oracle_retentions else None,
                    "avg_random_retention": float(np.mean(actual_random_retentions)) if actual_random_retentions else None,
                    "avg_gap_oracle_random_pct": float(np.mean(actual_gaps)) if actual_gaps else None,
                    "std_gap_oracle_random_pct": float(np.std(actual_gaps)) if actual_gaps else None,
                },
                "probability_estimates_summary": {
                    "avg_P_within_0.5pct": float(np.mean(prob_within_05pct)) if prob_within_05pct else None,
                    "avg_P_within_1pct": float(np.mean(prob_within_1pct)) if prob_within_1pct else None,
                    "avg_P_within_2pct": float(np.mean(prob_within_2pct)) if prob_within_2pct else None,
                    "median_P_within_1pct": float(np.median(prob_within_1pct)) if prob_within_1pct else None,
                    "tasks_with_P_gt_50pct_within_0.5pct": n_tasks_majority_05pct,
                    "tasks_with_P_gt_50pct_within_1pct": n_tasks_majority_1pct,
                    "tasks_with_P_gt_50pct_within_2pct": n_tasks_majority_2pct,
                    "fraction_tasks_P_gt_50pct_within_1pct": n_tasks_majority_1pct / len(prob_within_1pct) if prob_within_1pct else 0,
                },
                "importance_mc_summary": {
                    "avg_fraction_within_1pct": float(np.mean(mc_within_1pct)) if mc_within_1pct else None,
                    "avg_fraction_within_2pct": float(np.mean(mc_within_2pct)) if mc_within_2pct else None,
                    "avg_log10_near_optimal_masks": float(np.mean(mc_log10_masks)) if mc_log10_masks else None,
                },
                "per_task": task_analyses,
            }

            model_results["dimensions"][dim_str] = dim_summary

            # Print summary
            print(f"\n  Actual performance:")
            if actual_gaps:
                print(f"    Avg oracle retention: {np.mean(actual_oracle_retentions):.4f}")
                print(f"    Avg random retention: {np.mean(actual_random_retentions):.4f}")
                print(f"    Avg oracle-random gap: {np.mean(actual_gaps):.2f}%")

            print(f"\n  Estimated P(random within X% of oracle):")
            if prob_within_1pct:
                print(f"    Within 0.5%: avg={np.mean(prob_within_05pct)*100:.1f}%, "
                      f"median={np.median(prob_within_05pct)*100:.1f}%")
                print(f"    Within 1.0%: avg={np.mean(prob_within_1pct)*100:.1f}%, "
                      f"median={np.median(prob_within_1pct)*100:.1f}%")
                print(f"    Within 2.0%: avg={np.mean(prob_within_2pct)*100:.1f}%, "
                      f"median={np.median(prob_within_2pct)*100:.1f}%")

            print(f"\n  Tasks where P(within 1%) > 50%: "
                  f"{n_tasks_majority_1pct}/{len(prob_within_1pct)}")
            print(f"  Tasks where P(within 0.5%) > 50%: "
                  f"{n_tasks_majority_05pct}/{len(prob_within_05pct)}")
            print(f"  Tasks where P(within 2%) > 50%: "
                  f"{n_tasks_majority_2pct}/{len(prob_within_2pct)}")

            if mc_within_1pct:
                print(f"\n  Importance MC (proxy):")
                print(f"    Avg fraction within 1% of oracle importance: "
                      f"{np.mean(mc_within_1pct)*100:.1f}%")
                print(f"    Avg fraction within 2% of oracle importance: "
                      f"{np.mean(mc_within_2pct)*100:.1f}%")
                if mc_log10_masks:
                    print(f"    Avg log10(estimated near-optimal masks): "
                          f"{np.mean(mc_log10_masks):.1f}")

            # Print per-task detail for extreme tasks
            if prob_within_1pct:
                task_probs = [(name, t["probability_estimate"]["1.0%"]["probability"])
                              for name, t in task_analyses.items()
                              if t["probability_estimate"]]
                task_probs.sort(key=lambda x: x[1])
                print(f"\n  Top-5 hardest tasks (lowest P within 1%):")
                for name, p in task_probs[:5]:
                    print(f"    {name}: P={p*100:.1f}%")
                print(f"  Top-5 easiest tasks (highest P within 1%):")
                for name, p in task_probs[-5:]:
                    print(f"    {name}: P={p*100:.1f}%")

        all_results["models"][model_name] = model_results

    # Cross-model summary
    print(f"\n{'='*70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*70}")

    cross_summary = {}
    for dim_str in target_dims:
        probs_1pct = []
        probs_05pct = []
        probs_2pct = []
        fracs_1pct_tasks = []

        for model_name, mr in all_results["models"].items():
            if dim_str in mr["dimensions"]:
                ds = mr["dimensions"][dim_str]
                pe = ds.get("probability_estimates_summary", {})
                if pe.get("avg_P_within_1pct") is not None:
                    probs_1pct.append(pe["avg_P_within_1pct"])
                if pe.get("avg_P_within_0.5pct") is not None:
                    probs_05pct.append(pe["avg_P_within_0.5pct"])
                if pe.get("avg_P_within_2pct") is not None:
                    probs_2pct.append(pe["avg_P_within_2pct"])
                if pe.get("fraction_tasks_P_gt_50pct_within_1pct") is not None:
                    fracs_1pct_tasks.append(pe["fraction_tasks_P_gt_50pct_within_1pct"])

        cross_summary[dim_str] = {
            "avg_P_within_0.5pct": float(np.mean(probs_05pct)) if probs_05pct else None,
            "avg_P_within_1pct": float(np.mean(probs_1pct)) if probs_1pct else None,
            "avg_P_within_2pct": float(np.mean(probs_2pct)) if probs_2pct else None,
            "avg_fraction_tasks_with_majority_within_1pct": float(np.mean(fracs_1pct_tasks)) if fracs_1pct_tasks else None,
        }

        dim = int(dim_str)
        print(f"\n  dim={dim_str} ({dim} of 1024 dims kept = {dim/1024*100:.0f}%):")
        if probs_1pct:
            print(f"    Avg P(random within 0.5%): {np.mean(probs_05pct)*100:.1f}%")
            print(f"    Avg P(random within 1.0%): {np.mean(probs_1pct)*100:.1f}%")
            print(f"    Avg P(random within 2.0%): {np.mean(probs_2pct)*100:.1f}%")
            print(f"    Avg fraction of tasks where P(within 1%) > 50%: "
                  f"{np.mean(fracs_1pct_tasks)*100:.1f}%")

    all_results["cross_model_summary"] = cross_summary

    # Save
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Final takeaway
    print(f"\n{'='*70}")
    print("KEY FINDINGS FOR REVIEWER RESPONSE")
    print(f"{'='*70}")
    for dim_str, cs in cross_summary.items():
        dim = int(dim_str)
        print(f"\ndim={dim_str} ({dim/1024*100:.0f}% of dims kept):")
        if cs["avg_P_within_1pct"] is not None:
            print(f"  Across all models, the average probability that a random mask "
                  f"achieves within 1% of oracle: {cs['avg_P_within_1pct']*100:.1f}%")
        if cs["avg_P_within_2pct"] is not None:
            print(f"  Within 2%: {cs['avg_P_within_2pct']*100:.1f}%")
        if cs["avg_fraction_tasks_with_majority_within_1pct"] is not None:
            print(f"  Fraction of tasks where >50% of random masks are within 1%: "
                  f"{cs['avg_fraction_tasks_with_majority_within_1pct']*100:.1f}%")


if __name__ == "__main__":
    main()
