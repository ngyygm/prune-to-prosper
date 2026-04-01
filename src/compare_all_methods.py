"""
Final comparison: magnitude vs random vs sort vs optimized at dim=256.
Runs after MTEB results are available.
"""

import os
import json
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

EXCLUDE_TASKS = {"STS17"}  # STS17 has unreliable default scores


def compare_all_methods(output_dir):
    """Compare all pruning methods using analyze data + magnitude MTEB results."""

    # Load analyze data (original)
    analyze_dir = "/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/analyze"
    analyze_data = {}
    for fname in os.listdir(analyze_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(analyze_dir, fname), "r") as f:
                analyze_data[model_name] = json.load(f)

    # Load magnitude MTEB results (standalone files, preferred)
    mag_mteb = {}
    mag_file_map = {
        "stella_en_400M_v5": "magnitude_stella_mteb.json",
        "gte-large-en-v1.5": "magnitude_gte_mteb.json",
    }
    for model_name, mag_file in mag_file_map.items():
        path = os.path.join(output_dir, mag_file)
        if os.path.exists(path):
            with open(path, "r") as f:
                mag_mteb[model_name] = json.load(f)
            print(f"  Loaded magnitude MTEB: {model_name} ({len(mag_mteb[model_name]['scores'])} tasks)")

    # For each model, compute method comparison at dim=256
    comparison = {}

    for model_name, model_data in analyze_data.items():
        model_dim = model_data["model_dim"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name} (dim={model_dim})")
        print(f"{'='*60}")

        method_scores = {}

        for task_name, task_data in model_data["task_name"].items():
            if task_name in EXCLUDE_TASKS:
                continue

            default = task_data.get("defult_score", 0)
            if default <= 0:
                continue

            if task_name not in method_scores:
                method_scores[task_name] = {"default": default}

            # Random at dim=256
            random_data = task_data.get("random_score", {}).get("256", [])
            if random_data:
                method_scores[task_name]["random"] = float(np.mean(random_data))

            # Sort at dim=256
            sort_val = task_data.get("sort_score", {}).get("256")
            if sort_val is not None:
                method_scores[task_name]["sort"] = float(sort_val)

            # Best/Poor at dim=256 (from split_win_size)
            for ws_str, ws_data in task_data.get("split_win_size", {}).items():
                for td_str, td_data in ws_data.get("chunk_win_size", {}).items():
                    if td_str == "256":
                        head = td_data.get("head_score", {}).get("main_score")
                        end = td_data.get("end_score", {}).get("main_score")
                        if head is not None:
                            method_scores[task_name]["best"] = float(head)
                        if end is not None:
                            method_scores[task_name]["poor"] = float(end)

        # Add magnitude MTEB scores
        if model_name in mag_mteb:
            for task_name, score in mag_mteb[model_name].get("scores", {}).items():
                if task_name in method_scores:
                    method_scores[task_name]["magnitude"] = float(score)

        # Compute retention (excluding STS17)
        method_retentions = {"random": [], "sort": [], "best": [], "poor": [], "magnitude": []}

        for task_name, scores in method_scores.items():
            if task_name in EXCLUDE_TASKS:
                continue
            default = scores["default"]
            for method in ["random", "sort", "best", "poor", "magnitude"]:
                if method in scores and default > 0:
                    method_retentions[method].append(scores[method] / default)

        # Print results
        n_tasks = len(method_scores)
        print(f"\n  Method comparison at dim=256 ({n_tasks} tasks, excluding STS17):")
        print(f"  {'Method':<20} {'Mean Ret.':>10} {'Std':>8} {'Median':>8} {'n':>4}")
        print(f"  {'-'*52}")

        for method in ["random", "sort", "best", "magnitude", "poor"]:
            vals = method_retentions[method]
            if vals:
                print(f"  {method:<20} {np.mean(vals):>10.4f} {np.std(vals):>8.4f} "
                      f"{np.median(vals):>8.4f} {len(vals):>4}")
            else:
                print(f"  {method:<20} {'N/A':>10}")

        # Statistical tests: sort vs random
        if method_retentions["random"] and method_retentions["sort"]:
            _print_stats(method_retentions["random"], method_retentions["sort"], "sort vs random")

        # Statistical tests: magnitude vs random
        if method_retentions["random"] and method_retentions["magnitude"]:
            _print_stats(method_retentions["random"], method_retentions["magnitude"], "magnitude vs random")

            # Per-task win count
            mag_wins = sum(1 for t, s in method_scores.items()
                           if "magnitude" in s and "random" in s and t not in EXCLUDE_TASKS
                           and s["magnitude"] / s["default"] > s["random"] / s["default"])
            rnd_wins = sum(1 for t, s in method_scores.items()
                           if "magnitude" in s and "random" in s and t not in EXCLUDE_TASKS
                           and s["random"] / s["default"] > s["magnitude"] / s["default"])
            total = mag_wins + rnd_wins
            print(f"  Win count: Random={rnd_wins}, Magnitude={mag_wins} (of {total} tasks)")

        comparison[model_name] = {
            "n_tasks": n_tasks,
            "method_retentions": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                    "median": float(np.median(v)), "n": len(v)}
                for k, v in method_retentions.items() if v
            },
        }

    # Cross-model summary table
    print(f"\n\n{'='*80}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'stella':>12} {'gte-large':>12} {'roberta-IB':>12}")
    print(f"{'-'*58}")

    for method in ["random", "sort", "magnitude", "best", "poor"]:
        vals = []
        for model_name in ["stella_en_400M_v5", "gte-large-en-v1.5", "roberta-large-InBedder"]:
            if model_name in comparison:
                r = comparison[model_name]["method_retentions"].get(method)
                vals.append(f"{r['mean']:.4f}" if r else "N/A")
            else:
                vals.append("N/A")
        print(f"  {method:<18} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # Save
    output_path = os.path.join(output_dir, "all_methods_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return comparison


def _print_stats(ref_vals, comp_vals, label):
    """Print statistical comparison between two methods."""
    ref = np.array(ref_vals)
    comp = np.array(comp_vals)

    # Ensure paired: use only tasks present in both
    min_len = min(len(ref), len(comp))
    ref = ref[:min_len]
    comp = comp[:min_len]

    diffs = comp - ref
    pooled_std = np.sqrt(
        ((len(ref)-1)*ref.var() + (len(comp)-1)*comp.var()) /
        (len(ref)+len(comp)-2)
    )
    cd = diffs.mean() / pooled_std if pooled_std > 0 else 0

    t_stat, t_p = ttest_rel(comp, ref)
    try:
        w_stat, w_p = wilcoxon(comp, ref)
    except Exception:
        w_stat, w_p = 0, 1.0

    sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
    print(f"\n  {label}:")
    print(f"    Diff mean={diffs.mean():+.4f}, Cohen's d={cd:+.4f}")
    print(f"    t-test: t={t_stat:.3f}, p={t_p:.4f} {sig}")
    print(f"    Wilcoxon: W={w_stat:.1f}, p={w_p:.4f}")


if __name__ == "__main__":
    output_dir = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output"
    compare_all_methods(output_dir)
