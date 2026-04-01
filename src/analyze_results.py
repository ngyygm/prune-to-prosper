"""
Statistical analysis and visualization for the "Prune to Prosper" paper.

Combines Ideas 1+2:
1. Dimension interchangeability analysis (optimized vs random gap)
2. Cross-task transfer analysis (weak tasks as universal donors)
3. Statistical significance (bootstrap CIs)
4. Mechanistic analysis (dimension correlation)

Input: analyze/*.json, task_similar/*.json
Output: figures/, analysis_results.json
"""

import os
import json
import argparse
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


def compute_optimized_vs_random_gap(analyze_data):
    """Compute the gap between optimized (sort/best) and random selection."""
    results = {}

    for model_name, model_data in analyze_data.items():
        model_dim = model_data["model_dim"]
        results[model_name] = {"model_dim": model_dim, "tasks": {}}

        for task_name, task_data in model_data["task_name"].items():
            default_score = task_data["defult_score"]
            random_scores = task_data.get("random_score", {})
            sort_scores = task_data.get("sort_score", {})
            split_win_size = task_data.get("split_win_size", {})

            task_results = {
                "default": default_score,
                "random": {},
                "sort": {},
                "best": {},
                "poor": {},
                "gap": {},
            }

            for dim_str in random_scores:
                dim = int(dim_str)
                rand_vals = random_scores[dim_str]
                rand_mean = np.mean(rand_vals)
                rand_std = np.std(rand_vals)

                task_results["random"][dim] = {
                    "mean": rand_mean,
                    "std": rand_std,
                    "n": len(rand_vals),
                }

                if dim_str in sort_scores:
                    sort_val = sort_scores[dim_str]
                    task_results["sort"][dim] = sort_val
                    task_results["gap"][dim] = sort_val - rand_mean

            # Best/Poor from split_win_size
            for ws_str, ws_data in split_win_size.items():
                ws = int(ws_str)
                for target_dim_str, td_data in ws_data.get("chunk_win_size", {}).items():
                    target_dim = int(target_dim_str)
                    if "head_score" in td_data:
                        task_results["best"][target_dim] = td_data["head_score"]["main_score"]
                    if "end_score" in td_data:
                        task_results["poor"][target_dim] = td_data["end_score"]["main_score"]

            results[model_name]["tasks"][task_name] = task_results

    return results


def compute_cross_task_transfer_matrix(task_similar_data):
    """Compute the full cross-task transfer matrix."""
    results = {}

    for model_name, model_data in task_similar_data.items():
        if len(model_data) < 10:
            continue  # Skip incomplete data

        # Get all tasks
        all_tasks = set()
        for donor, targets in model_data.items():
            all_tasks.add(donor)
            all_tasks.update(targets.keys())
        all_tasks = sorted(all_tasks)

        results[model_name] = {
            "tasks": all_tasks,
            "self_transfer": {},
            "best_cross_transfer": {},
            "avg_cross_transfer": {},
            "donor_quality": {},
        }

        # For each target task, compute self-transfer and best cross-transfer
        for target in all_tasks:
            # Self-transfer: donor=target, evaluate on target
            if target in model_data and target in model_data.get(target, {}):
                self_score = model_data[target][target]
                if isinstance(self_score, (int, float)):
                    results[model_name]["self_transfer"][target] = self_score

                # Best cross-transfer: max score from any OTHER donor
                cross_scores = []
                for donor, targets in model_data.items():
                    if donor == target:
                        continue
                    if target in targets and isinstance(targets[target], (int, float)):
                        cross_scores.append(targets[target])

                if cross_scores:
                    results[model_name]["best_cross_transfer"][target] = max(cross_scores)
                    results[model_name]["avg_cross_transfer"][target] = np.mean(cross_scores)

        # Donor quality: average retention across all target tasks
        for donor in all_tasks:
            if donor not in model_data:
                continue
            retentions = []
            for target, score in model_data[donor].items():
                if target in results[model_name]["self_transfer"]:
                    self_score = results[model_name]["self_transfer"][target]
                    if self_score > 0 and isinstance(score, (int, float)):
                        retentions.append(score / self_score)
            if retentions:
                results[model_name]["donor_quality"][donor] = {
                    "avg_retention": np.mean(retentions),
                    "std_retention": np.std(retentions),
                    "n_targets": len(retentions),
                }

    return results


def bootstrap_confidence_interval(scores, n_bootstrap=10000, ci=95):
    """Compute bootstrap confidence interval for a list of scores."""
    scores = np.array(scores)
    if len(scores) < 2:
        return {"mean": float(np.mean(scores)), "ci_low": float(np.mean(scores)), "ci_high": float(np.mean(scores)), "std": 0.0}

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (100 - ci) / 2
    ci_low = np.percentile(bootstrap_means, alpha)
    ci_high = np.percentile(bootstrap_means, 100 - alpha)

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": len(scores),
    }


def compute_effect_size(group_a, group_b):
    """Compute Cohen's d effect size between two groups."""
    a, b = np.array(group_a), np.array(group_b)
    pooled_std = np.sqrt(((len(a)-1)*np.var(a) + (len(b)-1)*np.var(b)) / (len(a)+len(b)-2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def analyze_dimension_correlation(analyze_data):
    """Analyze correlation of chunk importance rankings across tasks."""
    results = {}

    for model_name, model_data in analyze_data.items():
        # Get chunk rankings for each task at win_size=2
        task_rankings = {}
        for task_name, task_data in model_data["task_name"].items():
            if "2" in task_data.get("split_win_size", {}):
                chunk_scores = task_data["split_win_size"]["2"]["chunk_result"]
                ranking = np.argsort(chunk_scores)[::-1]  # descending
                task_rankings[task_name] = ranking

        if len(task_rankings) < 2:
            continue

        # Compute pairwise ranking correlations (Spearman)
        from scipy.stats import spearmanr
        task_names = sorted(task_rankings.keys())
        corr_matrix = {}
        for t1, t2 in combinations(task_names, 2):
            r, p = spearmanr(task_rankings[t1], task_rankings[t2])
            corr_matrix[f"{t1}|||{t2}"] = {"rho": float(r), "p_value": float(p)}

        # Average correlation
        rhos = [v["rho"] for v in corr_matrix.values()]
        avg_corr = np.mean(rhos) if rhos else 0

        results[model_name] = {
            "n_tasks": len(task_rankings),
            "avg_ranking_correlation": float(avg_corr),
            "min_ranking_correlation": float(min(rhos)) if rhos else 0,
            "max_ranking_correlation": float(max(rhos)) if rhos else 0,
            "pairwise": corr_matrix,
        }

    return results


def classify_task_category(task_name, task_categories=None):
    """Classify a task into its MTEB category."""
    if task_categories is None:
        task_categories = {
            "Classification": [
                'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
                'Banking77Classification', 'EmotionClassification', 'ImdbClassification',
                'MTOPDomainClassification', 'MTOPIntentClassification',
                'MassiveIntentClassification', 'MassiveScenarioClassification',
                'ToxicConversationsClassification', 'TweetSentimentExtractionClassification'
            ],
            "Clustering": ['BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering'],
            "PairClassification": ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'],
            "Reranking": ['AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions'],
            "Retrieval": ['ArguAna', 'CQADupstackEnglishRetrieval', 'NFCorpus', 'SCIDOCS', 'SciFact'],
            "STS": ['BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STSBenchmark'],
            "Summarization": ['SummEval'],
        }

    for cat, tasks in task_categories.items():
        if task_name in tasks:
            return cat
    return "Unknown"


def compute_category_level_transfer(task_similar_data):
    """Compute cross-task transfer grouped by task category."""
    results = {}

    for model_name, model_data in task_similar_data.items():
        if len(model_data) < 10:
            continue

        cat_matrix = defaultdict(lambda: defaultdict(list))

        for donor, targets in model_data.items():
            donor_cat = classify_task_category(donor)
            for target, score in targets.items():
                target_cat = classify_task_category(target)
                if donor_cat != "Unknown" and target_cat != "Unknown":
                    cat_matrix[donor_cat][target_cat].append(score)

        results[model_name] = {}
        for donor_cat, target_cats in cat_matrix.items():
            results[model_name][donor_cat] = {}
            for target_cat, scores in target_cats.items():
                if scores:
                    results[model_name][donor_cat][target_cat] = bootstrap_confidence_interval(scores)

    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for Prune to Prosper")
    parser.add_argument("--analyze_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/analyze")
    parser.add_argument("--task_similar_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/task_similar")
    parser.add_argument("--output_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/experiments/analysis_output")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    analyze_data = load_analyze_data(args.analyze_dir)
    task_similar_data = load_task_similar_data(args.task_similar_dir)
    print(f"  Analyze data: {len(analyze_data)} models")
    print(f"  Task similar data: {len(task_similar_data)} models")

    all_results = {}

    # 1. Optimized vs Random gap analysis
    print("\n[1] Computing optimized vs random gap...")
    gap_results = compute_optimized_vs_random_gap(analyze_data)
    all_results["optimized_vs_random"] = gap_results

    # Summary statistics
    for model_name, model_data in gap_results.items():
        gaps = []
        for task_name, task_data in model_data["tasks"].items():
            for dim, gap in task_data.get("gap", {}).items():
                gaps.append(gap)
        if gaps:
            ci = bootstrap_confidence_interval(gaps, n_bootstrap=args.n_bootstrap)
            print(f"  {model_name}: mean gap = {ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    # 2. Cross-task transfer matrix
    print("\n[2] Computing cross-task transfer matrix...")
    transfer_results = compute_cross_task_transfer_matrix(task_similar_data)
    all_results["cross_task_transfer"] = transfer_results

    for model_name, model_data in transfer_results.items():
        retentions = [v["avg_retention"] for v in model_data.get("donor_quality", {}).values()]
        if retentions:
            ci = bootstrap_confidence_interval(retentions, n_bootstrap=args.n_bootstrap)
            print(f"  {model_name}: avg retention = {ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    # 3. Category-level transfer
    print("\n[3] Computing category-level transfer...")
    cat_results = compute_category_level_transfer(task_similar_data)
    all_results["category_transfer"] = cat_results

    # 4. Dimension correlation analysis
    print("\n[4] Analyzing dimension correlation across tasks...")
    corr_results = analyze_dimension_correlation(analyze_data)
    all_results["dimension_correlation"] = corr_results

    for model_name, model_data in corr_results.items():
        print(f"  {model_name}: avg ranking rho = {model_data['avg_ranking_correlation']:.3f}")

    # 5. Weak vs Strong donor analysis
    print("\n[5] Analyzing weak vs strong donor quality...")
    weak_vs_strong = {}
    for model_name, model_data in transfer_results.items():
        donor_quality = model_data.get("donor_quality", {})
        if not donor_quality:
            continue

        # Sort donors by self-transfer score (proxy for task difficulty)
        self_scores = model_data.get("self_transfer", {})
        donor_with_self = {}
        for d, q in donor_quality.items():
            if d in self_scores:
                try:
                    donor_with_self[d] = (float(q["avg_retention"]), float(self_scores[d]))
                except (ValueError, TypeError):
                    pass

        if not donor_with_self:
            continue

        sorted_donors = sorted(donor_with_self.items(), key=lambda x: x[1][1])
        n = len(sorted_donors)
        n_quartile = max(1, n // 4)

        weak_retentions = [float(v[1][0]) for v in sorted_donors[:n_quartile]]
        strong_retentions = [float(v[1][0]) for v in sorted_donors[-n_quartile:]]
        mid_retentions = [float(v[1][0]) for v in sorted_donors[n_quartile:2*n_quartile]]

        weak_vs_strong[model_name] = {
            "weak_donor_retention": bootstrap_confidence_interval(weak_retentions, n_bootstrap=args.n_bootstrap),
            "strong_donor_retention": bootstrap_confidence_interval(strong_retentions, n_bootstrap=args.n_bootstrap),
            "mid_donor_retention": bootstrap_confidence_interval(mid_retentions, n_bootstrap=args.n_bootstrap),
            "effect_size_weak_vs_strong": compute_effect_size(strong_retentions, weak_retentions),
            "n_donors": n,
            "weak_donors": [d[0] for d in sorted_donors[:n_quartile]],
            "strong_donors": [d[0] for d in sorted_donors[-n_quartile:]],
        }

        w_ci = weak_vs_strong[model_name]["weak_donor_retention"]
        s_ci = weak_vs_strong[model_name]["strong_donor_retention"]
        print(f"  {model_name}:")
        print(f"    Weak donors:  {w_ci['mean']:.3f} [{w_ci['ci_low']:.3f}, {w_ci['ci_high']:.3f}]")
        print(f"    Strong donors: {s_ci['mean']:.3f} [{s_ci['ci_low']:.3f}, {s_ci['ci_high']:.3f}]")
        print(f"    Effect size (Cohen's d): {weak_vs_strong[model_name]['effect_size_weak_vs_strong']:.3f}")

    all_results["weak_vs_strong_donors"] = weak_vs_strong

    # Save all results
    output_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
