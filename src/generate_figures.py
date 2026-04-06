"""
Generate publication-quality figures for the Prune to Prosper paper.

Creates:
1. Optimized vs Random gap across models (violin/box plot)
2. Cross-task transfer heatmap
3. Weak vs Strong donor comparison
4. Dimension correlation distribution
5. Category-level transfer matrix
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict


def load_results(data_dir):
    path = os.path.join(data_dir, "analysis_results.json")
    with open(path, "r") as f:
        return json.load(f)


def set_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def fig1_optimized_vs_random(results, output_dir):
    """Figure 1: Optimized vs Random gap distribution."""
    set_style()
    gap_data = results["optimized_vs_random"]

    models = sorted(gap_data.keys())
    all_gaps = {dim: [] for dim in [64, 128, 256, 512]}

    for model_name in models:
        for task_name, task_data in gap_data[model_name]["tasks"].items():
            for dim_str, gap in task_data.get("gap", {}).items():
                dim = int(dim_str)
                if dim in all_gaps:
                    all_gaps[dim].append(gap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: box plot of gaps per dimension
    dims = [64, 128, 256, 512]
    gap_values = [all_gaps[d] for d in dims]
    bp = axes[0].boxplot(gap_values, tick_labels=[str(d) for d in dims], patch_artist=True)
    colors = ['#4ECDC4', '#45B7D1', '#5B8DEE', '#7C6EF0']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference')
    axes[0].set_xlabel('Target Dimension')
    axes[0].set_ylabel('Gap (Optimized - Random) %')
    axes[0].set_title('(a) Optimized vs Random Selection Gap')
    axes[0].legend()

    # Right: per-model summary
    model_gaps = {}
    for model_name in models:
        gaps = []
        for task_name, task_data in gap_data[model_name]["tasks"].items():
            for dim_str, gap in task_data.get("gap", {}).items():
                if int(dim_str) == 256:
                    gaps.append(gap)
        if gaps:
            model_gaps[model_name] = np.mean(gaps)

    sorted_models = sorted(model_gaps.items(), key=lambda x: x[1])
    names = [m[0] for m in sorted_models]
    vals = [m[1] for m in sorted_models]
    colors_bar = ['#E74C3C' if v < 0 else '#27AE60' for v in vals]
    axes[1].barh(range(len(names)), vals, color=colors_bar, alpha=0.8)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Mean Gap at dim=256 (%)')
    axes[1].set_title('(b) Per-Model Optimized-Random Gap')

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_optimized_vs_random.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig2_cross_task_heatmap(results, output_dir):
    """Figure 2: Cross-task transfer retention heatmap."""
    set_style()
    transfer_data = results["cross_task_transfer"]

    for model_name, model_data in transfer_data.items():
        tasks = model_data["tasks"]
        n = len(tasks)

        # Build retention matrix
        matrix = np.zeros((n, n))
        for i, donor in enumerate(tasks):
            for j, target in enumerate(tasks):
                if donor in model_data and target in model_data.get(donor, {}):
                    donor_score = model_data[donor][target]
                    self_score = model_data.get("self_transfer", {}).get(target, 1)
                    if self_score and self_score != 0:
                        matrix[i, j] = donor_score / self_score
                    else:
                        matrix[i, j] = np.nan
                else:
                    matrix[i, j] = np.nan

        # Sort tasks by self-transfer (ascending = weakest first)
        self_scores = [model_data.get("self_transfer", {}).get(t, 0) for t in tasks]
        sort_idx = np.argsort(self_scores)
        tasks_sorted = [tasks[i] for i in sort_idx]
        matrix_sorted = matrix[sort_idx][:, sort_idx]

        # Shorten task names for display
        short_names = [t.replace("Classification", "Cls").replace("Clustering", "Clust")
                       .replace("PairClassification", "Pair").replace("Reranking", "Rerank")
                       .replace("Retrieval", "Retr").replace("Summarization", "Summ")[:20]
                      for t in tasks_sorted]

        fig, ax = plt.subplots(figsize=(14, 12))
        cmap = LinearSegmentedColormap.from_list("rg", ["#E74C3C", "#F39C12", "#27AE60"])
        im = ax.imshow(matrix_sorted, cmap=cmap, vmin=0.5, vmax=1.2, aspect='auto')
        plt.colorbar(im, ax=ax, label='Retention Ratio', shrink=0.8)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_names, rotation=90, fontsize=6)
        ax.set_yticklabels(short_names, fontsize=6)
        ax.set_title(f'{model_name}\nCross-Task Dimension Importance Transfer', fontsize=14)

        # Add category labels
        categories = {
            "Classification": "#3498DB", "Clustering": "#E67E22",
            "STS": "#9B59B6", "Retrieval": "#1ABC9C",
            "Reranking": "#E74C3C", "PairClassification": "#F1C40F",
            "Summarization": "#2ECC71"
        }
        def classify_task_category(task_name, task_categories=None):
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

        for i, task in enumerate(tasks_sorted):
            cat = classify_task_category(task)
            if cat in categories:
                ax.get_yticklabels()[i].set_color(categories[cat])

        plt.tight_layout()
        path = os.path.join(output_dir, f"fig2_transfer_heatmap_{model_name}.png")
        plt.savefig(path)
        plt.close()
        print(f"Saved: {path}")


def fig3_weak_vs_strong_donors(results, output_dir):
    """Figure 3: Weak vs Strong donor quality comparison."""
    set_style()
    ws_data = results.get("weak_vs_strong_donors", {})

    if not ws_data:
        print("No weak vs strong donor data available")
        return

    models = sorted(ws_data.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: grouped bar chart
    x = np.arange(n_models)
    width = 0.25

    weak_means = [ws_data[m]["weak_donor_retention"]["mean"] for m in models]
    weak_ci_low = [ws_data[m]["weak_donor_retention"]["ci_low"] for m in models]
    weak_ci_high = [ws_data[m]["weak_donor_retention"]["ci_high"] for m in models]

    mid_means = [ws_data[m]["mid_donor_retention"]["mean"] for m in models]
    mid_ci_low = [ws_data[m]["mid_donor_retention"]["ci_low"] for m in models]
    mid_ci_high = [ws_data[m]["mid_donor_retention"]["ci_high"] for m in models]

    strong_means = [ws_data[m]["strong_donor_retention"]["mean"] for m in models]
    strong_ci_low = [ws_data[m]["strong_donor_retention"]["ci_low"] for m in models]
    strong_ci_high = [ws_data[m]["strong_donor_retention"]["ci_high"] for m in models]

    weak_err = [[w - l for w, l in zip(weak_means, weak_ci_low)],
                [h - w for w, h in zip(weak_means, weak_ci_high)]]
    mid_err = [[m - l for m, l in zip(mid_means, mid_ci_low)],
               [h - m for m, h in zip(mid_means, mid_ci_high)]]
    strong_err = [[s - l for s, l in zip(strong_means, strong_ci_low)],
                  [h - s for s, h in zip(strong_means, strong_ci_high)]]

    short_models = [m[:20] for m in models]
    axes[0].bar(x - width, weak_means, width, yerr=weak_err, label='Weak Donors',
                color='#E74C3C', alpha=0.8, capsize=3)
    axes[0].bar(x, mid_means, width, yerr=mid_err, label='Mid Donors',
                color='#F39C12', alpha=0.8, capsize=3)
    axes[0].bar(x + width, strong_means, width, yerr=strong_err, label='Strong Donors',
                color='#27AE60', alpha=0.8, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_models, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('Average Cross-Task Retention')
    axes[0].set_title('(a) Donor Quality by Task Strength Quartile')
    axes[0].legend()
    axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)

    # Right: effect size
    effect_sizes = [ws_data[m]["effect_size_weak_vs_strong"] for m in models]
    colors = ['#27AE60' if es > 0 else '#E74C3C' for es in effect_sizes]
    axes[1].barh(range(n_models), effect_sizes, color=colors, alpha=0.8)
    axes[1].set_yticks(range(n_models))
    axes[1].set_yticklabels(short_models, fontsize=8)
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel("Cohen's d (Weak - Strong)")
    axes[1].set_title('(b) Effect Size: Weak vs Strong Donor Retention')
    axes[1].set_xlim(-1.5, 1.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_weak_vs_strong_donors.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig4_dimension_correlation(results, output_dir):
    """Figure 4: Distribution of pairwise ranking correlations."""
    set_style()
    corr_data = results.get("dimension_correlation", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (model_name, model_data) in enumerate(sorted(corr_data.items())):
        ax = axes[i % 2]
        rhos = [v["rho"] for v in model_data["pairwise"].values()]
        p_vals = [v["p_value"] for v in model_data["pairwise"].values()]

        ax.hist(rhos, bins=30, alpha=0.7, color='#5B8DEE', edgecolor='white')
        ax.axvline(x=np.mean(rhos), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(rhos):.3f}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Spearman Rank Correlation (rho)')
        ax.set_ylabel('Count')
        ax.set_title(f'{model_name}\nPairwise Dimension Ranking Correlation')
        ax.legend()

        # Annotate significance
        n_sig = sum(1 for p in p_vals if p < 0.05)
        n_total = len(p_vals)
        ax.text(0.95, 0.95, f'{n_sig}/{n_total} significant\n(p < 0.05)',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_dimension_correlation.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig5_category_transfer(results, output_dir):
    """Figure 5: Category-level transfer matrix."""
    set_style()
    cat_data = results.get("category_transfer", {})

    categories = ["Classification", "Clustering", "STS", "Retrieval", "Reranking",
                  "PairClassification", "Summarization"]

    # Aggregate across models
    agg_matrix = defaultdict(lambda: defaultdict(list))
    for model_name, model_cats in cat_data.items():
        for donor_cat, target_cats in model_cats.items():
            for target_cat, stats in target_cats.items():
                if isinstance(stats, dict) and "mean" in stats:
                    agg_matrix[donor_cat][target_cat].append(stats["mean"])

    n_cats = len(categories)
    matrix = np.zeros((n_cats, n_cats))
    for i, dc in enumerate(categories):
        for j, tc in enumerate(categories):
            vals = agg_matrix[dc][tc]
            matrix[i, j] = np.mean(vals) if vals else np.nan

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = LinearSegmentedColormap.from_list("rg", ["#E74C3C", "#F39C12", "#27AE60"])
    im = ax.imshow(matrix, cmap=cmap, vmin=0.7, vmax=1.1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Average Retention Ratio')

    ax.set_xticks(range(n_cats))
    ax.set_yticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_title('Category-Level Cross-Task Dimension Transfer\n(Averaged across models)', fontsize=14)

    # Annotate cells
    for i in range(n_cats):
        for j in range(n_cats):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.85 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig5_category_transfer.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--output_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/paper/figures")
    parser.add_argument("--data_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/data/experiment_results")
    args = parser.parse_args()

    results = load_results(args.data_dir)

    print("Generating figures...")
    fig1_optimized_vs_random(results, args.output_dir)
    fig2_cross_task_heatmap(results, args.output_dir)
    fig3_weak_vs_strong_donors(results, args.output_dir)
    fig4_dimension_correlation(results, args.output_dir)
    fig5_category_transfer(results, args.output_dir)
    print("All figures generated!")


if __name__ == "__main__":
    main()
