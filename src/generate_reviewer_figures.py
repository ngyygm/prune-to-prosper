"""
Generate additional figures addressing reviewer feedback.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(data_dir):
    paths = [
        os.path.join(data_dir, "analysis_results.json"),
        os.path.join(data_dir, "reviewer_response_analysis.json"),
    ]
    results = {}
    for path in paths:
        if os.path.exists(path):
            key = os.path.basename(path).replace('.json', '')
            with open(path, "r") as f:
                results[key] = json.load(f)
    return results


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


def fig6_pruning_ratio_sweep(results, output_dir):
    """Figure 6: Performance vs pruning ratio for sort vs random."""
    set_style()
    data = results.get("reviewer_response_analysis", {}).get("pruning_ratio_sweep", {})
    n_models = len(data)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    cmap = plt.cm.tab10
    model_colors = {name: cmap(i / max(n_models - 1, 1)) for i, name in enumerate(data)}

    for i, (model_name, model_data) in enumerate(data.items()):
        ax = axes[i]
        ratios = sorted(model_data["ratios"].keys())
        sort_vals = [model_data["ratios"][r]["avg_sort"] for r in ratios]
        sort_stds = [model_data["ratios"][r]["sort_std"] for r in ratios]
        rand_vals = [model_data["ratios"][r]["avg_random"] for r in ratios]
        rand_stds = [model_data["ratios"][r]["random_std"] for r in ratios]

        color = model_colors.get(model_name, '#333333')

        ax.fill_between(ratios,
                        np.array(sort_vals) - np.array(sort_stds),
                        np.array(sort_vals) + np.array(sort_stds),
                        alpha=0.15, color=color)
        ax.plot(ratios, sort_vals, 'o-', color=color, label='Sequential (Sort)', markersize=3)

        ax.fill_between(ratios,
                        np.array(rand_vals) - np.array(rand_stds),
                        np.array(rand_vals) + np.array(rand_stds),
                        alpha=0.15, color='#999999')
        ax.plot(ratios, rand_vals, 's--', color='#999999', label='Random', markersize=3)

        ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Full-dim baseline')

        ax.set_xlabel('Fraction of Dimensions Retained')
        ax.set_ylabel('Normalized Performance')
        short = model_name[:20]
        ax.set_title(f'{short}\n({model_data["model_dim"]}d)', fontsize=9)
        ax.legend(loc='lower right', fontsize=7)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0.15, 1.15)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig6_pruning_ratio_sweep.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig7_redundancy_mechanism(results, output_dir):
    """Figure 7: Visualizing the redundancy mechanism (why dimensions are interchangeable)."""
    set_style()
    data = results.get("reviewer_response_analysis", {}).get("redundancy_mechanism", {})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cmap = plt.cm.tab10
    model_colors = {name: cmap(i / max(len(data) - 1, 1)) for i, name in enumerate(data)}

    # (a) Normalized entropy distribution across tasks
    ax = axes[0]
    for model_name, model_data in data.items():
        entropies = [v["normalized_entropy"] for v in model_data["tasks"].values()]
        color = model_colors.get(model_name, '#333333')
        ax.hist(entropies, bins=20, alpha=0.5, color=color, label=model_name[:12])
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Maximum (uniform)')
    ax.set_xlabel('Normalized Entropy')
    ax.set_ylabel('Number of Tasks')
    ax.set_title('(a) Dimension Importance Entropy\n(close to 1.0 = all dimensions equally important)')
    ax.legend(fontsize=7)

    # (b) Top-K concentration curves
    ax = axes[1]
    for model_name, model_data in data.items():
        summary = model_data.get("model_summary", {})
        if not summary:
            continue
        x_labels = ['Top 10%', 'Top 25%', 'Top 50%']
        y_vals = [summary['avg_top_10pct_concentration'],
                  summary['avg_top_25pct_concentration'],
                  summary['avg_top_50pct_concentration']]
        color = model_colors.get(model_name, '#333333')
        ax.plot(x_labels, y_vals, 'o-', color=color, label=model_name[:12], markersize=8)

    # Reference lines for comparison
    ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=0.50, color='gray', linestyle=':', alpha=0.3)
    ax.set_ylabel('Fraction of Total Score')
    ax.set_title('(b) Score Concentration\n(lower = more redundant)')
    ax.legend(fontsize=7)

    # (c) Best-Poor gap vs number of chunks
    ax = axes[2]
    inter_data = results.get("reviewer_response_analysis", {}).get("interchangeability_evidence", {})
    for model_name, model_data in inter_data.get("models", {}).items():
        gaps = model_data.get("best_poor_gap_vs_n_chunks", {})
        if gaps:
            n_chunks = sorted([int(k) for k in gaps.keys()])
            gap_means = [gaps[str(k)]["mean"] for k in n_chunks]
            color = model_colors.get(model_name, '#333333')
            ax.plot(n_chunks, gap_means, 'o-', color=color, label=model_name[:12], markersize=5)

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Number of Chunks Selected')
    ax.set_ylabel('Best-Poor Gap (normalized)')
    ax.set_title('(c) Selection Quality Gap\n(gap → 0 as more chunks selected)')
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig7_redundancy_mechanism.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig8_evidence_summary(results, output_dir):
    """Figure 8: Summary of all evidence for dimension interchangeability."""
    set_style()

    inter_data = results.get("reviewer_response_analysis", {}).get("interchangeability_evidence", {})
    models_data = inter_data.get("models", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = sorted(models_data.keys())
    short_names = [n[:12] for n in model_names]
    x = np.arange(len(model_names))

    # (a) Optimized-Random gap
    ax = axes[0]
    gaps = [(models_data[m]["optimized_random_gap"] or {}).get("mean", 0) for m in model_names]
    gap_stds = [(models_data[m]["optimized_random_gap"] or {}).get("std", 0) for m in model_names]
    colors = ['#27AE60' if g <= 0 else '#E74C3C' for g in gaps]
    ax.barh(x, gaps, xerr=gap_stds, color=colors, alpha=0.7, capsize=3)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Optimized - Random Gap (%)')
    ax.set_title('(a) Task-Aware Selection Advantage')
    ax.set_xlim(-5, 5)

    # (b) Cross-task transfer retention
    ax = axes[1]
    retentions = [(models_data[m]["cross_task_transfer_retention"] or {}).get("mean", 0) for m in model_names]
    ret_stds = [(models_data[m]["cross_task_transfer_retention"] or {}).get("std", 0) for m in model_names]
    ax.barh(x, retentions, xerr=ret_stds, color='#5B8DEE', alpha=0.7, capsize=3)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Self-transfer')
    ax.set_yticks(x)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Cross-Task Retention Ratio')
    ax.set_title('(b) Cross-Task Transfer Quality')
    ax.legend()
    ax.set_xlim(0.8, 1.1)

    # (c) Chunk entropy (uniformity of dimension importance)
    ax = axes[2]
    entropies = [(models_data[m]["chunk_entropy"] or {}).get("mean", 0) for m in model_names]
    ent_stds = [(models_data[m]["chunk_entropy"] or {}).get("std", 0) for m in model_names]
    ax.barh(x, entropies, xerr=ent_stds, color='#9B59B6', alpha=0.7, capsize=3)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Perfectly uniform')
    ax.set_yticks(x)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Normalized Entropy')
    ax.set_title('(c) Dimension Importance Uniformity')
    ax.legend()
    ax.set_xlim(0.95, 1.0)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig8_evidence_summary.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    data_dir = "/home/linkco/exa/llm-usefulEeb/data/experiment_results"
    output_dir = "/home/linkco/exa/llm-usefulEeb/paper/figures"
    results = load_results(data_dir)

    print("Generating reviewer response figures...")
    fig6_pruning_ratio_sweep(results, output_dir)
    fig7_redundancy_mechanism(results, output_dir)
    fig8_evidence_summary(results, output_dir)
    print("All reviewer response figures generated!")


if __name__ == "__main__":
    main()
