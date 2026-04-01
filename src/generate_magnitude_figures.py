"""
Generate magnitude-based pruning comparison figures.
Updated with actual MTEB results.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def load_analyze_data(analyze_dir):
    """Load original analyze data from Useful-Embedding."""
    data = {}
    for fname in os.listdir(analyze_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(analyze_dir, fname), "r") as f:
                data[model_name] = json.load(f)
    return data


def get_method_scores_for_model(analyze_data, model_name, target_dim=256):
    """Extract random, sort, best, poor scores at a given dimension."""
    model_data = analyze_data.get(model_name)
    if model_data is None:
        return None

    task_scores = {}
    for task_name, task_data in model_data.get("task_name", {}).items():
        default = task_data.get("defult_score", 0)
        if default <= 0:
            continue

        scores = {"default": default}

        # Random
        random_data = task_data.get("random_score", {}).get(str(target_dim), [])
        if random_data:
            scores["random"] = float(np.mean(random_data))

        # Sort
        sort_val = task_data.get("sort_score", {}).get(str(target_dim))
        if sort_val is not None:
            scores["sort"] = float(sort_val)

        # Best/Poor from split_win_size
        for ws_str, ws_data in task_data.get("split_win_size", {}).items():
            for td_str, td_data in ws_data.get("chunk_win_size", {}).items():
                if td_str == str(target_dim):
                    head = td_data.get("head_score", {}).get("main_score")
                    end = td_data.get("end_score", {}).get("main_score")
                    if head is not None:
                        scores["best"] = float(head)
                    if end is not None:
                        scores["poor"] = float(end)

        if "random" in scores:
            task_scores[task_name] = scores

    return task_scores


def fig9_magnitude_vs_methods(output_dir, analyze_dir, mag_results_dir):
    """
    Figure 9: (a) Magnitude vs Task ranking correlation
              (b) Per-task magnitude vs random retention for gte-large
    """
    set_style()

    # Load magnitude analysis
    mag_analysis_path = os.path.join(mag_results_dir, "magnitude_analysis.json")
    with open(mag_analysis_path) as f:
        mag_analysis = json.load(f)

    # Load gte-large MTEB results
    gte_mteb_path = os.path.join(mag_results_dir, "magnitude_gte_mteb.json")
    gte_mteb = None
    if os.path.exists(gte_mteb_path):
        with open(gte_mteb_path) as f:
            gte_mteb = json.load(f)

    # Load stella MTEB results
    stella_mteb_path = os.path.join(mag_results_dir, "magnitude_stella_mteb.json")
    stella_mteb = None
    if os.path.exists(stella_mteb_path):
        with open(stella_mteb_path) as f:
            stella_mteb = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Magnitude vs Task ranking correlation
    ax = axes[0]
    mag_data = mag_analysis.get("magnitude_vs_task_correlation", {})
    models = sorted(mag_data.keys())
    x = np.arange(len(models))
    means = [mag_data[m]["summary"]["mean_rho"] for m in models]
    stds = [mag_data[m]["summary"]["std_rho"] for m in models]

    colors = ['#4ECDC4', '#5B8DEE', '#E74C3C']
    for i, (model_name, mean, std) in enumerate(zip(models, means, stds)):
        ax.barh(i, mean, xerr=std, color=colors[i % len(colors)], alpha=0.7, capsize=3)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(x)
    short_names = [n.split('-')[0][:15] for n in models]
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Spearman Rank Correlation ($\\rho$)")
    ax.set_title('(a) Magnitude Ranking vs Task-Specific Ranking\n($\\rho \\approx 0$: magnitude is uninformative)')
    ax.set_xlim(-0.15, 0.15)

    # (b) Per-task comparison: magnitude vs random retention
    ax = axes[1]

    if gte_mteb:
        analyze_data = load_analyze_data(analyze_dir)
        task_scores = get_method_scores_for_model(analyze_data, "gte-large-en-v1.5")
        mag_scores = gte_mteb.get("scores", {})

        if task_scores:
            # Build paired data, excluding STS17 outlier
            tasks, mag_rets, rnd_rets = [], [], []
            for tname, tscores in task_scores.items():
                if tname == "STS17":
                    continue
                if tname not in mag_scores:
                    continue
                mag_ret = mag_scores[tname] / tscores["default"]
                rnd_ret = tscores["random"] / tscores["default"]
                tasks.append(tname)
                mag_rets.append(mag_ret)
                rnd_rets.append(rnd_ret)

            tasks = np.array(tasks)
            mag_rets = np.array(mag_rets)
            rnd_rets = np.array(rnd_rets)

            # Sort by random retention
            sort_idx = np.argsort(rnd_rets)
            tasks = tasks[sort_idx]
            mag_rets = mag_rets[sort_idx]
            rnd_rets = rnd_rets[sort_idx]

            x = np.arange(len(tasks))
            ax.scatter(x, rnd_rets, c='#999999', s=20, alpha=0.7, label='Random', zorder=3)
            ax.scatter(x, mag_rets, c='#E67E22', s=20, alpha=0.7, label='Magnitude', zorder=3)
            ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3)

            # Add diagonal reference
            ax.plot([0, len(tasks)-1], [0.95, 0.95], 'k--', alpha=0.2, linewidth=0.5)

            ax.set_xticks(x[::4])
            short_tasks = [t[:12] for t in tasks[::4]]
            ax.set_xticklabels(short_tasks, rotation=90, fontsize=5)
            ax.set_ylabel('Retention (pruned / full-dim)')
            ax.set_xlabel('Tasks (sorted by random retention)')
            ax.set_title('(b) Magnitude vs Random per Task (gte-large, dim=256)\n(Random wins 24/34 tasks)')
            ax.legend(loc='lower left', fontsize=9)
            ax.set_ylim(0.85, 1.02)

            # Add annotation
            mag_wins = np.sum(mag_rets > rnd_rets)
            ax.annotate(f'Magnitude wins: {mag_wins}/{len(tasks)}',
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'MTEB results not yet available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('(b) Method Comparison (pending)')

    plt.tight_layout()
    path = os.path.join(output_dir, "fig9_magnitude_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def fig10_all_methods_with_magnitude(output_dir, analyze_dir, mag_results_dir):
    """
    Figure 10: Comprehensive comparison of all pruning methods at dim=256,
    including magnitude results for gte-large.
    """
    set_style()
    analyze_data = load_analyze_data(analyze_dir)

    # Load magnitude MTEB results
    gte_mteb = None
    gte_path = os.path.join(mag_results_dir, "magnitude_gte_mteb.json")
    if os.path.exists(gte_path):
        with open(gte_path) as f:
            gte_mteb = json.load(f)

    stella_mteb = None
    stella_path = os.path.join(mag_results_dir, "magnitude_stella_mteb.json")
    if os.path.exists(stella_path):
        with open(stella_path) as f:
            stella_mteb = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    model_configs = {
        "stella_en_400M_v5": "stella",
        "gte-large-en-v1.5": "gte-large",
        "roberta-large-InBedder": "roberta-InBedder",
    }

    mag_mteb_map = {}
    if stella_mteb:
        mag_mteb_map["stella_en_400M_v5"] = stella_mteb
    if gte_mteb:
        mag_mteb_map["gte-large-en-v1.5"] = gte_mteb

    models = list(model_configs.keys())
    n = len(models)
    x = np.arange(n)
    width = 0.15

    method_names = ["Random", "Sequential", "Magnitude", "Best (Oracle)", "Poor"]
    colors = ['#999999', '#5B8DEE', '#E67E22', '#27AE60', '#E74C3C']

    all_methods = {m: {method: [] for method in method_names} for m in models}

    for model_name in models:
        task_scores = get_method_scores_for_model(analyze_data, model_name)
        if task_scores is None:
            continue

        mag_scores = mag_mteb_map.get(model_name, {}).get("scores", {})

        for task_name, tscores in task_scores.items():
            if task_name == "STS17":
                continue
            default = tscores["default"]
            if default <= 0:
                continue

            if "random" in tscores:
                all_methods[model_name]["Random"].append(tscores["random"] / default)
            if "sort" in tscores:
                all_methods[model_name]["Sequential"].append(tscores["sort"] / default)
            if "best" in tscores:
                all_methods[model_name]["Best (Oracle)"].append(tscores["best"] / default)
            if "poor" in tscores:
                all_methods[model_name]["Poor"].append(tscores["poor"] / default)

            if task_name in mag_scores:
                all_methods[model_name]["Magnitude"].append(mag_scores[task_name] / default)

    for i, method in enumerate(method_names):
        vals = []
        for model_name in models:
            method_vals = all_methods[model_name][method]
            vals.append(np.mean(method_vals) if method_vals else 0)
        ax.bar(x + (i - 2) * width, vals, width, label=method, color=colors[i], alpha=0.8)

    ax.set_xticks(x)
    short_names = [model_configs[m] for m in models]
    ax.set_xticklabels(short_names, rotation=0, fontsize=10)
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3, label='Full-dim baseline')
    ax.set_ylabel('Mean Normalized Retention')
    ax.set_title('All Pruning Methods at dim=256\n(Magnitude $\leq$ Random: no heuristic beats random selection)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0.7, 1.15)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig10_all_methods_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def _build_scatter_data(analyze_data, model_name, mag_mteb, threshold=0.005):
    """Build scatter data for magnitude vs random comparison."""
    task_scores = get_method_scores_for_model(analyze_data, model_name)
    mag_scores = mag_mteb.get("scores", {})

    below, above, near = [], [], []
    for tname, tscores in task_scores.items():
        if tname == "STS17" or tname not in mag_scores:
            continue
        mag_ret = mag_scores[tname] / tscores["default"]
        rnd_ret = tscores["random"] / tscores["default"]
        if mag_ret > rnd_ret + threshold:
            above.append((tname, rnd_ret, mag_ret))
        elif rnd_ret > mag_ret + threshold:
            below.append((tname, rnd_ret, mag_ret))
        else:
            near.append((tname, rnd_ret, mag_ret))
    return below, above, near


def _plot_scatter(ax, below, above, near, lims, title):
    """Plot a single magnitude vs random scatter."""
    if below:
        x, y = zip(*[(t[1], t[2]) for t in below])
        ax.scatter(x, y, c='#E74C3C', s=35, alpha=0.7,
                   label=f'Random wins ({len(below)})', zorder=3)
    if above:
        x, y = zip(*[(t[1], t[2]) for t in above])
        ax.scatter(x, y, c='#27AE60', s=35, alpha=0.7,
                   label=f'Magnitude wins ({len(above)})', zorder=3)
    if near:
        x, y = zip(*[(t[1], t[2]) for t in near])
        ax.scatter(x, y, c='#999999', s=35, alpha=0.7,
                   label=f'Tie ({len(near)})', zorder=3)

    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal performance')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Random Selection Retention')
    ax.set_ylabel('Magnitude Selection Retention')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_aspect('equal')


def fig11_magnitude_scatter(output_dir, analyze_dir, mag_results_dir):
    """
    Figure 11: Scatter plots of magnitude retention vs random retention
    for both stella and gte-large side by side.
    """
    set_style()
    analyze_data = load_analyze_data(analyze_dir)

    gte_path = os.path.join(mag_results_dir, "magnitude_gte_mteb.json")
    stella_path = os.path.join(mag_results_dir, "magnitude_stella_mteb.json")

    if not os.path.exists(gte_path):
        print("No gte-large MTEB results, skipping fig11")
        return

    with open(gte_path) as f:
        gte_mteb = json.load(f)

    has_stella = os.path.exists(stella_path)
    if has_stella:
        with open(stella_path) as f:
            stella_mteb = json.load(f)

    lims = [0.85, 1.02]

    if has_stella:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # gte-large
        below, above, near = _build_scatter_data(
            analyze_data, "gte-large-en-v1.5", gte_mteb)
        _plot_scatter(ax1, below, above, near, lims,
                      '(a) gte-large (dim=256)\nRandom wins 24/34 tasks')
        ax1.annotate(f'$d$ = -0.28, $p$ = 0.002',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.8))

        # stella
        below, above, near = _build_scatter_data(
            analyze_data, "stella_en_400M_v5", stella_mteb)
        _plot_scatter(ax2, below, above, near, lims,
                      '(b) stella (dim=256)\nRandom wins 19/34 tasks')
        ax2.annotate(f'$d$ = -0.05, $p$ = 0.38',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        below, above, near = _build_scatter_data(
            analyze_data, "gte-large-en-v1.5", gte_mteb)
        _plot_scatter(ax, below, above, near, lims,
                      'Magnitude vs Random (gte-large, dim=256)\n'
                      'Most points below diagonal')

    fig.suptitle('Magnitude vs Random Dimension Selection (dim=256)\n'
                 'Points below diagonal = random outperforms magnitude',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig11_magnitude_scatter.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def main():
    output_dir = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output"
    analyze_dir = "/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/analyze"
    mag_results_dir = output_dir

    print("Generating magnitude comparison figures...")
    fig9_magnitude_vs_methods(output_dir, analyze_dir, mag_results_dir)
    fig10_all_methods_with_magnitude(output_dir, analyze_dir, mag_results_dir)
    fig11_magnitude_scatter(output_dir, analyze_dir, mag_results_dir)
    print("Done!")


if __name__ == "__main__":
    main()
