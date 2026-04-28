"""
Generate all data-driven paper figures with proper density.
Replaces the sparse lollipop/scatter approach with filled bar charts
and substantive visual elements.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "/home/linkco/exa/llm-usefulEeb/paper/figures"
ANALYZE_DIR = "/home/linkco/exa/llm-usefulEeb/data/analyze"
ANALYSIS_DIR = "/home/linkco/exa/llm-usefulEeb/data/experiment_results"

MODEL_DISPLAY = {
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella-400M",
    "roberta-large-InBedder": "RoBERTa-IB",
    "bge-m3": "BGE-M3",
    "instructor-large": "Instructor",
    "mxbai-embed-large-v1": "MxBai-Large",
    "gte-base": "GTE-Base",
    "gtr-t5-large": "GTR-T5",
    "bart-base": "BART-Base",
    "roberta-large": "RoBERTa-Large",
    "gte-Qwen2-1.5B-instruct": "GTE-Qwen2",
    "Qwen3-Embedding-0.6B": "Qwen3-Emb",
}

NON_CONTRASTIVE = {"BART-Base", "RoBERTa-Large"}

TABLE2_DATA = [
    ("MxBai-Large",    +2.24, 1.35, 3.29, 0.76, 0.001, True),
    ("GTE-Large",      +2.41, 1.49, 3.39, 0.83, 0.001, True),
    ("Instructor",     +2.89, 1.78, 4.09, 0.83, 0.001, True),
    ("GTE-Base",       +3.45, 2.11, 4.93, 0.81, 0.001, True),
    ("Stella-400M",    +3.22, 2.17, 4.44, 0.93, 0.001, True),
    ("BGE-M3",         +4.62, 3.21, 6.13, 1.04, 0.001, True),
    ("GTR-T5",         +4.75, 2.98, 6.67, 0.86, 0.001, True),
    ("Qwen3-Emb",      +5.01, 3.36, 6.84, 0.97, 0.001, True),
    ("RoBERTa-Large",  +8.56, -5.85, 18.88, 0.23, 0.191, False),
    ("RoBERTa-IB",     +8.19, 5.99, 10.56, 1.17, 0.001, True),
    ("BART-Base",      +10.08, 6.28, 13.73, 0.90, 0.001, True),
]

C_CONTRASTIVE = '#4A90D9'
C_NON_CONTRASTIVE = '#E74C3C'
C_METHODS = {
    'random': '#6C757D',
    'sequential': '#4A90D9',
    'optimized': '#27AE60',
    'anti_opt': '#E74C3C',
    'magnitude': '#E67E22',
}


def set_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_analyze_data():
    data = {}
    for fname in os.listdir(ANALYZE_DIR):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(ANALYZE_DIR, fname)) as f:
                data[model_name] = json.load(f)
    return data


def get_method_scores(analyze_data, model_name, target_dim=256):
    model_data = analyze_data.get(model_name)
    if model_data is None:
        return None
    task_scores = {}
    for task_name, task_data in model_data.get("task_name", {}).items():
        if task_name == "STS17":
            continue
        default = task_data.get("defult_score", 0)
        if default <= 0:
            continue
        scores = {"default": default}
        random_data = task_data.get("random_score", {}).get(str(target_dim), [])
        if random_data:
            scores["random"] = float(np.mean(random_data))
        sort_val = task_data.get("sort_score", {}).get(str(target_dim))
        if sort_val is not None:
            scores["sort"] = float(sort_val)
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


# ============================================================
# FIGURE: All Methods Comparison (replaces fig10)
# ============================================================
def fig_all_methods_comparison():
    """Grouped bar chart: retention by method across 3 representative models."""
    set_style()
    analyze_data = load_analyze_data()

    model_configs = [
        ("stella_en_400M_v5", "Stella-400M"),
        ("gte-large-en-v1.5", "GTE-Large"),
        ("roberta-large-InBedder", "RoBERTa-IB"),
    ]

    methods = ['Random', 'Sequential', 'Optimized', 'Anti-opt.']
    method_keys = ['random', 'sort', 'best', 'poor']
    method_colors = [C_METHODS['random'], C_METHODS['sequential'],
                     C_METHODS['optimized'], C_METHODS['anti_opt']]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    for ax_idx, (model_key, model_display) in enumerate(model_configs):
        ax = axes[ax_idx]
        ts = get_method_scores(analyze_data, model_key)
        if ts is None:
            continue

        ret_means = []
        ret_stds = []
        for key in method_keys:
            vals = []
            for tname, s in ts.items():
                if key in s:
                    vals.append(s[key] / s['default'] * 100)
            ret_means.append(np.mean(vals) if vals else 0)
            ret_stds.append(np.std(vals) if vals else 0)

        x = np.arange(len(methods))
        width = 0.6
        bars = ax.bar(x, ret_means, width, yerr=ret_stds, color=method_colors,
                      alpha=0.8, capsize=3, error_kw={'linewidth': 1},
                      edgecolor='white', linewidth=0.5)

        ax.axhline(y=100, color='#333333', linestyle=':', alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8, rotation=15, ha='right')
        ax.set_title(model_display, fontsize=12, fontweight='bold')
        ax.set_ylim(75, 110)

        # Value labels
        for bar, mean in zip(bars, ret_means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 1,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    axes[0].set_ylabel('Mean Retention (%)', fontsize=12)

    legend_elements = [mpatches.Patch(facecolor=c, alpha=0.8, label=m)
                       for m, c in zip(methods, method_colors)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig10_all_methods_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE: Magnitude Analysis (replaces fig9)
# ============================================================
def fig_magnitude_analysis():
    """Bar chart of magnitude correlation + scatter of magnitude vs random per task."""
    set_style()
    analyze_data = load_analyze_data()

    mag_path = os.path.join(ANALYSIS_DIR, "magnitude_analysis.json")
    with open(mag_path) as f:
        mag_analysis = json.load(f)

    gte_path = os.path.join(ANALYSIS_DIR, "magnitude_gte_mteb.json")
    with open(gte_path) as f:
        gte_mteb = json.load(f)

    stella_path = os.path.join(ANALYSIS_DIR, "magnitude_stella_mteb.json")
    with open(stella_path) as f:
        stella_mteb = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # (a) Magnitude correlation bar chart
    ax = axes[0]
    mag_corr = mag_analysis.get("magnitude_vs_task_correlation", {})
    models_sorted = sorted(mag_corr.keys())
    y = np.arange(len(models_sorted))
    means = [mag_corr[m]["summary"]["mean_rho"] for m in models_sorted]
    stds = [mag_corr[m]["summary"]["std_rho"] for m in models_sorted]
    colors = [C_NON_CONTRASTIVE if MODEL_DISPLAY.get(m, m) in NON_CONTRASTIVE
              else C_CONTRASTIVE for m in models_sorted]

    ax.barh(y, means, xerr=stds, color=colors, alpha=0.75, capsize=3,
            height=0.5, edgecolor='white')
    ax.axvline(x=0, color='#333333', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m)[:12] for m in models_sorted], fontsize=9)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("(a) Magnitude vs Task\nImportance Correlation", fontsize=11)
    ax.set_xlim(-0.15, 0.15)

    # (b) GTE scatter: magnitude retention vs random retention
    ax = axes[1]
    ts = get_method_scores(analyze_data, "gte-large-en-v1.5")
    mag_scores = gte_mteb.get("scores", {})
    if ts:
        rnd_rets, mag_rets = [], []
        for tname, s in ts.items():
            if tname in mag_scores:
                mag_rets.append(mag_scores[tname] / s["default"] * 100)
                rnd_rets.append(s["random"] / s["default"] * 100)
        mag_rets = np.array(mag_rets)
        rnd_rets = np.array(rnd_rets)
        ax.scatter(rnd_rets, mag_rets, c=C_CONTRASTIVE, s=40, alpha=0.7,
                   edgecolors='white', linewidth=0.3, zorder=3)
        lo, hi = min(rnd_rets.min(), mag_rets.min()) - 1, 101
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1)
        wins = np.sum(mag_rets > rnd_rets)
        ax.annotate(f'Mag wins: {wins}/{len(mag_rets)}',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    va='top', fontsize=9,
                    bbox=dict(facecolor='#F0F0F0', alpha=0.8, edgecolor='#CCC'))
    ax.set_xlabel("Random Retention (%)")
    ax.set_ylabel("Magnitude Retention (%)")
    ax.set_title("(b) GTE-Large\nPer-Task Comparison", fontsize=11)
    ax.set_aspect('equal')

    # (c) Stella scatter
    ax = axes[2]
    ts2 = get_method_scores(analyze_data, "stella_en_400M_v5")
    mag_scores2 = stella_mteb.get("scores", {})
    if ts2:
        rnd_rets2, mag_rets2 = [], []
        for tname, s in ts2.items():
            if tname in mag_scores2:
                mag_rets2.append(mag_scores2[tname] / s["default"] * 100)
                rnd_rets2.append(s["random"] / s["default"] * 100)
        mag_rets2 = np.array(mag_rets2)
        rnd_rets2 = np.array(rnd_rets2)
        ax.scatter(rnd_rets2, mag_rets2, c='#4ECDC4', s=40, alpha=0.7,
                   edgecolors='white', linewidth=0.3, zorder=3)
        lo2, hi2 = min(rnd_rets2.min(), mag_rets2.min()) - 1, 101
        ax.plot([lo2, hi2], [lo2, hi2], 'k--', alpha=0.3, linewidth=1)
        wins2 = np.sum(mag_rets2 > rnd_rets2)
        ax.annotate(f'Mag wins: {wins2}/{len(mag_rets2)}',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    va='top', fontsize=9,
                    bbox=dict(facecolor='#F0F0F0', alpha=0.8, edgecolor='#CCC'))
    ax.set_xlabel("Random Retention (%)")
    ax.set_ylabel("Magnitude Retention (%)")
    ax.set_title("(c) Stella-400M\nPer-Task Comparison", fontsize=11)
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig9_magnitude_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE: Magnitude Scatter (replaces fig11)
# ============================================================
def fig_magnitude_scatter():
    """Side-by-side scatter of magnitude vs random for GTE and Stella."""
    set_style()
    analyze_data = load_analyze_data()

    with open(os.path.join(ANALYSIS_DIR, "magnitude_gte_mteb.json")) as f:
        gte_mteb = json.load(f)
    with open(os.path.join(ANALYSIS_DIR, "magnitude_stella_mteb.json")) as f:
        stella_mteb = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    configs = [
        (ax1, "gte-large-en-v1.5", gte_mteb, "GTE-Large", C_CONTRASTIVE, -0.28, 0.002),
        (ax2, "stella_en_400M_v5", stella_mteb, "Stella-400M", '#4ECDC4', -0.05, 0.38),
    ]

    for ax, model_key, mag_data, display_name, color, d_val, p_val in configs:
        ts = get_method_scores(analyze_data, model_key)
        mag_scores = mag_data.get("scores", {})
        rnd_rets, mag_rets = [], []
        for tname, s in ts.items():
            if tname in mag_scores:
                mag_rets.append(mag_scores[tname] / s["default"] * 100)
                rnd_rets.append(s["random"] / s["default"] * 100)
        mag_rets = np.array(mag_rets)
        rnd_rets = np.array(rnd_rets)

        # Color by who wins
        for i in range(len(mag_rets)):
            c = C_METHODS['optimized'] if mag_rets[i] > rnd_rets[i] + 0.3 else (
                C_METHODS['anti_opt'] if rnd_rets[i] > mag_rets[i] + 0.3 else C_METHODS['random'])
            ax.scatter([rnd_rets[i]], [mag_rets[i]], c=c, s=50, alpha=0.7,
                       edgecolors='white', linewidth=0.3, zorder=3)

        lo = min(rnd_rets.min(), mag_rets.min()) - 1
        hi = max(rnd_rets.max(), mag_rets.max()) + 1
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel("Random Selection Retention (%)", fontsize=11)
        ax.set_ylabel("Magnitude Selection Retention (%)", fontsize=11)
        ax.set_title(f'{display_name} (dim=256)', fontsize=12, fontweight='bold')
        ax.annotate(f'd = {d_val:.2f}, p = {p_val:.3f}',
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=10,
                    bbox=dict(facecolor='#F5F5F5', alpha=0.8, edgecolor='#CCC'))
        ax.set_aspect('equal')

    legend_elements = [
        mpatches.Patch(facecolor=C_METHODS['optimized'], alpha=0.7, label='Magnitude wins'),
        mpatches.Patch(facecolor=C_METHODS['anti_opt'], alpha=0.7, label='Random wins'),
        mpatches.Patch(facecolor=C_METHODS['random'], alpha=0.7, label='Tie'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig11_magnitude_scatter.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE: Opt-Random Gap (replaces fig_opt_random_gap)
# ============================================================
def fig_opt_random_gap():
    """Horizontal bar chart: optimized-random gap for all 11 models."""
    set_style()

    data = sorted(TABLE2_DATA, key=lambda x: x[1])
    names = [d[0] for d in data]
    gaps = [d[1] for d in data]
    ci_lo = [d[2] for d in data]
    ci_hi = [d[3] for d in data]

    y = np.arange(len(names))
    err_lo = [g - lo for g, lo in zip(gaps, ci_lo)]
    err_hi = [hi - g for g, hi in zip(gaps, ci_hi)]

    colors = [C_NON_CONTRASTIVE if n in NON_CONTRASTIVE else C_CONTRASTIVE for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(y, gaps, xerr=[err_lo, err_hi], color=colors, alpha=0.75,
            capsize=3, error_kw={'linewidth': 1, 'color': '#333'},
            height=0.6, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='#333', linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Optimized − Random Gap (%)", fontsize=12)
    ax.set_xlim(-8, 16)

    for i, (name, gap) in enumerate(zip(names, gaps)):
        ax.text(gap + 0.3, i, f'+{gap:.1f}%', va='center', fontsize=8, color='#333')

    legend_elements = [
        mpatches.Patch(facecolor=C_CONTRASTIVE, alpha=0.75, label="Contrastive"),
        mpatches.Patch(facecolor=C_NON_CONTRASTIVE, alpha=0.75, label="Non-contrastive"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_opt_random_gap_all_models.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating data figures...")
    fig_opt_random_gap()
    fig_all_methods_comparison()
    fig_magnitude_analysis()
    fig_magnitude_scatter()
    print("Done!")
