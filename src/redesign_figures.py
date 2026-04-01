"""
Redesign all paper figures for better visual quality and completeness.
Addresses user feedback:
- Figure 3 (all methods): clean names, better design, more info
- Figure 2 (opt-random gap): clean names, better design
- Figure 4 (magnitude): more information
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUTPUT_DIR = "/home/linkco/exa/llm-usefulEeb/EACL__Prune_to_Prosper"
ANALYSIS_DIR = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output"
ANALYZE_DIR = "/home/linkco/exa/llm-usefulEeb/Useful-Embedding/data/analyze"

# Clean model display names
MODEL_DISPLAY = {
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella",
    "roberta-large-InBedder": "Roberta-InBedder",
    "bge-m3": "BGE-M3",
    "instructor-large": "Instructor",
    "mxbai-embed-large-v1": "MxBai-Large",
    "gte-base": "GTE-Base",
    "gtr-t5-large": "GTR-T5-Large",
    "bart-base": "BART-Base",
    "roberta-large": "Roberta-Large",
    "gte-Qwen2-1.5B-instruct": "GTE-Qwen2",
    "Qwen3-Embedding-0.6B": "Qwen3-Emb",
    "stella_en_400M_v5-GEDI-epoch_3": "Stella-GEDI",
}

# Color palette (professional, colorblind-friendly)
COLORS = {
    "random": "#6C757D",
    "sequential": "#4A90D9",
    "magnitude": "#E67E22",
    "optimized": "#27AE60",
    "anti_opt": "#E74C3C",
    "contrastive": "#4A90D9",
    "non_contrastive": "#E74C3C",
}

# Table 2 data from paper (already computed and verified)
TABLE2_DATA = [
    ("MxBai-Large",    +2.24, 1.35, 3.29, 0.76, 0.001, True),
    ("GTE-Large",      +2.41, 1.49, 3.39, 0.83, 0.001, True),
    ("Instructor",     +2.89, 1.78, 4.09, 0.83, 0.001, True),
    ("GTE-Base",       +3.45, 2.11, 4.93, 0.81, 0.001, True),
    ("Stella",         +3.22, 2.17, 4.44, 0.93, 0.001, True),
    ("BGE-M3",         +4.62, 3.21, 6.13, 1.04, 0.001, True),
    ("GTR-T5-Large",   +4.75, 2.98, 6.67, 0.86, 0.001, True),
    ("Qwen3-Emb",      +5.01, 3.36, 6.84, 0.97, 0.001, True),
    ("Roberta-Large",  +8.56, -5.85, 18.88, 0.23, 0.191, False),
    ("Roberta-InBedder", +8.19, 5.99, 10.56, 1.17, 0.001, True),
    ("BART-Base",      +10.08, 6.28, 13.73, 0.90, 0.001, True),
]

NON_CONTRASTIVE = {"BART-Base", "Roberta-Large"}


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
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_analyze_data():
    data = {}
    for fname in os.listdir(ANALYZE_DIR):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(ANALYZE_DIR, fname), "r") as f:
                data[model_name] = json.load(f)
    return data


def get_method_scores(analyze_data, model_name, target_dim=256):
    """Extract per-task scores for all methods at a given dimension."""
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
# FIGURE 2: Optimized-Random Gap (redesign)
# ============================================================
def fig2_opt_random_gap():
    """Redesigned Figure 2: Optimized-random gap across 11 models."""
    set_style()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Sort by gap
    data = sorted(TABLE2_DATA, key=lambda x: x[1])
    names = [d[0] for d in data]
    gaps = [d[1] for d in data]
    ci_lo = [d[2] for d in data]
    ci_hi = [d[3] for d in data]
    sig = [d[6] for d in data]

    y = np.arange(len(names))
    err_lo = [g - lo for g, lo in zip(gaps, ci_lo)]
    err_hi = [hi - g for g, hi in zip(gaps, ci_hi)]

    colors = [COLORS["non_contrastive"] if n in NON_CONTRASTIVE else COLORS["contrastive"]
              for n in names]

    bars = ax.barh(y, gaps, xerr=[err_lo, err_hi], color=colors, alpha=0.75,
                   capsize=3, error_kw={'linewidth': 1, 'color': '#333333'},
                   height=0.6, edgecolor='white', linewidth=0.5)

    ax.axvline(x=0, color='#333333', linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Optimized $-$ Random Retention Gap (%)", fontsize=12)
    ax.set_xlim(-8, 20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["contrastive"], alpha=0.75, label="Contrastively trained"),
        mpatches.Patch(facecolor=COLORS["non_contrastive"], alpha=0.75, label="Encoder-only (non-contrastive)"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#CCCCCC')

    # Add value labels
    for i, (name, gap) in enumerate(zip(names, gaps)):
        sig_mark = "" if sig[i] else " (n.s.)"
        ax.text(gap + 0.3, i, f"+{gap:.1f}%{sig_mark}", va='center',
                fontsize=8, color='#333333')

    # Add shading for the "strong models" zone
    ax.axvspan(-8, 6, alpha=0.04, color=COLORS["contrastive"])
    ax.axvspan(6, 20, alpha=0.04, color=COLORS["non_contrastive"])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_opt_random_gap_all_models.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE 3: All Methods Comparison (redesign)
# ============================================================
def fig3_all_methods():
    """Redesigned Figure 3: All 5 pruning methods at dim=256."""
    set_style()
    analyze_data = load_analyze_data()

    # Load magnitude MTEB results
    mag_mteb = {}
    for model_key, mag_file in [("stella_en_400M_v5", "magnitude_stella_mteb.json"),
                                 ("gte-large-en-v1.5", "magnitude_gte_mteb.json")]:
        mag_path = os.path.join(ANALYSIS_DIR, mag_file)
        if os.path.exists(mag_path):
            with open(mag_path) as f:
                mag_mteb[model_key] = json.load(f)

    model_configs = [
        ("stella_en_400M_v5", "Stella"),
        ("gte-large-en-v1.5", "GTE-Large"),
        ("roberta-large-InBedder", "Roberta-InBedder"),
    ]

    method_info = [
        ("Random",           COLORS["random"],     "dashed"),
        ("Sequential",       COLORS["sequential"],  "solid"),
        ("Magnitude",        COLORS["magnitude"],   "solid"),
        ("Optimized",        COLORS["optimized"],   "solid"),
        ("Anti-optimized",   COLORS["anti_opt"],    "solid"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, (model_key, model_display) in enumerate(model_configs):
        ax = axes[ax_idx]
        task_scores = get_method_scores(analyze_data, model_key)
        if task_scores is None:
            continue

        mag_scores = mag_mteb.get(model_key, {}).get("scores", {})

        # Compute mean retention per method
        method_rets = {}
        method_stds = {}
        for task_name, tscores in task_scores.items():
            default = tscores["default"]
            if default <= 0:
                continue

            for method, _, _ in method_info:
                key = method.lower().replace("-", "_").replace(" ", "_")
                if method == "Anti-optimized":
                    key = "poor"
                elif method == "Optimized":
                    key = "best"
                elif method == "Sequential":
                    key = "sort"
                elif method == "Magnitude":
                    if task_name in mag_scores:
                        val = mag_scores[task_name] / default
                        method_rets.setdefault(method, []).append(val)
                        continue
                    else:
                        continue

                if key in tscores:
                    val = tscores[key] / default
                    method_rets.setdefault(method, []).append(val)

        # Plot as lollipop chart
        methods_with_data = []
        means = []
        stds = []
        colors_list = []
        for method, color, ls in method_info:
            if method in method_rets:
                methods_with_data.append(method)
                means.append(np.mean(method_rets[method]) * 100)
                stds.append(np.std(method_rets[method]) * 100)
                colors_list.append(color)

        x = np.arange(len(methods_with_data))

        # Error bars + dots
        for i, (mean, std, color) in enumerate(zip(means, stds, colors_list)):
            ax.plot([i, i], [mean - std, mean + std], color=color, linewidth=1.5, alpha=0.5)
            ax.scatter([i], [mean], color=color, s=80, zorder=5, edgecolors='white', linewidth=0.5)

        # Connect with a light line
        ax.plot(x, means, '-', color='#BBBBBB', linewidth=0.8, zorder=1)

        # Reference line at 100%
        ax.axhline(y=100, color='#333333', linestyle=':', alpha=0.4, linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Anti-optimized", "Anti-opt.").replace("Sequential", "Seq.")
                            for m in methods_with_data], rotation=35, ha='right', fontsize=9)
        ax.set_title(model_display, fontsize=13, fontweight='bold', pad=10)

        # Add retention values
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.annotate(f'{mean:.1f}%', (i, mean + std + 0.3), ha='center', fontsize=8, color='#555555')

        ax.set_ylim(82, 112)

    axes[0].set_ylabel("Mean Retention (%)", fontsize=12)
    fig.suptitle("Five Pruning Strategies at dim=256 (75% Reduction)", fontsize=14,
                 y=1.02, fontweight='bold')

    # Common legend
    legend_elements = [mpatches.Patch(facecolor=c, alpha=0.8, label=m)
                       for m, c, _ in method_info]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.05), framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig10_all_methods_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE 4: Magnitude Analysis (redesign with more info)
# ============================================================
def fig4_magnitude_analysis():
    """Redesigned Figure 4: Magnitude analysis with more information."""
    set_style()
    analyze_data = load_analyze_data()

    # Load magnitude data
    mag_analysis_path = os.path.join(ANALYSIS_DIR, "magnitude_analysis.json")
    with open(mag_analysis_path) as f:
        mag_analysis = json.load(f)

    gte_mteb_path = os.path.join(ANALYSIS_DIR, "magnitude_gte_mteb.json")
    stella_mteb_path = os.path.join(ANALYSIS_DIR, "magnitude_stella_mteb.json")

    with open(gte_mteb_path) as f:
        gte_mteb = json.load(f)
    has_stella = os.path.exists(stella_mteb_path)
    if has_stella:
        with open(stella_mteb_path) as f:
            stella_mteb = json.load(f)

    fig = plt.figure(figsize=(15, 5))

    # (a) Magnitude vs Task ranking correlation - 3 models
    ax1 = fig.add_subplot(131)
    mag_corr = mag_analysis.get("magnitude_vs_task_correlation", {})
    models_sorted = sorted(mag_corr.keys())
    y = np.arange(len(models_sorted))
    for i, model in enumerate(models_sorted):
        summary = mag_corr[model]["summary"]
        mean_rho = summary["mean_rho"]
        std_rho = summary["std_rho"]
        display = MODEL_DISPLAY.get(model, model)
        color = COLORS["contrastive"] if model != "roberta-large-InBedder" else COLORS["non_contrastive"]
        ax1.barh(i, mean_rho, xerr=std_rho, color=color, alpha=0.7, capsize=3,
                 height=0.5, edgecolor='white')

    ax1.axvline(x=0, color='#333333', linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models_sorted], fontsize=9)
    ax1.set_xlabel("Spearman $\\rho$", fontsize=11)
    ax1.set_title("(a) Magnitude vs Task Ranking\n($\\rho \\approx 0$: uninformative)", fontsize=11)
    ax1.set_xlim(-0.12, 0.12)

    # (b) Per-task scatter: magnitude vs random for GTE-Large
    ax2 = fig.add_subplot(132)
    task_scores = get_method_scores(analyze_data, "gte-large-en-v1.5")
    mag_scores = gte_mteb.get("scores", {})

    if task_scores:
        tasks, mag_rets, rnd_rets, categories = [], [], [], []
        task_cat_map = {
            "Classification": ['AmazonCounterfactualClassification', 'AmazonReviewsClassification',
                              'Banking77Classification', 'EmotionClassification', 'ImdbClassification'],
            "Clustering": ['BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering'],
            "Retrieval": ['ArguAna', 'NFCorpus', 'SCIDOCS', 'SciFact', 'CQADupstackEnglishRetrieval'],
            "Reranking": ['MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions'],
            "STS": ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICK-R', 'BIOSSES'],
        }
        cat_colors = {
            "Classification": "#4A90D9", "Clustering": "#E67E22",
            "Retrieval": "#27AE60", "Reranking": "#E74C3C", "STS": "#9B59B6",
            "Other": "#6C757D"
        }

        for tname, tscores in task_scores.items():
            if tname == "STS17" or tname not in mag_scores:
                continue
            mag_ret = mag_scores[tname] / tscores["default"]
            rnd_ret = tscores["random"] / tscores["default"]
            tasks.append(tname)
            mag_rets.append(mag_ret)
            rnd_rets.append(rnd_ret)

            cat = "Other"
            for c, tlist in task_cat_map.items():
                if tname in tlist:
                    cat = c
                    break
            categories.append(cat)

        mag_rets = np.array(mag_rets)
        rnd_rets = np.array(rnd_rets)

        # Color by category
        for cat, color in cat_colors.items():
            mask = [i for i, c in enumerate(categories) if c == cat]
            if mask:
                ax2.scatter([rnd_rets[i] for i in mask], [mag_rets[i] for i in mask],
                           c=color, s=25, alpha=0.7, label=cat, zorder=3, edgecolors='white', linewidth=0.3)

        lims = [0.88, 1.02]
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_xlabel("Random Retention", fontsize=11)
        ax2.set_ylabel("Magnitude Retention", fontsize=11)
        ax2.set_title("(b) GTE-Large per Task\n(color = category)", fontsize=11)
        ax2.set_aspect('equal')
        ax2.legend(fontsize=7, loc='lower left', framealpha=0.9)

        # Add win count
        mag_wins = np.sum(mag_rets > rnd_rets)
        ax2.annotate(f'Random wins: {len(mag_rets)-mag_wins}/{len(mag_rets)}',
                    xy=(0.98, 0.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.8))

    # (c) Per-task scatter: magnitude vs random for Stella
    ax3 = fig.add_subplot(133)
    if has_stella:
        task_scores_stella = get_method_scores(analyze_data, "stella_en_400M_v5")
        mag_scores_stella = stella_mteb.get("scores", {})

        if task_scores_stella:
            tasks_s, mag_rets_s, rnd_rets_s, cats_s = [], [], [], []
            for tname, tscores in task_scores_stella.items():
                if tname == "STS17" or tname not in mag_scores_stella:
                    continue
                mag_ret = mag_scores_stella[tname] / tscores["default"]
                rnd_ret = tscores["random"] / tscores["default"]
                tasks_s.append(tname)
                mag_rets_s.append(mag_ret)
                rnd_rets_s.append(rnd_ret)

                cat = "Other"
                for c, tlist in task_cat_map.items():
                    if tname in tlist:
                        cat = c
                        break
                cats_s.append(cat)

            mag_rets_s = np.array(mag_rets_s)
            rnd_rets_s = np.array(rnd_rets_s)

            for cat, color in cat_colors.items():
                mask = [i for i, c in enumerate(cats_s) if c == cat]
                if mask:
                    ax3.scatter([rnd_rets_s[i] for i in mask], [mag_rets_s[i] for i in mask],
                               c=color, s=25, alpha=0.7, label=cat, zorder=3, edgecolors='white', linewidth=0.3)

            ax3.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
            ax3.set_xlim(lims)
            ax3.set_ylim(lims)
            ax3.set_xlabel("Random Retention", fontsize=11)
            ax3.set_ylabel("Magnitude Retention", fontsize=11)
            ax3.set_title("(c) Stella per Task\n(color = category)", fontsize=11)
            ax3.set_aspect('equal')
            ax3.legend(fontsize=7, loc='lower left', framealpha=0.9)

            mag_wins_s = np.sum(mag_rets_s > rnd_rets_s)
            ax3.annotate(f'Random wins: {len(mag_rets_s)-mag_wins_s}/{len(mag_rets_s)}',
                        xy=(0.98, 0.02), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig9_magnitude_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# FIGURE 11: Magnitude Scatter (redesign)
# ============================================================
def fig11_magnitude_scatter():
    """Redesigned Figure 11: Scatter plots of magnitude vs random."""
    set_style()
    analyze_data = load_analyze_data()

    gte_mteb_path = os.path.join(ANALYSIS_DIR, "magnitude_gte_mteb.json")
    stella_mteb_path = os.path.join(ANALYSIS_DIR, "magnitude_stella_mteb.json")

    if not os.path.exists(gte_mteb_path):
        print("No GTE MTEB results, skipping fig11")
        return

    with open(gte_mteb_path) as f:
        gte_mteb = json.load(f)

    has_stella = os.path.exists(stella_mteb_path)
    if has_stella:
        with open(stella_mteb_path) as f:
            stella_mteb = json.load(f)

    def build_scatter(analyze_data, model_name, mag_mteb):
        task_scores = get_method_scores(analyze_data, model_name)
        mag_scores = mag_mteb.get("scores", {})

        below, above, near = [], [], []
        for tname, tscores in task_scores.items():
            if tname == "STS17" or tname not in mag_scores:
                continue
            mag_ret = mag_scores[tname] / tscores["default"]
            rnd_ret = tscores["random"] / tscores["default"]
            if mag_ret > rnd_ret + 0.005:
                above.append((tname, rnd_ret, mag_ret))
            elif rnd_ret > mag_ret + 0.005:
                below.append((tname, rnd_ret, mag_ret))
            else:
                near.append((tname, rnd_ret, mag_ret))
        return below, above, near

    lims = [0.88, 1.02]

    if has_stella:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

        # GTE-Large
        below, above, near = build_scatter(analyze_data, "gte-large-en-v1.5", gte_mteb)
        if below:
            x, y = zip(*[(t[1], t[2]) for t in below])
            ax1.scatter(x, y, c=COLORS["anti_opt"], s=35, alpha=0.7,
                       label=f'Random wins ({len(below)})', zorder=3, edgecolors='white', linewidth=0.3)
        if above:
            x, y = zip(*[(t[1], t[2]) for t in above])
            ax1.scatter(x, y, c=COLORS["optimized"], s=35, alpha=0.7,
                       label=f'Magnitude wins ({len(above)})', zorder=3, edgecolors='white', linewidth=0.3)
        if near:
            x, y = zip(*[(t[1], t[2]) for t in near])
            ax1.scatter(x, y, c=COLORS["random"], s=35, alpha=0.7,
                       label=f'Tie ({len(near)})', zorder=3, edgecolors='white', linewidth=0.3)
        ax1.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal performance')
        ax1.set_xlim(lims); ax1.set_ylim(lims)
        ax1.set_xlabel('Random Selection Retention', fontsize=11)
        ax1.set_ylabel('Magnitude Selection Retention', fontsize=11)
        ax1.set_title('(a) GTE-Large (dim=256)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax1.set_aspect('equal')
        ax1.annotate('$d = -0.28$, $p = 0.002$',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEBD0', alpha=0.8))

        # Stella
        below, above, near = build_scatter(analyze_data, "stella_en_400M_v5", stella_mteb)
        if below:
            x, y = zip(*[(t[1], t[2]) for t in below])
            ax2.scatter(x, y, c=COLORS["anti_opt"], s=35, alpha=0.7,
                       label=f'Random wins ({len(below)})', zorder=3, edgecolors='white', linewidth=0.3)
        if above:
            x, y = zip(*[(t[1], t[2]) for t in above])
            ax2.scatter(x, y, c=COLORS["optimized"], s=35, alpha=0.7,
                       label=f'Magnitude wins ({len(above)})', zorder=3, edgecolors='white', linewidth=0.3)
        if near:
            x, y = zip(*[(t[1], t[2]) for t in near])
            ax2.scatter(x, y, c=COLORS["random"], s=35, alpha=0.7,
                       label=f'Tie ({len(near)})', zorder=3, edgecolors='white', linewidth=0.3)
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal performance')
        ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_xlabel('Random Selection Retention', fontsize=11)
        ax2.set_ylabel('Magnitude Selection Retention', fontsize=11)
        ax2.set_title('(b) Stella (dim=256)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)
        ax2.set_aspect('equal')
        ax2.annotate('$d = -0.05$, $p = 0.38$',
                     xy=(0.98, 0.02), xycoords='axes fraction',
                     ha='right', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig11_magnitude_scatter.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Redesigning all figures...")
    print("\n[1] Figure 2: Optimized-Random Gap")
    fig2_opt_random_gap()
    print("\n[2] Figure 3: All Methods Comparison")
    fig3_all_methods()
    print("\n[3] Figure 4: Magnitude Analysis")
    fig4_magnitude_analysis()
    print("\n[4] Figure 11: Magnitude Scatter")
    fig11_magnitude_scatter()
    print("\nAll figures redesigned!")
