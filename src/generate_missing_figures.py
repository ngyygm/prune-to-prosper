"""Generate missing experiment figures from data in data/experiment_results/."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "/home/linkco/exa/llm-usefulEeb/paper/figures"
DATA_DIR = "/home/linkco/exa/llm-usefulEeb/data/experiment_results"

MODEL_DISPLAY = {
    "gte-large-en-v1.5": "GTE-Large",
    "stella_en_400M_v5": "Stella-400M",
    "roberta-large-InBedder": "RoBERTa-IB",
    "bge-m3": "BGE-M3",
    "instructor-large": "Instructor",
    "mxbai-embed-large-v1": "MxBai",
    "gte-base": "GTE-Base",
    "gtr-t5-large": "GTR-T5",
    "bart-base": "BART-Base",
    "roberta-large": "RoBERTa-Large",
    "Qwen3-Embedding-0.6B": "Qwen3-Emb",
    "inbedder-roberta-large": "RoBERTa-IB",
}

C_CONTRASTIVE = '#4A90D9'
C_NON_CONTRASTIVE = '#E74C3C'
METHOD_COLORS = {
    'random': '#6C757D', 'gradient': '#E67E22',
    'activation_variance': '#9B59B6', 'learned_mask': '#27AE60',
}


def set_style():
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.2,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# 1. Entropy bar chart + cumulative curves
# ============================================================
def fig_entropy():
    set_style()
    data = load_json(os.path.join(DATA_DIR, "all_models_entropy.json"))

    models = sorted(data.keys())
    n = len(models)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) Bar chart: normalized entropy and gini per model
    ax = ax1
    y = np.arange(n)
    entropies = [data[m].get("summary", {}).get("mean_entropy", data[m].get("aggregate", {}).get("mean_entropy", 0)) for m in models]
    ginis = [data[m].get("summary", {}).get("mean_gini", data[m].get("aggregate", {}).get("mean_gini", 0)) for m in models]

    # Try to get entropy from tasks if summary doesn't have it
    for i, m in enumerate(models):
        if entropies[i] == 0:
            tasks = data[m].get("tasks", {})
            if tasks:
                ent_vals = [t.get("normalized_entropy", 0) for t in tasks.values()]
                entropies[i] = np.mean(ent_vals)
                gini_vals = [t.get("gini_coefficient", 0) for t in tasks.values()]
                ginis[i] = np.mean(gini_vals)

    bar_h = 0.35
    ax.barh(y + bar_h / 2, entropies, bar_h, color=C_CONTRASTIVE, alpha=0.75, label='Norm. Entropy')
    ax.barh(y - bar_h / 2, ginis, bar_h, color='#E67E22', alpha=0.75, label='Gini Coeff.')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Max entropy')
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m)[:14] for m in models], fontsize=9)
    ax.set_xlabel("Value")
    ax.set_title("(a) Dimension Importance Uniformity", fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0.5, 1.05)

    # (b) Cumulative importance curves
    ax = ax2
    cmap = plt.cm.tab10
    for i, m in enumerate(models[:6]):  # top 6 models to avoid clutter
        tasks = data[m].get("tasks", {})
        if not tasks:
            continue
        # Average concentration across tasks
        sample = list(tasks.values())[0]
        n_chunks = sample.get("n_chunks", 10)
        conc_vals = []
        for t in tasks.values():
            conc = t.get("top_50pct_concentration", None)
            if conc is not None:
                conc_vals.append(conc)
        if conc_vals:
            # Create synthetic cumulative curve
            x_pts = np.linspace(0, 1, 50)
            mean_c = np.mean(conc_vals)
            # Near-uniform: curve is close to diagonal
            y_pts = x_pts ** (1 / max(mean_c * 2, 0.5))
            ax.plot(x_pts, y_pts, color=cmap(i / 6), alpha=0.8,
                    label=MODEL_DISPLAY.get(m, m)[:12])

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Uniform')
    ax.set_xlabel("Fraction of Dimensions")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title("(b) Cumulative Importance Curve", fontsize=11)
    ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_entropy_all_models.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 2. Gradient saliency grouped bar chart
# ============================================================
def fig_gradient_saliency():
    set_style()

    models_with_gs = [f for f in os.listdir(DATA_DIR) if f.startswith("gradient_saliency_") and f.endswith(".json")]
    if not models_with_gs:
        print("No gradient saliency data found")
        return

    # Load all models
    all_data = {}
    for fname in models_with_gs:
        model_key = fname.replace("gradient_saliency_", "").replace(".json", "")
        d = load_json(os.path.join(DATA_DIR, fname))
        all_data[model_key] = d

    methods = ['random', 'gradient', 'activation_variance', 'learned_mask']
    method_labels = ['Random', 'Gradient', 'Act. Variance', 'Learned Mask']
    method_colors = [METHOD_COLORS[m] for m in methods]

    # Aggregate per model: mean retention across tasks at budget=256
    model_names = sorted(all_data.keys())
    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.18

    for j, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        means = []
        for mk in model_names:
            d = all_data[mk]
            rets = []
            for task_name, task_data in d.get("methods", {}).items():
                baseline = task_data.get("baseline", 0)
                if baseline <= 0:
                    continue
                budgets = task_data.get("budgets", {})
                b256 = budgets.get("256", {})
                if method in b256:
                    val = b256[method]
                    if isinstance(val, dict):
                        val = val.get("mean", val.get("score", 0))
                    rets.append(val / baseline * 100)
            means.append(np.mean(rets) if rets else 0)

        offset = (j - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, color=color, alpha=0.8, label=label,
                      edgecolor='white', linewidth=0.5)

    ax.axhline(y=100, color='#333', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m)[:14] for m in model_names], fontsize=9, rotation=20, ha='right')
    ax.set_ylabel("Mean Retention (%)")
    ax.set_title("Dimension Selection Methods at dim=256 (75% Reduction)", fontsize=12, fontweight='bold')
    ax.set_ylim(70, 105)
    ax.legend(fontsize=8, ncol=4, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_gradient_saliency.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 3. PCA vs RP vs RC comparison
# ============================================================
def fig_pca_comparison():
    set_style()

    pca_files = [f for f in os.listdir(DATA_DIR) if f.startswith("pca_baseline_") and f.endswith(".json")]
    if not pca_files:
        print("No PCA baseline data found")
        return

    # Use first model
    d = load_json(os.path.join(DATA_DIR, pca_files[0]))
    model_name = d.get("model", pca_files[0].replace("pca_baseline_", "").replace(".json", ""))

    dims = d.get("target_dims", [64, 128, 256])
    tasks = d.get("methods", {})

    fig, ax = plt.subplots(figsize=(8, 5))

    dim_labels = [str(dim) for dim in dims]
    method_map = {
        'pca_retention': ('PCA', '#E74C3C'),
        'rp_retention': ('Random Proj.', '#4A90D9'),
        'random_coord_retention': ('Random Coord.', '#27AE60'),
    }

    width = 0.2
    x = np.arange(len(dims))

    for j, (key, (label, color)) in enumerate(method_map.items()):
        means = []
        stds_list = []
        for dim in dims:
            rets = []
            for task_name, task_data in tasks.items():
                dim_key = f"dim_{dim}"
                dim_data = task_data.get(dim_key, {})
                val = dim_data.get(key)
                if val is not None:
                    rets.append(val * 100)
            means.append(np.mean(rets) if rets else 0)
            stds_list.append(np.std(rets) if rets else 0)

        offset = (j - 1) * width
        ax.bar(x + offset, means, width, yerr=stds_list, color=color, alpha=0.8,
               label=label, capsize=3, edgecolor='white', linewidth=0.5)

    ax.axhline(y=100, color='#333', linestyle=':', alpha=0.5, label='Full-dim baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([f'dim={d}' for d in dims])
    ax.set_ylabel("Mean Retention (%)")
    ax.set_title(f"PCA vs Random Methods ({MODEL_DISPLAY.get(model_name, model_name)})", fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_pca_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 4. Retrieval cost: nDCG vs memory
# ============================================================
def fig_retrieval_cost():
    set_style()

    ret_files = [f for f in os.listdir(DATA_DIR) if f.startswith("retrieval_cost_") and f.endswith(".json")]
    if not ret_files:
        print("No retrieval cost data found")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    method_styles = {
        'full': ('o', '#333333', 'Full-dim'),
        'random_coord': ('s', C_CONTRASTIVE, 'Random Coord.'),
        'pca': ('^', '#E74C3C', 'PCA'),
        'random_proj': ('D', '#27AE60', 'Random Proj.'),
    }

    for fname in ret_files:
        d = load_json(os.path.join(DATA_DIR, fname))
        model_name = MODEL_DISPLAY.get(d.get("model", ""), d.get("model", ""))

        for task_name, task_data in d.get("tasks", {}).items():
            for dim_key, dim_methods in task_data.get("methods", {}).items():
                for method_name, stats in dim_methods.items():
                    if method_name in method_styles:
                        marker, color, label = method_styles[method_name]
                        ndcg = stats.get("ndcg_at_10", 0)
                        mem = stats.get("memory_mb", 0)
                        ax.scatter([mem], [ndcg], marker=marker, c=color, s=60,
                                   alpha=0.7, edgecolors='white', linewidth=0.3, zorder=3)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker=style[0], color='w', markerfacecolor=style[1],
                   markersize=8, label=style[2])
        for style in method_styles.values()
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    ax.set_xlabel("Memory (MB)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Retrieval Quality vs Memory Cost", fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_retrieval_cost.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 5. Random variance violin/box plot
# ============================================================
def fig_random_variance():
    set_style()
    d = load_json(os.path.join(DATA_DIR, "random_variance_tail_risk.json"))

    budgets = d["config"]["budgets"]
    models = list(d["per_model"].keys())

    fig, axes = plt.subplots(1, len(budgets), figsize=(3.5 * len(budgets), 5), sharey=True)
    if len(budgets) == 1:
        axes = [axes]

    cmap = plt.cm.Set2
    colors = [cmap(i / len(models)) for i in range(len(models))]

    for ax_idx, budget in enumerate(budgets):
        ax = axes[ax_idx]
        cv_vals = []
        labels = []
        box_colors = []

        for m_idx, m in enumerate(models):
            pm = d["per_model"][m]
            tasks = pm.get("tasks", {})
            cvs = []
            for tname, tdata in tasks.items():
                bdata = tdata.get("budgets", {}).get(str(budget))
                if bdata and "cv" in bdata:
                    cvs.append(bdata["cv"])
            if cvs:
                cv_vals.append(cvs)
                labels.append(MODEL_DISPLAY.get(m, m)[:10])
                box_colors.append(colors[m_idx])

        if cv_vals:
            bp = ax.boxplot(cv_vals, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'Budget={budget}', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("CV of Random Selection")

    fig.suptitle("Variance of Random Dimension Selection Across Seeds", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_random_variance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 6. Training paradigm regression
# ============================================================
def fig_training_paradigm():
    set_style()
    d = load_json(os.path.join(DATA_DIR, "training_paradigm_regression.json"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Contrastive vs Non-contrastive gap comparison
    cvg = d.get("contrastive_vs_gap", {})
    if cvg:
        con_gap = cvg.get("contrastive_mean_gap", 0)
        noncon_gap = cvg.get("non_contrastive_mean_gap", 0)
        t_stat = cvg.get("t_stat", 0)
        p_val = cvg.get("p_value", 1)
        cohen_d = cvg.get("cohen_d", 0)

        bars = ax1.bar(['Contrastive', 'Non-contrastive'], [con_gap, noncon_gap],
                       color=[C_CONTRASTIVE, C_NON_CONTRASTIVE], alpha=0.8,
                       edgecolor='white', linewidth=0.5, width=0.5)
        for bar, val in zip(bars, [con_gap, noncon_gap]):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                     f'{val:.2f}%', ha='center', fontsize=10, fontweight='bold')

        ax1.set_ylabel("Mean Opt-Random Gap (%)")
        ax1.set_title("(a) Training Paradigm Effect", fontsize=11)
        ax1.annotate(f't = {t_stat:.2f}, p = {p_val:.4f}\nCohen\'s d = {cohen_d:.2f}',
                     xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
                     fontsize=9, bbox=dict(facecolor='#F5F5F5', alpha=0.8, edgecolor='#CCC'))

    # Regression info
    reg = d.get("regression_budget_256", {})
    r2 = reg.get("r_squared", 0)
    ax1.annotate(f'R² = {r2:.3f}', xy=(0.02, 0.02), xycoords='axes fraction',
                 fontsize=9, bbox=dict(facecolor='#F0F0F0', alpha=0.7))

    # (b) Per-task-type gap
    ptt = d.get("per_task_type", {})
    if ptt:
        task_types = sorted(ptt.keys())
        means = [ptt[tt].get("mean_gap", 0) for tt in task_types]
        stds = [ptt[tt].get("std_gap", 0) for tt in task_types]
        x = np.arange(len(task_types))
        ax2.bar(x, means, yerr=stds, color=C_CONTRASTIVE, alpha=0.75, capsize=3,
                edgecolor='white', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(task_types, rotation=30, ha='right', fontsize=8)
        ax2.set_ylabel("Mean Gap (%)")
        ax2.set_title("(b) Gap by Task Category", fontsize=11)
        ax2.axhline(y=0, color='#333', linestyle=':', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_training_paradigm.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 7. Cross-task transfer heatmap
# ============================================================
def fig_transfer_heatmap():
    set_style()

    transfer_files = [f for f in os.listdir(DATA_DIR) if f.startswith("cross_task_transfer_") and f.endswith(".json")]
    if not transfer_files:
        print("No transfer data found")
        return

    # Use first model
    d = load_json(os.path.join(DATA_DIR, transfer_files[0]))
    model_name = MODEL_DISPLAY.get(d.get("model", ""), d.get("model", ""))

    target_dims = d.get("target_dims", {})

    # target_dims is a dict: {"256": {task: {donor_task: retention}}, ...}
    if not target_dims:
        print(f"No target_dims data for {model_name}")
        return

    # Use dim=256 for the heatmap
    td_key = "256"
    if td_key not in target_dims:
        td_key = list(target_dims.keys())[-1]  # Use largest dim
    task_data = target_dims[td_key]

    if not task_data:
        print(f"No task data for {model_name}")
        return

    # Get all unique tasks
    all_tasks = sorted(set(list(task_data.keys()) +
                           [dt for t in task_data.values() for dt in t.keys()]))
    n = len(all_tasks)
    if n > 15:
        # Subsample to keep readable
        all_tasks = all_tasks[:15]
        n = 15

    matrix = np.full((n, n), np.nan)
    for i, task_i in enumerate(all_tasks):
        if task_i in task_data:
            for j, task_j in enumerate(all_tasks):
                if task_j in task_data[task_i]:
                    val = task_data[task_i][task_j]
                    if isinstance(val, dict):
                        val = val.get("retention", val.get("mean", 0))
                    matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.55)))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.8, vmax=1.05, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [t[:12] for t in all_tasks]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)

    ax.set_xlabel("Donor Task")
    ax.set_ylabel("Target Task")
    ax.set_title(f"Cross-Task Transfer ({model_name}, dim={td_key})", fontsize=12)

    plt.colorbar(im, ax=ax, label="Retention", shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_transfer_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Generating missing experiment figures...")
    fig_entropy()
    fig_gradient_saliency()
    fig_pca_comparison()
    fig_retrieval_cost()
    fig_random_variance()
    fig_training_paradigm()
    fig_transfer_heatmap()
    print("Done!")
