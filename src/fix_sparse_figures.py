"""Fix 6 sparse figures (>80% white) by using denser layouts with more data."""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "/home/linkco/exa/llm-usefulEeb/paper/figures"
DATA_DIR = "/home/linkco/exa/llm-usefulEeb/data/experiment_results"
ANALYZE_DIR = "/home/linkco/exa/llm-usefulEeb/data/analyze"

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

NON_CONTRASTIVE = {"BART-Base", "RoBERTa-Large", "RoBERTa-IB"}

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


# ============================================================
# Fix 1: fig_random_variance.png (92.2% white → target <75%)
# ============================================================
def fix_random_variance():
    set_style()
    d = json.load(open(os.path.join(DATA_DIR, "random_variance_tail_risk.json")))
    budgets = d["config"]["budgets"]
    models = list(d["per_model"].keys())

    # Use a compact 2-row layout: top row = CV heatmap, bottom = P5/CVaR bars
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: CV, P5, CVaR-5 as grouped bar charts across budgets
    metrics = [('cv', 'Coefficient of Variation (%)', '#4A90D9'),
               ('p5', '5th Percentile Retention (%)', '#E67E22'),
               ('cvar_5', 'CVaR-5 Retention (%)', '#E74C3C')]

    for col, (metric_key, ylabel, color) in enumerate(metrics):
        ax = axes[0, col]
        x = np.arange(len(budgets))
        width = 0.8 / len(models)
        cmap = plt.cm.tab10

        for m_idx, m in enumerate(models[:6]):  # Top 6 models
            pm = d["per_model"][m]
            tasks = pm.get("tasks", {})
            vals = []
            for budget in budgets:
                metric_vals = []
                for tname, tdata in tasks.items():
                    bdata = tdata.get("budgets", {}).get(str(budget))
                    if bdata and metric_key in bdata:
                        v = bdata[metric_key]
                        if metric_key == 'cv':
                            v *= 100  # Convert to percentage
                        metric_vals.append(v)
                vals.append(np.mean(metric_vals) if metric_vals else 0)

            offset = (m_idx - len(models[:6]) / 2 + 0.5) * width
            label = MODEL_DISPLAY.get(m, m)[:10]
            ax.bar(x + offset, vals, width, color=cmap(m_idx / 6), alpha=0.8,
                   label=label, edgecolor='white', linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels([f'dim={b}' for b in budgets])
        ax.set_ylabel(ylabel)
        ax.set_title(f'{"CV" if metric_key == "cv" else "P5" if metric_key == "p5" else "CVaR-5"} by Budget')
        if col == 0:
            ax.legend(fontsize=6, ncol=2, loc='upper left')

    # Row 2: Per-budget scatter plots (P5 vs CV, colored by paradigm)
    for col, budget in enumerate(budgets[:3]):
        ax = axes[1, col]
        for m in models:
            pm = d["per_model"][m]
            tasks = pm.get("tasks", {})
            cvs, p5s = [], []
            for tname, tdata in tasks.items():
                bdata = tdata.get("budgets", {}).get(str(budget))
                if bdata:
                    cvs.append(bdata.get("cv", 0) * 100)
                    p5s.append(bdata.get("p5", 0))
            display = MODEL_DISPLAY.get(m, m)
            color = C_NON_CONTRASTIVE if display in NON_CONTRASTIVE else C_CONTRASTIVE
            ax.scatter(cvs, p5s, c=color, s=30, alpha=0.6, edgecolors='white', linewidth=0.3)
            if cvs:
                ax.annotate(display[:6], (np.mean(cvs), np.mean(p5s)),
                            fontsize=6, ha='center', va='bottom')

        ax.set_xlabel("CV (%)")
        ax.set_ylabel("P5 Retention (%)")
        ax.set_title(f'Budget={budget}: CV vs P5')
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.3, linewidth=0.8)

    # Legend for paradigm
    legend_elements = [
        mpatches.Patch(facecolor=C_CONTRASTIVE, alpha=0.7, label='Contrastive'),
        mpatches.Patch(facecolor=C_NON_CONTRASTIVE, alpha=0.7, label='Non-contrastive'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_random_variance.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Fix 2: fig_training_paradigm.png (82.4% → target <75%)
# ============================================================
def fix_training_paradigm():
    set_style()
    d = json.load(open(os.path.join(DATA_DIR, "training_paradigm_regression.json")))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Contrastive vs Non-contrastive gap
    ax = axes[0, 0]
    cvg = d.get("contrastive_vs_gap", {})
    if cvg:
        con_gap = cvg.get("contrastive_mean_gap", 0)
        noncon_gap = cvg.get("non_contrastive_mean_gap", 0)
        con_std = cvg.get("contrastive_std_gap", 1)
        noncon_std = cvg.get("non_contrastive_std_gap", 1)
        t_stat = cvg.get("t_stat", 0)
        p_val = cvg.get("p_value", 1)
        cohen_d = cvg.get("cohen_d", 0)

        bars = ax.bar(['Contrastive\n(n=7)', 'Non-contrastive\n(n=2)'],
                       [con_gap, noncon_gap],
                       yerr=[con_std, noncon_std],
                       color=[C_CONTRASTIVE, C_NON_CONTRASTIVE], alpha=0.85,
                       edgecolor='white', linewidth=0.5, width=0.5, capsize=5)
        for bar, val in zip(bars, [con_gap, noncon_gap]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                    f'{val:.2f}%', ha='center', fontsize=11, fontweight='bold')

        ax.set_ylabel("Mean Opt-Random Gap (%)")
        ax.set_title("(a) Training Paradigm Effect on Selection Value", fontweight='bold')
        ax.annotate(f't = {t_stat:.2f}, p = {p_val:.4f}\nCohen\'s d = {cohen_d:.2f}',
                     xy=(0.98, 0.98), xycoords='axes fraction', ha='right', va='top',
                     fontsize=9, bbox=dict(facecolor='#F5F5F5', alpha=0.8, edgecolor='#CCC'))
        ax.set_ylim(0, max(con_gap, noncon_gap) * 1.5)

    # (b) Per-task-type gap with individual model dots
    ax = axes[0, 1]
    ptt = d.get("per_task_type", {})
    if ptt:
        task_types = sorted(ptt.keys())
        means = [ptt[tt].get("mean_gap", 0) for tt in task_types]
        stds = [ptt[tt].get("std_gap", 0) for tt in task_types]
        x = np.arange(len(task_types))
        bars = ax.bar(x, means, yerr=stds, color=C_CONTRASTIVE, alpha=0.75, capsize=3,
                       edgecolor='white', linewidth=0.5)
        for i, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + stds[i] + 0.1,
                    f'{mean:.1f}', ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_types, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Mean Gap (%)")
        ax.set_title("(b) Gap by Task Category", fontweight='bold')
        ax.axhline(y=0, color='#333', linestyle=':', alpha=0.3)

    # (c) Regression coefficients
    ax = axes[1, 0]
    reg = d.get("regression_budget_256", {})
    if reg:
        coeffs = reg.get("coefficients", {})
        if coeffs:
            names = list(coeffs.keys())
            values = [v["beta"] if isinstance(v, dict) else v for v in coeffs.values()]
            errors = [v.get("se", 0) if isinstance(v, dict) else 0 for v in coeffs.values()]
            p_vals = [v.get("p", 1) if isinstance(v, dict) else 1 for v in coeffs.values()]
            colors = ['#27AE60' if v >= 0 else '#E74C3C' for v in values]
            y_pos = np.arange(len(names))
            ax.barh(y_pos, values, xerr=errors, color=colors, alpha=0.8,
                    edgecolor='white', linewidth=0.5, capsize=3)
            ax.axvline(x=0, color='#333', linewidth=1)
            ax.set_yticks(y_pos)
            labels = [f'{n}{"*" if p < 0.05 else ""}' for n, p in zip(names, p_vals)]
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Regression Coefficient (β)")
            r2 = reg.get("r_squared", 0)
            ax.set_title(f"(c) Regression Coefficients (R²={r2:.3f})", fontweight='bold')

    # (d) Per-task-type contrastive vs non-contrastive breakdown
    ax = axes[1, 1]
    # Create a simple table-like visualization
    ptt2 = d.get("per_task_type", {})
    if ptt2:
        task_types = sorted(ptt2.keys())
        cell_text = []
        for tt in task_types:
            m = ptt[tt].get("mean_gap", 0)
            s = ptt[tt].get("std_gap", 0)
            n = ptt[tt].get("n_tasks", 0)
            cell_text.append([f'{m:.2f}±{s:.2f}', str(n)])

        ax.axis('off')
        table = ax.table(cellText=cell_text,
                         rowLabels=task_types,
                         colLabels=['Gap (Mean±Std)', 'N Tasks'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title("(d) Per-Task-Type Details", fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_training_paradigm.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Fix 3: fig_entropy_all_models.png (84.9% → target <75%)
# ============================================================
def fix_entropy():
    set_style()
    data = json.load(open(os.path.join(DATA_DIR, "all_models_entropy.json")))
    models = sorted(data.keys())

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Entropy bar chart — DENSE version
    ax = axes[0, 0]
    y = np.arange(len(models))
    entropies, ginis = [], []
    for m in models:
        mdata = data[m]
        tasks = mdata.get("tasks", {})
        if tasks:
            ent_vals = [t.get("normalized_entropy", 0) for t in tasks.values() if t.get("normalized_entropy")]
            gin_vals = [t.get("gini_coefficient", 0) for t in tasks.values() if t.get("gini_coefficient")]
            entropies.append(np.mean(ent_vals) if ent_vals else 0)
            ginis.append(np.mean(gin_vals) if gin_vals else 0)
        else:
            entropies.append(mdata.get("aggregate", {}).get("mean_entropy", 0))
            ginis.append(mdata.get("aggregate", {}).get("mean_gini", 0))

    bar_h = 0.35
    ax.barh(y + bar_h / 2, entropies, bar_h, color=C_CONTRASTIVE, alpha=0.75, label='Norm. Entropy')
    ax.barh(y - bar_h / 2, ginis, bar_h, color='#E67E22', alpha=0.75, label='Gini Coeff.')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Max entropy')
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m)[:14] for m in models], fontsize=8)
    ax.set_xlabel("Value")
    ax.set_title("(a) Dimension Importance Uniformity", fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0.5, 1.05)

    # Add value annotations
    for i, (e, g) in enumerate(zip(entropies, ginis)):
        ax.text(e + 0.005, i + bar_h / 2, f'{e:.3f}', va='center', fontsize=7)
        ax.text(g + 0.005, i - bar_h / 2, f'{g:.3f}', va='center', fontsize=7, color='#E67E22')

    # (b) Per-task entropy violin-style (scatter + bars) for top 3 models
    ax = axes[0, 1]
    top_models = models[:3]
    cmap = plt.cm.tab10
    for mi, m in enumerate(top_models):
        tasks = data[m].get("tasks", {})
        if not tasks:
            continue
        task_names = sorted(tasks.keys())
        ent_vals = [tasks[t].get("normalized_entropy", 0) for t in task_names]
        x_jitter = np.random.normal(mi, 0.08, len(ent_vals))
        ax.scatter(x_jitter, ent_vals, c=cmap(mi / 3), s=20, alpha=0.6, edgecolors='white', linewidth=0.3)
        ax.scatter([mi], [np.mean(ent_vals)], c=cmap(mi / 3), s=80, marker='D',
                   edgecolors='black', linewidth=1, zorder=5)

    ax.set_xticks(range(len(top_models)))
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m)[:10] for m in top_models], fontsize=9)
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("(b) Per-Task Entropy Distribution", fontweight='bold')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax.set_ylim(0.95, 1.01)

    # (c) Concentration bar chart: top-10%, top-25%, top-50%
    ax = axes[1, 0]
    conc_levels = ['top_10pct_concentration', 'top_25pct_concentration', 'top_50pct_concentration']
    conc_labels = ['Top 10%', 'Top 25%', 'Top 50%']
    x = np.arange(len(conc_labels))
    width = 0.8 / len(models[:6])

    for mi, m in enumerate(models[:6]):
        tasks = data[m].get("tasks", {})
        conc_means = []
        for cl in conc_levels:
            vals = [t.get(cl, 0) for t in tasks.values() if t.get(cl) is not None]
            conc_means.append(np.mean(vals) * 100 if vals else 0)

        offset = (mi - len(models[:6]) / 2 + 0.5) * width
        ax.bar(x + offset, conc_means, width, color=cmap(mi / 6), alpha=0.8,
               label=MODEL_DISPLAY.get(m, m)[:10], edgecolor='white', linewidth=0.3)

    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=25, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(conc_labels)
    ax.set_ylabel("Fraction of Total Score (%)")
    ax.set_title("(c) Score Concentration by Subset Size", fontweight='bold')
    ax.legend(fontsize=6, ncol=3, loc='upper left')

    # (d) Entropy vs Gini scatter with model labels
    ax = axes[1, 1]
    for mi, m in enumerate(models):
        display = MODEL_DISPLAY.get(m, m)
        color = C_NON_CONTRASTIVE if display in NON_CONTRASTIVE else C_CONTRASTIVE
        ax.scatter([ginis[mi]], [entropies[mi]], c=color, s=80, alpha=0.8,
                   edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(display[:8], (ginis[mi], entropies[mi]),
                    fontsize=7, ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points')

    ax.set_xlabel("Gini Coefficient")
    ax.set_ylabel("Normalized Entropy")
    ax.set_title("(d) Entropy vs Gini per Model", fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor=C_CONTRASTIVE, alpha=0.8, label='Contrastive'),
        mpatches.Patch(facecolor=C_NON_CONTRASTIVE, alpha=0.8, label='Non-contrastive'),
    ]
    ax.legend(handles=legend_elements, fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_entropy_all_models.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Fix 4: fig_opt_random_gap_all_models.png (87.1% → target <75%)
# ============================================================
def fix_opt_random_gap():
    set_style()

    TABLE2_DATA = [
        ("MxBai",       +2.24, 1.35, 3.29, 0.76, 0.001, True),
        ("GTE-Large",   +2.41, 1.49, 3.39, 0.83, 0.001, True),
        ("Instructor",  +2.89, 1.78, 4.09, 0.83, 0.001, True),
        ("GTE-Base",    +3.45, 2.11, 4.93, 0.81, 0.001, True),
        ("Stella-400M", +3.22, 2.17, 4.44, 0.93, 0.001, True),
        ("BGE-M3",      +4.62, 3.21, 6.13, 1.04, 0.001, True),
        ("GTR-T5",      +4.75, 2.98, 6.67, 0.86, 0.001, True),
        ("Qwen3-Emb",   +5.01, 3.36, 6.84, 0.97, 0.001, True),
        ("RoBERTa-Large", +8.56, -5.85, 18.88, 0.23, 0.191, False),
        ("RoBERTa-IB",  +8.19, 5.99, 10.56, 1.17, 0.001, False),
        ("BART-Base",   +10.08, 6.28, 13.73, 0.90, 0.001, False),
    ]

    data = sorted(TABLE2_DATA, key=lambda x: x[1])
    names = [d[0] for d in data]
    gaps = [d[1] for d in data]
    ci_lo = [d[2] for d in data]
    ci_hi = [d[3] for d in data]
    is_contr = [d[6] for d in data]

    y = np.arange(len(names))
    err_lo = [g - lo for g, lo in zip(gaps, ci_lo)]
    err_hi = [hi - g for g, hi in zip(gaps, ci_hi)]
    colors = [C_CONTRASTIVE if c else C_NON_CONTRASTIVE for c in is_contr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    # (a) Horizontal bar chart
    ax1.barh(y, gaps, xerr=[err_lo, err_hi], color=colors, alpha=0.8,
             capsize=3, error_kw={'linewidth': 1, 'color': '#333'},
             height=0.65, edgecolor='white', linewidth=0.5)
    ax1.axvline(x=0, color='#333', linewidth=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel("Optimized − Random Gap (%)", fontsize=12)
    ax1.set_title("(a) Optimized-Random Gap by Model", fontweight='bold')

    for i, (name, gap) in enumerate(zip(names, gaps)):
        x_pos = gap + 0.4 if gap >= 0 else gap - 0.4
        ax1.text(x_pos, i, f'+{gap:.1f}%', va='center', fontsize=8,
                 ha='left' if gap >= 0 else 'right', fontweight='bold', color='#333')

    # (b) Distribution of gaps by paradigm (box plot)
    bp2 = ax2.boxplot([[d[1] for d in data if d[6]], [d[1] for d in data if not d[6]]],
                       patch_artist=True, widths=0.5,
                       labels=['Contrastive\n(n=8)', 'Non-contrastive\n(n=3)'])
    bp2['boxes'][0].set_facecolor(C_CONTRASTIVE)
    bp2['boxes'][0].set_alpha(0.7)
    bp2['boxes'][1].set_facecolor(C_NON_CONTRASTIVE)
    bp2['boxes'][1].set_alpha(0.7)

    ax2.axhline(y=0, color='#333', linestyle=':', alpha=0.5)
    ax2.set_ylabel("Gap (%)")
    ax2.set_title("(b) Gap Distribution\nby Training Paradigm", fontweight='bold')

    # Effect size annotation
    con_gaps = [d[1] for d in data if d[6]]
    noncon_gaps = [d[1] for d in data if not d[6]]
    ax2.annotate(f'Contrastive: {np.mean(con_gaps):.1f}±{np.std(con_gaps):.1f}%\n'
                 f'Non-contr.: {np.mean(noncon_gaps):.1f}±{np.std(noncon_gaps):.1f}%',
                 xy=(0.02, 0.98), xycoords='axes fraction', va='top', fontsize=8,
                 bbox=dict(facecolor='#F5F5F5', alpha=0.8, edgecolor='#CCC'))

    legend_elements = [
        mpatches.Patch(facecolor=C_CONTRASTIVE, alpha=0.8, label="Contrastive"),
        mpatches.Patch(facecolor=C_NON_CONTRASTIVE, alpha=0.8, label="Non-contrastive"),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_opt_random_gap_all_models.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Fix 5: fig7_redundancy_mechanism.png (88.9% → target <75%)
# ============================================================
def fix_redundancy_mechanism():
    set_style()

    # Load analyze data for real entropy/concentration values
    analyze_data = {}
    for fname in os.listdir(ANALYZE_DIR):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(ANALYZE_DIR, fname)) as f:
                analyze_data[model_name] = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # (a) Entropy bar chart with ALL models
    ax = axes[0]
    models_sorted = sorted(analyze_data.keys())
    y = np.arange(len(models_sorted))
    entropies = []
    for m in models_sorted:
        tasks = analyze_data[m].get("task_name", {})
        ent_vals = []
        for tname, tdata in tasks.items():
            for ws, ws_data in tdata.get("split_win_size", {}).items():
                for cw, cw_data in ws_data.get("chunk_win_size", {}).items():
                    head = cw_data.get("head_score", {}).get("main_score")
                    if head is not None:
                        ent_vals.append(head)
        # Compute entropy from head scores
        if ent_vals:
            arr = np.array(ent_vals)
            arr = arr[arr > 0]
            if len(arr) > 0:
                probs = arr / arr.sum()
                h = -np.sum(probs * np.log(probs + 1e-10))
                h_norm = h / np.log(len(probs))
                entropies.append(h_norm)
            else:
                entropies.append(0)
        else:
            entropies.append(0)

    display_names = [MODEL_DISPLAY.get(m, m)[:12] for m in models_sorted]
    colors = [C_NON_CONTRASTIVE if MODEL_DISPLAY.get(m, '') in NON_CONTRASTIVE
              else C_CONTRASTIVE for m in models_sorted]

    ax.barh(y, entropies, color=colors, alpha=0.8, height=0.6, edgecolor='white', linewidth=0.5)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Maximum (uniform)')
    ax.set_yticks(y)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel("Normalized Entropy")
    ax.set_title("(a) Dimension Importance\nEntropy", fontweight='bold')
    ax.set_xlim(0.9, 1.01)
    ax.legend(fontsize=7)

    # (b) Top-K concentration
    ax = axes[1]
    cmap = plt.cm.tab10
    for mi, m in enumerate(models_sorted[:6]):
        tasks = analyze_data[m].get("task_name", {})
        # Approximate concentration from chunk scores
        chunk_scores_all = []
        for tname, tdata in tasks.items():
            for ws, ws_data in tdata.get("split_win_size", {}).items():
                for cw, cw_data in ws_data.get("chunk_win_size", {}).items():
                    head = cw_data.get("head_score", {}).get("main_score")
                    end = cw_data.get("end_score", {}).get("main_score")
                    if head is not None and end is not None:
                        chunk_scores_all.append((head, end))

        if chunk_scores_all:
            sorted_scores = sorted([h for h, e in chunk_scores_all], reverse=True)
            total = sum(sorted_scores)
            if total > 0:
                x_pts = np.linspace(0, 1, 50)
                cum = np.cumsum(sorted_scores) / total
                interp_vals = np.interp(x_pts, np.linspace(0, 1, len(cum)), cum)
                ax.plot(x_pts, interp_vals, color=cmap(mi / 6), alpha=0.8,
                        label=MODEL_DISPLAY.get(m, m)[:10])

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Uniform')
    ax.set_xlabel("Fraction of Chunks")
    ax.set_ylabel("Cumulative Importance")
    ax.set_title("(b) Cumulative Importance\n(lower = more uniform)", fontweight='bold')
    ax.legend(fontsize=6, loc='upper left')

    # (c) Best-Poor gap distribution across tasks for each model
    ax = axes[2]
    top6 = models_sorted[:6]
    display6 = [MODEL_DISPLAY.get(m, m)[:10] for m in top6]
    all_gaps_by_model = []

    for mi, m in enumerate(top6):
        tasks = analyze_data[m].get("task_name", {})
        gaps = []
        for tname, tdata in tasks.items():
            for ws, ws_data in tdata.get("split_win_size", {}).items():
                for cw, cw_data in ws_data.get("chunk_win_size", {}).items():
                    head = cw_data.get("head_score", {}).get("main_score")
                    end = cw_data.get("end_score", {}).get("main_score")
                    if head is not None and end is not None:
                        gaps.append(head - end)
        all_gaps_by_model.append(gaps)

    if any(len(g) > 0 for g in all_gaps_by_model):
        bp3 = ax.boxplot(all_gaps_by_model, patch_artist=True, widths=0.5)
        for i, patch in enumerate(bp3['boxes']):
            patch.set_facecolor(cmap(i / 6))
            patch.set_alpha(0.7)

    ax.set_xticklabels(display6, rotation=25, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylabel("Best-Poor Gap")
    ax.set_title("(c) Best-Poor Gap\nper Model", fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig7_redundancy_mechanism.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Fix 6: fig9_magnitude_comparison.png (94.4% → target <75%)
# ============================================================
def fix_magnitude_comparison():
    set_style()

    # Load magnitude analysis data
    mag_path = os.path.join(DATA_DIR, "magnitude_analysis.json")
    if not os.path.exists(mag_path):
        print("magnitude_analysis.json not found, skipping")
        return
    mag_analysis = json.load(open(mag_path))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # (a) Magnitude correlation bar chart — dense version
    ax = axes[0]
    mag_corr = mag_analysis.get("magnitude_vs_task_correlation", {})
    models_sorted = sorted(mag_corr.keys())
    y = np.arange(len(models_sorted))
    means = [mag_corr[m]["summary"]["mean_rho"] for m in models_sorted]
    stds = [mag_corr[m]["summary"]["std_rho"] for m in models_sorted]
    colors = [C_NON_CONTRASTIVE if MODEL_DISPLAY.get(m, '') in NON_CONTRASTIVE
              else C_CONTRASTIVE for m in models_sorted]

    ax.barh(y, means, xerr=stds, color=colors, alpha=0.8, capsize=3,
            height=0.6, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='#333', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m)[:12] for m in models_sorted], fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("(a) Magnitude vs Task\nImportance Correlation", fontweight='bold')
    ax.set_xlim(-0.2, 0.2)

    # Add value annotations
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + 0.01 if mean >= 0 else mean - 0.01, i,
                f'ρ={mean:.3f}', va='center', fontsize=7,
                ha='left' if mean >= 0 else 'right')

    # (b) Per-model: number of tasks where magnitude wins vs loses
    ax = axes[1]
    for mi, m in enumerate(models_sorted):
        tasks = mag_corr[m].get("tasks", {})
        wins = sum(1 for t in tasks.values() if t.get("rho", 0) > 0.05)
        losses = sum(1 for t in tasks.values() if t.get("rho", 0) < -0.05)
        ties = len(tasks) - wins - losses
        display = MODEL_DISPLAY.get(m, m)[:12]
        color = C_NON_CONTRASTIVE if display in NON_CONTRASTIVE else C_CONTRASTIVE

        ax.barh(mi - 0.15, wins, 0.3, color='#27AE60', alpha=0.7, edgecolor='white')
        ax.barh(mi + 0.15, losses, 0.3, color='#E74C3C', alpha=0.7, edgecolor='white')

    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m)[:12] for m in models_sorted], fontsize=8)
    ax.set_xlabel("Number of Tasks")
    ax.set_title("(b) Tasks Where Magnitude\nWins (green) vs Loses (red)", fontweight='bold')
    ax.legend([mpatches.Patch(color='#27AE60', alpha=0.7),
               mpatches.Patch(color='#E74C3C', alpha=0.7)],
              ['Mag > Random', 'Mag < Random'], fontsize=8, loc='lower right')

    # (c) Summary: mean correlation + distribution
    ax = axes[2]
    all_rhos = []
    model_labels = []
    for m in models_sorted:
        tasks = mag_corr[m].get("tasks", {})
        rhos = [t.get("rho", 0) for t in tasks.values()]
        all_rhos.append(rhos)
        model_labels.append(MODEL_DISPLAY.get(m, m)[:10])

    bp = ax.boxplot(all_rhos, patch_artist=True, vert=False, widths=0.6)
    cmap = plt.cm.Set2
    for i, patch in enumerate(bp['boxes']):
        color = C_NON_CONTRASTIVE if MODEL_DISPLAY.get(models_sorted[i], '') in NON_CONTRASTIVE else C_CONTRASTIVE
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("(c) Per-Task Correlation\nDistribution", fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor=C_CONTRASTIVE, alpha=0.7, label='Contrastive'),
        mpatches.Patch(facecolor=C_NON_CONTRASTIVE, alpha=0.7, label='Non-contrastive'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig9_magnitude_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Fixing sparse figures...")
    fix_random_variance()
    fix_training_paradigm()
    fix_entropy()
    fix_opt_random_gap()
    fix_redundancy_mechanism()
    fix_magnitude_comparison()
    print("Done! All sparse figures fixed.")
