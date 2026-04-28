"""Generate fig_concept_overview.png — Research Framework Diagram.
A schematic showing: WHAT we study → HOW we do it → WHAT we find.
Combines text labels with graphical illustrations (not raw result charts).
Wide flat layout for double-column EMNLP paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches
import numpy as np

# ── Color palette ─────────────────────────────────────────────
C_CONTR   = '#4A90D9'
C_NONCONTR = '#E74C3C'
C_FINE    = '#E67E22'
C_INSTR   = '#7B68EE'
C_AGNO    = '#6C757D'   # task-agnostic methods
C_AWARE   = '#27AE60'   # task-aware methods
C_ADV     = '#9B59B6'   # advanced methods
C_ARROW   = '#2C3E50'
C_FIND    = '#E67E22'
C_DARK    = '#2C3E50'
C_LIGHTBG = '#F8F9FA'
C_GREENBG = '#E8F5E9'
C_BLUEBG  = '#E3F2FD'
C_REDBG   = '#FFEBEE'
C_PURPBG  = '#F3E5F5'
C_YELLOW  = '#FFF8E1'

fig = plt.figure(figsize=(16, 6), facecolor='white')

# ── Helper functions ──────────────────────────────────────────
def rounded_box(ax, x, y, w, h, text, fc, tc='white', fs=7, bold=True,
                ec=None, lw=0.6, alpha=0.92, zorder=2):
    ec = ec or fc
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal', color=tc, zorder=zorder+1)

def section_bg(ax, color=C_LIGHTBG, ec='#D0D0D0'):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    b = FancyBboxPatch((0.05, 0.05), 9.9, 9.85, boxstyle='round,pad=0.12',
                        facecolor=color, edgecolor=ec, linewidth=1, alpha=0.6, zorder=0)
    ax.add_patch(b)


# ══════════════════════════════════════════════════════════════
# PANEL A (left): WHAT WE STUDY — Models + Tasks
# ══════════════════════════════════════════════════════════════
ax_a = fig.add_axes([0.01, 0.03, 0.19, 0.92])
section_bg(ax_a)
ax_a.text(5, 9.6, 'A. What We Study', ha='center', fontsize=11,
          fontweight='bold', color=C_DARK)

# ── Embedding vector illustration ──
# Show a colorful embedding vector (heatmap strip) being compressed
ax_vec = fig.add_axes([0.025, 0.76, 0.16, 0.10])
np.random.seed(7)
vec = np.concatenate([np.random.randn(1, 32)*0.6 + 0.3,
                      np.random.randn(1, 32)*0.4,
                      np.random.randn(1, 32)*0.6 - 0.2,
                      np.random.randn(1, 32)*0.5 + 0.1], axis=1)
ax_vec.imshow(vec, cmap='coolwarm', aspect='auto', vmin=-1.5, vmax=1.5)
ax_vec.set_title('Embedding Vector  e ∈ ℝᴰ', fontsize=7, fontweight='bold', pad=2)
ax_vec.set_xlabel('D = 768~1024 dims', fontsize=5.5)
ax_vec.set_yticks([])
ax_vec.tick_params(axis='x', labelsize=4.5)
for s in ax_vec.spines.values(): s.set_visible(False)

# ── Arrow: vector → pruned ──
fig.text(0.105, 0.73, '▼  Select k dims  ▼', ha='center', fontsize=7,
         color=C_ARROW, fontweight='bold')

# ── Pruned vector illustration ──
ax_prune = fig.add_axes([0.025, 0.59, 0.16, 0.10])
mask = np.zeros_like(vec)
keep_idx = np.sort(np.random.choice(128, 32, replace=False))
mask[0, keep_idx] = vec[0, keep_idx]
ax_prune.imshow(np.abs(mask), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1.5)
# Gray out removed dims
for j in range(128):
    if j not in keep_idx:
        ax_prune.axvline(x=j, color='#E0E0E0', linewidth=0.3, alpha=0.7)
ax_prune.set_title('Pruned Vector  e′ ∈ ℝᵏ  (k=256)', fontsize=7, fontweight='bold', pad=2)
ax_prune.set_xlabel('Only k ≪ D dimensions kept', fontsize=5.5)
ax_prune.set_yticks([])
ax_prune.tick_params(axis='x', labelsize=4.5)
for s in ax_prune.spines.values(): s.set_visible(False)

# ── Model paradigm legend ──
ax_a.text(5, 5.4, '11 Embedding Models', ha='center', fontsize=8.5,
          fontweight='bold', color=C_DARK)

paradigms = [
    (0.3, 4.5, 4.2, 0.7, 'Contrastive\n(7 models)', C_CONTR),
    (4.8, 4.5, 4.8, 0.7, 'Non-contrastive\n(2 models)', C_NONCONTR),
    (0.3, 3.3, 4.2, 0.7, 'Fine-tuned\n(1 model)', C_FINE),
    (4.8, 3.3, 4.8, 0.7, 'Instruction-tuned\n(1 model)', C_INSTR),
]
for x, y, w, h, label, color in paradigms:
    rounded_box(ax_a, x, y, w, h, label, color, fs=6.5)

# Model names (small text)
ax_a.text(5, 2.65, 'GTE-Large · Stella · BGE-M3 · Qwen3\n'
          'BART · RoBERTa · Instructor · ...',
          ha='center', fontsize=5.5, color='#666', linespacing=1.3)

# ── Task categories ──
ax_a.text(5, 1.8, '35 MTEB Tasks', ha='center', fontsize=8.5,
          fontweight='bold', color=C_DARK)
tasks = [
    (0.3, 0.8, 2.1, 0.7, 'Classif.\n(11)', '#5D6D7E'),
    (2.6, 0.8, 2.1, 0.7, 'Retriev.\n(5)', '#5D6D7E'),
    (4.9, 0.8, 2.1, 0.7, 'STS\n(7)', '#5D6D7E'),
    (7.2, 0.8, 2.5, 0.7, 'Others\n(12)', '#5D6D7E'),
]
for x, y, w, h, label, color in tasks:
    rounded_box(ax_a, x, y, w, h, label, color, fs=6)


# ══════════════════════════════════════════════════════════════
# PANEL B (center-left): HOW — 8 Selection Methods
# ══════════════════════════════════════════════════════════════
ax_b = fig.add_axes([0.215, 0.03, 0.27, 0.92])
section_bg(ax_b, '#FAFBFF', '#C5CAE9')
ax_b.text(5, 9.6, 'B. How: 8 Selection Methods', ha='center', fontsize=11,
          fontweight='bold', color=C_DARK)

# ── Method Category 1: Task-Agnostic ──
rounded_box(ax_b, 0.3, 8.3, 9.3, 0.55, 'Task-Agnostic (no task data needed)',
            C_AGNO, fs=7.5)

# Three method boxes with mini-illustrations
# Random
rounded_box(ax_b, 0.3, 7.0, 2.8, 1.0, '① Random\nSelect k dims\nuniformly', '#95A5A6', fs=6.5)
# Mini random selection visualization
ax_rand = fig.add_axes([0.225, 0.745, 0.07, 0.05])
np.random.seed(42)
dims_r = np.arange(20)
sel_r = np.random.choice(20, 8, replace=False)
cols_r = ['#E74C3C' if i in sel_r else '#D5D8DC' for i in dims_r]
ax_rand.bar(dims_r, np.ones(20), color=cols_r, width=1, edgecolor='white', linewidth=0.2)
ax_rand.axis('off')
ax_rand.set_xlim(-0.5, 19.5)

# Sequential
rounded_box(ax_b, 3.5, 7.0, 2.8, 1.0, '② Sequential\nKeep first k\ndimensions', '#95A5A6', fs=6.5)
# Mini sequential visualization
ax_seq = fig.add_axes([0.335, 0.745, 0.07, 0.05])
cols_s = ['#4A90D9']*8 + ['#D5D8DC']*12
ax_seq.bar(dims_r, np.ones(20), color=cols_s, width=1, edgecolor='white', linewidth=0.2)
ax_seq.axis('off')
ax_seq.set_xlim(-0.5, 19.5)

# Magnitude
rounded_box(ax_b, 6.7, 7.0, 3.0, 1.0, '③ Magnitude\nRank by ‖w_d‖₂\n(weight norm)', '#95A5A6', fs=6.5)
# Mini magnitude visualization
ax_mag = fig.add_axes([0.605, 0.745, 0.07, 0.05])
np.random.seed(99)
heights = np.random.uniform(0.3, 1.0, 20)
top8 = np.argsort(heights)[-8:]
cols_m = ['#E67E22' if i in top8 else '#D5D8DC' for i in range(20)]
ax_mag.bar(dims_r, heights, color=cols_m, width=1, edgecolor='white', linewidth=0.2)
ax_mag.axis('off')
ax_mag.set_xlim(-0.5, 19.5)

# ── Method Category 2: Task-Aware ──
rounded_box(ax_b, 0.3, 5.7, 9.3, 0.55, 'Task-Aware (requires task evaluation)',
            C_AWARE, fs=7.5)

# Oracle/Optimized
rounded_box(ax_b, 0.3, 4.4, 4.5, 1.0,
            '④ Oracle (Optimized)\nScore each chunk on task T\nSelect top-k chunks',
            C_AWARE, fs=6.5)
# Mini oracle illustration
ax_orc = fig.add_axes([0.25, 0.465, 0.07, 0.05])
scores = np.array([0.3, 0.9, 0.5, 0.8, 0.2, 0.7, 0.6, 0.95, 0.4, 0.85,
                   0.35, 0.75, 0.55, 0.88, 0.25, 0.65, 0.45, 0.92, 0.38, 0.72])
top_o = np.argsort(scores)[-8:]
cols_o = ['#27AE60' if i in top_o else '#D5D8DC' for i in range(20)]
ax_orc.bar(dims_r, scores, color=cols_o, width=1, edgecolor='white', linewidth=0.2)
ax_orc.axis('off')
ax_orc.set_xlim(-0.5, 19.5)

# Anti-optimized
rounded_box(ax_b, 5.1, 4.4, 4.5, 1.0,
            '⑤ Anti-optimized\nSelect worst chunks\n(lower bound)',
            C_NONCONTR, fs=6.5)

# ── Method Category 3: Advanced ──
rounded_box(ax_b, 0.3, 3.1, 9.3, 0.55, 'Advanced (gradient / learning-based)',
            C_ADV, fs=7.5)

rounded_box(ax_b, 0.3, 1.8, 3.0, 1.0, '⑥ Gradient\nSaliency\n∇_e L per dim', C_ADV, fs=6.5)
rounded_box(ax_b, 3.5, 1.8, 3.0, 1.0, '⑦ Activation\nVariance\nVar(e_d) ranking', C_ADV, fs=6.5)
rounded_box(ax_b, 6.7, 1.8, 3.0, 1.0, '⑧ Learned\nMask\nTrainable binary', C_ADV, fs=6.5)

# ── Evaluation metric box at bottom ──
rounded_box(ax_b, 0.3, 0.3, 9.3, 1.2,
            'Retention: R = S(k)/S(D) × 100%\n'
            'S(k): task score with k dims  |  S(D): full-dim score\n'
            'Budgets tested: k = 64, 128, 256, 512',
            '#ECF0F1', tc='#333', fs=6.5, bold=False, ec='#BDC3C7', lw=1)


# ══════════════════════════════════════════════════════════════
# PANEL C (center-right): WHAT WE VERIFY — Experiments
# ══════════════════════════════════════════════════════════════
ax_c = fig.add_axes([0.505, 0.03, 0.23, 0.92])
section_bg(ax_c, '#FFFAF0', '#FFE0B2')
ax_c.text(5, 9.6, 'C. What We Verify', ha='center', fontsize=11,
          fontweight='bold', color=C_DARK)

experiments = [
    # (y_pos, icon_char, title, description, color)
    (8.0, '📊', 'Exp 1: Pruning Ratio Sweep',
     'How does retention\ndecline as k decreases?\nk = 512→128→64', '#4A90D9'),
    (6.0, '🔄', 'Exp 2: Cross-Task Transfer',
     'Does Task A ranking\nwork for Task B?\nRank correlation ρ ≈ 0', '#27AE60'),
    (4.0, '📐', 'Exp 3: Information Uniformity',
     'Is importance spread\nuniformly?\nH_norm, Gini coefficient', '#E67E22'),
    (2.0, '🎯', 'Exp 4: Paradigm Dependence',
     'Does training objective\nshape dimension structure?\nContrastive vs others', '#9B59B6'),
    (0.4, '⚡', 'Exp 5: Advanced Methods',
     'Do gradient/mask methods\nbeat random?\nMarginal gains only', '#E74C3C'),
]

for y, icon, title, desc, color in experiments:
    # Experiment card
    bg = FancyBboxPatch((0.3, y - 0.5), 9.3, 1.5, boxstyle='round,pad=0.08',
                         facecolor='white', edgecolor=color, linewidth=1.2, alpha=0.9, zorder=2)
    ax_c.add_patch(bg)
    # Left color stripe
    stripe = FancyBboxPatch((0.3, y - 0.5), 0.35, 1.5, boxstyle='round,pad=0.0',
                             facecolor=color, edgecolor=color, linewidth=0, alpha=0.9, zorder=3)
    ax_c.add_patch(stripe)
    ax_c.text(0.5, y + 0.6, title, fontsize=7.5, fontweight='bold', color=color, zorder=4)
    ax_c.text(1.0, y - 0.15, desc, fontsize=6, color='#555', linespacing=1.2, zorder=4)


# ══════════════════════════════════════════════════════════════
# PANEL D (right): WHAT WE FIND — Key Findings
# ══════════════════════════════════════════════════════════════
ax_d = fig.add_axes([0.755, 0.03, 0.24, 0.92])
section_bg(ax_d, C_GREENBG, '#A5D6A7')
ax_d.text(5, 9.6, 'D. Key Findings', ha='center', fontsize=11,
          fontweight='bold', color=C_DARK)

# ── Finding 1: Random ≈ Oracle ──
bg1 = FancyBboxPatch((0.3, 7.2), 9.3, 2.0, boxstyle='round,pad=0.1',
                      facecolor='white', edgecolor=C_FIND, linewidth=1.5, zorder=2)
ax_d.add_patch(bg1)
ax_d.text(5, 8.85, 'F1: Random Is Near-Optimal', ha='center',
          fontsize=8, fontweight='bold', color=C_FIND, zorder=3)
# Mini comparison illustration: two bars side by side
ax_f1 = fig.add_axes([0.77, 0.82, 0.08, 0.06])
categories_f1 = ['Random', 'Oracle']
contr_vals = [97.5, 99.7]
noncontr_vals = [87.2, 97.3]
x_f1 = np.arange(2)
w = 0.35
ax_f1.barh(x_f1 - w/2, contr_vals, w, color=C_CONTR, alpha=0.8, label='Contrast.')
ax_f1.barh(x_f1 + w/2, noncontr_vals, w, color=C_NONCONTR, alpha=0.8, label='Non-contr.')
ax_f1.set_yticks(x_f1)
ax_f1.set_yticklabels(categories_f1, fontsize=5)
ax_f1.set_xlim(80, 102)
ax_f1.tick_params(axis='x', labelsize=4)
ax_f1.set_title('Retention (%)', fontsize=5, pad=1)
for s in ['top', 'right']: ax_f1.spines[s].set_visible(False)

ax_d.text(5, 7.55,
          'Gap: +2–5% (contrastive)\n'
          'Gap: +8–10% (non-contrastive)\n'
          '→ Paradigm matters!',
          ha='center', fontsize=6, color='#333', linespacing=1.3, zorder=3)

# ── Finding 2: Uniform importance ──
bg2 = FancyBboxPatch((0.3, 4.8), 9.3, 2.1, boxstyle='round,pad=0.1',
                      facecolor='white', edgecolor=C_CONTR, linewidth=1.5, zorder=2)
ax_d.add_patch(bg2)
ax_d.text(5, 6.5, 'F2: Near-Uniform Importance', ha='center',
          fontsize=8, fontweight='bold', color=C_CONTR, zorder=3)

# Mini heatmap: uniform vs skewed
ax_unif = fig.add_axes([0.77, 0.60, 0.08, 0.05])
# Uniform importance (left half) vs skewed (right half)
uniform_imp = np.ones((1, 16)) * 0.95 + np.random.uniform(-0.05, 0.05, (1, 16))
skewed_imp = np.array([[0.95, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5,
                         0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]])
combined = np.hstack([uniform_imp, skewed_imp])
ax_unif.imshow(combined, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
ax_unif.axvline(x=15.5, color='red', linewidth=1, linestyle='--')
ax_unif.text(8, -0.5, 'H=0.99', fontsize=4, ha='center', color=C_CONTR, fontweight='bold')
ax_unif.text(24, -0.5, 'Skewed', fontsize=4, ha='center', color=C_NONCONTR, fontweight='bold')
ax_unif.axis('off')

ax_d.text(5, 5.15,
          'H_norm = 0.991 (contrastive)\n'
          'Gini = 0.10–0.12\n'
          '→ Any large subset works',
          ha='center', fontsize=6, color='#333', linespacing=1.3, zorder=3)

# ── Finding 3: Transfer paradox ──
bg3 = FancyBboxPatch((0.3, 2.5), 9.3, 1.9, boxstyle='round,pad=0.1',
                      facecolor='white', edgecolor=C_AWARE, linewidth=1.5, zorder=2)
ax_d.add_patch(bg3)
ax_d.text(5, 4.1, 'F3: Rankings Don\'t Transfer', ha='center',
          fontsize=8, fontweight='bold', color=C_AWARE, zorder=3)

# Mini: two different rankings with arrows
ax_xfer = fig.add_axes([0.77, 0.36, 0.08, 0.05])
# Two mini bar charts showing different rank orderings
rank_a = np.array([5, 2, 8, 1, 7, 3, 6, 4, 10, 9])
rank_b = np.array([9, 7, 1, 5, 3, 10, 2, 8, 4, 6])
x_r = np.arange(10)
ax_xfer.bar(x_r - 0.2, rank_a/10, 0.35, color=C_CONTR, alpha=0.7)
ax_xfer.bar(x_r + 0.2, rank_b/10, 0.35, color=C_NONCONTR, alpha=0.7)
ax_xfer.text(5, -0.15, 'ρ ≈ 0.001', fontsize=4, ha='center', color=C_FIND, fontweight='bold')
ax_xfer.axis('off')

ax_d.text(5, 2.8,
          'ρ ≈ 0.001 (no rank transfer)\n'
          'But either ranking retains 95–100%\n'
          '→ Paradox resolved by uniformity',
          ha='center', fontsize=6, color='#333', linespacing=1.3, zorder=3)

# ── Finding 4: Advanced methods ≈ random ──
bg4 = FancyBboxPatch((0.3, 0.3), 9.3, 1.9, boxstyle='round,pad=0.1',
                      facecolor='white', edgecolor=C_ADV, linewidth=1.5, zorder=2)
ax_d.add_patch(bg4)
ax_d.text(5, 1.9, 'F4: Advanced ≈ Random', ha='center',
          fontsize=8, fontweight='bold', color=C_ADV, zorder=3)

# Mini bar: 4 methods nearly equal
ax_adv = fig.add_axes([0.77, 0.08, 0.08, 0.05])
methods = ['Rand', 'Grad', 'ActV', 'Mask']
rets = [99.88, 99.96, 99.85, 100.27]
cols_adv = [C_AGNO, C_ADV, C_ADV, C_ADV]
ax_adv.bar(range(4), rets, color=cols_adv, alpha=0.7, width=0.6)
ax_adv.set_ylim(98.5, 101)
ax_adv.set_xticks(range(4))
ax_adv.set_xticklabels(methods, fontsize=4)
ax_adv.tick_params(axis='y', labelsize=3.5)
for s in ['top', 'right']: ax_adv.spines[s].set_visible(False)
ax_adv.axhline(y=100, color='#333', linestyle=':', linewidth=0.5)

ax_d.text(5, 0.6,
          'Gradient, ActVar, Mask:\n'
          'Marginal gains over random\n'
          '→ Not worth the compute cost',
          ha='center', fontsize=6, color='#333', linespacing=1.3, zorder=3)


# ══════════════════════════════════════════════════════════════
# FLOW ARROWS connecting sections
# ══════════════════════════════════════════════════════════════
arrow_y = 0.50
arrow_props = dict(arrowstyle='->', color=C_ARROW, lw=2.5,
                   connectionstyle='arc3,rad=0')

# A → B
fig.patches.append(FancyArrowPatch(
    (0.205, arrow_y), (0.22, arrow_y),
    **arrow_props, transform=fig.transFigure, figure=fig, zorder=10))

# B → C
fig.patches.append(FancyArrowPatch(
    (0.49, arrow_y), (0.505, arrow_y),
    **arrow_props, transform=fig.transFigure, figure=fig, zorder=10))

# C → D
fig.patches.append(FancyArrowPatch(
    (0.74, arrow_y), (0.755, arrow_y),
    **arrow_props, transform=fig.transFigure, figure=fig, zorder=10))

# Arrow labels
fig.text(0.213, arrow_y + 0.02, 'apply', fontsize=6, color=C_ARROW,
         ha='center', fontweight='bold')
fig.text(0.497, arrow_y + 0.02, 'verify', fontsize=6, color=C_ARROW,
         ha='center', fontweight='bold')
fig.text(0.748, arrow_y + 0.02, 'reveal', fontsize=6, color=C_ARROW,
         ha='center', fontweight='bold')


plt.savefig('/home/linkco/exa/llm-usefulEeb/paper/figures/fig_concept_overview.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("fig_concept_overview.png (framework diagram v3) generated.")
