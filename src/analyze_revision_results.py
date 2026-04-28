#!/usr/bin/env python3
"""
Analyze all revision experiment results and generate summary tables.
Reads from data/experiment_results/ and outputs LaTeX tables and summaries.
"""

import json
import os
import numpy as np
from collections import defaultdict

RESULT_DIR = "data/experiment_results"


def load_result(prefix, model_name):
    """Load a result file by prefix and model name."""
    path = os.path.join(RESULT_DIR, f"{prefix}_{model_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================================
# 1. PCA Baseline Comparison
# ============================================================================

def analyze_pca_baselines():
    """Compare PCA, Random Projection, and Random Coordinate selection."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("PCA BASELINE COMPARISON")
    print("="*70)

    for model in models:
        data = load_result("pca_baseline", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model} (dim={data['model_dim']})")
        print(f"  {'Task':<35} {'dim':>4} {'PCA%':>6} {'RP%':>6} {'RandCoord%':>10}")
        print("  " + "-"*65)

        for task, tr in data['methods'].items():
            baseline = tr['baseline']
            for dim_key in sorted([k for k in tr if k.startswith('dim_')]):
                dim = dim_key.replace('dim_', '')
                dr = tr[dim_key]
                print(f"  {task:<35} {dim:>4} {dr['pca_retention']:>6.1f} "
                      f"{dr['rp_retention']:>6.1f} {dr['random_coord_retention']:>10.1f}")

    # Aggregate: mean retention per model per method per dim
    print("\n\n  AGGREGATE (mean across tasks)")
    print(f"  {'Model':<25} {'dim':>4} {'PCA%':>6} {'RP%':>6} {'RandCoord%':>10}")
    print("  " + "-"*55)

    for model in models:
        data = load_result("pca_baseline", model)
        if data is None:
            continue

        for target_dim in data['target_dims']:
            pca_rets, rp_rets, rc_rets = [], [], []
            for task, tr in data['methods'].items():
                dim_key = f"dim_{target_dim}"
                if dim_key in tr:
                    pca_rets.append(tr[dim_key]['pca_retention'])
                    rp_rets.append(tr[dim_key]['rp_retention'])
                    rc_rets.append(tr[dim_key]['random_coord_retention'])

            if pca_rets:
                print(f"  {model:<25} {target_dim:>4} {np.mean(pca_rets):>6.1f} "
                      f"{np.mean(rp_rets):>6.1f} {np.mean(rc_rets):>10.1f}")


# ============================================================================
# 2. Gradient / Saliency Baselines
# ============================================================================

def analyze_gradient_saliency():
    """Compare gradient, activation variance, learned mask vs random."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("GRADIENT / SALIENCY BASELINES")
    print("="*70)

    for model in models:
        data = load_result("gradient_saliency", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model} (dim={data['model_dim']})")
        print(f"  {'Task':<35} {'budget':>6} {'Rand%':>6} {'Grad%':>6} {'Var%':>6} {'Mask%':>6}")
        print("  " + "-"*70)

        for task, tr in data['methods'].items():
            baseline = tr['baseline']
            for budget, br in sorted(tr['budgets'].items()):
                parts = [f"  {task:<35} {budget:>6}"]
                for method in ['random', 'gradient', 'activation_variance', 'learned_mask']:
                    if method in br:
                        parts.append(f"{br[method]['retention']:>6.1f}")
                    else:
                        parts.append(f"{'N/A':>6}")
                print(" ".join(parts))

    # Aggregate
    print("\n\n  AGGREGATE (mean retention across tasks)")
    print(f"  {'Model':<25} {'budget':>6} {'Rand%':>6} {'Grad%':>6} {'Var%':>6} {'Mask%':>6}")
    print("  " + "-"*60)

    for model in models:
        data = load_result("gradient_saliency", model)
        if data is None:
            continue

        for budget in ['64', '128', '256']:
            rets = defaultdict(list)
            for task, tr in data['methods'].items():
                if budget in tr['budgets']:
                    for method in ['random', 'gradient', 'activation_variance', 'learned_mask']:
                        if method in tr['budgets'][budget]:
                            rets[method].append(tr['budgets'][budget][method]['retention'])

            parts = [f"  {model:<25} {budget:>6}"]
            for method in ['random', 'gradient', 'activation_variance', 'learned_mask']:
                if rets[method]:
                    parts.append(f"{np.mean(rets[method]):>6.1f}")
                else:
                    parts.append(f"{'N/A':>6}")
            print(" ".join(parts))


# ============================================================================
# 3. Leave-One-Out Analysis
# ============================================================================

def analyze_loo():
    """Analyze LOO results: importance definition agreement."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("LEAVE-ONE-OUT / IMPORTANCE DEFINITION ANALYSIS")
    print("="*70)

    for model in models:
        data = load_result("leave_one_out", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model} (dim={data['model_dim']}, win_size={data['win_size']}, n_chunks={data['n_chunks']})")
        print(f"  {'Task':<35} {'rho(stand,loo)':>14} {'rho(stand,marg)':>15} {'rho(stand,shap)':>15}")
        print("  " + "-"*85)

        for task, tr in data['tasks'].items():
            rho_loo = tr.get('rho_standalone_loo', 'N/A')
            rho_marg = tr.get('rho_standalone_marginal', 'N/A')
            rho_shap = tr.get('rho_standalone_shapley', 'N/A')

            loo_str = f"{rho_loo:.3f}" if isinstance(rho_loo, (int, float)) else str(rho_loo)
            marg_str = f"{rho_marg:.3f}" if isinstance(rho_marg, (int, float)) else str(rho_marg)
            shap_str = f"{rho_shap:.3f}" if isinstance(rho_shap, (int, float)) else str(rho_shap)

            print(f"  {task:<35} {loo_str:>14} {marg_str:>15} {shap_str:>15}")

        # Aggregate entropies
        print(f"\n  Entropy across importance definitions:")
        for defn in ['standalone', 'loo', 'marginal', 'shapley']:
            entropies = [tr.get(f'entropy_{defn}') for tr in data['tasks'].values()
                        if f'entropy_{defn}' in tr]
            if entropies:
                print(f"    {defn}: mean={np.mean(entropies):.4f}, std={np.std(entropies):.4f}")

        # Top-k Jaccard overlap
        print(f"\n  Top-k Jaccard overlap (budget={data['budget']}):")
        pairs = [('standalone', 'loo'), ('standalone', 'marginal'), ('standalone', 'shapley'),
                 ('loo', 'marginal'), ('loo', 'shapley'), ('marginal', 'shapley')]
        for n1, n2 in pairs:
            jaccards = [tr.get(f'jaccard_{n1}_{n2}', tr.get(f'jaccard_{n2}_{n1}'))
                       for tr in data['tasks'].values()]
            jaccards = [j for j in jaccards if j is not None]
            if jaccards:
                print(f"    {n1} vs {n2}: mean={np.mean(jaccards):.3f}")


# ============================================================================
# 4. Chunk Size Sweep
# ============================================================================

def analyze_chunk_sweep():
    """Analyze chunk size sweep results."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("CHUNK SIZE SWEEP")
    print("="*70)

    for model in models:
        data = load_result("chunk_size_sweep", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model} (dim={data.get('model_dim', '?')})")

        if 'tasks' in data:
            # Get the sweep data structure
            for task, tr in list(data['tasks'].items())[:2]:
                print(f"  Task: {task}")
                if 'sweep_results' in tr:
                    for ws, sr in sorted(tr['sweep_results'].items(), key=lambda x: int(x[0])):
                        opt_ret = sr.get('optimized_retention', sr.get('opt_retention', 'N/A'))
                        rnd_ret = sr.get('random_retention', sr.get('rnd_retention', 'N/A'))
                        gap = sr.get('gap', 'N/A')
                        print(f"    win_size={ws}: opt={opt_ret}%, rnd={rnd_ret}%, gap={gap}")


# ============================================================================
# 5. Cross-Task Transfer (fast version)
# ============================================================================

def analyze_cross_task_transfer():
    """Analyze cross-task transfer results."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("CROSS-TASK TRANSFER")
    print("="*70)

    for model in models:
        data = load_result("cross_task_transfer", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model}")
        # Structure depends on the specific output format
        if 'tasks' in data:
            for task, tr in data['tasks'].items():
                if 'transfer_retention' in tr:
                    rets = list(tr['transfer_retention'].values()) if isinstance(tr['transfer_retention'], dict) else tr['transfer_retention']
                    if rets:
                        print(f"    {task}: mean_transfer_ret={np.mean(rets):.1f}%")


# ============================================================================
# 6. Retrieval Cost Analysis
# ============================================================================

def analyze_retrieval_cost():
    """Analyze retrieval system cost with compression."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("RETRIEVAL COST ANALYSIS")
    print("="*70)

    for model in models:
        data = load_result("retrieval_cost", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model} (dim={data.get('model_dim', '?')})")
        for task, tr in data.get('tasks', {}).items():
            print(f"  Task: {task}")
            for dim_str, dim_results in tr.get('methods', {}).items():
                for method, bench in dim_results.items():
                    print(f"    dim={dim_str} {method}: "
                          f"nDCG@10={bench['ndcg_at_10']:.3f} "
                          f"Recall@10={bench['recall_at_10']:.3f} "
                          f"P50={bench['latency_p50_ms']:.1f}ms "
                          f"Mem={bench['memory_mb']:.0f}MB")


# ============================================================================
# 7. OOD Robustness
# ============================================================================

def analyze_ood_robustness():
    """Analyze OOD robustness of dimension selection."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
              'roberta-large-InBedder']

    print("\n" + "="*70)
    print("OOD ROBUSTNESS")
    print("="*70)

    for model in models:
        data = load_result("ood_robustness", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model}")
        # Structure depends on output format
        for key, val in data.items():
            if key not in ['model', 'model_dim', 'total_time_s']:
                if isinstance(val, dict):
                    print(f"  {key}: {list(val.keys())[:5]}")


# ============================================================================
# 8. Non-Contiguous Selection
# ============================================================================

def analyze_non_contiguous():
    """Analyze contiguous vs non-contiguous selection."""
    models = ['gte-large-en-v1.5', 'stella_en_400M_v5',
              'roberta-large', 'roberta-large-InBedder']

    print("\n" + "="*70)
    print("NON-CONTIGUOUS SELECTION")
    print("="*70)

    for model in models:
        data = load_result("non_contiguous", model)
        if data is None:
            print(f"\n  {model}: NOT AVAILABLE")
            continue

        print(f"\n  {model}")


# ============================================================================
# 9. Training Paradigm Regression
# ============================================================================

def analyze_training_paradigm():
    """Analyze training paradigm regression."""
    print("\n" + "="*70)
    print("TRAINING PARADIGM REGRESSION")
    print("="*70)

    data = load_result("training_paradigm_regression", "")
    # Try alternate loading
    if data is None:
        path = os.path.join(RESULT_DIR, "training_paradigm_regression.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

    if data is None:
        print("  NOT AVAILABLE")
        return

    print(f"  Keys: {list(data.keys())}")
    for key in ['r_squared', 'coefficients', 'p_values', 'model_results']:
        if key in data:
            print(f"  {key}: {data[key]}")


# ============================================================================
# 10. Random Variance / Tail Risk
# ============================================================================

def analyze_random_variance():
    """Analyze random selection variance and tail risk."""
    print("\n" + "="*70)
    print("RANDOM VARIANCE / TAIL RISK")
    print("="*70)

    path = os.path.join(RESULT_DIR, "random_variance_tail_risk.json")
    if not os.path.exists(path):
        print("  NOT AVAILABLE")
        return

    with open(path) as f:
        data = json.load(f)

    print(f"  Keys: {list(data.keys())}")
    for key in ['models', 'overall_stats', 'tail_risk']:
        if key in data:
            print(f"  {key}: {json.dumps(data[key], indent=4)[:500]}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("REVISION EXPERIMENT RESULTS ANALYSIS")
    print("="*70)

    # Count available results
    prefixes = ['pca_baseline', 'gradient_saliency', 'leave_one_out',
                'chunk_size_sweep', 'cross_task_transfer', 'retrieval_cost',
                'ood_robustness', 'non_contiguous']
    standalone = ['training_paradigm_regression', 'random_variance_tail_risk']

    print("\nAvailable results:")
    for prefix in prefixes:
        files = [f for f in os.listdir(RESULT_DIR) if f.startswith(prefix) and f.endswith('.json')]
        print(f"  {prefix}: {len(files)} files")
    for name in standalone:
        path = os.path.join(RESULT_DIR, f"{name}.json")
        print(f"  {name}: {'YES' if os.path.exists(path) else 'NO'}")

    # Run analyses
    analyze_pca_baselines()
    analyze_gradient_saliency()
    analyze_loo()
    analyze_chunk_sweep()
    analyze_cross_task_transfer()
    analyze_retrieval_cost()
    analyze_ood_robustness()
    analyze_non_contiguous()
    analyze_training_paradigm()
    analyze_random_variance()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
