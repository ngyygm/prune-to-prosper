#!/usr/env python3
"""
H. Training-paradigm controlled analysis.

Runs regression: gap ~ contrastive_training + embedding_native + dim + model_family
Uses existing experiment data (no GPU needed).

Usage:
    python src/training_paradigm_regression.py
"""

import os
import json
import argparse
import numpy as np
from scipy import stats
from collections import defaultdict

DATA_DIR = 'data/experiment_results'

# Model metadata
MODEL_INFO = {
    'gte-large-en-v1.5': {
        'family': 'GTE',
        'architecture': 'BERT',
        'dim': 1024,
        'contrastive': 1,
        'embedding_native': 1,
        'pooling': 'mean',
        'params_M': 335,
    },
    'stella_en_400M_v5': {
        'family': 'Stella',
        'architecture': 'BERT',
        'dim': 1024,
        'contrastive': 1,
        'embedding_native': 1,
        'pooling': 'mean',
        'params_M': 335,
    },
    'bge-m3': {
        'family': 'BGE',
        'architecture': 'XLM-R',
        'dim': 1024,
        'contrastive': 1,
        'embedding_native': 1,
        'pooling': 'cls',
        'params_M': 568,
    },
    'mxbai-embed-large-v1': {
        'family': 'MxBai',
        'architecture': 'BERT',
        'dim': 1024,
        'contrastive': 1,
        'embedding_native': 1,
        'pooling': 'mean',
        'params_M': 335,
    },
    'instructor-large': {
        'family': 'Instructor',
        'architecture': 'BERT',
        'dim': 768,
        'contrastive': 1,  # instruction-tuned with contrastive-like objectives
        'embedding_native': 1,
        'pooling': 'mean',
        'params_M': 335,
    },
    'Qwen3-Embedding-0.6B': {
        'family': 'Qwen',
        'architecture': 'Qwen2',
        'dim': 1024,
        'contrastive': 1,
        'embedding_native': 1,
        'pooling': 'last',
        'params_M': 600,
    },
    'roberta-large': {
        'family': 'RoBERTa',
        'architecture': 'RoBERTa',
        'dim': 1024,
        'contrastive': 0,
        'embedding_native': 0,
        'pooling': 'mean',
        'params_M': 355,
    },
    'bart-base': {
        'family': 'BART',
        'architecture': 'BART',
        'dim': 768,
        'contrastive': 0,
        'embedding_native': 0,
        'pooling': 'mean',
        'params_M': 139,
    },
    'inbedder-roberta-large': {
        'family': 'RoBERTa',
        'architecture': 'RoBERTa',
        'dim': 1024,
        'contrastive': 1,  # fine-tuned with contrastive training
        'embedding_native': 0.5,  # adapted but not natively embedding model
        'pooling': 'mean',
        'params_M': 355,
    },
}

TASK_CATEGORIES = {
    "Clustering": ['BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering'],
    "PairClassification": ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'],
    "Reranking": ['AskUbuntuDupQuestions', 'SciDocsRR', 'StackOverflowDupQuestions'],
    "STS": ['BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STSBenchmark'],
    "Summarization": ['SummEval'],
    "Retrieval": ['ArguAna', 'CQADupstackEnglishRetrieval', 'NFCorpus', 'SCIDOCS', 'SciFact'],
}

TASK_TYPE_MAP = {}
for cat, tasks in TASK_CATEGORIES.items():
    for t in tasks:
        TASK_TYPE_MAP[t] = cat


def collect_data():
    """Collect per-task oracle-random gap data across all models."""
    rows = []

    for model_name, info in MODEL_INFO.items():
        filepath = os.path.join(DATA_DIR, f'train_test_split_{model_name}.json')
        if not os.path.exists(filepath):
            print(f"  Missing: {model_name}")
            continue

        with open(filepath) as f:
            data = json.load(f)

        for task_name, task_data in data.get('tasks', {}).items():
            if 'STS17' in task_name:
                continue

            budgets = task_data.get('budgets', {})
            for budget_str, budget_data in budgets.items():
                budget = int(budget_str)
                oracle_adv = budget_data.get('oracle_advantage')

                if oracle_adv is not None:
                    task_type = TASK_TYPE_MAP.get(task_name, 'Unknown')
                    rows.append({
                        'model': model_name,
                        'task': task_name,
                        'task_type': task_type,
                        'budget': budget,
                        'gap': oracle_adv,
                        'contrastive': info['contrastive'],
                        'embedding_native': info['embedding_native'],
                        'dim': info['dim'],
                        'family': info['family'],
                        'params_M': info['params_M'],
                    })

    return rows


def run_regression(rows):
    """Run OLS regression on collected data."""
    import pandas as pd

    df = pd.DataFrame(rows)

    results = {"n_observations": len(df), "n_models": df['model'].nunique()}

    # 1. Simple correlation: contrastive training vs gap
    contrastive_gaps = df[df['contrastive'] == 1]['gap']
    non_contrastive_gaps = df[df['contrastive'] == 0]['gap']
    t_stat, p_val = stats.ttest_ind(contrastive_gaps, non_contrastive_gaps)
    cohen_d = (contrastive_gaps.mean() - non_contrastive_gaps.mean()) / np.sqrt(
        (contrastive_gaps.var() * (len(contrastive_gaps)-1) +
         non_contrastive_gaps.var() * (len(non_contrastive_gaps)-1)) /
        (len(contrastive_gaps) + len(non_contrastive_gaps) - 2)
    )

    results["contrastive_vs_gap"] = {
        "contrastive_mean_gap": float(contrastive_gaps.mean()),
        "non_contrastive_mean_gap": float(non_contrastive_gaps.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohen_d": float(cohen_d),
    }
    print(f"\nContrastive vs gap:")
    print(f"  Contrastive mean gap: {contrastive_gaps.mean():.3f}")
    print(f"  Non-contrastive mean gap: {non_contrastive_gaps.mean():.3f}")
    print(f"  t={t_stat:.2f}, p={p_val:.6f}, d={cohen_d:.2f}")

    # 2. OLS regression at budget=256
    df_256 = df[df['budget'] == 256].copy()

    if len(df_256) > 20:
        # Encode categorical variables
        df_256['is_retrieval'] = (df_256['task_type'] == 'Retrieval').astype(int)
        df_256['is_sts'] = (df_256['task_type'] == 'STS').astype(int)

        # Simple regression: gap ~ contrastive + params_M
        # Note: embedding_native dropped due to collinearity with contrastive (r>.95)
        from numpy.linalg import lstsq

        X_vars = ['contrastive', 'params_M']
        X = df_256[X_vars].values.astype(float)
        X = np.column_stack([np.ones(len(X)), X])  # add intercept
        y = df_256['gap'].values.astype(float)

        # OLS
        beta, residuals, rank, sv = lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Standard errors
        n, k = X.shape
        mse = ss_res / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se_beta = np.sqrt(var_beta)
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        results["regression_budget_256"] = {
            "r_squared": float(r_squared),
            "n": int(n),
            "coefficients": {
                "intercept": {"beta": float(beta[0]), "se": float(se_beta[0]),
                              "t": float(t_stats[0]), "p": float(p_values[0])},
                "contrastive": {"beta": float(beta[1]), "se": float(se_beta[1]),
                                "t": float(t_stats[1]), "p": float(p_values[1])},
                "params_M": {"beta": float(beta[2]), "se": float(se_beta[2]),
                              "t": float(t_stats[2]), "p": float(p_values[2])},
            }
        }

        print(f"\nRegression (budget=256): R²={r_squared:.3f}")
        for i, name in enumerate(['intercept'] + X_vars):
            sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
            print(f"  {name}: beta={beta[i]:.3f} (SE={se_beta[i]:.3f}, t={t_stats[i]:.2f}, p={p_values[i]:.4f}) {sig}")

    # 3. Per-task-type analysis
    print(f"\nPer-task-type gap:")
    task_type_results = {}
    for tt in sorted(df['task_type'].unique()):
        tt_data = df[(df['task_type'] == tt) & (df['budget'] == 256)]
        if len(tt_data) < 5:
            continue
        cont = tt_data[tt_data['contrastive'] == 1]['gap']
        noncont = tt_data[tt_data['contrastive'] == 0]['gap']

        result = {
            "n_contrastive": len(cont),
            "n_non_contrastive": len(noncont),
            "contrastive_mean": float(cont.mean()),
            "non_contrastive_mean": float(noncont.mean()) if len(noncont) > 0 else None,
        }
        task_type_results[tt] = result
        print(f"  {tt}: contrastive={cont.mean():.2f} ({len(cont)}), "
              f"non-contrastive={noncont.mean():.2f} ({len(noncont)})" if len(noncont) > 0
              else f"  {tt}: contrastive={cont.mean():.2f} ({len(cont)}), no non-contrastive")

    results["per_task_type"] = task_type_results

    # 4. Correlation matrix of predictors with gap
    print(f"\nCorrelations with gap (budget=256):")
    df_256_numeric = df_256[['gap', 'contrastive', 'embedding_native', 'dim', 'params_M']].copy()
    corr_matrix = df_256_numeric.corr()
    results["correlation_matrix"] = corr_matrix.to_dict()
    for col in ['contrastive', 'embedding_native', 'dim', 'params_M']:
        rho, p = stats.spearmanr(df_256[col], df_256['gap'])
        print(f"  {col}: rho={rho:.3f}, p={p:.4f}")

    # 5. Effective rank / singular value spectrum analysis
    try:
        entropy_file = os.path.join(DATA_DIR, 'all_models_entropy.json')
        if os.path.exists(entropy_file):
            with open(entropy_file) as f:
                entropy_data = json.load(f)
            results["entropy_analysis"] = {}

            for model_name in entropy_data:
                model_entropy = entropy_data[model_name]
                if isinstance(model_entropy, dict):
                    mean_ent = model_entropy.get('mean_entropy')
                else:
                    mean_ent = model_entropy

                info = MODEL_INFO.get(model_name, {})
                if mean_ent and info:
                    results["entropy_analysis"][model_name] = {
                        "entropy": mean_ent,
                        "contrastive": info['contrastive'],
                    }

            # Correlate entropy with gap
            entropies = []
            mean_gaps = []
            for model_name in MODEL_INFO:
                if model_name in results.get('entropy_analysis', {}):
                    ent = results['entropy_analysis'][model_name]['entropy']
                    # Mean gap for this model at budget=256
                    model_df = df_256[df_256['model'] == model_name]
                    if len(model_df) > 0:
                        entropies.append(ent)
                        mean_gaps.append(model_df['gap'].mean())

            if len(entropies) >= 3:
                rho_ent, p_ent = stats.spearmanr(entropies, mean_gaps)
                results["entropy_gap_correlation"] = {
                    "rho": float(rho_ent),
                    "p": float(p_ent),
                    "n_models": len(entropies),
                }
                print(f"\nEntropy-gap correlation: rho={rho_ent:.3f}, p={p_ent:.4f}")
    except Exception as e:
        print(f"Entropy analysis error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/experiment_results/training_paradigm_regression.json')
    args = parser.parse_args()

    print("Collecting experiment data...")
    rows = collect_data()
    print(f"Collected {len(rows)} observations from {len(set(r['model'] for r in rows))} models")

    print("\nRunning regression analysis...")
    results = run_regression(rows)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
