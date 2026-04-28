#!/usr/bin/env python3
"""
PCA and Random Projection baselines for embedding compression.

Compares coordinate selection (random/oracle) vs transformation-based compression (PCA, RP).

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/pca_baseline_experiment.py --models gte-large-en-v1.5 stella_en_400M_v5
    CUDA_VISIBLE_DEVICES=2 python src/pca_baseline_experiment.py --models roberta-large roberta-large-InBedder bge-m3
"""

import os
import sys
import argparse
import json
import gc
import numpy as np
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA
import mteb
from sentence_transformers import SentenceTransformer

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

REPRESENTATIVE_TASKS = [
    'ImdbClassification', 'Banking77Classification',
    'TwentyNewsgroupsClustering', 'MedrxivClusteringS2S',
    'NFCorpus', 'ArguAna',
    'SciDocsRR', 'StackOverflowDupQuestions',
    'STSBenchmark', 'BIOSSES',
    'SprintDuplicateQuestions', 'TwitterURLCorpus',
    'SummEval', 'SciFact',
]


class TransformModel:
    """Wrapper that applies dimensionality reduction transform before evaluation."""

    def __init__(self, base_model, transform_func, device='cuda:0'):
        self.base_model = base_model
        self.transform_func = transform_func
        self.device = device
        self.cache = {}
        self.model_card_data = {
            "model_name": "TransformModel",
            "description": "Model with dimensionality transform",
        }

    def encode(self, input_texts, **kwargs):
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device

        embs = self.base_model.encode(input_texts, **kwargs)
        if isinstance(embs, torch.Tensor):
            embs_np = embs.cpu().float().numpy()
        else:
            embs_np = np.array(embs, dtype=np.float32)

        transformed = self.transform_func(embs_np)
        return transformed


def compute_pca_transform(train_embeddings, target_dim):
    """Fit PCA on training embeddings and return transform function."""
    pca = PCA(n_components=target_dim)
    pca.fit(train_embeddings)
    print(f"  PCA explained variance ratio (top 5): {pca.explained_variance_ratio_[:5]}")
    print(f"  PCA total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    def transform(embs):
        return pca.transform(embs)

    return transform, pca


def compute_random_projection(input_dim, target_dim, seed=42):
    """Create random projection matrix (Gaussian)."""
    rng = np.random.RandomState(seed)
    proj = rng.randn(input_dim, target_dim) / np.sqrt(target_dim)

    def transform(embs):
        return embs @ proj

    return transform


def evaluate_task_with_model(model, task_name, task, split='test'):
    """Evaluate a model on a task, return main_score * 100."""
    try:
        evaluation = mteb.MTEB(tasks=[task])
        results = evaluation.run(model, output_folder=None, verbosity=0)
        return results[0].scores[split][0]['main_score'] * 100
    except Exception as e:
        print(f"  Error evaluating {task_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5',
                                 'roberta-large', 'roberta-large-InBedder', 'bge-m3'])
    parser.add_argument('--target-dims', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
        'roberta-large-InBedder': '/home/linkco/exa/models/roberta-large-InBedder',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
    }

    # Load tasks
    print("Loading tasks...")
    all_tasks = mteb.get_tasks(tasks=REPRESENTATIVE_TASKS)
    task_map = {t.metadata.name: t for t in all_tasks}

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}: path not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        base_model = SentenceTransformer(model_path, trust_remote_code=True).to(args.device)
        model_dim = base_model.encode("hello").shape[-1]
        print(f"Dimension: {model_dim}")

        results = {
            "model": model_name,
            "model_dim": model_dim,
            "target_dims": args.target_dims,
            "methods": {},
        }

        # Generate sample embeddings for PCA fitting
        print("Generating sample embeddings for PCA...")
        sample_texts = [
            "This is a sample sentence for fitting PCA.",
            "Another example text to capture embedding distribution.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models process high-dimensional vectors.",
            "Text embeddings capture semantic meaning of sentences.",
        ] * 20  # 100 samples
        sample_embs = base_model.encode(sample_texts, convert_to_tensor=False)
        sample_embs = np.array(sample_embs, dtype=np.float32)

        for task_name in REPRESENTATIVE_TASKS:
            if task_name not in task_map:
                continue

            task = task_map[task_name]
            print(f"\n  Task: {task_name}")

            task_results = {}

            # 1. Full-dim baseline
            baseline = evaluate_task_with_model(base_model, task_name, task)
            if baseline is None:
                continue
            task_results["baseline"] = baseline
            print(f"    Baseline: {baseline:.2f}")

            for target_dim in args.target_dims:
                dim_results = {}

                # 2. PCA
                transform_fn, pca = compute_pca_transform(sample_embs, target_dim)
                pca_model = TransformModel(base_model, transform_fn, device=args.device)
                pca_score = evaluate_task_with_model(pca_model, task_name, task)
                if pca_score:
                    dim_results["pca_retention"] = pca_score / baseline * 100
                    dim_results["pca_score"] = pca_score
                    print(f"    PCA-{target_dim}: {pca_score/baseline*100:.1f}%")

                # 3. Random Projection (average of 5 seeds)
                rp_scores = []
                for seed in range(5):
                    rp_transform = compute_random_projection(model_dim, target_dim, seed=seed + 42)
                    rp_model = TransformModel(base_model, rp_transform, device=args.device)
                    rp_score = evaluate_task_with_model(rp_model, task_name, task)
                    if rp_score:
                        rp_scores.append(rp_score)
                if rp_scores:
                    rp_mean = np.mean(rp_scores)
                    dim_results["rp_retention"] = rp_mean / baseline * 100
                    dim_results["rp_mean_score"] = rp_mean
                    print(f"    RP-{target_dim}: {rp_mean/baseline*100:.1f}% (mean of {len(rp_scores)})")

                # 4. Random coordinate selection (from existing data or re-compute)
                # Load existing train_test_split data if available
                existing_file = os.path.join(args.output_dir, f"train_test_split_{model_name}.json")
                if os.path.exists(existing_file):
                    with open(existing_file) as f:
                        existing_data = json.load(f)
                    task_data = existing_data.get("tasks", {}).get(task_name, {})
                    # Get budget closest to target_dim
                    budget_data = task_data.get("budgets", {}).get(str(target_dim), task_data.get("budgets", {}).get(str(target_dim * 2 // 2), {}))
                    if budget_data:
                        random_ret = budget_data.get("random_mean_retention", budget_data.get("random_retention"))
                        oracle_ret = budget_data.get("oracle_retention")
                        if random_ret:
                            dim_results["random_coord_retention"] = random_ret
                            print(f"    Random-coord-{target_dim}: {random_ret:.1f}% (from existing data)")
                        if oracle_ret:
                            dim_results["oracle_coord_retention"] = oracle_ret
                            print(f"    Oracle-coord-{target_dim}: {oracle_ret:.1f}% (from existing data)")

                task_results[f"dim_{target_dim}"] = dim_results

            results["methods"][task_name] = task_results

        # Cleanup
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"pca_baseline_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
