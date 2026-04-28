#!/usr/bin/env python3
"""
D supplement: Gradient-based importance, activation saliency, and learned mask baselines.

Methods:
  1. Gradient-based importance: avg |grad * activation| on dev set
  2. Activation variance / Fisher score per dimension
  3. Learned diagonal mask (sigmoid gate, trained on dev split)

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/gradient_saliency_baseline.py --models gte-large-en-v1.5 stella_en_400M_v5
    CUDA_VISIBLE_DEVICES=2 python src/gradient_saliency_baseline.py --models roberta-large roberta-large-InBedder bge-m3
"""

import os
import sys
import argparse
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import mteb
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def get_embeddings(model, texts, device='cuda:0', batch_size=64):
    """Get full-dim embeddings in batches."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_tensor=True, device=device, show_progress_bar=False)
        all_embs.append(embs.detach().cpu().float())
    return torch.cat(all_embs, dim=0)


def compute_gradient_importance(model, texts, labels, model_dim, device='cuda:0'):
    """Gradient-based importance using a linear probe.

    Train a simple linear classifier on top of frozen embeddings,
    then compute |grad * activation| for each dimension.
    """
    # Get embeddings with grad tracking
    model.eval()
    with torch.no_grad():
        embs = get_embeddings(model, texts, device)

    embs_np = embs.numpy()
    labels_np = np.array(labels)

    # Train a simple linear classifier
    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs_np)

    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    clf.fit(embs_scaled, labels_np)

    # Importance = |weight| * std(feature)
    # This approximates gradient * activation importance
    if len(clf.classes_) <= 2:
        weights = np.abs(clf.coef_[0])
    else:
        weights = np.mean(np.abs(clf.coef_), axis=0)

    feature_std = np.std(embs_np, axis=0)
    importance = weights * feature_std

    return importance


def compute_activation_importance(model, texts, device='cuda:0'):
    """Activation-based importance: variance, mean abs, Fisher score per dim."""
    with torch.no_grad():
        embs = get_embeddings(model, texts, device)

    embs_np = embs.numpy()

    # Per-dimension statistics
    mean_abs = np.mean(np.abs(embs_np), axis=0)
    variance = np.var(embs_np, axis=0)
    std = np.std(embs_np, axis=0)

    # Fisher discriminant score (ratio of between-class to within-class variance)
    # Simplified: use signal-to-noise ratio = mean^2 / variance
    snr = (mean_abs ** 2) / (variance + 1e-8)

    return {
        "mean_abs": mean_abs.tolist(),
        "variance": variance.tolist(),
        "std": std.tolist(),
        "snr": snr.tolist(),
    }


class DiagonalMask(nn.Module):
    """Learnable diagonal mask for dimension selection."""

    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim) * 0.1)
        self.temperature = temperature

    def forward(self, x):
        mask = torch.sigmoid(self.logits / self.temperature)
        return x * mask.unsqueeze(0)

    def get_mask(self):
        return torch.sigmoid(self.logits / self.temperature).detach().cpu().numpy()

    def get_top_k(self, k):
        mask = self.get_mask()
        return np.argsort(mask)[::-1][:k].tolist()


def train_diagonal_mask(model, task, model_dim, budget, device='cuda:0',
                        n_epochs=20, lr=0.01, temperature=0.1):
    """Train a learnable diagonal mask using the MTEB task.

    Uses the task's train split to learn which dimensions to keep.
    """
    from datasets import load_dataset

    # Get task data
    try:
        task_datasets = task.load_data()
    except Exception as e:
        print(f"    Cannot load task data for mask learning: {e}")
        return None

    # Get train split texts
    if hasattr(task_datasets, 'keys'):
        splits = list(task_datasets.keys())
        train_split = 'train' if 'train' in splits else splits[0]
        dataset = task_datasets[train_split]
    else:
        dataset = task_datasets

    # Extract texts
    if isinstance(dataset, dict):
        texts = dataset.get('text', dataset.get('sentence', []))
        if isinstance(texts, dict):
            texts = texts.get('list', [])
    else:
        texts = []
        for item in dataset:
            if isinstance(item, dict):
                text = item.get('text', item.get('sentence', ''))
                if isinstance(text, list):
                    texts.extend(text)
                else:
                    texts.append(str(text))

    if len(texts) < 10:
        print(f"    Not enough texts for mask learning ({len(texts)})")
        return None

    texts = texts[:2000]  # Cap for efficiency

    # Get embeddings
    with torch.no_grad():
        embs = get_embeddings(model, texts, device)

    # Train mask
    mask_module = DiagonalMask(model_dim, temperature=temperature).to(device)
    optimizer = optim.Adam(mask_module.parameters(), lr=lr)

    # Simple objective: maximize variance of masked embeddings (encourages useful dims)
    # Plus a sparsity penalty
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        masked = mask_module(embs.to(device))

        # Variance objective: we want high variance in masked output
        var = torch.var(masked, dim=0).mean()

        # Sparsity: encourage using exactly budget dimensions
        mask_probs = torch.sigmoid(mask_module.logits / temperature)
        sparsity_loss = torch.abs(mask_probs.sum() - budget) * 0.1

        loss = -var + sparsity_loss
        loss.backward()
        optimizer.step()

    return mask_module.get_top_k(budget)


class MaskedModel:
    """Wrapper for MTEB evaluation with dimension mask."""

    def __init__(self, base_model, dim_indices=None, device='cuda:0'):
        self.base_model = base_model
        self.dim_indices = dim_indices
        self.device = device

    def encode(self, input_texts, **kwargs):
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device

        embs = self.base_model.encode(input_texts, **kwargs)
        if isinstance(embs, torch.Tensor):
            embs = embs.detach().cpu().float().numpy()
        else:
            embs = np.array(embs, dtype=np.float32)

        if self.dim_indices is not None:
            embs = embs[:, self.dim_indices]

        return embs


def evaluate_with_dims(base_model, task, dim_indices, device='cuda:0'):
    """Evaluate with selected dimensions."""
    wrapped = MaskedModel(base_model, dim_indices=dim_indices, device=device)
    evaluation = mteb.MTEB(tasks=[task])
    try:
        results = evaluation.run(wrapped, output_folder=None, verbosity=0)
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Eval error: {e}")
        return None


def evaluate_full(base_model, task, device='cuda:0'):
    """Full-dim baseline."""
    wrapped = MaskedModel(base_model, dim_indices=None, device=device)
    evaluation = mteb.MTEB(tasks=[task])
    try:
        results = evaluation.run(wrapped, output_folder=None, verbosity=0)
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Baseline error: {e}")
        return None


def load_tasks(task_names):
    all_tasks = mteb.get_tasks(tasks=task_names)
    return {t.metadata.name: t for t in all_tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5',
                                 'roberta-large', 'roberta-large-InBedder', 'bge-m3'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--n-random', type=int, default=5)
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

    print("Loading tasks...")
    task_map = load_tasks(REPRESENTATIVE_TASKS)
    print(f"Loaded {len(task_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
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
            "methods": {},
        }

        for task_name in REPRESENTATIVE_TASKS:
            if task_name not in task_map:
                continue

            task = task_map[task_name]
            print(f"\n  Task: {task_name}")

            # Baseline
            baseline = evaluate_full(base_model, task, args.device)
            if baseline is None:
                continue
            print(f"    Baseline: {baseline:.2f}")

            task_results = {"baseline": baseline, "budgets": {}}

            for budget in args.budgets:
                print(f"    Budget: {budget}")
                b_results = {}

                # 1. Random selection (baseline)
                random_scores = []
                for seed in range(args.n_random):
                    rng = np.random.RandomState(seed + 100)
                    dims = sorted(rng.choice(model_dim, size=budget, replace=False).tolist())
                    score = evaluate_with_dims(base_model, task, dims, args.device)
                    if score:
                        random_scores.append(score)
                random_mean = np.mean(random_scores) if random_scores else None
                b_results["random"] = {
                    "score": random_mean,
                    "retention": random_mean / baseline * 100 if random_mean else None,
                }
                print(f"      Random: {random_mean / baseline * 100:.1f}%" if random_mean else "      Random: FAILED")

                # 2. Gradient-based importance
                print("      Gradient importance...")
                try:
                    # Get some labeled data for gradient computation
                    task_data = task.load_data()
                    if hasattr(task_data, 'keys'):
                        splits = list(task_data.keys())
                        split_name = 'train' if 'train' in splits else splits[0]
                        ds = task_data[split_name]
                    else:
                        ds = task_data

                    texts_for_grad = []
                    labels_for_grad = []
                    for item in (ds if isinstance(ds, list) else list(ds)):
                        if isinstance(item, dict):
                            text = item.get('text', item.get('sentence', ''))
                            label = item.get('label', 0)
                            if isinstance(text, list):
                                texts_for_grad.extend(text)
                                labels_for_grad.extend([label] * len(text))
                            else:
                                texts_for_grad.append(str(text))
                                labels_for_grad.append(label)

                    if len(texts_for_grad) >= 50:
                        grad_imp = compute_gradient_importance(
                            base_model, texts_for_grad[:2000], labels_for_grad[:2000],
                            model_dim, args.device)
                        grad_dims = sorted(np.argsort(grad_imp)[::-1][:budget].tolist())
                        grad_score = evaluate_with_dims(base_model, task, grad_dims, args.device)
                        b_results["gradient"] = {
                            "score": grad_score,
                            "retention": grad_score / baseline * 100 if grad_score else None,
                        }
                        print(f"      Gradient: {grad_score / baseline * 100:.1f}%" if grad_score else "      Gradient: FAILED")
                    else:
                        print(f"      Gradient: insufficient data ({len(texts_for_grad)})")
                except Exception as e:
                    print(f"      Gradient: error ({e})")

                # 3. Activation variance importance
                print("      Activation variance...")
                try:
                    if len(texts_for_grad) >= 20:
                        act_imp = compute_activation_importance(
                            base_model, texts_for_grad[:1000], args.device)
                        # Use variance as importance
                        var_imp = np.array(act_imp["variance"])
                        var_dims = sorted(np.argsort(var_imp)[::-1][:budget].tolist())
                        var_score = evaluate_with_dims(base_model, task, var_dims, args.device)
                        b_results["activation_variance"] = {
                            "score": var_score,
                            "retention": var_score / baseline * 100 if var_score else None,
                        }
                        print(f"      Act-Var: {var_score / baseline * 100:.1f}%" if var_score else "      Act-Var: FAILED")
                    else:
                        print(f"      Act-Var: insufficient data")
                except Exception as e:
                    print(f"      Act-Var: error ({e})")

                # 4. Learned diagonal mask
                print("      Learned mask...")
                try:
                    mask_dims = train_diagonal_mask(
                        base_model, task, model_dim, budget, args.device)
                    if mask_dims:
                        mask_score = evaluate_with_dims(base_model, task, mask_dims, args.device)
                        b_results["learned_mask"] = {
                            "score": mask_score,
                            "retention": mask_score / baseline * 100 if mask_score else None,
                        }
                        print(f"      Mask: {mask_score / baseline * 100:.1f}%" if mask_score else "      Mask: FAILED")
                except Exception as e:
                    print(f"      Mask: error ({e})")

                task_results["budgets"][str(budget)] = b_results

            results["methods"][task_name] = task_results

        # Cleanup
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"gradient_saliency_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
