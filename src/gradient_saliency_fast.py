#!/usr/bin/env python3
"""
Fast gradient-based importance, activation saliency, and learned mask baselines.

Uses pre-computed embeddings for evaluation, gradient/activation methods
compute importance scores directly on embeddings.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/gradient_saliency_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 --device cuda:0
"""

import os
import argparse
import json
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from chunk_size_sweep_fast import load_task_data, evaluate_with_dims
from embedding_cache import EmbeddingCache

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


def compute_gradient_importance(embs, labels):
    """Gradient-based importance: |weight| * std(feature) from linear probe."""
    embs_np = embs if isinstance(embs, np.ndarray) else embs.cpu().numpy()
    labels_np = np.array(labels)

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(embs_np)

    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    clf.fit(embs_scaled, labels_np)

    if len(clf.classes_) <= 2:
        weights = np.abs(clf.coef_[0])
    else:
        weights = np.mean(np.abs(clf.coef_), axis=0)

    feature_std = np.std(embs_np, axis=0)
    return weights * feature_std


def compute_activation_importance(embs):
    """Activation-based importance: variance, mean abs, SNR per dimension."""
    embs_np = embs if isinstance(embs, np.ndarray) else embs.cpu().numpy()

    mean_abs = np.mean(np.abs(embs_np), axis=0)
    variance = np.var(embs_np, axis=0)
    snr = (mean_abs ** 2) / (variance + 1e-8)

    return {"mean_abs": mean_abs, "variance": variance, "snr": snr}


def train_diagonal_mask(embs, model_dim, budget, n_epochs=20, lr=0.01, device='cuda:0'):
    """Train learnable diagonal mask by maximizing masked variance."""
    mask_module = DiagonalMask(model_dim).to(device)
    optimizer = torch.optim.Adam(mask_module.parameters(), lr=lr)

    embs_t = torch.tensor(embs, dtype=torch.float32, device=device) if not isinstance(embs, torch.Tensor) else embs.to(device)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        masked = mask_module(embs_t)
        var = torch.var(masked, dim=0).mean()
        mask_probs = torch.sigmoid(mask_module.logits)
        sparsity = torch.abs(mask_probs.sum() - budget) * 0.1
        loss = -var + sparsity
        loss.backward()
        optimizer.step()

    return mask_module.get_top_k(budget)


class DiagonalMask(nn.Module):
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim) * 0.1)
        self.temperature = temperature

    def forward(self, x):
        mask = torch.sigmoid(self.logits / self.temperature)
        return x * mask.unsqueeze(0)

    def get_top_k(self, k):
        mask = torch.sigmoid(self.logits / self.temperature).detach().cpu().numpy()
        return np.argsort(mask)[::-1][:k].tolist()


def get_task_texts_and_labels(task_data):
    """Extract texts and labels from task data for gradient computation."""
    eval_fn = task_data["eval_fn"]
    if eval_fn == "classification":
        return task_data["texts"], task_data["labels"]
    elif eval_fn == "clustering":
        return task_data["texts"], task_data["labels"]
    elif eval_fn == "sts":
        return task_data["texts1"] + task_data["texts2"], list(range(len(task_data["texts1"]))) * 2
    elif eval_fn == "retrieval":
        return task_data["corpus_texts"] + task_data["query_texts"], \
               [0] * len(task_data["corpus_texts"]) + [1] * len(task_data["query_texts"])
    return None, None


def run_gradient_saliency(model_name, model_path, task_data_map, budgets, n_random,
                          device='cuda:0', cache_dir='data/embeddings_cache'):
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
    model_dim = model.encode("hello").shape[-1]
    print(f"Dimension: {model_dim}")

    cache = EmbeddingCache(cache_dir)

    results = {
        "model": model_name,
        "model_dim": model_dim,
        "methods": {},
    }

    for task_name, task_data in task_data_map.items():
        print(f"\n  Task: {task_name}")
        t0 = time.time()

        # Get embeddings
        eval_fn = task_data["eval_fn"]
        if eval_fn == "sts":
            all_texts = task_data["texts1"] + task_data["texts2"]
        elif eval_fn == "retrieval":
            all_texts = task_data["corpus_texts"] + task_data["query_texts"]
        else:
            all_texts = task_data["texts"]

        embs = cache.get_or_compute(model_name, task_name, all_texts, model, device=device)
        baseline = evaluate_with_dims(embs, task_data, list(range(model_dim)))

        if baseline == 0:
            print(f"    Baseline=0, skipping")
            continue
        print(f"    Baseline: {baseline:.2f}")

        # Compute importance scores (one-time per task)
        task_texts, task_labels = get_task_texts_and_labels(task_data)

        # Gradient importance
        grad_imp = None
        if task_labels and len(set(task_labels)) > 1 and len(task_texts) >= 50:
            grad_imp = compute_gradient_importance(embs, task_labels)
            print(f"    Gradient importance computed")

        # Activation importance
        act_imp = compute_activation_importance(embs)
        print(f"    Activation importance computed")

        task_results = {"baseline": baseline, "budgets": {}}

        for budget in budgets:
            b_results = {}

            # Random baseline
            random_scores = []
            for seed in range(n_random):
                rng = np.random.RandomState(seed + 100)
                dims = sorted(rng.choice(model_dim, size=budget, replace=False).tolist())
                rs = evaluate_with_dims(embs, task_data, dims)
                random_scores.append(rs)
            random_mean = np.mean(random_scores)
            b_results["random"] = {"score": random_mean, "retention": random_mean / baseline * 100}

            # Gradient importance
            if grad_imp is not None:
                grad_dims = sorted(np.argsort(grad_imp)[::-1][:budget].tolist())
                grad_score = evaluate_with_dims(embs, task_data, grad_dims)
                b_results["gradient"] = {"score": grad_score, "retention": grad_score / baseline * 100}

            # Activation variance
            var_imp = act_imp["variance"]
            var_dims = sorted(np.argsort(var_imp)[::-1][:budget].tolist())
            var_score = evaluate_with_dims(embs, task_data, var_dims)
            b_results["activation_variance"] = {"score": var_score, "retention": var_score / baseline * 100}

            # Learned mask
            try:
                mask_dims = train_diagonal_mask(embs, model_dim, budget, device=device)
                mask_score = evaluate_with_dims(embs, task_data, mask_dims)
                b_results["learned_mask"] = {"score": mask_score, "retention": mask_score / baseline * 100}
            except Exception as e:
                print(f"      Mask error: {e}")

            # Summary
            parts = [f"rand={random_mean/baseline*100:.1f}%"]
            if grad_imp is not None:
                parts.append(f"grad={b_results['gradient']['retention']:.1f}%")
            parts.append(f"var={b_results['activation_variance']['retention']:.1f}%")
            if "learned_mask" in b_results:
                parts.append(f"mask={b_results['learned_mask']['retention']:.1f}%")
            print(f"    budget={budget}: {' '.join(parts)}")

            task_results["budgets"][str(budget)] = b_results

        results["methods"][task_name] = task_results
        print(f"    Done in {time.time()-t0:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


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
        'roberta-large-InBedder': '/home/linkco/exa/models/inbedder-roberta-large',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
    }

    print("Loading task data...")
    task_data_map = {}
    for task_name in REPRESENTATIVE_TASKS:
        td = load_task_data(task_name)
        if td is not None:
            task_data_map[task_name] = td
    print(f"Loaded {len(task_data_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        t0 = time.time()
        results = run_gradient_saliency(
            model_name, model_path, task_data_map,
            budgets=args.budgets, n_random=args.n_random,
            device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"gradient_saliency_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
