#!/usr/bin/env python3
"""
Supervised learnable mask with proper train/dev/test split.

Trains a differentiable diagonal mask (hard concrete / sigmoid gate) using
task-specific loss on train set, selects mask on dev set, evaluates on test set.
Uses EmbeddingCache — encode once per model+task.

Methods compared:
  - Random selection (5 seeds)
  - Magnitude-based selection
  - Gradient importance (from linear probe)
  - Unsupervised mask (variance maximization)
  - Supervised mask (task-specific loss, train/dev/test split)

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/learnable_mask_supervised.py \
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
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from chunk_size_sweep_fast import (
    load_task_data, evaluate_with_dims,
    compute_classification_accuracy, compute_sts_spearman,
    compute_clustering_vmeasure, compute_retrieval_ndcg,
)
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


class HardConcreteMask(nn.Module):
    """Hard concrete distribution for L0 regularization (Louizos et al. 2018)."""
    def __init__(self, dim, temperature=0.5, init_mean=0.5):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim) * 0.01 + self._inverse_sigmoid(init_mean))
        self.temperature = temperature
        # Stretch parameters for hard concrete
        self.l, self.r = -0.1, 1.1

    def _inverse_sigmoid(self, x):
        return torch.log(torch.tensor(x) / (1 - torch.tensor(x)))

    def forward(self, x):
        if self.training:
            u = torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)
            s = torch.log(u) - torch.log(1 - u) + self.logits
            s = torch.sigmoid(s / self.temperature)
            # Stretch
            s = s * (self.r - self.l) + self.l
            s = s.clamp(0, 1)
        else:
            s = torch.sigmoid(self.logits).clamp(0, 1)
        return x * s.unsqueeze(0)

    def get_top_k(self, k):
        probs = torch.sigmoid(self.logits).detach().cpu().numpy()
        return np.argsort(probs)[::-1][:k].tolist()


class SigmoidMask(nn.Module):
    """Simple sigmoid mask baseline."""
    def __init__(self, dim, temperature=0.1):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim) * 0.01)
        self.temperature = temperature

    def forward(self, x):
        mask = torch.sigmoid(self.logits / self.temperature)
        return x * mask.unsqueeze(0)

    def get_top_k(self, k):
        mask = torch.sigmoid(self.logits / self.temperature).detach().cpu().numpy()
        return np.argsort(mask)[::-1][:k].tolist()


def make_task_loss(eval_fn):
    """Create a differentiable loss function for the task."""
    if eval_fn == "classification":
        def loss_fn(masked_embs, labels):
            # Cosine similarity-based loss
            masked_embs = masked_embs / (masked_embs.norm(dim=1, keepdim=True) + 1e-8)
            sim = masked_embs @ masked_embs.T
            labels_t = torch.tensor(labels, device=sim.device)
            same_class = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1)).float()
            # Contrastive-style loss
            pos = (sim * same_class).sum(dim=1) / (same_class.sum(dim=1) + 1e-8)
            neg = (sim * (1 - same_class)).sum(dim=1) / ((1 - same_class).sum(dim=1) + 1e-8)
            return -torch.mean(pos - neg)
        return loss_fn
    elif eval_fn == "sts":
        def loss_fn(masked_embs, gold_scores):
            # Negative Spearman proxy: maximize correlation
            n = len(gold_scores)
            half = n
            embs1 = masked_embs[:n]
            embs2 = masked_embs[n:2*n]
            cos = torch.nn.functional.cosine_similarity(embs1, embs2)
            # Pearson as differentiable proxy for Spearman
            cos_centered = cos - cos.mean()
            gs = torch.tensor(gold_scores, device=cos.device, dtype=torch.float32)
            gs_centered = gs - gs.mean()
            corr = (cos_centered * gs_centered).sum() / (
                cos_centered.norm() * gs_centered.norm() + 1e-8)
            return -corr
        return loss_fn
    elif eval_fn == "clustering":
        def loss_fn(masked_embs, labels):
            # Maximize inter-class distance, minimize intra-class distance
            labels_t = torch.tensor(labels, device=masked_embs.device)
            unique_labels = labels_t.unique()
            centers = []
            for lbl in unique_labels:
                mask = labels_t == lbl
                centers.append(masked_embs[mask].mean(dim=0))
            centers = torch.stack(centers)
            # Spread centers apart
            sim = torch.nn.functional.cosine_similarity(
                centers.unsqueeze(1), centers.unsqueeze(0), dim=2)
            # Penalize similarity between different clusters
            diff_mask = 1 - torch.eye(len(centers), device=sim.device)
            return (sim * diff_mask).sum()
        return loss_fn
    elif eval_fn == "retrieval":
        def loss_fn(masked_embs, n_corpus, relevant_docs):
            # Simple contrastive: pull relevant pairs together
            corpus_embs = masked_embs[:n_corpus]
            query_embs = masked_embs[n_corpus:]
            loss = torch.tensor(0.0, device=masked_embs.device)
            count = 0
            for qi, docs in relevant_docs.items():
                if qi >= len(query_embs):
                    continue
                q = query_embs[qi]
                for di in docs:
                    if di < len(corpus_embs):
                        sim = torch.nn.functional.cosine_similarity(
                            q.unsqueeze(0), corpus_embs[di].unsqueeze(0))
                        loss = loss - sim.mean()
                        count += 1
            return loss / max(count, 1)
        return loss_fn
    return None


def split_data(embs, task_data, train_ratio=0.5, dev_ratio=0.25):
    """Split embeddings into train/dev/test for supervised mask training."""
    eval_fn = task_data["eval_fn"]
    n = len(embs)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]
    test_idx = idx[n_train + n_dev:]

    if eval_fn in ("classification", "clustering"):
        labels = np.array(task_data["labels"])
        return {
            "train": {"embs": embs[train_idx], "labels": labels[train_idx].tolist()},
            "dev": {"embs": embs[dev_idx], "labels": labels[dev_idx].tolist()},
            "test": {"embs": embs[test_idx], "labels": labels[test_idx].tolist()},
        }
    elif eval_fn == "sts":
        n_pairs = len(task_data["gold_scores"])
        half = n // 2
        # Split by pair index
        pair_idx = rng.permutation(n_pairs)
        n_train_p = int(n_pairs * train_ratio)
        n_dev_p = int(n_pairs * dev_ratio)
        train_pairs = pair_idx[:n_train_p]
        dev_pairs = pair_idx[n_train_p:n_train_p + n_dev_p]
        test_pairs = pair_idx[n_train_p + n_dev_p:]

        def get_pair_data(pairs):
            embs1 = embs[pairs]
            embs2 = embs[half + pairs]
            pair_embs = np.vstack([embs1, embs2])
            gs = [task_data["gold_scores"][i] for i in pairs]
            return {"embs": pair_embs, "gold_scores": gs,
                    "n_pairs": len(pairs)}

        return {
            "train": get_pair_data(train_pairs),
            "dev": get_pair_data(dev_pairs),
            "test": get_pair_data(test_pairs),
        }
    elif eval_fn == "retrieval":
        n_corpus = len(task_data["corpus_texts"])
        # Keep all corpus, split queries
        n_queries = n - n_corpus
        query_idx = rng.permutation(n_queries)
        n_train_q = int(n_queries * train_ratio)
        n_dev_q = int(n_queries * dev_ratio)

        train_qi = query_idx[:n_train_q]
        dev_qi = query_idx[n_train_q:n_train_q + n_dev_q]
        test_qi = query_idx[n_train_q + n_dev_q:]

        def get_retr_data(qi_arr):
            qi = qi_arr.tolist()
            new_rel = {}
            for i_new, qi_orig in enumerate(qi):
                if qi_orig in task_data["relevant_docs"]:
                    new_rel[i_new] = task_data["relevant_docs"][qi_orig]
            return {
                "embs": np.vstack([embs[:n_corpus], embs[n_corpus + qi_arr]]),
                "n_corpus": n_corpus, "relevant_docs": new_rel,
            }

        return {
            "train": get_retr_data(train_qi),
            "dev": get_retr_data(dev_qi),
            "test": get_retr_data(test_qi),
        }
    return None


def train_supervised_mask(train_data, dev_data, model_dim, budget, eval_fn,
                          mask_type='hard_concrete', n_epochs=30, lr=0.05,
                          device='cuda:0'):
    """Train a supervised mask on train set, validate on dev set."""
    if mask_type == 'hard_concrete':
        mask_module = HardConcreteMask(model_dim).to(device)
    else:
        mask_module = SigmoidMask(model_dim).to(device)

    optimizer = torch.optim.Adam(mask_module.parameters(), lr=lr)

    train_embs = torch.tensor(train_data["embs"], dtype=torch.float32, device=device)
    dev_embs = torch.tensor(dev_data["embs"], dtype=torch.float32, device=device)

    loss_fn = make_task_loss(eval_fn)
    if loss_fn is None:
        return None

    # Build loss args
    def get_loss_args(data, embs_t):
        if eval_fn in ("classification", "clustering"):
            return [embs_t, data["labels"]]
        elif eval_fn == "sts":
            return [embs_t, data["gold_scores"]]
        elif eval_fn == "retrieval":
            return [embs_t, data["n_corpus"], data["relevant_docs"]]
        return [embs_t]

    best_dev_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        # Train
        mask_module.train()
        optimizer.zero_grad()
        masked = mask_module(train_embs)
        args = get_loss_args(train_data, masked)
        loss = loss_fn(*args)
        # Sparsity penalty
        mask_probs = torch.sigmoid(mask_module.logits)
        sparsity = torch.abs(mask_probs.sum() - budget) * 0.01
        total_loss = loss + sparsity
        total_loss.backward()
        optimizer.step()

        # Dev eval
        mask_module.eval()
        with torch.no_grad():
            masked_dev = mask_module(dev_embs)
            dev_args = get_loss_args(dev_data, masked_dev)
            dev_loss = loss_fn(*dev_args).item()

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in mask_module.state_dict().items()}

    if best_state is not None:
        mask_module.load_state_dict(best_state)

    return mask_module.get_top_k(budget)


def run_supervised_mask(model_name, model_path, task_data_map, budgets, n_random,
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
        "model": model_name, "model_dim": model_dim,
        "methods": {},
    }

    for task_name, task_data in task_data_map.items():
        print(f"\n  Task: {task_name}")
        t0 = time.time()

        eval_fn = task_data["eval_fn"]
        if eval_fn == "sts":
            all_texts = task_data["texts1"] + task_data["texts2"]
        elif eval_fn == "retrieval":
            all_texts = task_data["corpus_texts"] + task_data["query_texts"]
        else:
            all_texts = task_data["texts"]

        embs = cache.get_or_compute(model_name, task_name, all_texts, model, device=device)

        # Test set baseline
        baseline = evaluate_with_dims(embs, task_data, list(range(model_dim)))
        if baseline == 0:
            print(f"    Baseline=0, skipping")
            continue
        print(f"    Baseline: {baseline:.2f}")

        # Split data
        splits = split_data(embs, task_data)
        if splits is None:
            print(f"    Cannot split {eval_fn} data, skipping")
            continue

        task_results = {"baseline": baseline, "budgets": {}}

        for budget in budgets:
            b_results = {}

            # 1. Random (5 seeds)
            random_scores = []
            for seed in range(n_random):
                rng = np.random.RandomState(seed + 100)
                dims = sorted(rng.choice(model_dim, size=budget, replace=False).tolist())
                rs = evaluate_with_dims(embs, task_data, dims)
                random_scores.append(rs)
            random_mean = np.mean(random_scores)
            b_results["random"] = {"score": float(random_mean),
                                   "retention": float(random_mean / baseline * 100)}

            # 2. Magnitude
            mag = np.mean(np.abs(embs), axis=0)
            mag_dims = sorted(np.argsort(mag)[::-1][:budget].tolist())
            mag_score = evaluate_with_dims(embs, task_data, mag_dims)
            b_results["magnitude"] = {"score": float(mag_score),
                                      "retention": float(mag_score / baseline * 100)}

            # 3. Gradient importance (from linear probe on train set)
            train_embs_np = splits["train"]["embs"]
            if eval_fn in ("classification", "clustering"):
                train_labels = splits["train"]["labels"]
                if len(set(train_labels)) > 1:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(train_embs_np)
                    clf = LogisticRegression(max_iter=1000, multi_class='auto')
                    clf.fit(scaled, train_labels)
                    if len(clf.classes_) <= 2:
                        grad_imp = np.abs(clf.coef_[0]) * np.std(train_embs_np, axis=0)
                    else:
                        grad_imp = np.mean(np.abs(clf.coef_), axis=0) * np.std(train_embs_np, axis=0)
                    grad_dims = sorted(np.argsort(grad_imp)[::-1][:budget].tolist())
                    grad_score = evaluate_with_dims(embs, task_data, grad_dims)
                    b_results["gradient"] = {"score": float(grad_score),
                                             "retention": float(grad_score / baseline * 100)}

            # 4. Supervised mask — hard concrete
            try:
                hc_dims = train_supervised_mask(
                    splits["train"], splits["dev"], model_dim, budget,
                    eval_fn, mask_type='hard_concrete', device=device)
                if hc_dims:
                    hc_score = evaluate_with_dims(embs, task_data, hc_dims)
                    b_results["hard_concrete_mask"] = {
                        "score": float(hc_score),
                        "retention": float(hc_score / baseline * 100)}
            except Exception as e:
                print(f"      HC mask error: {e}")

            # 5. Supervised mask — sigmoid
            try:
                sig_dims = train_supervised_mask(
                    splits["train"], splits["dev"], model_dim, budget,
                    eval_fn, mask_type='sigmoid', device=device)
                if sig_dims:
                    sig_score = evaluate_with_dims(embs, task_data, sig_dims)
                    b_results["sigmoid_mask"] = {
                        "score": float(sig_score),
                        "retention": float(sig_score / baseline * 100)}
            except Exception as e:
                print(f"      Sigmoid mask error: {e}")

            # Summary
            parts = [f"rand={random_mean/baseline*100:.1f}%"]
            for key in ["magnitude", "gradient", "hard_concrete_mask", "sigmoid_mask"]:
                if key in b_results:
                    parts.append(f"{key[:4]}={b_results[key]['retention']:.1f}%")
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
                                 'roberta-large', 'roberta-large-InBedder'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--n-random', type=int, default=5)
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
        'roberta-large-InBedder': '/home/linkco/exa/models/inbedder-roberta-large',
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
        results = run_supervised_mask(
            model_name, model_path, task_data_map,
            budgets=args.budgets, n_random=args.n_random,
            device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"learnable_mask_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
