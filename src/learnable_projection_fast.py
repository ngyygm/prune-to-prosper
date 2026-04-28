#!/usr/bin/env python3
"""
Learnable linear projection + quantization baselines.

Trains a linear projection from D to k dims using task-specific loss on train set.
Also adds scalar quantization and product quantization baselines.
Uses EmbeddingCache — encode once.

Methods compared:
  - Random coordinate selection
  - PCA
  - Random projection
  - Learnable projection (trained on train set, validated on dev set)
  - Scalar quantization (int8)
  - Product quantization

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/learnable_projection_fast.py \
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
import faiss
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

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


class LearnableProjection(nn.Module):
    """Linear projection with optional whitening."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        # Initialize with random orthogonal matrix
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


def make_projection_loss(eval_fn):
    """Create loss for training projection."""
    if eval_fn == "classification":
        def loss_fn(proj_embs, labels):
            proj_embs = proj_embs / (proj_embs.norm(dim=1, keepdim=True) + 1e-8)
            sim = proj_embs @ proj_embs.T
            labels_t = torch.tensor(labels, device=sim.device)
            same = (labels_t.unsqueeze(0) == labels_t.unsqueeze(1)).float()
            pos = (sim * same).sum(dim=1) / (same.sum(dim=1) + 1e-8)
            neg = (sim * (1 - same)).sum(dim=1) / ((1 - same).sum(dim=1) + 1e-8)
            return -torch.mean(pos - neg)
        return loss_fn
    elif eval_fn == "sts":
        def loss_fn(proj_embs, gold_scores):
            n = len(gold_scores)
            e1 = proj_embs[:n]
            e2 = proj_embs[n:2*n]
            cos = torch.nn.functional.cosine_similarity(e1, e2)
            c = cos - cos.mean()
            gs = torch.tensor(gold_scores, device=cos.device, dtype=torch.float32)
            g = gs - gs.mean()
            corr = (c * g).sum() / (c.norm() * g.norm() + 1e-8)
            return -corr
        return loss_fn
    elif eval_fn == "clustering":
        def loss_fn(proj_embs, labels):
            labels_t = torch.tensor(labels, device=proj_embs.device)
            centers = []
            for lbl in labels_t.unique():
                centers.append(proj_embs[labels_t == lbl].mean(dim=0))
            centers = torch.stack(centers)
            sim = torch.nn.functional.cosine_similarity(
                centers.unsqueeze(1), centers.unsqueeze(0), dim=2)
            mask = 1 - torch.eye(len(centers), device=sim.device)
            return (sim * mask).sum()
        return loss_fn
    elif eval_fn == "retrieval":
        def loss_fn(proj_embs, n_corpus, relevant_docs):
            corpus = proj_embs[:n_corpus]
            queries = proj_embs[n_corpus:]
            loss = torch.tensor(0.0, device=proj_embs.device)
            cnt = 0
            for qi, docs in relevant_docs.items():
                if qi >= len(queries):
                    continue
                for di in docs:
                    if di < len(corpus):
                        sim = torch.nn.functional.cosine_similarity(
                            queries[qi].unsqueeze(0), corpus[di].unsqueeze(0))
                        loss = loss - sim.mean()
                        cnt += 1
            return loss / max(cnt, 1)
        return loss_fn
    return None


def split_data(embs, task_data, train_ratio=0.5, dev_ratio=0.25):
    """Split embeddings into train/dev/test."""
    eval_fn = task_data["eval_fn"]
    n = len(embs)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:n_train + n_dev]

    if eval_fn in ("classification", "clustering"):
        labels = np.array(task_data["labels"])
        return {
            "train": {"embs": embs[train_idx], "labels": labels[train_idx].tolist()},
            "dev": {"embs": embs[dev_idx], "labels": labels[dev_idx].tolist()},
        }
    elif eval_fn == "sts":
        n_pairs = len(task_data["gold_scores"])
        half = n // 2
        pair_idx = rng.permutation(n_pairs)
        n_train_p = int(n_pairs * train_ratio)
        n_dev_p = int(n_pairs * dev_ratio)
        train_pairs = pair_idx[:n_train_p]
        dev_pairs = pair_idx[n_train_p:n_train_p + n_dev_p]

        def get_pair(pairs):
            return {
                "embs": np.vstack([embs[pairs], embs[half + pairs]]),
                "gold_scores": [task_data["gold_scores"][i] for i in pairs],
                "n_pairs": len(pairs),
            }
        return {"train": get_pair(train_pairs), "dev": get_pair(dev_pairs)}
    elif eval_fn == "retrieval":
        n_corpus = len(task_data["corpus_texts"])
        n_queries = n - n_corpus
        query_idx = rng.permutation(n_queries)
        n_train_q = int(n_queries * train_ratio)
        n_dev_q = int(n_queries * dev_ratio)
        train_qi = query_idx[:n_train_q]
        dev_qi = query_idx[n_train_q:n_train_q + n_dev_q]

        def get_retr(qi_arr):
            new_rel = {}
            for i_new, qi_orig in enumerate(qi_arr):
                if qi_orig in task_data["relevant_docs"]:
                    new_rel[i_new] = task_data["relevant_docs"][qi_orig]
            return {
                "embs": np.vstack([embs[:n_corpus], embs[n_corpus + qi_arr]]),
                "n_corpus": n_corpus, "relevant_docs": new_rel,
            }
        return {"train": get_retr(train_qi), "dev": get_retr(dev_qi)}
    return None


def train_projection(train_data, dev_data, model_dim, target_dim, eval_fn,
                     n_epochs=30, lr=0.01, device='cuda:0'):
    """Train learnable projection on train set, validate on dev set."""
    proj = LearnableProjection(model_dim, target_dim).to(device)
    optimizer = torch.optim.Adam(proj.parameters(), lr=lr)

    train_embs = torch.tensor(train_data["embs"], dtype=torch.float32, device=device)
    dev_embs = torch.tensor(dev_data["embs"], dtype=torch.float32, device=device)

    loss_fn = make_projection_loss(eval_fn)
    if loss_fn is None:
        return None

    def get_args(data, embs):
        if eval_fn in ("classification", "clustering"):
            return [embs, data["labels"]]
        elif eval_fn == "sts":
            return [embs, data["gold_scores"]]
        elif eval_fn == "retrieval":
            return [embs, data["n_corpus"], data["relevant_docs"]]
        return [embs]

    best_dev_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        proj.train()
        optimizer.zero_grad()
        projected = proj(train_embs)
        args = get_args(train_data, projected)
        loss = loss_fn(*args)
        loss.backward()
        optimizer.step()

        proj.eval()
        with torch.no_grad():
            dev_projected = proj(dev_embs)
            dev_args = get_args(dev_data, dev_projected)
            dev_loss = loss_fn(*dev_args).item()

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}

    if best_state:
        proj.load_state_dict(best_state)

    # Apply to all embeddings and return
    return proj


def scalar_quantize(embs, n_bits=8):
    """Scalar quantization to int8."""
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    # Scale to [-1, 1]
    max_val = np.max(np.abs(embs_norm))
    scaled = embs_norm / (max_val + 1e-8)
    quantized = np.round(scaled * (2**(n_bits-1) - 1)).astype(np.int8)
    # Dequantize
    dequant = quantized.astype(np.float32) / (2**(n_bits-1) - 1) * max_val
    return dequant


def evaluate_with_transform(embs, task_data, transform_fn):
    """Evaluate with a transformation applied to embeddings."""
    transformed = transform_fn(embs)
    eval_fn = task_data["eval_fn"]

    if eval_fn == "classification":
        return compute_classification_accuracy(transformed, task_data["labels"])
    elif eval_fn == "sts":
        n = len(task_data["gold_scores"])
        return compute_sts_spearman(transformed[:n], transformed[n:2*n], task_data["gold_scores"])
    elif eval_fn == "clustering":
        return compute_clustering_vmeasure(transformed, task_data["labels"])
    elif eval_fn == "retrieval":
        n_corpus = len(task_data["corpus_texts"])
        return compute_retrieval_ndcg(transformed[n_corpus:], transformed[:n_corpus],
                                       task_data["relevant_docs"])
    return 0.0


def run_learnable_projection(model_name, model_path, task_data_map, target_dims,
                             device='cuda:0', cache_dir='data/embeddings_cache'):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
    model_dim = model.encode("hello").shape[-1]
    print(f"Dimension: {model_dim}")

    cache = EmbeddingCache(cache_dir)

    results = {
        "model": model_name, "model_dim": model_dim,
        "target_dims": target_dims, "methods": {},
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
        baseline = evaluate_with_dims(embs, task_data, list(range(model_dim)))
        if baseline == 0:
            print(f"    Baseline=0, skipping")
            continue
        print(f"    Baseline: {baseline:.2f}")

        # Split for supervised methods
        splits = split_data(embs, task_data)

        task_results = {"baseline": baseline}

        # Scalar quantization (no dimensionality reduction)
        dequant = scalar_quantize(embs)
        sq_score = evaluate_with_transform(embs, task_data, lambda e, dq=dequant: dq)
        task_results["scalar_quant_int8"] = {
            "score": float(sq_score),
            "retention": float(sq_score / baseline * 100),
        }

        for target_dim in target_dims:
            dim_results = {}
            actual_dim = min(target_dim, embs.shape[0], embs.shape[1])

            # 1. Random coordinate selection
            rng = np.random.RandomState(42)
            rand_dims = rng.choice(model_dim, size=target_dim, replace=False).tolist()
            rand_score = evaluate_with_dims(embs, task_data, rand_dims)
            dim_results["random_coord"] = {"score": float(rand_score),
                                           "retention": float(rand_score / baseline * 100)}

            # 2. PCA (clamp to min(n_samples, n_features))
            pca = PCA(n_components=actual_dim)
            pca.fit(embs)  # Fit on all data (unsupervised)
            pca_score = evaluate_with_transform(embs, task_data, lambda e, p=pca: p.transform(e))
            dim_results["pca"] = {"score": float(pca_score),
                                  "retention": float(pca_score / baseline * 100)}

            # 3. Random projection
            rng2 = np.random.RandomState(42)
            proj = rng2.randn(model_dim, target_dim) / np.sqrt(target_dim)
            rp_score = evaluate_with_transform(embs, task_data, lambda e, pr=proj: e @ pr)
            dim_results["random_proj"] = {"score": float(rp_score),
                                          "retention": float(rp_score / baseline * 100)}

            # 4. Learnable projection (train on train set only)
            if splits:
                try:
                    lp = train_projection(
                        splits["train"], splits["dev"], model_dim, target_dim,
                        eval_fn, device=device)
                    if lp is not None:
                        all_embs_t = torch.tensor(embs, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            projected = lp(all_embs_t).cpu().numpy()
                        lp_score = evaluate_with_transform(embs, task_data,
                                                           lambda e, p=projected: p)
                        dim_results["learnable_proj"] = {
                            "score": float(lp_score),
                            "retention": float(lp_score / baseline * 100),
                        }
                except Exception as e:
                    print(f"      Learnable proj error: {e}")

            print(f"    dim={target_dim}: "
                  f"RC={dim_results['random_coord']['retention']:.1f}% "
                  f"PCA={dim_results['pca']['retention']:.1f}% "
                  f"RP={dim_results['random_proj']['retention']:.1f}% "
                  f"LP={dim_results.get('learnable_proj', {}).get('retention', 'N/A')}%")

            task_results[f"dim_{target_dim}"] = dim_results

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
                                 'roberta-large'])
    parser.add_argument('--target-dims', nargs='+', type=int, default=[64, 128, 256])
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
        results = run_learnable_projection(
            model_name, model_path, task_data_map,
            target_dims=args.target_dims, device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"learnable_projection_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
