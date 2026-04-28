#!/usr/bin/env python3
"""
Fast leave-one-out / marginal contribution analysis using pre-computed embeddings.

Compares multiple importance definitions using embedding slicing (seconds per chunk
instead of minutes with full MTEB pipeline):

  1. Standalone contribution: Eval(chunk_i alone)
  2. Leave-one-out drop: Eval(full) - Eval(full \ chunk_i)
  3. Marginal gain: avg Eval(B ∪ chunk_i) - Eval(B) for random subsets B
  4. Approximate Shapley value (permutation sampling)

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/leave_one_out_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 \
        --win-size 4 --device cuda:0
"""

import os
import argparse
import json
import gc
import time
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr

# Reuse functions from chunk_size_sweep_fast
from chunk_size_sweep_fast import (
    load_task_data, evaluate_with_dims, cosine_similarity,
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

MAX_SAMPLES = 5000  # Subsample large datasets for tractable LOO computation


def subsample_for_loo(embs, task_data, max_samples=5000):
    """Subsample large datasets to make LOO computation tractable.

    kNN classification is O(n^2), so ImdbClassification (25k samples)
    makes each evaluate_with_dims call take ~60s. Subsampling to 5k
    reduces this to ~3s per call, making LOO/marginal/Shapley feasible.
    """
    eval_fn = task_data["eval_fn"]
    n = len(embs)

    if n <= max_samples:
        return embs, task_data

    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=max_samples, replace=False)
    embs_sub = embs[idx]

    # Create subsampled task_data copy
    td = dict(task_data)

    if eval_fn == "classification":
        td["texts"] = None  # Not needed with pre-computed embeddings
        td["labels"] = [task_data["labels"][i] for i in idx]
    elif eval_fn == "clustering":
        td["texts"] = None
        td["labels"] = [task_data["labels"][i] for i in idx]
    elif eval_fn == "sts":
        n_pairs = len(task_data["gold_scores"])
        half = n // 2
        idx1 = idx[idx < half]
        idx2 = idx[idx >= half] - half
        # Use matching pairs only
        pair_idx = sorted(set(idx1) & set(idx2))
        if len(pair_idx) < max_samples // 2:
            pair_idx = rng.choice(min(half, n_pairs), size=min(max_samples // 2, n_pairs), replace=False)
        td["texts1"] = None
        td["texts2"] = None
        td["gold_scores"] = [task_data["gold_scores"][i] for i in pair_idx]
        embs_sub = np.vstack([embs[pair_idx], embs[half + pair_idx]])
    elif eval_fn == "retrieval":
        n_corpus = len(task_data["corpus_texts"])
        # Keep all corpus, subsample queries
        corpus_idx = idx[idx < n_corpus]
        query_idx = idx[idx >= n_corpus] - n_corpus
        if len(corpus_idx) == 0:
            corpus_idx = np.arange(min(n_corpus, max_samples))
        if len(query_idx) == 0:
            query_idx = rng.choice(n_corpus, size=max(1, max_samples - n_corpus), replace=False)
        # Rebuild with full corpus + subsampled queries
        td["n_corpus"] = len(corpus_idx)
        td["corpus_texts"] = None
        td["query_texts"] = None
        new_relevant = {}
        for qi_orig in query_idx:
            new_qi = n_corpus + list(query_idx).index(qi_orig)
            if qi_orig in task_data["relevant_docs"]:
                new_relevant[new_qi] = task_data["relevant_docs"][qi_orig]
        td["relevant_docs"] = new_relevant
        embs_sub = np.vstack([embs[corpus_idx], embs[n_corpus + query_idx]])

    return embs_sub, td


def evaluate_chunk(embs, task_data, chunk_id, win_size):
    """Evaluate a single chunk's standalone contribution."""
    dims = list(range(chunk_id * win_size, (chunk_id + 1) * win_size))
    return evaluate_with_dims(embs, task_data, dims)


def evaluate_without_chunk(embs, task_data, chunk_id, win_size, model_dim):
    """Evaluate with one chunk dropped (LOO)."""
    all_dims = list(range(model_dim))
    drop_start = chunk_id * win_size
    drop_end = (chunk_id + 1) * win_size
    keep_dims = [d for d in all_dims if d < drop_start or d >= drop_end]
    return evaluate_with_dims(embs, task_data, keep_dims)


def evaluate_chunks(embs, task_data, chunk_ids, win_size):
    """Evaluate using selected chunks concatenated."""
    dims = []
    for cid in chunk_ids:
        dims.extend(range(cid * win_size, (cid + 1) * win_size))
    return evaluate_with_dims(embs, task_data, dims)


def compute_standalone(embs, task_data, win_size, n_chunks):
    """Standalone contribution: Eval(chunk_i alone)."""
    scores = np.zeros(n_chunks)
    for cid in range(n_chunks):
        scores[cid] = evaluate_chunk(embs, task_data, cid, win_size)
    return scores


def compute_loo(embs, task_data, win_size, n_chunks, model_dim):
    """Leave-one-out drop: Eval(full) - Eval(full \\ chunk_i)."""
    full_score = evaluate_with_dims(embs, task_data, list(range(model_dim)))
    if full_score == 0:
        return full_score, np.zeros(n_chunks)

    drops = np.zeros(n_chunks)
    for cid in range(n_chunks):
        score_without = evaluate_without_chunk(embs, task_data, cid, win_size, model_dim)
        drops[cid] = full_score - score_without
    return full_score, drops


def compute_marginal(embs, task_data, win_size, n_chunks, budget, n_samples=30):
    """Marginal gain of adding each chunk to a random base subset."""
    n_select = max(budget // win_size, 1)
    rng = np.random.RandomState(42)
    marginal_gains = np.zeros(n_chunks)

    for _ in tqdm(range(n_samples), desc="Marginal", leave=False):
        base_size = max(n_select - 1, 1)
        base_chunks = rng.choice(n_chunks, size=base_size, replace=False).tolist()
        base_score = evaluate_chunks(embs, task_data, base_chunks, win_size)

        remaining = [c for c in range(n_chunks) if c not in base_chunks]
        for cid in remaining:
            extended = base_chunks + [cid]
            ext_score = evaluate_chunks(embs, task_data, extended, win_size)
            marginal_gains[cid] += (ext_score - base_score)

    marginal_gains /= n_samples
    return marginal_gains


def compute_shapley(embs, task_data, win_size, n_chunks, n_permutations=32, budget=256):
    """Approximate Shapley values via permutation sampling."""
    n_select = max(budget // win_size, 1)
    shapley = np.zeros(n_chunks)
    max_set_size = n_select * 2

    for _ in tqdm(range(n_permutations), desc="Shapley", leave=False):
        perm = np.random.permutation(n_chunks)
        prev_score = 0.0

        for i, cid in enumerate(perm):
            if i >= max_set_size:
                break
            current_set = perm[:i+1].tolist()
            score = evaluate_chunks(embs, task_data, current_set, win_size)
            shapley[cid] += (score - prev_score)
            prev_score = score

    shapley /= n_permutations
    return shapley


def run_loo_analysis(model_name, model_path, task_data_map, win_size, budget,
                     n_marginal, n_shapley, device='cuda:0', cache_dir='data/embeddings_cache'):
    """Run fast LOO analysis for one model."""
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
    model_dim = model.encode("hello").shape[-1]
    n_chunks = model_dim // win_size
    print(f"Dimension: {model_dim}, win_size: {win_size}, n_chunks: {n_chunks}")

    cache = EmbeddingCache(cache_dir)

    results = {
        "model": model_name,
        "model_dim": model_dim,
        "win_size": win_size,
        "budget": budget,
        "n_chunks": n_chunks,
        "tasks": {},
    }

    for task_name, task_data in task_data_map.items():
        print(f"\n  Task: {task_name}")
        t0 = time.time()

        # Encode all texts once
        eval_fn = task_data["eval_fn"]
        if eval_fn == "sts":
            all_texts = task_data["texts1"] + task_data["texts2"]
        elif eval_fn == "retrieval":
            all_texts = task_data["corpus_texts"] + task_data["query_texts"]
        else:
            all_texts = task_data["texts"]

        embs = cache.get_or_compute(model_name, task_name, all_texts, model, device=device)
        print(f"    Embeddings: {embs.shape}")

        # Subsample large datasets for tractable LOO computation
        embs_eval, task_data_eval = subsample_for_loo(embs, task_data, MAX_SAMPLES)
        if embs_eval.shape[0] < embs.shape[0]:
            print(f"    Subsampled: {embs.shape[0]} -> {embs_eval.shape[0]} for LOO")

        task_results = {}

        # 1. Standalone
        print("    Computing standalone scores...")
        standalone = compute_standalone(embs_eval, task_data_eval, win_size, n_chunks)
        task_results["standalone"] = standalone.tolist()

        # 2. LOO
        print("    Computing LOO scores...")
        full_score, loo_drops = compute_loo(embs_eval, task_data_eval, win_size, n_chunks, model_dim)
        task_results["full_score"] = full_score
        task_results["loo_drops"] = loo_drops.tolist()

        # 3. Marginal
        print("    Computing marginal gains...")
        marginal = compute_marginal(embs_eval, task_data_eval, win_size, n_chunks, budget, n_marginal)
        task_results["marginal"] = marginal.tolist()

        # 4. Shapley
        if n_shapley > 0:
            print("    Computing approximate Shapley...")
            shapley = compute_shapley(embs_eval, task_data_eval, win_size, n_chunks, n_shapley, budget)
            task_results["shapley"] = shapley.tolist()

        # Rank correlations between importance definitions
        for name2, scores2 in [("loo", loo_drops), ("marginal", marginal)]:
            if np.sum(np.abs(scores2)) > 0:
                rho, p = spearmanr(standalone, scores2)
                task_results[f"rho_standalone_{name2}"] = float(rho)
                task_results[f"p_standalone_{name2}"] = float(p)

        if n_shapley > 0 and np.sum(np.abs(shapley)) > 0:
            rho, p = spearmanr(standalone, shapley)
            task_results["rho_standalone_shapley"] = float(rho)
            task_results["p_standalone_shapley"] = float(p)

        # Entropy for each definition
        for name, scores in [("standalone", standalone), ("loo", loo_drops),
                              ("marginal", marginal)]:
            if n_shapley > 0:
                scores_all = {"standalone": standalone, "loo": loo_drops,
                              "marginal": marginal, "shapley": shapley}
            else:
                scores_all = {"standalone": standalone, "loo": loo_drops,
                              "marginal": marginal}

        for name, scores in scores_all.items():
            abs_scores = np.abs(scores)
            if np.sum(abs_scores) > 0:
                p = abs_scores / np.sum(abs_scores)
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p)) / np.log2(len(scores))
                task_results[f"entropy_{name}"] = float(entropy)

        # Top-k overlap (Jaccard)
        n_select = max(budget // win_size, 1)
        rankings = {}
        for name, scores in scores_all.items():
            rankings[name] = set(np.argsort(scores)[::-1][:n_select].tolist())

        ranking_names = list(rankings.keys())
        for i in range(len(ranking_names)):
            for j in range(i+1, len(ranking_names)):
                n1, n2 = ranking_names[i], ranking_names[j]
                s1, s2 = rankings[n1], rankings[n2]
                jaccard = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
                task_results[f"jaccard_{n1}_{n2}"] = float(jaccard)

        results["tasks"][task_name] = task_results
        print(f"    Full: {full_score:.2f}, done in {time.time()-t0:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5',
                                 'roberta-large', 'roberta-large-InBedder'])
    parser.add_argument('--win-size', type=int, default=4)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--n-marginal', type=int, default=5)
    parser.add_argument('--n-shapley', type=int, default=4)
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

    # Load task data
    print("Loading task data...")
    task_data_map = {}
    for task_name in REPRESENTATIVE_TASKS:
        td = load_task_data(task_name)
        if td is not None:
            task_data_map[task_name] = td
            print(f"  {task_name}: {td['eval_fn']}")
        else:
            print(f"  {task_name}: FAILED")
    print(f"Loaded {len(task_data_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}: path not found")
            continue

        t0 = time.time()
        results = run_loo_analysis(
            model_name, model_path, task_data_map,
            win_size=args.win_size, budget=args.budget,
            n_marginal=args.n_marginal, n_shapley=args.n_shapley,
            device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        # Use the filename the queue expects
        output_path = os.path.join(args.output_dir, f"leave_one_out_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
