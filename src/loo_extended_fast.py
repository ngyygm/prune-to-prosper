#!/usr/bin/env python3
"""
Extended LOO analysis with full marginal (50 samples) and Shapley (64 permutations).

Reuses EmbeddingCache — encode once per model+task.
Runs on 14 representative tasks, 4 models.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/loo_extended_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 --device cuda:0
"""

import os
import argparse
import json
import gc
import time
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

from chunk_size_sweep_fast import (
    load_task_data, evaluate_with_dims,
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

MAX_SAMPLES = 3000  # Aggressive subsample for speed


def subsample(embs, task_data, max_samples=3000):
    """Subsample for tractable marginal/Shapley computation."""
    eval_fn = task_data["eval_fn"]
    n = len(embs)
    if n <= max_samples:
        return embs, task_data

    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=max_samples, replace=False)
    embs_sub = embs[idx]
    td = dict(task_data)

    if eval_fn in ("classification", "clustering"):
        td["texts"] = None
        td["labels"] = [task_data["labels"][i] for i in idx]
    elif eval_fn == "sts":
        half = n // 2
        idx1 = idx[idx < half]
        idx2 = idx[idx >= half] - half
        pair_idx = sorted(set(idx1) & set(idx2))
        if len(pair_idx) < max_samples // 2:
            pair_idx = rng.choice(min(half, len(task_data["gold_scores"])),
                                  size=min(max_samples // 2, len(task_data["gold_scores"])),
                                  replace=False)
        td["texts1"] = None
        td["texts2"] = None
        td["gold_scores"] = [task_data["gold_scores"][i] for i in pair_idx]
        embs_sub = np.vstack([embs[pair_idx], embs[half + pair_idx]])
    elif eval_fn == "retrieval":
        n_corpus = len(task_data["corpus_texts"])
        corpus_idx = idx[idx < n_corpus]
        query_idx = idx[idx >= n_corpus] - n_corpus
        if len(corpus_idx) == 0:
            corpus_idx = np.arange(min(n_corpus, max_samples))
        if len(query_idx) == 0:
            query_idx = rng.choice(n_corpus, size=max(1, max_samples - n_corpus), replace=False)
        td["corpus_texts"] = None
        td["query_texts"] = None
        td["n_corpus"] = len(corpus_idx)
        new_rel = {}
        for i_q, qi_orig in enumerate(query_idx):
            new_qi = len(corpus_idx) + i_q
            if qi_orig in task_data["relevant_docs"]:
                new_rel[new_qi] = task_data["relevant_docs"][qi_orig]
        td["relevant_docs"] = new_rel
        embs_sub = np.vstack([embs[corpus_idx], embs[n_corpus + query_idx]])

    return embs_sub, td


def eval_chunks(embs, task_data, chunk_ids, win_size):
    dims = []
    for cid in chunk_ids:
        dims.extend(range(cid * win_size, (cid + 1) * win_size))
    return evaluate_with_dims(embs, task_data, dims)


def compute_standalone(embs, task_data, win_size, n_chunks):
    scores = np.zeros(n_chunks)
    for cid in range(n_chunks):
        dims = list(range(cid * win_size, (cid + 1) * win_size))
        scores[cid] = evaluate_with_dims(embs, task_data, dims)
    return scores


def compute_loo(embs, task_data, win_size, n_chunks, model_dim):
    full_score = evaluate_with_dims(embs, task_data, list(range(model_dim)))
    if full_score == 0:
        return full_score, np.zeros(n_chunks)
    drops = np.zeros(n_chunks)
    for cid in range(n_chunks):
        keep = [d for d in range(model_dim)
                if d < cid * win_size or d >= (cid + 1) * win_size]
        drops[cid] = full_score - evaluate_with_dims(embs, task_data, keep)
    return full_score, drops


def compute_marginal_full(embs, task_data, win_size, n_chunks, budget, n_samples=50):
    """Marginal gain of adding each chunk to random base subsets (50 samples)."""
    n_select = max(budget // win_size, 1)
    rng = np.random.RandomState(42)
    marginal_gains = np.zeros(n_chunks)

    for _ in tqdm(range(n_samples), desc="Marginal(50)", leave=False):
        base_size = max(n_select - 1, 1)
        base_chunks = rng.choice(n_chunks, size=base_size, replace=False).tolist()
        base_score = eval_chunks(embs, task_data, base_chunks, win_size)

        remaining = [c for c in range(n_chunks) if c not in base_chunks]
        # Batch: compute marginal for all remaining chunks
        for cid in remaining:
            extended = base_chunks + [cid]
            ext_score = eval_chunks(embs, task_data, extended, win_size)
            marginal_gains[cid] += (ext_score - base_score)

    marginal_gains /= n_samples
    return marginal_gains


def compute_shapley_full(embs, task_data, win_size, n_chunks, budget, n_permutations=64):
    """Approximate Shapley via permutation sampling (64 permutations)."""
    n_select = max(budget // win_size, 1)
    shapley = np.zeros(n_chunks)
    max_set_size = n_select * 2

    for _ in tqdm(range(n_permutations), desc="Shapley(64)", leave=False):
        perm = np.random.permutation(n_chunks)
        prev_score = 0.0
        for i, cid in enumerate(perm):
            if i >= max_set_size:
                break
            current_set = perm[:i+1].tolist()
            score = eval_chunks(embs, task_data, current_set, win_size)
            shapley[cid] += (score - prev_score)
            prev_score = score

    shapley /= n_permutations
    return shapley


def run_extended_loo(model_name, model_path, task_data_map, win_size, budget,
                     n_marginal, n_shapley, device='cuda:0', cache_dir='data/embeddings_cache'):
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
        "model": model_name, "model_dim": model_dim,
        "win_size": win_size, "budget": budget,
        "n_marginal": n_marginal, "n_shapley": n_shapley,
        "n_chunks": n_chunks, "tasks": {},
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
        embs_eval, task_data_eval = subsample(embs, task_data, MAX_SAMPLES)
        if embs_eval.shape[0] < embs.shape[0]:
            print(f"    Subsampled: {embs.shape[0]} -> {embs_eval.shape[0]}")

        task_results = {}

        # 1. Standalone
        print("    Computing standalone...")
        standalone = compute_standalone(embs_eval, task_data_eval, win_size, n_chunks)
        task_results["standalone"] = standalone.tolist()

        # 2. LOO
        print("    Computing LOO...")
        full_score, loo_drops = compute_loo(embs_eval, task_data_eval, win_size, n_chunks, model_dim)
        task_results["full_score"] = full_score
        task_results["loo_drops"] = loo_drops.tolist()

        # 3. Marginal (50 samples)
        print(f"    Computing marginal ({n_marginal} samples)...")
        marginal = compute_marginal_full(embs_eval, task_data_eval, win_size, n_chunks, budget, n_marginal)
        task_results["marginal"] = marginal.tolist()

        # 4. Shapley (64 permutations)
        if n_shapley > 0:
            print(f"    Computing Shapley ({n_shapley} permutations)...")
            shapley = compute_shapley_full(embs_eval, task_data_eval, win_size, n_chunks, budget, n_shapley)
            task_results["shapley"] = shapley.tolist()

        # Rank correlations
        scores_all = {"standalone": standalone, "loo": loo_drops, "marginal": marginal}
        if n_shapley > 0:
            scores_all["shapley"] = shapley

        for n1 in scores_all:
            for n2 in scores_all:
                if n1 >= n2:
                    continue
                s1, s2 = scores_all[n1], scores_all[n2]
                if np.sum(np.abs(s1)) > 0 and np.sum(np.abs(s2)) > 0:
                    rho, p = spearmanr(s1, s2)
                    task_results[f"rho_{n1}_{n2}"] = float(rho)
                    task_results[f"p_{n1}_{n2}"] = float(p)

        # Entropy for each definition
        for name, scores in scores_all.items():
            abs_scores = np.abs(scores)
            if np.sum(abs_scores) > 0:
                p_dist = abs_scores / np.sum(abs_scores)
                p_dist = p_dist[p_dist > 0]
                entropy = -np.sum(p_dist * np.log2(p_dist)) / np.log2(len(scores))
                task_results[f"entropy_{name}"] = float(entropy)

        # Top-k Jaccard overlap
        n_select = max(budget // win_size, 1)
        rankings = {n: set(np.argsort(np.abs(s))[::-1][:n_select].tolist())
                    for n, s in scores_all.items()}
        rnames = list(rankings.keys())
        for i in range(len(rnames)):
            for j in range(i+1, len(rnames)):
                s1, s2 = rankings[rnames[i]], rankings[rnames[j]]
                jaccard = len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0
                task_results[f"jaccard_{rnames[i]}_{rnames[j]}"] = float(jaccard)

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
    parser.add_argument('--n-marginal', type=int, default=50)
    parser.add_argument('--n-shapley', type=int, default=64)
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
        'roberta-large-InBedder': '/home/linkco/exa/models/inbedder-roberta-large',
    }

    print("Loading task data...")
    task_data_map = {}
    for task_name in REPRESENTATIVE_TASKS:
        td = load_task_data(task_name)
        if td is not None:
            task_data_map[task_name] = td
            print(f"  {task_name}: {td['eval_fn']}")
    print(f"Loaded {len(task_data_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        t0 = time.time()
        results = run_extended_loo(
            model_name, model_path, task_data_map,
            win_size=args.win_size, budget=args.budget,
            n_marginal=args.n_marginal, n_shapley=args.n_shapley,
            device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"loo_extended_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
