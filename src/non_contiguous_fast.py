#!/usr/bin/env python3
"""
Fast non-contiguous selection using pre-computed embeddings.

Compares contiguous chunk oracle vs greedy forward selection vs random.
Uses pre-computed embeddings — orders of magnitude faster than full MTEB.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/non_contiguous_fast.py \
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

MAX_SAMPLES = 5000  # Subsample large datasets for tractable greedy computation


def subsample_for_eval(embs, task_data, max_samples=5000):
    """Subsample large datasets to make greedy forward feasible.

    kNN classification is O(n^2), so ImdbClassification (25k samples)
    makes each evaluate_with_dims call take ~4s. Subsampling to 5k
    reduces this to ~0.3s, making greedy forward tractable.
    """
    eval_fn = task_data["eval_fn"]
    n = len(embs)

    if n <= max_samples:
        return embs, task_data

    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=max_samples, replace=False)
    embs_sub = embs[idx]

    td = dict(task_data)

    if eval_fn == "classification":
        td["texts"] = None
        td["labels"] = [task_data["labels"][i] for i in idx]
    elif eval_fn == "clustering":
        td["texts"] = None
        td["labels"] = [task_data["labels"][i] for i in idx]
    elif eval_fn == "sts":
        n_pairs = len(task_data["gold_scores"])
        half = n // 2
        idx1 = idx[idx < half]
        idx2 = idx[idx >= half] - half
        pair_idx = sorted(set(idx1) & set(idx2))
        if len(pair_idx) < max_samples // 2:
            pair_idx = rng.choice(min(half, n_pairs), size=min(max_samples // 2, n_pairs), replace=False)
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
        td["n_corpus"] = len(corpus_idx)  # preserve for evaluate_with_dims
        new_relevant = {}
        for i_q, qi_orig in enumerate(query_idx):
            new_qi = len(corpus_idx) + i_q
            if qi_orig in task_data["relevant_docs"]:
                new_relevant[new_qi] = task_data["relevant_docs"][qi_orig]
        td["relevant_docs"] = new_relevant
        embs_sub = np.vstack([embs[corpus_idx], embs[n_corpus + query_idx]])

    return embs_sub, td


def contiguous_oracle(embs, task_data, model_dim, budget, win_size):
    """Score chunks standalone, select top-k contiguous chunks."""
    n_chunks = model_dim // win_size
    n_select = max(budget // win_size, 1)

    chunk_scores = np.zeros(n_chunks)
    for cid in range(n_chunks):
        dims = list(range(cid * win_size, (cid + 1) * win_size))
        chunk_scores[cid] = evaluate_with_dims(embs, task_data, dims)

    ranked = np.argsort(chunk_scores)[::-1]
    selected_dims = []
    for cid in ranked[:n_select]:
        selected_dims.extend(range(cid * win_size, (cid + 1) * win_size))

    return selected_dims, chunk_scores


def greedy_forward(embs, task_data, model_dim, budget, win_size):
    """Greedy forward selection: add best chunk each step."""
    n_chunks = model_dim // win_size
    n_select = max(budget // win_size, 1)

    selected = []
    remaining = list(range(n_chunks))
    history = []

    for step in range(n_select):
        best_score = -float('inf')
        best_chunk = None

        for cid in remaining:
            trial = selected + [cid]
            dims = []
            for c in trial:
                dims.extend(range(c * win_size, (c + 1) * win_size))
            score = evaluate_with_dims(embs, task_data, dims)
            if score > best_score:
                best_score = score
                best_chunk = cid

        if best_chunk is not None:
            selected.append(best_chunk)
            remaining.remove(best_chunk)
            history.append(best_score)

    selected_dims = []
    for cid in selected:
        selected_dims.extend(range(cid * win_size, (cid + 1) * win_size))

    return selected_dims, history


def random_dims(model_dim, budget, seed=42):
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(model_dim, size=budget, replace=False).tolist())


def run_non_contiguous(model_name, model_path, task_data_map, win_size, budgets,
                       n_random, device='cuda:0', cache_dir='data/embeddings_cache'):
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
        "win_size": win_size,
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

        # Subsample large datasets for tractable greedy forward computation
        embs_eval, task_data_eval = subsample_for_eval(embs, task_data, MAX_SAMPLES)
        if embs_eval.shape[0] < embs.shape[0]:
            print(f"    Subsampled: {embs.shape[0]} -> {embs_eval.shape[0]}")

        baseline = evaluate_with_dims(embs_eval, task_data_eval, list(range(model_dim)))
        if baseline == 0:
            print(f"    Baseline=0, skipping")
            continue
        print(f"    Baseline: {baseline:.2f}")

        task_results = {"baseline": baseline, "budgets": {}}

        # Compute chunk standalone scores ONCE (reuse across budgets)
        n_chunks = model_dim // win_size
        print(f"    Computing {n_chunks} chunk scores...")
        chunk_scores = np.zeros(n_chunks)
        for cid in range(n_chunks):
            dims = list(range(cid * win_size, (cid + 1) * win_size))
            chunk_scores[cid] = evaluate_with_dims(embs_eval, task_data_eval, dims)
        ranked_chunks = np.argsort(chunk_scores)[::-1]

        # Greedy forward for budget=64 only (key comparison, most expensive)
        greedy_budget = min(budgets)
        n_select_greedy = max(greedy_budget // win_size, 1)
        print(f"    Running greedy forward (budget={greedy_budget}, {n_select_greedy} steps)...")
        greedy_selected = []
        greedy_remaining = list(range(n_chunks))
        greedy_history = []
        for step in range(n_select_greedy):
            best_score = -float('inf')
            best_chunk = None
            for cid in greedy_remaining:
                trial = greedy_selected + [cid]
                dims = []
                for c in trial:
                    dims.extend(range(c * win_size, (c + 1) * win_size))
                score = evaluate_with_dims(embs_eval, task_data_eval, dims)
                if score > best_score:
                    best_score = score
                    best_chunk = cid
            if best_chunk is not None:
                greedy_selected.append(best_chunk)
                greedy_remaining.remove(best_chunk)
                greedy_history.append(best_score)

        for budget in budgets:
            b_results = {}
            t_budget = time.time()
            n_select = max(budget // win_size, 1)

            # 1. Contiguous chunk oracle (reuse precomputed chunk_scores)
            cont_chunks = ranked_chunks[:n_select].tolist()
            cont_dims = []
            for cid in cont_chunks:
                cont_dims.extend(range(cid * win_size, (cid + 1) * win_size))
            cont_score = evaluate_with_dims(embs_eval, task_data_eval, cont_dims)
            b_results["contiguous_oracle"] = {
                "score": cont_score,
                "retention": cont_score / baseline * 100,
            }

            # 2. Greedy forward (only for smallest budget)
            if budget == greedy_budget:
                greedy_dims_list = []
                for cid in greedy_selected:
                    greedy_dims_list.extend(range(cid * win_size, (cid + 1) * win_size))
                greedy_score = greedy_history[-1] if greedy_history else 0
                b_results["greedy_forward"] = {
                    "score": greedy_score,
                    "retention": greedy_score / baseline * 100,
                    "n_steps": len(greedy_history),
                }
                # Overlap
                cont_set = set(cont_dims)
                greedy_set = set(greedy_dims_list)
                jaccard = len(cont_set & greedy_set) / len(cont_set | greedy_set) if cont_set | greedy_set else 0
                b_results["selection_overlap_jaccard"] = float(jaccard)
            else:
                b_results["greedy_forward"] = {"retention": None, "note": "only computed for smallest budget"}

            # 3. Random (average of n_random seeds)
            random_scores = []
            for seed in range(n_random):
                rand_d = random_dims(model_dim, budget, seed + 100)
                rs = evaluate_with_dims(embs_eval, task_data_eval, rand_d)
                random_scores.append(rs)
            random_mean = np.mean(random_scores)
            b_results["random"] = {
                "mean_score": random_mean,
                "retention": random_mean / baseline * 100,
                "n_seeds": len(random_scores),
            }

            gf_ret = b_results['greedy_forward'].get('retention')
            gf_str = f"{gf_ret:.1f}%" if gf_ret is not None else "skip"
            print(f"    budget={budget}: cont={b_results['contiguous_oracle']['retention']:.1f}% "
                  f"greedy={gf_str} "
                  f"rand={b_results['random']['retention']:.1f}% "
                  f"({time.time()-t_budget:.0f}s)")

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
    parser.add_argument('--win-size', type=int, default=4)
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
        results = run_non_contiguous(
            model_name, model_path, task_data_map,
            win_size=args.win_size, budgets=args.budgets,
            n_random=args.n_random, device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"non_contiguous_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
