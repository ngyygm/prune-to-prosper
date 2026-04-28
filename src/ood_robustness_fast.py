#!/usr/bin/env python3
"""
Fast OOD / domain robustness analysis using pre-computed embeddings.

Train importance on one domain, test on another.
Compares random vs task-aware selection under distribution shift.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/ood_robustness_fast.py \
        --models gte-large-en-v1.5 bge-m3 --device cuda:0
"""

import os
import argparse
import json
import gc
import time
import numpy as np
import torch
from tqdm import tqdm

from chunk_size_sweep_fast import load_task_data, evaluate_with_dims
from embedding_cache import EmbeddingCache

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

# OOD experiment pairs: source → target
OOD_PAIRS = {
    "retrieval": {
        "source": ["NFCorpus"],
        "target": ["ArguAna", "SciFact"],
    },
    "classification": {
        "source": ["ImdbClassification"],
        "target": ["Banking77Classification"],
    },
    "sts": {
        "source": ["STSBenchmark"],
        "target": ["BIOSSES"],
    },
}


def compute_chunk_scores(embs, task_data, model_dim, win_size):
    """Score each chunk standalone on a task."""
    n_chunks = model_dim // win_size
    scores = np.zeros(n_chunks)
    for cid in range(n_chunks):
        dims = list(range(cid * win_size, (cid + 1) * win_size))
        scores[cid] = evaluate_with_dims(embs, task_data, dims)
    return scores


def select_top_k(scores, budget, win_size):
    """Select top-k chunks from importance scores."""
    n_select = max(budget // win_size, 1)
    ranked = np.argsort(scores)[::-1]
    dims = []
    for cid in ranked[:n_select]:
        dims.extend(range(cid * win_size, (cid + 1) * win_size))
    return dims


def run_ood(model_name, model_path, task_data_map, win_size, budgets, n_random,
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
        "win_size": win_size,
        "ood_pairs": {},
    }

    for domain_name, pair_config in OOD_PAIRS.items():
        print(f"\n  Domain: {domain_name}")

        source_names = pair_config["source"]
        target_names = pair_config["target"]

        # Check all tasks are available
        all_task_names = source_names + target_names
        available = {n: task_data_map[n] for n in all_task_names if n in task_data_map}
        if len(available) < len(all_task_names):
            missing = [n for n in all_task_names if n not in task_data_map]
            print(f"    Missing tasks: {missing}, skipping domain")
            continue

        domain_results = {"source": source_names, "target": target_names, "budgets": {}}

        # Encode and score source tasks
        source_scores_list = []
        for st_name in source_names:
            st_data = available[st_name]
            eval_fn = st_data["eval_fn"]
            if eval_fn == "sts":
                texts = st_data["texts1"] + st_data["texts2"]
            elif eval_fn == "retrieval":
                texts = st_data["corpus_texts"] + st_data["query_texts"]
            else:
                texts = st_data["texts"]

            embs = cache.get_or_compute(model_name, st_name, texts, model, device=device)
            scores = compute_chunk_scores(embs, st_data, model_dim, win_size)
            source_scores_list.append(scores)

        avg_source_scores = np.mean(source_scores_list, axis=0)

        # For each budget, test on target tasks
        for budget in budgets:
            budget_results = {}
            opt_dims = select_top_k(avg_source_scores, budget, win_size)

            for tt_name in target_names:
                tt_data = available[tt_name]
                eval_fn = tt_data["eval_fn"]
                if eval_fn == "sts":
                    texts = tt_data["texts1"] + tt_data["texts2"]
                elif eval_fn == "retrieval":
                    texts = tt_data["corpus_texts"] + tt_data["query_texts"]
                else:
                    texts = tt_data["texts"]

                embs = cache.get_or_compute(model_name, tt_name, texts, model, device=device)

                baseline = evaluate_with_dims(embs, tt_data, list(range(model_dim)))
                if baseline == 0:
                    continue

                # Source-optimized
                opt_score = evaluate_with_dims(embs, tt_data, opt_dims)

                # Random (avg of seeds)
                random_scores = []
                for seed in range(n_random):
                    rng = np.random.RandomState(seed + 100)
                    rand_dims = sorted(rng.choice(model_dim, size=budget, replace=False).tolist())
                    rs = evaluate_with_dims(embs, tt_data, rand_dims)
                    random_scores.append(rs)
                random_mean = np.mean(random_scores)

                # Oracle (target-optimized)
                target_scores = compute_chunk_scores(embs, tt_data, model_dim, win_size)
                oracle_dims = select_top_k(target_scores, budget, win_size)
                oracle_score = evaluate_with_dims(embs, tt_data, oracle_dims)

                budget_results[tt_name] = {
                    "baseline": baseline,
                    "source_opt_score": opt_score,
                    "source_opt_retention": opt_score / baseline * 100,
                    "random_mean": random_mean,
                    "random_retention": random_mean / baseline * 100,
                    "oracle_score": oracle_score,
                    "oracle_retention": oracle_score / baseline * 100,
                }

                print(f"    budget={budget} {tt_name}: "
                      f"source-opt={opt_score/baseline*100:.1f}% "
                      f"random={random_mean/baseline*100:.1f}% "
                      f"oracle={oracle_score/baseline*100:.1f}%")

            domain_results["budgets"][str(budget)] = budget_results

        results["ood_pairs"][domain_name] = domain_results

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'bge-m3',
                                 'stella_en_400M_v5', 'roberta-large-InBedder'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--win-size', type=int, default=2)
    parser.add_argument('--n-random', type=int, default=5)
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large-InBedder': '/home/linkco/exa/models/inbedder-roberta-large',
    }

    # Load all needed task data
    all_task_names = set()
    for pair in OOD_PAIRS.values():
        all_task_names.update(pair["source"])
        all_task_names.update(pair["target"])

    print("Loading task data...")
    task_data_map = {}
    for tn in sorted(all_task_names):
        td = load_task_data(tn)
        if td is not None:
            task_data_map[tn] = td
            print(f"  {tn}: {td['eval_fn']}")
        else:
            print(f"  {tn}: FAILED")
    print(f"Loaded {len(task_data_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        t0 = time.time()
        results = run_ood(
            model_name, model_path, task_data_map,
            win_size=args.win_size, budgets=args.budgets,
            n_random=args.n_random, device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"ood_robustness_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
