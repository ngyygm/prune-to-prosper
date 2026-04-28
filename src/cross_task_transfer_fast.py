#!/usr/bin/env python3
"""
Fast cross-task dimension importance transfer using pre-computed embeddings.

Tests whether donor task rankings transfer at various compression levels.
Uses pre-computed embeddings instead of full MTEB evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/cross_task_transfer_fast.py \
        --models gte-large-en-v1.5 --device cuda:0
"""

import os
import json
import argparse
import gc
import time
import numpy as np
import torch
from tqdm import tqdm

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


def compute_chunk_scores(embs, task_data, model_dim, win_size):
    """Score each chunk standalone on a task."""
    n_chunks = model_dim // win_size
    scores = np.zeros(n_chunks)
    for cid in range(n_chunks):
        dims = list(range(cid * win_size, (cid + 1) * win_size))
        scores[cid] = evaluate_with_dims(embs, task_data, dims)
    return scores


def run_cross_task(model_name, model_path, task_data_map, win_size, dims,
                   n_random, device='cuda:0', cache_dir='data/embeddings_cache'):
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
    model_dim = model.encode("hello").shape[-1]
    cache = EmbeddingCache(cache_dir)

    print(f"Dimension: {model_dim}, win_size: {win_size}")

    # Phase 1: Compute chunk rankings for ALL tasks
    print("\nPhase 1: Computing chunk rankings for all tasks...")
    all_rankings = {}
    all_embs = {}

    for task_name, task_data in task_data_map.items():
        eval_fn = task_data["eval_fn"]
        if eval_fn == "sts":
            texts = task_data["texts1"] + task_data["texts2"]
        elif eval_fn == "retrieval":
            texts = task_data["corpus_texts"] + task_data["query_texts"]
        else:
            texts = task_data["texts"]

        embs = cache.get_or_compute(model_name, task_name, texts, model, device=device)
        all_embs[task_name] = embs

        scores = compute_chunk_scores(embs, task_data, model_dim, win_size)
        ranking = np.argsort(scores)[::-1].tolist()
        all_rankings[task_name] = ranking
        print(f"  {task_name}: ranked {len(ranking)} chunks")

    # Phase 2: For each target dim, test transfer
    results = {
        "model": model_name,
        "model_dim": model_dim,
        "win_size": win_size,
        "target_dims": {},
    }

    for target_dim in dims:
        n_chunks_select = target_dim // win_size
        if n_chunks_select < 1:
            continue

        print(f"\nPhase 2: target_dim={target_dim} ({n_chunks_select} chunks)")
        dim_results = {}

        for donor_task in task_data_map:
            # Use donor's top chunks
            donor_chunks = all_rankings[donor_task][:n_chunks_select]
            donor_dims = []
            for cid in donor_chunks:
                donor_dims.extend(range(cid * win_size, (cid + 1) * win_size))

            dim_results[donor_task] = {}

            for target_task, task_data in task_data_map.items():
                # Evaluate donor's selection on target task
                score = evaluate_with_dims(all_embs[target_task], task_data, donor_dims)
                dim_results[donor_task][target_task] = score

            # Self-transfer score
            self_score = dim_results[donor_task][donor_task]

            # Random baseline for this donor's budget
            random_scores = []
            for seed in range(n_random):
                rng = np.random.RandomState(seed + 100)
                rand_chunks = rng.choice(model_dim // win_size, size=n_chunks_select, replace=False).tolist()
                rand_dims = []
                for cid in rand_chunks:
                    rand_dims.extend(range(cid * win_size, (cid + 1) * win_size))
                rs = evaluate_with_dims(all_embs[donor_task], task_data_map[donor_task], rand_dims)
                random_scores.append(rs)
            random_mean = np.mean(random_scores)

            # Transfer quality: self-transfer vs random
            dim_results[donor_task]["_random_mean"] = random_mean
            dim_results[donor_task]["_self_transfer"] = self_score

            # Cross-task transfer: average score across all target tasks
            target_scores = [dim_results[donor_task][t] for t in task_data_map if t != donor_task]
            dim_results[donor_task]["_avg_cross_transfer"] = np.mean(target_scores) if target_scores else 0

        results["target_dims"][str(target_dim)] = dim_results

        # Print transfer matrix summary
        print(f"  Transfer matrix ({target_dim}d):")
        for donor in sorted(task_data_map.keys()):
            dr = dim_results[donor]
            print(f"    {donor}: self={dr['_self_transfer']:.1f} "
                  f"cross_avg={dr['_avg_cross_transfer']:.1f} "
                  f"random={dr['_random_mean']:.1f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5',
                                 'bge-m3', 'roberta-large', 'roberta-large-InBedder'])
    parser.add_argument('--win-size', type=int, default=4)
    parser.add_argument('--dims', nargs='+', type=int, default=[64, 128, 256])
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
        results = run_cross_task(
            model_name, model_path, task_data_map,
            win_size=args.win_size, dims=args.dims,
            n_random=args.n_random, device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"cross_task_transfer_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
