#!/usr/bin/env python3
"""
Fast retrieval system cost analysis using pre-computed embeddings.

Benchmarks FAISS indexing with different compression methods:
  - Full-dim (baseline)
  - Random-256 coordinate selection
  - PCA-256
  - Random projection-256

Metrics: nDCG@10, Recall@10, index build time, query latency, memory footprint.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/retrieval_cost_analysis_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 --device cuda:0
"""

import os
import argparse
import json
import gc
import time
import numpy as np
import torch
import faiss
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from chunk_size_sweep_fast import load_task_data
from embedding_cache import EmbeddingCache

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

RETRIEVAL_TASKS = ['NFCorpus', 'ArguAna', 'SciFact']


def benchmark_retrieval(corpus_embs, query_embs, relevant_docs, n_queries=100):
    """Benchmark retrieval with FlatIP index."""
    results = {}
    n_corpus = corpus_embs.shape[0]

    # Select queries that have relevance judgments (not just first n)
    judged_queries = [qi for qi in relevant_docs if qi < query_embs.shape[0]]
    if len(judged_queries) > n_queries:
        rng = np.random.RandomState(42)
        judged_queries = sorted(rng.choice(judged_queries, size=n_queries, replace=False).tolist())
    if not judged_queries:
        return {"ndcg_at_10": 0, "recall_at_10": 0, "build_time_s": 0,
                "latency_p50_ms": 0, "throughput_qps": 0, "memory_mb": 0,
                "n_corpus": n_corpus, "n_queries": 0, "dim": corpus_embs.shape[1]}

    queries = query_embs[judged_queries]
    n_q = len(queries)
    dim = corpus_embs.shape[1]

    # Build index
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embs.astype(np.float32))
    build_time = time.time() - t0

    memory_mb = corpus_embs.nbytes / (1024 * 1024)

    # Query
    t0 = time.time()
    k = 10
    scores, indices = index.search(queries.astype(np.float32), k)
    query_time = time.time() - t0
    latency_p50 = query_time / n_q * 1000
    throughput = n_q / query_time

    # Compute metrics — use actual query indices from judged_queries
    recalls, ndcgs = [], []
    for i, qi in enumerate(judged_queries):
        retrieved = indices[i].tolist()
        relevant = relevant_docs.get(qi, set())
        if not relevant:
            continue

        hit = len(set(retrieved) & relevant)
        recall = hit / len(relevant)
        recalls.append(recall)

        dcg = sum(1.0 / np.log2(i + 2) for i, doc in enumerate(retrieved) if doc in relevant)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    return {
        "recall_at_10": float(np.mean(recalls)) if recalls else 0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0,
        "build_time_s": float(build_time),
        "latency_p50_ms": float(latency_p50),
        "throughput_qps": float(throughput),
        "memory_mb": float(memory_mb),
        "n_corpus": n_corpus,
        "n_queries": n_q,
        "dim": dim,
    }


def apply_compression(embs, method, target_dim, model_dim, seed=42):
    if method == 'full':
        return embs
    elif method == 'random_coord':
        rng = np.random.RandomState(seed)
        indices = sorted(rng.choice(model_dim, size=target_dim, replace=False).tolist())
        return embs[:, indices]
    elif method == 'pca':
        pca = PCA(n_components=target_dim)
        return pca.fit_transform(embs)
    elif method == 'random_proj':
        rng = np.random.RandomState(seed)
        proj = rng.randn(embs.shape[1], target_dim) / np.sqrt(target_dim)
        return embs @ proj
    else:
        raise ValueError(f"Unknown method: {method}")


def run_retrieval_cost(model_name, model_path, task_data_map, target_dims,
                       device='cuda:0', cache_dir='data/embeddings_cache'):
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
        "tasks": {},
    }

    for task_name, task_data in task_data_map.items():
        print(f"\n  Task: {task_name}")
        t0 = time.time()

        all_texts = task_data["corpus_texts"] + task_data["query_texts"]
        embs = cache.get_or_compute(model_name, task_name, all_texts, model, device=device)

        n_corpus = len(task_data["corpus_texts"])
        corpus_embs_full = embs[:n_corpus]
        query_embs_full = embs[n_corpus:]
        relevant_docs = task_data["relevant_docs"]

        print(f"    Corpus: {n_corpus}, Queries: {len(query_embs_full)}")

        task_results = {"methods": {}}

        for target_dim in target_dims:
            dim_results = {}

            for method in ['full', 'random_coord', 'pca', 'random_proj']:
                comp_dim = model_dim if method == 'full' else target_dim
                corpus_comp = np.ascontiguousarray(apply_compression(corpus_embs_full, method, target_dim, model_dim), dtype=np.float32)
                query_comp = np.ascontiguousarray(apply_compression(query_embs_full, method, target_dim, model_dim), dtype=np.float32)

                faiss.normalize_L2(corpus_comp)
                faiss.normalize_L2(query_comp)

                bench = benchmark_retrieval(corpus_comp, query_comp, relevant_docs)
                dim_results[method] = bench

                print(f"    dim={target_dim} {method}: "
                      f"nDCG@10={bench['ndcg_at_10']:.3f} "
                      f"Recall@10={bench['recall_at_10']:.3f} "
                      f"P50={bench['latency_p50_ms']:.1f}ms")

            task_results["methods"][str(target_dim)] = dim_results

        results["tasks"][task_name] = task_results
        print(f"    Done in {time.time()-t0:.1f}s")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5', 'bge-m3',
                                 'roberta-large-InBedder'])
    parser.add_argument('--target-dims', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
        'roberta-large-InBedder': '/home/linkco/exa/models/inbedder-roberta-large',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
    }

    print("Loading task data...")
    task_data_map = {}
    for task_name in RETRIEVAL_TASKS:
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
        results = run_retrieval_cost(
            model_name, model_path, task_data_map,
            target_dims=args.target_dims, device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"retrieval_cost_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
