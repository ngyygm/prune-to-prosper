#!/usr/bin/env python3
"""
Fast PCA and Random Projection baselines using pre-computed embeddings.

Compares coordinate selection vs transformation-based compression (PCA, RP).
Uses pre-computed embeddings — encode once, apply transforms directly.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/pca_baseline_fast.py \
        --models gte-large-en-v1.5 stella_en_400M_v5 --device cuda:0
"""

import os
import argparse
import json
import gc
import time
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from chunk_size_sweep_fast import (
    load_task_data, evaluate_with_dims,
    compute_classification_accuracy, compute_sts_spearman,
    compute_clustering_vmeasure, compute_retrieval_ndcg,
    cosine_similarity,
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


def evaluate_with_transform(embs, task_data, transformed_embs_fn):
    """Evaluate task performance using transformed embeddings."""
    eval_fn = task_data["eval_fn"]

    # Get transformed embeddings (all at once)
    transformed = transformed_embs_fn(embs)

    if eval_fn == "classification":
        return compute_classification_accuracy(transformed, task_data["labels"])
    elif eval_fn == "sts":
        n = len(task_data["gold_scores"])
        embs1 = transformed[:n]
        embs2 = transformed[n:2*n]
        return compute_sts_spearman(embs1, embs2, task_data["gold_scores"])
    elif eval_fn == "clustering":
        return compute_clustering_vmeasure(transformed, task_data["labels"])
    elif eval_fn == "retrieval":
        n_corpus = len(task_data["corpus_texts"])
        corpus_embs = transformed[:n_corpus]
        query_embs = transformed[n_corpus:]
        return compute_retrieval_ndcg(query_embs, corpus_embs, task_data["relevant_docs"])
    return 0.0


def run_pca_baseline(model_name, model_path, task_data_map, target_dims,
                     device='cuda:0', cache_dir='data/embeddings_cache'):
    """Run PCA/RP baselines for one model."""
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
        "target_dims": target_dims,
        "methods": {},
    }

    # Fit PCA on sample texts
    print("Fitting PCA on sample texts...")
    sample_texts = [
        "This is a sample sentence for fitting PCA.",
        "Another example text to capture embedding distribution.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process high-dimensional vectors.",
        "Text embeddings capture semantic meaning of sentences.",
    ] * 60  # 300 samples — enough for PCA up to 300 components
    sample_embs = model.encode(sample_texts, convert_to_tensor=False)
    sample_embs = np.array(sample_embs, dtype=np.float32)

    pca_models = {}
    for td in target_dims:
        n_comp = min(td, sample_embs.shape[0], sample_embs.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(sample_embs)
        pca_models[td] = pca
        print(f"  PCA-{n_comp}: explained variance = {sum(pca.explained_variance_ratio_):.4f}")

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

        # Full-dim baseline
        baseline = evaluate_with_dims(embs, task_data, list(range(model_dim)))
        if baseline == 0:
            print(f"    Baseline=0, skipping")
            continue
        print(f"    Baseline: {baseline:.2f}")

        task_results = {"baseline": baseline}

        for target_dim in target_dims:
            dim_results = {}

            # PCA
            pca = pca_models[target_dim]
            pca_fn = lambda e, p=pca: p.transform(e)
            pca_score = evaluate_with_transform(embs, task_data, pca_fn)
            dim_results["pca_score"] = pca_score
            dim_results["pca_retention"] = pca_score / baseline * 100

            # Random Projection (average of 5 seeds)
            rp_scores = []
            for seed in range(5):
                rng = np.random.RandomState(seed + 42)
                proj = rng.randn(model_dim, target_dim) / np.sqrt(target_dim)
                rp_fn = lambda e, pr=proj: e @ pr
                rp_score = evaluate_with_transform(embs, task_data, rp_fn)
                rp_scores.append(rp_score)
            rp_mean = np.mean(rp_scores)
            dim_results["rp_mean_score"] = rp_mean
            dim_results["rp_retention"] = rp_mean / baseline * 100

            # Random coordinate selection (compute directly)
            n_select = target_dim // 1  # each coord is dim=1
            rng = np.random.RandomState(42)
            rand_dims = rng.choice(model_dim, size=target_dim, replace=False).tolist()
            rand_score = evaluate_with_dims(embs, task_data, rand_dims)
            dim_results["random_coord_score"] = rand_score
            dim_results["random_coord_retention"] = rand_score / baseline * 100

            print(f"    dim={target_dim}: PCA={dim_results['pca_retention']:.1f}% "
                  f"RP={dim_results['rp_retention']:.1f}% "
                  f"RandCoord={dim_results['random_coord_retention']:.1f}%")

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
                                 'roberta-large', 'roberta-large-InBedder', 'bge-m3'])
    parser.add_argument('--target-dims', nargs='+', type=int, default=[64, 128, 256])
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
            print(f"  {task_name}: {td['eval_fn']}")
    print(f"Loaded {len(task_data_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        t0 = time.time()
        results = run_pca_baseline(
            model_name, model_path, task_data_map,
            target_dims=args.target_dims, device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"pca_baseline_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
