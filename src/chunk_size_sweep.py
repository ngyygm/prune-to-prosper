#!/usr/bin/env python3
"""
Chunk-size sensitivity sweep: test whether conclusions change with chunk granularity.

Tests w ∈ {1, 2, 4, 8, 16} for 5 representative models on a subset of tasks.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/chunk_size_sweep.py --models gte-large-en-v1.5 bge-m3
    CUDA_VISIBLE_DEVICES=2 python src/chunk_size_sweep.py --models roberta-large roberta-large-InBedder stella_en_400M_v5
"""

import os
import sys
import argparse
import json
import random
import gc
import time
import numpy as np
import torch
import mteb
from tqdm import tqdm
from datasets import DatasetDict
from sentence_transformers import SentenceTransformer

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

# Representative task subset (14 tasks, 2 per category)
REPRESENTATIVE_TASKS = [
    'ImdbClassification', 'Banking77Classification',
    'TwentyNewsgroupsClustering', 'MedrxivClusteringS2S',
    'NFCorpus', 'ArguAna',
    'SciDocsRR', 'StackOverflowDupQuestions',
    'STSBenchmark', 'BIOSSES',
    'SprintDuplicateQuestions', 'TwitterURLCorpus',
    'SummEval',
    'SciFact',
]

TASK_CATEGORIES = {
    "Clustering": ['BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering'],
    "PairClassification": ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'],
    "Reranking": ['AskUbuntuDupQuestions', 'SciDocsRR', 'StackOverflowDupQuestions'],
    "STS": ['BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STSBenchmark'],
    "Summarization": ['SummEval'],
    "Retrieval": ['ArguAna', 'CQADupstackEnglishRetrieval', 'NFCorpus', 'SCIDOCS', 'SciFact'],
}

TASK_TYPE_MAP = {}
for cat, tasks in TASK_CATEGORIES.items():
    for t in tasks:
        TASK_TYPE_MAP[t] = cat


class MyModel:
    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}

    def get_dim(self):
        return self.model.encode("hello", convert_to_tensor=True).shape[-1]

    def set_config(self, batch_size=4, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def'):
        self.batch_size = batch_size
        self.emb_len = emb_len
        self.seed = seed
        self.win_size = win_size
        self.chunk_ids = chunk_ids
        self.dtype = dtype

    def encode(self, input_texts, **kwargs):
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device

        results = [None] * len(input_texts)
        texts_to_encode = []
        idx_to_encode = []

        for i, text in enumerate(input_texts):
            if text in self.cache:
                results[i] = self.cache[text]
            else:
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            new_embeddings = self.model.encode(texts_to_encode, max_len=1024, **kwargs)
            for i, text, emb in zip(idx_to_encode, texts_to_encode, new_embeddings):
                self.cache[text] = emb.detach()
                results[i] = emb.detach()

        embeddings = torch.stack(results, dim=0)
        embeddings = get_texts_embeddings(
            embeddings, self.get_dim(),
            emb_len=self.emb_len, seed=self.seed,
            win_size=self.win_size, chunk_ids=self.chunk_ids,
            dtype=self.dtype, device=self.device,
        )
        return embeddings.detach().to(torch.float32).cpu().numpy()

    def clear_cache(self):
        self.cache = {}


def get_texts_embeddings(embeddings, model_dim, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def', device="cuda:0"):
    if dtype == 'def':
        return embeddings
    elif dtype == 'random':
        return get_texts_embeddings_random(embeddings, model_dim, emb_len=emb_len, seed=seed, device=device)
    elif dtype == 'chunk':
        return get_texts_embeddings_chunk(embeddings, win_size=win_size, chunk_ids=chunk_ids)


def get_texts_embeddings_random(embeddings, model_dim, emb_len=64, seed=42, device="cuda:0"):
    g = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randperm(model_dim, generator=g, device=device)[:emb_len]
    return embeddings.index_select(1, indices)


def get_texts_embeddings_chunk(embeddings, win_size=64, chunk_ids=[0]):
    if not chunk_ids:
        return torch.empty((embeddings.shape[0], 0), device=embeddings.device)
    chunks = [embeddings[:, i * win_size:(i + 1) * win_size] for i in chunk_ids]
    return torch.cat(chunks, dim=1)


def is_retrieval_task(task_name):
    return TASK_TYPE_MAP.get(task_name) == "Retrieval"


def is_pair_classification_task(task_name):
    return TASK_TYPE_MAP.get(task_name) == "PairClassification"


def load_all_tasks(task_names=None):
    """Load all MTEB tasks and filter to specified subset."""
    if task_names is None:
        task_names = REPRESENTATIVE_TASKS
    all_tasks = mteb.get_tasks(tasks=task_names)
    task_map = {}
    for t in all_tasks:
        task_map[t.metadata.name] = t
    return task_map


def evaluate_model_task(model, task, batch_size=8,
                        win_size=2, chunk_ids=None, dtype='chunk',
                        emb_len=None, seed=None):
    """Run MTEB evaluation and return main_score * 100."""
    model.set_config(
        batch_size=batch_size,
        emb_len=emb_len or 64,
        seed=seed or 42,
        win_size=win_size,
        chunk_ids=chunk_ids or [0],
        dtype=dtype,
    )

    evaluation = mteb.MTEB(tasks=[task])
    try:
        results = evaluation.run(model, output_folder=None, verbosity=0)
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


def run_chunk_sweep(model_name, model_path, task_map, win_sizes, budget_sizes, n_random=10, device='cuda:0'):
    """Run chunk-size sweep for one model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    model = MyModel(model_path, device=device)
    model_dim = model.get_dim()
    print(f"Dimension: {model_dim}")

    results = {
        "model": model_name,
        "model_dim": model_dim,
        "win_sizes": win_sizes,
        "budget_sizes": budget_sizes,
        "tasks": {},
    }

    for task_name in REPRESENTATIVE_TASKS:
        if task_name not in task_map:
            print(f"\n  Task {task_name} not found, skipping")
            continue

        task = task_map[task_name]
        print(f"\n  Task: {task_name}")

        # Get baseline (full-dim)
        model.clear_cache()
        baseline_score = evaluate_model_task(model, task, dtype='def')
        if baseline_score is None:
            print(f"    Baseline failed, skipping")
            continue
        print(f"    Baseline: {baseline_score:.2f}")

        task_results = {
            "baseline": baseline_score,
            "win_sizes": {},
        }

        for w in win_sizes:
            n_chunks = model_dim // w
            if n_chunks < max(budget_sizes):
                print(f"    w={w}: n_chunks={n_chunks}, skipping budgets > {n_chunks}")
                usable_budgets = [b for b in budget_sizes if b <= n_chunks]
            else:
                usable_budgets = budget_sizes

            w_results = {"budgets": {}}

            # Score each chunk
            chunk_scores = []
            print(f"    w={w}: scoring {n_chunks} chunks...")
            for chunk_id in tqdm(range(n_chunks), desc=f"    w={w}", leave=False):
                model.clear_cache()
                score = evaluate_model_task(
                    model, task, win_size=w, chunk_ids=[chunk_id], dtype='chunk'
                )
                chunk_scores.append(score if score else 0.0)

            chunk_scores = np.array(chunk_scores)
            sorted_indices = np.argsort(chunk_scores)[::-1]

            for budget in usable_budgets:
                n_select = budget // w  # number of chunks to select

                # Oracle: top chunks
                oracle_ids = sorted_indices[:n_select].tolist()
                model.clear_cache()
                oracle_score = evaluate_model_task(
                    model, task, win_size=w, chunk_ids=oracle_ids, dtype='chunk'
                )

                # Anti-oracle: bottom chunks
                anti_ids = sorted_indices[-n_select:].tolist()
                model.clear_cache()
                anti_score = evaluate_model_task(
                    model, task, win_size=w, chunk_ids=anti_ids, dtype='chunk'
                )

                # Random: average of n_random trials
                random_scores = []
                for seed in range(n_random):
                    rng = np.random.RandomState(seed + 100)
                    rand_ids = rng.choice(n_chunks, size=n_select, replace=False).tolist()
                    model.clear_cache()
                    rs = evaluate_model_task(
                        model, task, win_size=w, chunk_ids=rand_ids, dtype='chunk'
                    )
                    if rs is not None:
                        random_scores.append(rs)

                random_mean = np.mean(random_scores) if random_scores else None

                if oracle_score and baseline_score > 0:
                    oracle_ret = oracle_score / baseline_score * 100
                    anti_ret = anti_score / baseline_score * 100 if anti_score else None
                    random_ret = random_mean / baseline_score * 100 if random_mean else None
                    gap = oracle_ret - random_ret if (oracle_ret and random_ret) else None
                else:
                    oracle_ret = anti_ret = random_ret = gap = None

                w_results["budgets"][str(budget)] = {
                    "oracle_score": oracle_score,
                    "anti_score": anti_score,
                    "random_mean": random_mean,
                    "oracle_retention": oracle_ret,
                    "anti_retention": anti_ret,
                    "random_retention": random_ret,
                    "oracle_random_gap": gap,
                    "chunk_scores": chunk_scores.tolist(),
                }

                print(f"      budget={budget}: oracle={oracle_ret:.1f}% random={random_ret:.1f}% gap={gap:.2f}%")

            # Compute entropy for this w
            if np.sum(chunk_scores) > 0:
                p = chunk_scores / np.sum(chunk_scores)
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p)) / np.log2(len(chunk_scores))
                gini = np.sum(np.abs(np.subtract.outer(chunk_scores, chunk_scores))) / (2 * len(chunk_scores) * np.sum(chunk_scores))
            else:
                entropy = 0
                gini = 0

            w_results["entropy"] = float(entropy)
            w_results["gini"] = float(gini)
            task_results["win_sizes"][str(w)] = w_results

        results["tasks"][task_name] = task_results

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'bge-m3', 'stella_en_400M_v5',
                                 'roberta-large-InBedder', 'roberta-large'])
    parser.add_argument('--win-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--budgets', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--n-random', type=int, default=10)
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Override task list (default: REPRESENTATIVE_TASKS)')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large-InBedder': '/home/linkco/exa/models/roberta-large-InBedder',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
    }

    # Load tasks once
    print("Loading tasks...")
    task_names = args.tasks if args.tasks else REPRESENTATIVE_TASKS
    task_map = load_all_tasks(task_names)
    print(f"Loaded {len(task_map)} tasks: {list(task_map.keys())}")

    for model_name in args.models:
        if model_name not in MODEL_PATHS:
            print(f"Unknown model: {model_name}")
            continue

        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path):
            print(f"Model path not found: {model_path}")
            continue

        results = run_chunk_sweep(
            model_name, model_path, task_map,
            win_sizes=args.win_sizes,
            budget_sizes=args.budgets,
            n_random=args.n_random,
            device=args.device,
        )

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"chunk_size_sweep_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
