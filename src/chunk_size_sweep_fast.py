#!/usr/bin/env python3
"""
Fast chunk-size sweep using pre-computed embeddings.

Instead of running full MTEB evaluation per chunk (10 min each),
we:
  1. Encode all texts once (~5 min)
  2. For each chunk/window: slice embeddings, recompute metrics (~seconds)

Uses task-specific metric functions from MTEB metadata.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/chunk_size_sweep_fast.py \
        --models gte-large-en-v1.5 bge-m3 stella_en_400M_v5 \
        --win-sizes 4 8 16 --tasks ImdbClassification STSBenchmark \
        --device cuda:0
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
from embedding_cache import EmbeddingCache

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"


def cosine_similarity(a, b):
    """Compute cosine similarity between two sets of vectors."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def compute_classification_accuracy(embs, labels, test_size=0.2, seed=42):
    """kNN classification accuracy using cosine similarity."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    n = len(labels)
    if n < 20:
        return 0.0

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels if len(set(labels)) > 1 else None)

    if len(train_idx) < 5 or len(test_idx) < 5:
        return 0.0

    train_embs = embs[train_idx]
    test_embs = embs[test_idx]
    train_labels = np.array(labels)[train_idx]
    test_labels = np.array(labels)[test_idx]

    clf = KNeighborsClassifier(n_neighbors=min(5, len(train_idx)), metric='cosine')
    clf.fit(train_embs, train_labels)
    return clf.score(test_embs, test_labels) * 100


def compute_sts_spearman(embs1, embs2, gold_scores):
    """Spearman correlation for STS tasks."""
    sim = np.array([cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0, 0]
                    for e1, e2 in zip(embs1, embs2)])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(sim, gold_scores)
    return rho * 100


def compute_clustering_vmeasure(embs, labels, n_clusters=None):
    """V-measure for clustering tasks."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import v_measure_score

    if n_clusters is None:
        n_clusters = len(set(labels))
    n_clusters = min(n_clusters, len(embs) // 2)
    if n_clusters < 2:
        return 0.0

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=1, max_iter=100)
    pred = km.fit_predict(embs)
    return v_measure_score(labels, pred) * 100


def compute_retrieval_ndcg(query_embs, corpus_embs, relevant_docs, k=10):
    """nDCG@10 for retrieval tasks."""
    # Compute similarity matrix
    sim = cosine_similarity(query_embs, corpus_embs)

    ndcgs = []
    for qi in range(len(query_embs)):
        if qi not in relevant_docs or not relevant_docs[qi]:
            continue

        top_k = np.argsort(sim[qi])[::-1][:k]
        rel = relevant_docs[qi]

        dcg = sum(1.0 / np.log2(i + 2) for i, doc in enumerate(top_k) if doc in rel)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return np.mean(ndcgs) * 100 if ndcgs else 0.0


def load_retrieval_task_direct(task_name, task=None):
    """Load retrieval task data directly from HuggingFace datasets.

    Fallback for when MTEB's load_data() fails due to cache conflicts.
    """
    from datasets import load_dataset

    # Map task names to HuggingFace dataset IDs
    DATASET_MAP = {
        'NFCorpus': 'mteb/nfcorpus',
        'ArguAna': 'mteb/arguana',
        'SciFact': 'mteb/scifact',
        'FiQA2018': 'mteb/fiqa',
        'QuoraRetrieval': 'mteb/quora',
        'TRECCOVID': 'mteb/trec-covid',
    }

    hf_id = DATASET_MAP.get(task_name)
    if not hf_id:
        print(f"    No direct load mapping for {task_name}")
        return None

    # MTEB retrieval datasets use config-based splits: corpus, queries, default (qrels)
    try:
        corpus = load_dataset(hf_id, 'corpus', split='corpus', trust_remote_code=True)
    except Exception:
        try:
            corpus = load_dataset(hf_id, 'corpus', split='train', trust_remote_code=True)
        except Exception:
            try:
                corpus = load_dataset(hf_id, 'corpus', split='test', trust_remote_code=True)
            except Exception as e:
                print(f"    Cannot load corpus for {task_name}: {e}")
                return None

    try:
        queries = load_dataset(hf_id, 'queries', split='queries', trust_remote_code=True)
    except Exception:
        try:
            queries = load_dataset(hf_id, 'queries', split='train', trust_remote_code=True)
        except Exception:
            try:
                queries = load_dataset(hf_id, 'queries', split='test', trust_remote_code=True)
            except Exception as e:
                print(f"    Cannot load queries for {task_name}: {e}")
                return None

    # Load qrels from default config
    qrels = None
    for split in ['test', 'train', 'default']:
        try:
            qrels = load_dataset(hf_id, 'default', split=split, trust_remote_code=True)
            break
        except Exception:
            continue

    corpus_texts = [item.get('text', '') for item in corpus]
    corpus_ids = [item.get('_id', item.get('id', str(i))) for i, item in enumerate(corpus)]
    query_texts = [item.get('text', '') for item in queries]
    query_ids = [item.get('_id', item.get('id', str(i))) for i, item in enumerate(queries)]

    c2i = {k: i for i, k in enumerate(corpus_ids)}
    q2i = {k: i for i, k in enumerate(query_ids)}

    # Build relevance mapping
    rel_idx = {}
    if qrels is not None:
        for item in qrels:
            qid = item.get('query-id', '')
            cid = item.get('corpus-id', '')
            qi = q2i.get(qid)
            ci = c2i.get(cid)
            if qi is not None and ci is not None:
                if qi not in rel_idx:
                    rel_idx[qi] = set()
                rel_idx[qi].add(ci)
    elif task and hasattr(task, 'relevant_docs') and task.relevant_docs:
        for qi, qk in enumerate(query_ids):
            if qk in task.relevant_docs:
                rel_idx[qi] = set(c2i[dk] for dk in task.relevant_docs[qk] if dk in c2i)

    if not rel_idx:
        print(f"    {task_name}: no relevance judgments found")
        return None

    # Limit corpus size for tractability
    max_corpus = 5000
    if len(corpus_texts) > max_corpus:
        corpus_texts = corpus_texts[:max_corpus]
        rel_idx_filtered = {}
        for qi, docs in rel_idx.items():
            filtered = {d for d in docs if d < max_corpus}
            if filtered:
                rel_idx_filtered[qi] = filtered
        rel_idx = rel_idx_filtered

    result = {
        "task_type": "Retrieval",
        "task_name": task_name,
        "corpus_texts": corpus_texts,
        "query_texts": query_texts,
        "relevant_docs": rel_idx,
        "eval_fn": "retrieval",
    }
    print(f"    {task_name}: direct-loaded {len(corpus_texts)} corpus, {len(query_texts)} queries, {len(rel_idx)} qrels")
    return result


def load_task_data(task_name):
    """Load task data from MTEB."""
    import mteb

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        return None
    task = tasks[0]

    try:
        task.load_data()
    except Exception as e:
        print(f"    Failed to load {task_name}: {e}")
        return None

    if not task.data_loaded or task.dataset is None:
        # For retrieval tasks, try direct loading as fallback
        task_type = task.metadata.type
        if task_type == 'Retrieval':
            return load_retrieval_task_direct(task_name, task)
        print(f"    {task_name}: data not available after load_data()")
        return None

    dataset = task.dataset
    task_type = task.metadata.type

    # Get test split
    if hasattr(dataset, 'keys'):
        splits = list(dataset.keys())
        split_name = 'test' if 'test' in splits else splits[-1]
        data = dataset[split_name]
    else:
        data = dataset

    items = list(data) if not isinstance(data, list) else data
    if not items:
        return None

    result = {"task_type": task_type, "task_name": task_name}

    if task_type == 'Classification':
        result["texts"] = [item.get('text', '') for item in items]
        result["labels"] = [item.get('label', 0) for item in items]
        result["eval_fn"] = "classification"

    elif task_type == 'STS':
        result["texts1"] = [item.get('sentence1', '') for item in items]
        result["texts2"] = [item.get('sentence2', '') for item in items]
        result["gold_scores"] = [float(item.get('score', 0.0)) for item in items]
        result["eval_fn"] = "sts"

    elif task_type == 'Clustering':
        # Handle both flat format and cluster-level format
        if 'text' in items[0]:
            result["texts"] = [item.get('text', '') for item in items]
            result["labels"] = [item.get('label', 0) for item in items]
        elif 'sentences' in items[0]:
            # Cluster-level: {'sentences': [...], 'labels': [...]}
            texts, labels = [], []
            for item in items:
                sents = item.get('sentences', [])
                labs = item.get('labels', [])
                for s, l in zip(sents, labs):
                    texts.append(s)
                    labels.append(l)
            result["texts"] = texts
            result["labels"] = labels
        else:
            print(f"    Unknown clustering format for {task_name}")
            return None
        result["eval_fn"] = "clustering"

    elif task_type == 'Retrieval':
        # Retrieval tasks have corpus/queries structure
        try:
            if hasattr(data, 'corpus'):
                corpus_dict = data.corpus
                queries_dict = data.queries
                relevant_docs = data.relevant_docs
            else:
                print(f"    Retrieval {task_name}: non-standard format, skipping")
                return None

            corpus_texts = [corpus_dict[k].get('text', '') for k in corpus_dict]
            query_texts = [queries_dict[k].get('text', '') for k in queries_dict]
            corpus_keys = list(corpus_dict.keys())
            query_keys = list(queries_dict.keys())
            c2i = {k: i for i, k in enumerate(corpus_keys)}

            rel_idx = {}
            for qi, qk in enumerate(query_keys):
                if qk in relevant_docs:
                    rel_idx[qi] = set(c2i[dk] for dk in relevant_docs[qk] if dk in c2i)

            result["corpus_texts"] = corpus_texts[:5000]
            result["query_texts"] = query_texts
            result["relevant_docs"] = rel_idx
            result["eval_fn"] = "retrieval"
        except Exception as e:
            print(f"    Retrieval {task_name} error: {e}")
            return None

    elif task_type in ('Reranking', 'PairClassification'):
        texts, labels = [], []
        for item in items:
            t1 = item.get('sentence1', item.get('text', ''))
            t2 = item.get('sentence2', '')
            label = item.get('label', 0)
            texts.append(f"{t1} [SEP] {t2}")
            labels.append(label)
        result["texts"] = texts
        result["labels"] = labels
        result["eval_fn"] = "classification"

    else:
        print(f"    Unsupported task type: {task_type}")
        return None

    return result


def evaluate_with_dims(embs_full, task_data, dim_indices):
    """Evaluate task performance using selected dimensions."""
    eval_fn = task_data["eval_fn"]

    if eval_fn == "classification":
        embs = embs_full[:, dim_indices]
        return compute_classification_accuracy(embs, task_data["labels"])

    elif eval_fn == "sts":
        n = len(task_data["gold_scores"])
        dim = len(dim_indices)
        embs1 = embs_full[:n, dim_indices]
        embs2 = embs_full[n:2*n, dim_indices]
        return compute_sts_spearman(embs1, embs2, task_data["gold_scores"])

    elif eval_fn == "clustering":
        embs = embs_full[:, dim_indices]
        return compute_clustering_vmeasure(embs, task_data["labels"])

    elif eval_fn == "retrieval":
        n_corpus = task_data.get("n_corpus")
        if n_corpus is None:
            n_corpus = len(task_data["corpus_texts"])
        corpus_embs = embs_full[:n_corpus, dim_indices]
        query_embs = embs_full[n_corpus:, dim_indices]
        return compute_retrieval_ndcg(query_embs, corpus_embs, task_data["relevant_docs"])

    return 0.0


def run_sweep(model_name, model_path, task_data_map, win_sizes, budget_sizes,
              n_random=10, device='cuda:0', cache_dir='data/embeddings_cache'):
    """Run fast chunk-size sweep for one model."""
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
        "win_sizes": win_sizes,
        "budget_sizes": budget_sizes,
        "tasks": {},
    }

    for task_name, task_data in task_data_map.items():
        print(f"\n  Task: {task_name}")
        t0 = time.time()

        # Encode all texts once (with cache)
        eval_fn = task_data["eval_fn"]
        if eval_fn == "sts":
            all_texts = task_data["texts1"] + task_data["texts2"]
        elif eval_fn == "retrieval":
            all_texts = task_data["corpus_texts"] + task_data["query_texts"]
        else:
            all_texts = task_data["texts"]

        embs = cache.get_or_compute(model_name, task_name, all_texts, model, device=device)
        print(f"    Embeddings: {embs.shape}, took {time.time()-t0:.1f}s")

        # Baseline (full dim)
        baseline = evaluate_with_dims(embs, task_data, list(range(model_dim)))
        print(f"    Baseline: {baseline:.2f}%")

        task_results = {
            "baseline": baseline,
            "n_texts": len(all_texts),
            "encode_time_s": time.time() - t0,
            "win_sizes": {},
        }

        for w in win_sizes:
            n_chunks = model_dim // w
            if n_chunks < max(budget_sizes):
                usable_budgets = [b for b in budget_sizes if b <= n_chunks]
            else:
                usable_budgets = budget_sizes

            if not usable_budgets:
                continue

            wt0 = time.time()

            # Score each chunk (fast: just slice + evaluate)
            chunk_scores = []
            for cid in range(n_chunks):
                dims = list(range(cid * w, (cid + 1) * w))
                score = evaluate_with_dims(embs, task_data, dims)
                chunk_scores.append(score)

            chunk_scores = np.array(chunk_scores)
            sorted_idx = np.argsort(chunk_scores)[::-1]

            w_results = {"budgets": {}, "chunk_scores": chunk_scores.tolist()}

            for budget in usable_budgets:
                n_select = budget // w

                # Oracle: top chunks
                oracle_dims = []
                for cid in sorted_idx[:n_select]:
                    oracle_dims.extend(range(cid * w, (cid + 1) * w))
                oracle_score = evaluate_with_dims(embs, task_data, oracle_dims)

                # Anti-oracle: bottom chunks
                anti_dims = []
                for cid in sorted_idx[-n_select:]:
                    anti_dims.extend(range(cid * w, (cid + 1) * w))
                anti_score = evaluate_with_dims(embs, task_data, anti_dims)

                # Random: average of n_random trials
                random_scores = []
                for seed in range(n_random):
                    rng = np.random.RandomState(seed + 100)
                    rand_ids = rng.choice(n_chunks, size=n_select, replace=False)
                    rand_dims = []
                    for cid in rand_ids:
                        rand_dims.extend(range(cid * w, (cid + 1) * w))
                    rs = evaluate_with_dims(embs, task_data, rand_dims)
                    random_scores.append(rs)
                random_mean = np.mean(random_scores)

                oracle_ret = oracle_score / baseline * 100 if baseline > 0 else None
                random_ret = random_mean / baseline * 100 if baseline > 0 else None
                gap = oracle_ret - random_ret if oracle_ret and random_ret else None

                w_results["budgets"][str(budget)] = {
                    "oracle_score": oracle_score,
                    "anti_score": anti_score,
                    "random_mean": random_mean,
                    "oracle_retention": oracle_ret,
                    "random_retention": random_ret,
                    "gap": gap,
                }

                if oracle_ret is not None and random_ret is not None:
                    print(f"    w={w} b={budget}: oracle={oracle_ret:.1f}% random={random_ret:.1f}% gap={gap:.2f}pp")
                else:
                    print(f"    w={w} b={budget}: baseline=0, skip retention")

            # Entropy
            if np.sum(chunk_scores) > 0:
                p = chunk_scores / np.sum(chunk_scores)
                p = p[p > 0]
                entropy = -np.sum(p * np.log2(p)) / np.log2(len(chunk_scores))
            else:
                entropy = 0
            w_results["entropy"] = float(entropy)

            wt1 = time.time()
            w_results["time_s"] = wt1 - wt0
            print(f"    w={w}: done in {wt1-wt0:.1f}s (entropy={entropy:.3f})")

            task_results["win_sizes"][str(w)] = w_results

        results["tasks"][task_name] = task_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'bge-m3', 'stella_en_400M_v5',
                                 'roberta-large-InBedder', 'roberta-large'])
    parser.add_argument('--win-sizes', nargs='+', type=int, default=[2, 4, 8, 16])
    parser.add_argument('--budgets', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--n-random', type=int, default=10)
    parser.add_argument('--tasks', nargs='+',
                        default=['ImdbClassification', 'STSBenchmark',
                                 'TwentyNewsgroupsClustering', 'NFCorpus',
                                 'SciDocsRR', 'SprintDuplicateQuestions', 'SummEval'])
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'bge-m3': '/home/linkco/exa/models/bge-m3',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large-InBedder': '/home/linkco/exa/models/inbedder-roberta-large',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
    }

    # Load all task data upfront
    print("Loading task data...")
    task_data_map = {}
    for task_name in args.tasks:
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
            print(f"Skipping {model_name}")
            continue

        t0 = time.time()
        results = run_sweep(
            model_name, model_path, task_data_map,
            win_sizes=args.win_sizes,
            budget_sizes=args.budgets,
            n_random=args.n_random,
            device=args.device,
        )
        results["total_time_s"] = time.time() - t0

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"chunk_size_sweep_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path} (took {time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
