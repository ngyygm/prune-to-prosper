#!/usr/bin/env python3
"""
Train/test split experiment for oracle dimension selection generalization.

Splits each MTEB task's test data into 70% (rank) / 30% (holdout).
Ranks chunks on 70%, evaluates oracle vs random on 30%.

Tests whether oracle advantage comes from data leakage or genuine signal.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python src/train_test_split_experiment.py /path/to/model

    # 3 GPUs parallel (split tasks across GPUs)
    CUDA_VISIBLE_DEVICES=0 python src/train_test_split_experiment.py /path/to/model --task-range 0:8
    CUDA_VISIBLE_DEVICES=1 python src/train_test_split_experiment.py /path/to/model --task-range 8:16
    CUDA_VISIBLE_DEVICES=2 python src/train_test_split_experiment.py /path/to/model --task-range 16:25

    # Specific tasks only
    python src/train_test_split_experiment.py /path/to/model --tasks BIOSSES SICK-R
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

# ============================================================
# Non-Classification task categories
# ============================================================
TASK_CATEGORIES = {
    "Clustering": [
        'BiorxivClusteringS2S',
        'MedrxivClusteringS2S',
        'TwentyNewsgroupsClustering',
    ],
    "PairClassification": [
        'SprintDuplicateQuestions',
        'TwitterSemEval2015',
        'TwitterURLCorpus',
    ],
    "Reranking": [
        'AskUbuntuDupQuestions',
        'SciDocsRR',
        'StackOverflowDupQuestions',
    ],
    "STS": [
        'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15',
        'STS16', 'STS17', 'STSBenchmark',
    ],
    "Summarization": [
        'SummEval',
    ],
    "Retrieval": [
        'ArguAna', 'CQADupstackEnglishRetrieval',
        'NFCorpus', 'SCIDOCS', 'SciFact',
    ],
}

ALL_TASKS = []
TASK_TYPE_MAP = {}
for cat, tasks in TASK_CATEGORIES.items():
    for t in tasks:
        ALL_TASKS.append(t)
        TASK_TYPE_MAP[t] = cat


# ============================================================
# MyModel class (from rank_chunk_mteb.py)
# ============================================================
class MyModel:
    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}

        self.model_card_data = {
            "model_name": "MyModel",
            "description": "Custom embedding model",
            "version": "1.0",
        }

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
            embeddings,
            self.get_dim(),
            emb_len=self.emb_len,
            seed=self.seed,
            win_size=self.win_size,
            chunk_ids=self.chunk_ids,
            dtype=self.dtype,
            device=self.device,
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

    original_dim = embeddings.shape[1]
    max_requested_index = max(chunk_ids)
    max_possible_end_index = (max_requested_index + 1) * win_size

    if max_possible_end_index > original_dim:
        raise IndexError(
            f"Requested chunk_id {max_requested_index} (ending at index {max_possible_end_index}) "
            f"is out of bounds for embedding dimension {original_dim}."
        )

    chunks = [embeddings[:, i * win_size:(i + 1) * win_size] for i in chunk_ids]
    return torch.cat(chunks, dim=1)


# ============================================================
# Task type detection
# ============================================================
def is_retrieval_task(task_name):
    return TASK_TYPE_MAP.get(task_name) == "Retrieval"


def is_pair_classification_task(task_name):
    return TASK_TYPE_MAP.get(task_name) == "PairClassification"


# ============================================================
# Data splitting functions
# ============================================================
def split_standard_task(task, task_name, rank_ratio=0.7, seed=42):
    """
    Split test data for STS, Clustering, Reranking, Summarization.
    These tasks use DatasetDict with 'test' key, where each row is a data point.
    """
    test_data = task.dataset["test"]
    original_size = len(test_data)

    shuffled = test_data.shuffle(seed=seed)
    n_rank = int(len(shuffled) * rank_ratio)

    rank_data = shuffled.select(range(n_rank))
    holdout_data = shuffled.select(range(n_rank, len(shuffled)))

    # Add new split keys to the dataset dict
    task.dataset["test_rank"] = rank_data
    task.dataset["test_holdout"] = holdout_data

    return {
        "original_size": original_size,
        "rank_size": n_rank,
        "holdout_size": len(shuffled) - n_rank,
    }


def split_pair_classification_task(task, task_name, rank_ratio=0.7, seed=42):
    """
    Split PairClassification tasks.
    These have 1 row with list columns: sentence1, sentence2, labels (each ~10k-100k items).
    We split by index into the lists.
    """
    from datasets import Dataset

    test_data = task.dataset["test"]
    row = test_data[0]

    # Get list length
    n = len(row['sentence1'])
    original_size = n

    # Shuffle indices
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_rank = int(n * rank_ratio)
    rank_idx = indices[:n_rank]
    holdout_idx = indices[n_rank:]

    # Create new datasets by slicing the lists
    rank_data = Dataset.from_dict({
        'sentence1': [row['sentence1'][i] for i in rank_idx],
        'sentence2': [row['sentence2'][i] for i in rank_idx],
        'labels': [row['labels'][i] for i in rank_idx],
    })
    holdout_data = Dataset.from_dict({
        'sentence1': [row['sentence1'][i] for i in holdout_idx],
        'sentence2': [row['sentence2'][i] for i in holdout_idx],
        'labels': [row['labels'][i] for i in holdout_idx],
    })

    # Add new split keys
    task.dataset["test_rank"] = rank_data
    task.dataset["test_holdout"] = holdout_data

    return {
        "original_size": original_size,
        "rank_size": n_rank,
        "holdout_size": len(holdout_idx),
    }


def split_retrieval_task(task, task_name, rank_ratio=0.7, seed=42):
    """
    Split retrieval task by query IDs.
    Corpus is shared between rank and holdout.
    Queries and relevant_docs are split.
    """
    corpus = task.corpus["test"]
    queries = task.queries["test"]
    relevant_docs = task.relevant_docs["test"]

    original_query_count = len(queries)
    query_ids = list(queries.keys())

    rng = random.Random(seed)
    rng.shuffle(query_ids)

    n_rank = int(len(query_ids) * rank_ratio)
    rank_query_ids = set(query_ids[:n_rank])
    holdout_query_ids = set(query_ids[n_rank:])

    # Split queries
    rank_queries = {qid: queries[qid] for qid in rank_query_ids}
    holdout_queries = {qid: queries[qid] for qid in holdout_query_ids}

    # Split relevant_docs
    rank_relevant_docs = {qid: relevant_docs[qid] for qid in rank_query_ids if qid in relevant_docs}
    holdout_relevant_docs = {qid: relevant_docs[qid] for qid in holdout_query_ids if qid in relevant_docs}

    # Add custom splits (corpus is shared)
    task.corpus["test_rank"] = corpus
    task.queries["test_rank"] = rank_queries
    task.relevant_docs["test_rank"] = rank_relevant_docs

    task.corpus["test_holdout"] = corpus
    task.queries["test_holdout"] = holdout_queries
    task.relevant_docs["test_holdout"] = holdout_relevant_docs

    return {
        "original_size": original_query_count,
        "rank_size": n_rank,
        "holdout_size": len(query_ids) - n_rank,
    }


def split_task_data(task, task_name, rank_ratio=0.7, seed=42):
    """
    Unified dispatcher: split task data based on task type.
    Modifies task object in-place by adding custom splits.
    """
    if is_retrieval_task(task_name):
        return split_retrieval_task(task, task_name, rank_ratio, seed)
    elif is_pair_classification_task(task_name):
        return split_pair_classification_task(task, task_name, rank_ratio, seed)
    else:
        return split_standard_task(task, task_name, rank_ratio, seed)


# ============================================================
# MTEB task patching (prevent data reload)
# ============================================================
def patch_task(task):
    """Patch task to prevent data reload during evaluation."""
    task.data_loaded = True
    original_load_data = task.load_data
    task.load_data = lambda *a, **k: None
    task.dataset_transform = lambda *a, **k: None


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_split(model, task, split_name, batch_size=8,
                   win_size=2, chunk_ids=None, dtype='chunk',
                   emb_len=None, seed=None):
    """
    Run MTEB evaluation on a specific split.
    Returns main_score * 100.
    """
    model.set_config(
        batch_size=batch_size,
        emb_len=emb_len or 64,
        seed=seed or 42,
        win_size=win_size,
        chunk_ids=chunk_ids or [0],
        dtype=dtype,
    )

    max_retries = 2
    for attempt in range(max_retries):
        try:
            scores = task.evaluate(
                model,
                split=split_name,
                encode_kwargs={'batch_size': batch_size},
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM retry {attempt+1}/{max_retries}] Clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                import time; time.sleep(2)
                if attempt == max_retries - 1:
                    print(f"  [ERROR] OOM after {max_retries} retries: {e}")
                    return None
            else:
                raise
        except Exception as e:
            print(f"  [ERROR] evaluate failed: {e}")
            return None

    # scores is dict[HFSubset, ScoresDict]
    # For non-multilingual: scores["default"] or just scores
    if "default" in scores:
        main_score = scores["default"]["main_score"] * 100
    else:
        # Try first key
        first_key = list(scores.keys())[0]
        main_score = scores[first_key]["main_score"] * 100

    return main_score


def rank_chunks(model, task, split_name, model_dim, win_size=2, batch_size=8):
    """
    Evaluate each chunk individually on the given split.
    Returns list of scores, one per chunk.
    Failed chunks get retried once at the end.
    """
    n_chunks = model_dim // win_size
    chunk_scores = [None] * n_chunks

    for chunk_id in tqdm(range(n_chunks), desc="  Ranking chunks", leave=False):
        score = evaluate_split(
            model, task, split_name,
            batch_size=batch_size,
            win_size=win_size,
            chunk_ids=[chunk_id],
            dtype='chunk',
        )
        chunk_scores[chunk_id] = score

    # Retry failed chunks (OOM etc.)
    failed_ids = [i for i, s in enumerate(chunk_scores) if s is None]
    if failed_ids:
        print(f"  Retrying {len(failed_ids)} failed chunks...")
        torch.cuda.empty_cache()
        gc.collect()
        for chunk_id in failed_ids:
            score = evaluate_split(
                model, task, split_name,
                batch_size=max(1, batch_size // 2),
                win_size=win_size,
                chunk_ids=[chunk_id],
                dtype='chunk',
            )
            chunk_scores[chunk_id] = score

    # Fill remaining None with 0
    still_failed = sum(1 for s in chunk_scores if s is None)
    if still_failed:
        print(f"  WARNING: {still_failed} chunks still failed after retry, filling with 0")
    chunk_scores = [s if s is not None else 0.0 for s in chunk_scores]

    return chunk_scores


# ============================================================
# Main experiment logic
# ============================================================
def run_single_task(model, task_name, model_name, config):
    """
    Run the full train/test split experiment for one task.
    Returns result dict.
    """
    win_size = config['win_size']
    budget_sizes = config['budget_sizes']
    n_random_trials = config['n_random_trials']
    rank_ratio = config['rank_ratio']
    batch_size = config['batch_size']
    split_seed = config['split_seed']
    model_dim = model.get_dim()

    task_type = TASK_TYPE_MAP.get(task_name, "Unknown")

    result = {
        "task_name": task_name,
        "task_type": task_type,
        "model_dim": model_dim,
        "rank_size": 0,
        "holdout_size": 0,
        "original_size": 0,
        "chunk_scores_rank": [],
        "budgets": {},
    }

    # Step 1: Load task
    print(f"  Loading task: {task_name} ({task_type})")
    try:
        task = mteb.get_tasks(tasks=[task_name], languages=["eng"])[0]
        task.load_data()
    except Exception as e:
        print(f"  [ERROR] Failed to load task {task_name}: {e}")
        result["error"] = str(e)
        return result

    # Step 2: Split data and patch task
    try:
        split_info = split_task_data(task, task_name, rank_ratio, split_seed)
        result.update(split_info)
        print(f"  Split: {split_info['rank_size']} rank / {split_info['holdout_size']} holdout "
              f"(original: {split_info['original_size']})")
    except Exception as e:
        print(f"  [ERROR] Failed to split data: {e}")
        result["error"] = str(e)
        return result

    patch_task(task)

    # Step 3: Rank chunks on rank split
    print(f"  Ranking {model_dim // win_size} chunks on rank split...")
    t0 = time.time()
    chunk_scores = rank_chunks(model, task, "test_rank", model_dim, win_size, batch_size)
    rank_time = time.time() - t0
    result["chunk_scores_rank"] = chunk_scores
    print(f"  Ranking done in {rank_time:.1f}s")

    # Sort chunks by score (descending) → oracle ranking
    sorted_chunk_ids = sorted(
        range(len(chunk_scores)),
        key=lambda i: chunk_scores[i],
        reverse=True,
    )

    # Step 4: Evaluate on holdout split for each budget
    print(f"  Evaluating {len(budget_sizes)} budgets on holdout split...")
    for budget in budget_sizes:
        n_chunks_to_use = budget // win_size
        if n_chunks_to_use < 1 or n_chunks_to_use * win_size > model_dim:
            continue

        budget_key = str(budget)
        oracle_chunk_ids = sorted_chunk_ids[:n_chunks_to_use]

        # Oracle on holdout
        oracle_score = evaluate_split(
            model, task, "test_holdout",
            batch_size=batch_size,
            win_size=win_size,
            chunk_ids=oracle_chunk_ids,
            dtype='chunk',
        )

        # Random on holdout (multiple trials)
        random_scores = []
        for trial_seed in range(n_random_trials):
            r_score = evaluate_split(
                model, task, "test_holdout",
                batch_size=batch_size,
                emb_len=budget,
                seed=trial_seed,
                win_size=win_size,
                dtype='random',
            )
            if r_score is not None:
                random_scores.append(r_score)

        if random_scores:
            r_mean = float(np.mean(random_scores))
            r_std = float(np.std(random_scores))
        else:
            r_mean = None
            r_std = None

        oracle_advantage = (
            float(oracle_score - r_mean)
            if oracle_score is not None and r_mean is not None
            else None
        )

        result["budgets"][budget_key] = {
            "oracle_score": oracle_score,
            "random_scores": random_scores,
            "random_mean": r_mean,
            "random_std": r_std,
            "oracle_advantage": oracle_advantage,
            "n_chunks_used": n_chunks_to_use,
        }

        adv_str = f"{oracle_advantage:+.2f}" if oracle_advantage is not None else "N/A"
        o_str = f"{oracle_score:.2f}" if oracle_score is not None else "N/A"
        r_str = f"{r_mean:.2f}±{r_std:.2f}" if r_mean is not None else "N/A"
        print(f"    Budget {budget:>4d}: oracle={o_str}, "
              f"random={r_str}, advantage={adv_str}")

    return result


def run_experiment_for_model(model_path, config, task_list=None, task_range=None, task_indices=None):
    """Run experiment for a single model."""
    model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')
    output_dir = config['output_dir']

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Determine task list
    if task_list:
        tasks_to_run = task_list
    else:
        tasks_to_run = list(ALL_TASKS)

    if task_range:
        start, end = map(int, task_range.split(':'))
        tasks_to_run = tasks_to_run[start:end]

    if task_indices:
        indices = [int(x) for x in task_indices.split(',')]
        all_tasks = list(ALL_TASKS)
        tasks_to_run = [all_tasks[i] for i in indices if i < len(all_tasks)]

    print(f"Tasks to run ({len(tasks_to_run)}): {tasks_to_run}")

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MyModel(model_path, device)
    model_dim = model.get_dim()
    print(f"Model dim: {model_dim}")

    # Filter budgets by model dim
    valid_budgets = [b for b in config['budget_sizes'] if b < model_dim]
    config = dict(config)
    config['budget_sizes'] = valid_budgets

    # Check for existing results (resume support)
    result_file = os.path.join(output_dir, f"train_test_split_{model_name}.json")
    existing_results = {}
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            existing_data = json.load(f)
            existing_results = existing_data.get('tasks', {})
        print(f"Found existing results with {len(existing_results)} tasks. Resuming...")

    results = {"tasks": existing_results}

    def save_safe(results_dict, include_summary=False):
        """Save results, merging with on-disk data from other workers first."""
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    on_disk = json.load(f)
                for task_name, task_data in on_disk.get('tasks', {}).items():
                    if task_name not in results_dict["tasks"]:
                        results_dict["tasks"][task_name] = task_data
            except (json.JSONDecodeError, IOError):
                pass
        output = {
            "config": config,
            "model_name": model_name,
            "model_dim": model_dim,
            "tasks": results_dict["tasks"],
        }
        if include_summary:
            output["summary"] = compute_summary(results_dict["tasks"])
        os.makedirs(output_dir, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    # Run experiment for each task
    for task_name in tasks_to_run:
        if task_name in existing_results:
            print(f"\n  Skipping {task_name} (already done)")
            continue

        print(f"\n--- Task: {task_name} ---")
        task_result = run_single_task(model, task_name, model_name, config)
        results["tasks"][task_name] = task_result

        # Save after each task (crash-safe + race-safe)
        save_safe(results)
        print(f"  Saved to {result_file}")

    # Final save
    save_safe(results, include_summary=True)

    # Print summary
    print_summary(output)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return output


# ============================================================
# Summary statistics
# ============================================================
def compute_summary(tasks):
    """Compute summary statistics across all tasks."""
    summary = {
        "per_budget": {},
        "overall": {},
    }

    # Collect oracle advantages per budget
    budget_advantages = {}
    for task_name, task_data in tasks.items():
        if "error" in task_data:
            continue
        for budget_key, budget_data in task_data.get("budgets", {}).items():
            adv = budget_data.get("oracle_advantage")
            if adv is not None:
                budget_advantages.setdefault(budget_key, []).append({
                    "task": task_name,
                    "advantage": adv,
                    "oracle_score": budget_data["oracle_score"],
                    "random_mean": budget_data["random_mean"],
                })

    # Per-budget stats
    for budget_key, entries in sorted(budget_advantages.items(), key=lambda x: int(x[0])):
        advantages = [e["advantage"] for e in entries]
        summary["per_budget"][budget_key] = {
            "mean_oracle_advantage": float(np.mean(advantages)),
            "std_oracle_advantage": float(np.std(advantages)),
            "median_oracle_advantage": float(np.median(advantages)),
            "n_tasks": len(advantages),
            "oracle_wins": sum(1 for a in advantages if a > 0),
            "random_wins": sum(1 for a in advantages if a < 0),
            "ties": sum(1 for a in advantages if a == 0),
        }

    # Overall stats
    all_advantages = []
    for entries in budget_advantages.values():
        all_advantages.extend([e["advantage"] for e in entries])

    if all_advantages:
        summary["overall"] = {
            "mean_oracle_advantage": float(np.mean(all_advantages)),
            "std_oracle_advantage": float(np.std(all_advantages)),
            "median_oracle_advantage": float(np.median(all_advantages)),
            "total_comparisons": len(all_advantages),
            "oracle_wins_total": sum(1 for a in all_advantages if a > 0),
            "random_wins_total": sum(1 for a in all_advantages if a < 0),
        }

    return summary


def print_summary(output):
    """Print a human-readable summary table."""
    summary = output.get("summary", {})
    model_name = output.get("model_name", "unknown")
    model_dim = output.get("model_dim", 0)

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {model_name} (dim={model_dim})")
    print(f"  Split: {output.get('config', {}).get('rank_ratio', 0.7)*100:.0f}% rank / "
          f"{(1-output.get('config', {}).get('rank_ratio', 0.7))*100:.0f}% holdout")
    print(f"{'='*70}")

    per_budget = summary.get("per_budget", {})
    if not per_budget:
        print("  No results to summarize.")
        return

    print(f"  {'Budget':>7s} | {'Oracle':>7s} | {'Random':>7s} | {'Advantage':>9s} | {'Oracle Wins':>11s}")
    print(f"  {'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*11}")

    for budget_key in sorted(per_budget.keys(), key=int):
        b = per_budget[budget_key]
        # Compute mean scores from the raw task data
        mean_oracle = 0
        mean_random = 0
        count = 0
        for task_data in output.get("tasks", {}).values():
            if budget_key in task_data.get("budgets", {}):
                bd = task_data["budgets"][budget_key]
                if bd.get("oracle_score") is not None and bd.get("random_mean") is not None:
                    mean_oracle += bd["oracle_score"]
                    mean_random += bd["random_mean"]
                    count += 1
        if count > 0:
            mean_oracle /= count
            mean_random /= count

        wins = f"{b['oracle_wins']}/{b['n_tasks']}"
        adv = b['mean_oracle_advantage']
        print(f"  {budget_key:>7s} | {mean_oracle:7.2f} | {mean_random:7.2f} | {adv:+9.2f} | {wins:>11s}")

    overall = summary.get("overall", {})
    if overall:
        print(f"  {'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*11}")
        print(f"  Overall mean advantage: {overall['mean_oracle_advantage']:+.2f} ± {overall['std_oracle_advantage']:.2f}")
        print(f"  Oracle wins: {overall['oracle_wins_total']} / {overall['total_comparisons']} "
              f"({overall['oracle_wins_total']/overall['total_comparisons']*100:.1f}%)")
    print(f"{'='*70}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train/test split experiment for oracle dimension selection'
    )
    parser.add_argument('model_paths', nargs='+', help='Model path(s)')
    parser.add_argument('--output-dir', default='data/experiment_results',
                        help='Output directory for results')
    parser.add_argument('--win-size', type=int, default=2,
                        help='Chunk window size (default: 2)')
    parser.add_argument('--rank-ratio', type=float, default=0.7,
                        help='Ratio for rank split (default: 0.7)')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Random seed for data split (default: 42)')
    parser.add_argument('--budgets', nargs='+', type=int,
                        default=[2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512],
                        help='Budget sizes to test')
    parser.add_argument('--n-random-trials', type=int, default=10,
                        help='Number of random trials per (task, budget)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for encoding')
    parser.add_argument('--tasks', nargs='+', default=None,
                        help='Specific tasks to run (default: all non-Classification)')
    parser.add_argument('--task-range', type=str, default=None,
                        help='Task range, e.g., "0:8" (for splitting across GPUs)')
    parser.add_argument('--task-indices', type=str, default=None,
                        help='Comma-separated task indices, e.g., "0,2,4,6,8,10,12,14,16,18,20,22"')
    args = parser.parse_args()

    config = {
        'win_size': args.win_size,
        'rank_ratio': args.rank_ratio,
        'split_seed': args.split_seed,
        'budget_sizes': args.budgets,
        'n_random_trials': args.n_random_trials,
        'batch_size': args.batch_size,
        'output_dir': args.output_dir,
    }

    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Tasks available ({len(ALL_TASKS)}): {ALL_TASKS}")

    for model_path in args.model_paths:
        run_experiment_for_model(
            model_path, config,
            task_list=args.tasks,
            task_range=args.task_range,
            task_indices=args.task_indices,
        )


if __name__ == "__main__":
    main()
