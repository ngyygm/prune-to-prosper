"""
Universal Mask Experiment — Key Paper Experiment

Tests whether a single universal mask (derived from a subset of tasks) transfers
to held-out tasks. If successful, proves "task-aware pruning is unnecessary."

Experiment design:
1. Split 35 MTEB tasks into train (24) / test (11)
2. Derive "universal mask" = dimensions that are important across ALL train tasks
3. Evaluate universal mask on held-out tasks
4. Compare against: random, sequential, task-specific oracle, task-specific anti-oracle

Also tests across multiple budgets: k=64, 128, 256, 512

Usage:
  CUDA_VISIBLE_DEVICES=0 python universal_mask_experiment.py --model gte-large --gpu 0
  CUDA_VISIBLE_DEVICES=1 python universal_mask_experiment.py --model stella --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from itertools import combinations

def get_model_path(model_name):
    paths = {
        "gte-large": "/home/linkco/exa/models/gte-large-en-v1.5",
        "stella": "/home/linkco/exa/models/stella_en_400M_v5",
        "inbedder": "/home/linkco/exa/models/roberta-large-InBedder",
    }
    return paths.get(model_name)

def get_model_dim(model_name):
    return {"gte-large": 1024, "stella": 1024, "inbedder": 1024}[model_name]

def compute_retention(pruned_scores, full_scores):
    """Compute retention ratio per task, then average."""
    retentions = []
    for task in full_scores:
        if task in pruned_scores and full_scores[task] != 0:
            retentions.append(pruned_scores[task] / full_scores[task])
    return np.mean(retentions) if retentions else 0.0

def get_task_split():
    """Split 35 MTEB tasks into train (24) and test (11), stratified by category."""
    train_tasks = [
        # Classification (8 train)
        "AmazonCounterfactualClassification", "AmazonReviewsClassification",
        "Banking77Classification", "EmotionClassification",
        "ImdbClassification", "MassiveIntentClassification",
        "MassiveScenarioClassification", "MTOPDomainClassification",
        # Clustering (2 train)
        "BiorxivClusteringS2S", "MedrxivClusteringS2S",
        # Retrieval (3 train)
        "ArguAna", "NFCorpus", "SCIDOCS",
        # Reranking (2 train)
        "AskUbuntuDupQuestions", "SciDocsRR",
        # STS (5 train)
        "BIOSSES", "SICK-R", "STS12", "STS14", "STSBenchmark",
        # Pair classification (2 train)
        "SprintDuplicateQuestions", "TwitterSemEval2015",
        # Summarization (1 train)
        "SummEval",
        # Additional (1 train)
        "TweetSentimentExtractionClassification",
    ]
    test_tasks = [
        # Classification (3 test)
        "MTOPIntentClassification", "ToxicConversationsClassification",
        "TwitterURLCorpus",
        # Clustering (1 test)
        "TwentyNewsgroupsClustering",
        # Retrieval (2 test)
        "CQADupstackEnglishRetrieval", "SciFact",
        # Reranking (1 test)
        "StackOverflowDupQuestions",
        # STS (3 test)
        "STS13", "STS15", "STS16",
        # Pair classification (1 test)
        "TwitterURLCorpus",
    ]
    # De-duplicate test tasks
    test_tasks = list(dict.fromkeys(test_tasks))
    return train_tasks, test_tasks


def run_experiment(model_name, gpu_id, output_dir):
    """Run universal mask experiment for one model."""
    from sentence_transformers import SentenceTransformer
    import torch
    from mteb import MTEB

    print(f"\n{'='*60}")
    print(f"Universal Mask Experiment: {model_name}")
    print(f"GPU: {gpu_id}, Time: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    model_path = get_model_path(model_name)
    D = get_model_dim(model_name)

    # Load analyze data for this model
    analyze_dir = "/home/linkco/exa/llm-usefulEeb/Useful-Embedding/data/analyze"
    analyze_file = os.path.join(analyze_dir, f"{model_name.replace('gte-large', 'gte-large-en-v1.5').replace('stella', 'stella_en_400M_v5').replace('inbedder', 'roberta-large-InBedder')}.json")

    if not os.path.exists(analyze_file):
        print(f"WARNING: Analyze file not found at {analyze_file}")
        # Try alternative names
        for alt_name in [model_name, model_name.replace("-", "_"),
                         "gte-large-en-v1.5", "stella_en_400M_v5", "roberta-large-InBedder"]:
            alt_path = os.path.join(analyze_dir, f"{alt_name}.json")
            if os.path.exists(alt_path):
                analyze_file = alt_path
                print(f"Found at: {analyze_file}")
                break

    with open(analyze_file) as f:
        analyze_data = json.load(f)

    print(f"Loaded analyze data: {len(analyze_data.get('task_name', {}))} tasks")

    # Get chunk importance scores per task
    task_chunk_scores = {}
    for task_name, task_data in analyze_data.get("task_name", {}).items():
        if "2" in task_data.get("split_win_size", {}):
            scores = task_data["split_win_size"]["2"]["chunk_result"]
            task_chunk_scores[task_name] = np.array(scores)

    N_chunks = len(next(iter(task_chunk_scores.values())))  # 512

    # Define train/test split
    train_tasks, test_tasks = get_task_split()
    # Filter to tasks we have data for
    train_tasks = [t for t in train_tasks if t in task_chunk_scores]
    test_tasks = [t for t in test_tasks if t in task_chunk_scores]

    print(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")
    print(f"Train: {train_tasks[:5]}...")
    print(f"Test: {test_tasks}")

    results = {
        "model": model_name, "D": D, "N_chunks": N_chunks,
        "train_tasks": train_tasks, "test_tasks": test_tasks,
    }

    # Budgets to test
    budgets = [64, 128, 256, 512]  # k chunks to keep

    for budget in budgets:
        k = budget  # chunks to keep
        print(f"\n{'='*40}")
        print(f"Budget: k={k} chunks ({k*2} dims, {k*2/D*100:.1f}% kept)")
        print(f"{'='*40}")

        budget_results = {}

        # 1. Derive universal mask from train tasks
        # Universal = chunks that are important across ALL train tasks
        # Strategy: average importance ranking across train tasks, take top-k
        importance_matrix = np.zeros((len(train_tasks), N_chunks))
        for i, task in enumerate(train_tasks):
            scores = task_chunk_scores[task]
            # Normalize scores
            importance_matrix[i] = scores / (scores.sum() + 1e-10)

        avg_importance = importance_matrix.mean(axis=0)
        universal_ranking = np.argsort(avg_importance)[::-1]
        universal_top_k = set(universal_ranking[:k].tolist())

        # 2. For each test task, compute masks
        for test_task in test_tasks:
            scores = task_chunk_scores[test_task]

            # Oracle (task-specific best)
            task_ranking = np.argsort(scores)[::-1]
            oracle_chunks = set(task_ranking[:k].tolist())

            # Anti-oracle (worst)
            anti_chunks = set(task_ranking[-k:].tolist())

            # Sequential (first-k)
            seq_chunks = set(range(k))

            # Universal (from train tasks)
            uni_chunks = universal_top_k

            # Random (average over 10 trials)
            rng = np.random.default_rng(42)
            random_retentions = []
            for trial in range(10):
                random_idx = rng.choice(N_chunks, size=k, replace=False)
                random_retentions.append(
                    np.sum(scores[random_idx]) / np.sum(scores)
                )

            # Compute importance retention for each mask
            total_importance = np.sum(scores)

            def mask_retention(chunk_set):
                return np.sum(scores[list(chunk_set)]) / total_importance

            oracle_ret = mask_retention(oracle_chunks)
            anti_ret = mask_retention(anti_chunks)
            seq_ret = mask_retainment(seq_chunks)
            uni_ret = mask_retention(uni_chunks)
            random_mean = np.mean(random_retentions)
            random_std = np.std(random_retentions)

            # Overlap metrics
            uni_oracle_overlap = len(uni_chunks & oracle_chunks) / k
            seq_oracle_overlap = len(seq_chunks & oracle_chunks) / k

            budget_results[test_task] = {
                "oracle_retention": float(oracle_ret),
                "anti_retention": float(anti_ret),
                "sequential_retention": float(seq_ret),
                "universal_retention": float(uni_ret),
                "random_mean_retention": float(random_mean),
                "random_std": float(random_std),
                "universal_oracle_overlap": float(uni_oracle_overlap),
                "sequential_oracle_overlap": float(seq_oracle_overlap),
                "universal_vs_random_gap": float(uni_ret - random_mean),
                "universal_vs_oracle_gap": float(oracle_ret - uni_ret),
            }

            print(f"  {test_task}:")
            print(f"    Oracle={oracle_ret:.4f} Uni={uni_ret:.4f} Seq={seq_ret:.4f} Rnd={random_mean:.4f}±{random_std:.4f} Anti={anti_ret:.4f}")
            print(f"    Uni-Oracle gap={oracle_ret-uni_ret:.4f} Uni-Rnd gap={uni_ret-random_mean:.4f}")
            print(f"    Uni-Oracle overlap={uni_oracle_overlap:.3f}")

        # Aggregate
        oracles = [budget_results[t]["oracle_retention"] for t in test_tasks]
        unis = [budget_results[t]["universal_retention"] for t in test_tasks]
        seqs = [budget_results[t]["sequential_retention"] for t in test_tasks]
        rands = [budget_results[t]["random_mean_retention"] for t in test_tasks]
        antis = [budget_results[t]["anti_retention"] for t in test_tasks]
        uni_oracle_gaps = [budget_results[t]["universal_vs_oracle_gap"] for t in test_tasks]
        uni_rnd_gaps = [budget_results[t]["universal_vs_random_gap"] for t in test_tasks]

        budget_summary = {
            "k": k,
            "mean_oracle": float(np.mean(oracles)),
            "mean_universal": float(np.mean(unis)),
            "mean_sequential": float(np.mean(seqs)),
            "mean_random": float(np.mean(rands)),
            "mean_anti": float(np.mean(antis)),
            "mean_uni_oracle_gap": float(np.mean(uni_oracle_gaps)),
            "mean_uni_random_gap": float(np.mean(uni_rnd_gaps)),
            "per_task": budget_results,
        }

        results[f"budget_{k}"] = budget_summary

        print(f"\n  BUDGET k={k} SUMMARY:")
        print(f"    Oracle:    {np.mean(oracles):.4f}")
        print(f"    Universal: {np.mean(unis):.4f}")
        print(f"    Sequential:{np.mean(seqs):.4f}")
        print(f"    Random:    {np.mean(rands):.4f}")
        print(f"    Anti:      {np.mean(antis):.4f}")
        print(f"    Uni-Oracle gap: {np.mean(uni_oracle_gaps):.4f}")
        print(f"    Uni-Random gap: {np.mean(uni_rnd_gaps):.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"universal_mask_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["gte-large", "stella", "inbedder"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_dir = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output"
    run_experiment(args.model, args.gpu, output_dir)


if __name__ == "__main__":
    main()
