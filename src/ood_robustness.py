#!/usr/bin/env python3
"""
F. OOD / domain robustness analysis.

Train importance on one domain, test on another:
  - Retrieval: learn on MS MARCO, test on BEIR subsets
  - Classification: learn on one domain, test on another
  - STS: learn on STSBenchmark, test on BIOSSES/SICK-R

Compares random vs task-aware selection under distribution shift.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/ood_robustness.py --models gte-large-en-v1.5 bge-m3
    CUDA_VISIBLE_DEVICES=2 python src/ood_robustness.py --models stella_en_400M_v5 roberta-large-InBedder
"""

import os
import sys
import argparse
import json
import gc
import numpy as np
import torch
from tqdm import tqdm
import mteb
from sentence_transformers import SentenceTransformer

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"


class DimSelectModel:
    """Wrapper for MTEB evaluation with dimension selection."""

    def __init__(self, base_model, dim_indices=None, device='cuda:0'):
        self.base_model = base_model
        self.dim_indices = dim_indices
        self.device = device

    def encode(self, input_texts, **kwargs):
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device

        embs = self.base_model.encode(input_texts, **kwargs)
        if isinstance(embs, torch.Tensor):
            embs = embs.detach().cpu().float().numpy()
        else:
            embs = np.array(embs, dtype=np.float32)

        if self.dim_indices is not None:
            embs = embs[:, self.dim_indices]

        return embs


def compute_chunk_importance(model, task, model_dim, win_size=2, device='cuda:0'):
    """Compute standalone chunk importance on a task."""
    n_chunks = model_dim // win_size
    scores = []

    for cid in tqdm(range(n_chunks), desc="Scoring chunks", leave=False):
        dim_start = cid * win_size
        dim_end = (cid + 1) * win_size
        dims = list(range(dim_start, dim_end))

        wrapped = DimSelectModel(model, dim_indices=dims, device=device)
        evaluation = mteb.MTEB(tasks=[task])
        try:
            results = evaluation.run(wrapped, output_folder=None, verbosity=0)
            score = results[0].scores['test'][0]['main_score'] * 100
        except:
            score = 0.0
        scores.append(score)

    return np.array(scores)


def select_top_k_chunks(chunk_scores, budget, win_size):
    """Select top-k chunks based on importance scores."""
    n_select = budget // win_size
    ranked = np.argsort(chunk_scores)[::-1]
    selected_dims = []
    for cid in ranked[:n_select]:
        selected_dims.extend(range(cid * win_size, (cid + 1) * win_size))
    return sorted(selected_dims)


def random_select(model_dim, budget, seed=42):
    """Random dimension selection."""
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(model_dim, size=budget, replace=False).tolist())


def evaluate_on_task(model, task, dim_indices=None, device='cuda:0'):
    """Evaluate model on task with optional dimension selection."""
    wrapped = DimSelectModel(model, dim_indices=dim_indices, device=device)
    evaluation = mteb.MTEB(tasks=[task])
    try:
        results = evaluation.run(wrapped, output_folder=None, verbosity=0)
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"    Eval error: {e}")
        return None


# Define OOD experiment pairs
OOD_PAIRS = {
    "retrieval": {
        "source_tasks": ["NFCorpus"],
        "target_tasks": ["ArguAna", "SciFact", "SCIDOCS"],
    },
    "classification": {
        "source_tasks": ["ImdbClassification"],
        "target_tasks": ["Banking77Classification", "ToxicConversationsClassification"],
    },
    "sts": {
        "source_tasks": ["STSBenchmark"],
        "target_tasks": ["BIOSSES", "SICK-R"],
    },
}


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
        'roberta-large-InBedder': '/home/linkco/exa/models/roberta-large-InBedder',
    }

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        base_model = SentenceTransformer(model_path, trust_remote_code=True).to(args.device)
        model_dim = base_model.encode("hello").shape[-1]
        print(f"Dimension: {model_dim}")

        results = {
            "model": model_name,
            "model_dim": model_dim,
            "win_size": args.win_size,
            "ood_pairs": {},
        }

        for domain_name, pair_config in OOD_PAIRS.items():
            print(f"\n  Domain: {domain_name}")
            domain_results = {"source_tasks": pair_config["source_tasks"],
                              "target_tasks": pair_config["target_tasks"],
                              "budgets": {}}

            # Compute importance on source tasks
            print("    Computing source importance...")
            all_source_tasks = []
            for st_name in pair_config["source_tasks"]:
                try:
                    tasks = mteb.get_tasks(tasks=[st_name])
                    all_source_tasks.extend(tasks)
                except:
                    print(f"    Source task {st_name} not found")

            if not all_source_tasks:
                print(f"    No source tasks found for {domain_name}, skipping")
                continue

            # Average chunk importance across source tasks
            source_scores = []
            for st in all_source_tasks:
                scores = compute_chunk_importance(base_model, st, model_dim,
                                                   args.win_size, args.device)
                source_scores.append(scores)

            avg_source_scores = np.mean(source_scores, axis=0)

            for budget in args.budgets:
                print(f"    Budget: {budget}")
                budget_results = {}

                for tt_name in pair_config["target_tasks"]:
                    print(f"      Target: {tt_name}")
                    try:
                        target_tasks = mteb.get_tasks(tasks=[tt_name])
                        target_task = target_tasks[0]
                    except:
                        print(f"      Target {tt_name} not found")
                        continue

                    # Full-dim baseline
                    baseline = evaluate_on_task(base_model, target_task, device=args.device)
                    if baseline is None:
                        continue

                    # 1. Source-optimized: use source importance
                    opt_dims = select_top_k_chunks(avg_source_scores, budget, args.win_size)
                    opt_score = evaluate_on_task(base_model, target_task, opt_dims, args.device)

                    # 2. Random (avg of n_random seeds)
                    random_scores = []
                    for seed in range(args.n_random):
                        rand_dims = random_select(model_dim, budget, seed + 100)
                        rs = evaluate_on_task(base_model, target_task, rand_dims, args.device)
                        if rs:
                            random_scores.append(rs)
                    random_mean = np.mean(random_scores) if random_scores else None

                    # 3. Target-optimized (oracle, for reference)
                    target_scores = compute_chunk_importance(base_model, target_task, model_dim,
                                                              args.win_size, args.device)
                    oracle_dims = select_top_k_chunks(target_scores, budget, args.win_size)
                    oracle_score = evaluate_on_task(base_model, target_task, oracle_dims, args.device)

                    budget_results[tt_name] = {
                        "baseline": baseline,
                        "source_optimized_score": opt_score,
                        "source_optimized_retention": opt_score / baseline * 100 if opt_score else None,
                        "random_mean_score": random_mean,
                        "random_retention": random_mean / baseline * 100 if random_mean else None,
                        "oracle_score": oracle_score,
                        "oracle_retention": oracle_score / baseline * 100 if oracle_score else None,
                        "ood_drop_source": (opt_score - random_mean) if opt_score and random_mean else None,
                    }

                    opt_ret = opt_score / baseline * 100 if opt_score else None
                    rnd_ret = random_mean / baseline * 100 if random_mean else None
                    print(f"        Source-opt: {opt_ret:.1f}%  Random: {rnd_ret:.1f}%  Oracle: {oracle_score/baseline*100:.1f}%" if all([opt_ret, rnd_ret, oracle_score]) else "        FAILED")

                domain_results["budgets"][str(budget)] = budget_results

            results["ood_pairs"][domain_name] = domain_results

        # Cleanup
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"ood_robustness_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
