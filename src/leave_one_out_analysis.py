#!/usr/bin/env python3
"""
B. Leave-one-out / marginal contribution analysis.

Compares multiple importance definitions:
  1. Standalone contribution (original): Eval(chunk_i alone)
  2. Leave-one-out drop: Eval(full) - Eval(full \ chunk_i)
  3. Marginal gain over random subset: avg Eval(B ∪ chunk_i) - Eval(B)
  4. Approximate Shapley value (permutation sampling)

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/leave_one_out_analysis.py --models gte-large-en-v1.5 stella_en_400M_v5
    CUDA_VISIBLE_DEVICES=2 python src/leave_one_out_analysis.py --models roberta-large roberta-large-InBedder
"""

import os
import sys
import argparse
import json
import gc
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import mteb
from sentence_transformers import SentenceTransformer

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

TASK_CATEGORIES = {
    "Clustering": ['BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering'],
    "PairClassification": ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'],
    "Reranking": ['AskUbuntuDupQuestions', 'SciDocsRR', 'StackOverflowDupQuestions'],
    "STS": ['BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STSBenchmark'],
    "Summarization": ['SummEval'],
    "Retrieval": ['ArguAna', 'CQADupstackEnglishRetrieval', 'NFCorpus', 'SCIDOCS', 'SciFact'],
}


class ChunkedModel:
    """Model wrapper that supports chunk selection and LOO evaluation."""

    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}

    def get_dim(self):
        return self.model.encode("hello", convert_to_tensor=True).shape[-1]

    def encode(self, input_texts, win_size=2, keep_chunks=None, drop_chunks=None,
               dtype='full', **kwargs):
        """Encode texts with optional chunk selection or dropping."""
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device

        # Check cache
        cache_key = (dtype, win_size,
                     tuple(sorted(keep_chunks)) if keep_chunks else None,
                     tuple(sorted(drop_chunks)) if drop_chunks else None)
        if cache_key in self.cache and len(self.cache[cache_key]) == len(input_texts):
            return self.cache[cache_key]

        # Get full embeddings
        texts_to_encode = []
        idx_to_encode = []
        results = [None] * len(input_texts)

        for i, text in enumerate(input_texts):
            full_cache_key = ('full', 0, None, None)
            if full_cache_key in self.cache and text in self._text_cache:
                results[i] = self._text_cache[text]
            else:
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            new_embs = self.model.encode(texts_to_encode, max_len=1024, **kwargs)
            for i, text, emb in zip(idx_to_encode, texts_to_encode, new_embs):
                results[i] = emb.detach()

        embs = torch.stack(results, dim=0) if isinstance(results[0], torch.Tensor) else torch.tensor(results)

        # Apply chunk selection/dropping
        dim = embs.shape[-1]
        n_chunks = dim // win_size

        if dtype == 'chunk_only':
            # Keep only specified chunks
            mask = torch.zeros(dim, device=embs.device, dtype=torch.bool)
            for cid in keep_chunks:
                mask[cid * win_size:(cid + 1) * win_size] = True
            embs = embs * mask.unsqueeze(0).float()

        elif dtype == 'drop_chunks':
            # Drop specified chunks (set to zero)
            mask = torch.ones(dim, device=embs.device, dtype=torch.bool)
            for cid in drop_chunks:
                mask[cid * win_size:(cid + 1) * win_size] = False
            embs = embs * mask.unsqueeze(0).float()

        return embs.detach().to(torch.float32).cpu().numpy()

    def clear_cache(self):
        self.cache = {}
        self._text_cache = {}


def evaluate_task(model, task, win_size=2, keep_chunks=None,
                  drop_chunks=None, dtype='full', batch_size=8):
    """Run MTEB evaluation and return main_score * 100."""
    model.clear_cache()
    evaluation = mteb.MTEB(tasks=[task])
    try:
        # We need to wrap the encode call with our parameters
        original_encode = model.encode

        def custom_encode(texts, **kwargs):
            return original_encode(texts, win_size=win_size,
                                   keep_chunks=keep_chunks,
                                   drop_chunks=drop_chunks,
                                   dtype=dtype, **kwargs)

        model.encode = custom_encode
        results = evaluation.run(model, output_folder=None, verbosity=0)
        model.encode = original_encode
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


def load_tasks(task_names):
    """Load MTEB tasks by name."""
    all_tasks = mteb.get_tasks(tasks=task_names)
    return {t.metadata.name: t for t in all_tasks}


def compute_standalone_scores(model, task, win_size=4, n_chunks=None):
    """Compute standalone contribution score for each chunk."""
    scores = []
    for cid in tqdm(range(n_chunks), desc="Standalone", leave=False):
        score = evaluate_task(model, task, win_size=win_size,
                              keep_chunks=[cid], dtype='chunk_only')
        scores.append(score if score else 0.0)
    return np.array(scores)


def compute_loo_scores(model, task, win_size=4, n_chunks=None):
    """Compute leave-one-out drop for each chunk."""
    # First get full performance
    full_score = evaluate_task(model, task, dtype='full')
    if full_score is None:
        return None, None

    drops = []
    for cid in tqdm(range(n_chunks), desc="LOO", leave=False):
        score_without = evaluate_task(model, task, win_size=win_size,
                                       drop_chunks=[cid], dtype='drop_chunks')
        if score_without is not None:
            drops.append(full_score - score_without)
        else:
            drops.append(0.0)

    return full_score, np.array(drops)


def compute_marginal_scores(model, task, win_size=4, n_chunks=None,
                            budget=256, n_samples=50):
    """Compute marginal gain of adding each chunk to a random base subset."""
    n_select = budget // win_size  # number of chunks for budget
    rng = np.random.RandomState(42)
    marginal_gains = np.zeros(n_chunks)

    for trial in tqdm(range(n_samples), desc="Marginal", leave=False):
        # Sample a random base subset of size n_select-1
        if n_select <= 1:
            base_size = 1
        else:
            base_size = n_select - 1

        base_chunks = rng.choice(n_chunks, size=base_size, replace=False).tolist()
        base_score = evaluate_task(model, task, win_size=win_size,
                                    keep_chunks=base_chunks, dtype='chunk_only')

        if base_score is None:
            continue

        # For each chunk not in base, compute gain of adding it
        remaining = [c for c in range(n_chunks) if c not in base_chunks]
        for cid in remaining:
            extended = base_chunks + [cid]
            ext_score = evaluate_task(model, task, win_size=win_size,
                                       keep_chunks=extended, dtype='chunk_only')
            if ext_score is not None:
                marginal_gains[cid] += (ext_score - base_score)

    marginal_gains /= n_samples
    return marginal_gains


def compute_shapley_approx(model, task, win_size=4, n_chunks=None,
                           n_permutations=64, budget=256):
    """Approximate Shapley values using permutation sampling."""
    n_select = budget // win_size
    shapley = np.zeros(n_chunks)

    for _ in tqdm(range(n_permutations), desc="Shapley", leave=False):
        perm = np.random.permutation(n_chunks)
        prev_score = 0.0  # empty set score

        for i, cid in enumerate(perm):
            current_set = perm[:i+1].tolist()
            # Only evaluate if we have a reasonable subset size
            if len(current_set) > n_select * 2:
                break

            score = evaluate_task(model, task, win_size=win_size,
                                   keep_chunks=current_set, dtype='chunk_only')
            if score is not None:
                shapley[cid] += (score - prev_score)
                prev_score = score
            else:
                shapley[cid] += 0

    shapley /= n_permutations
    return shapley


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5',
                                 'roberta-large', 'roberta-large-InBedder'])
    parser.add_argument('--win-size', type=int, default=4)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--n-marginal-samples', type=int, default=30)
    parser.add_argument('--n-shapley-perms', type=int, default=32)
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
        'roberta-large-InBedder': '/home/linkco/exa/models/roberta-large-InBedder',
    }

    # Load tasks
    print("Loading tasks...")
    task_map = load_tasks(REPRESENTATIVE_TASKS)
    print(f"Loaded {len(task_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}: path not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model = ChunkedModel(model_path, device=args.device)
        model_dim = model.get_dim()
        n_chunks = model_dim // args.win_size
        print(f"Dimension: {model_dim}, n_chunks: {n_chunks}")

        results = {
            "model": model_name,
            "model_dim": model_dim,
            "win_size": args.win_size,
            "budget": args.budget,
            "n_chunks": n_chunks,
            "tasks": {},
        }

        for task_name in REPRESENTATIVE_TASKS:
            if task_name not in task_map:
                continue

            task = task_map[task_name]
            print(f"\n  Task: {task_name}")

            task_results = {}

            # 1. Standalone contribution
            print("    Computing standalone scores...")
            standalone = compute_standalone_scores(model, task, args.win_size, n_chunks)

            # 2. Leave-one-out
            print("    Computing LOO scores...")
            full_score, loo_drops = compute_loo_scores(model, task, args.win_size, n_chunks)

            # 3. Marginal gain
            print("    Computing marginal gains...")
            marginal = compute_marginal_scores(model, task, args.win_size, n_chunks,
                                               args.budget, args.n_marginal_samples)

            # 4. Shapley (optional, expensive)
            if args.n_shapley_perms > 0:
                print("    Computing approximate Shapley...")
                shapley = compute_shapley_approx(model, task, args.win_size, n_chunks,
                                                  args.n_shapley_perms, args.budget)
            else:
                shapley = np.zeros(n_chunks)

            task_results["full_score"] = full_score
            task_results["standalone"] = standalone.tolist()
            task_results["loo_drops"] = loo_drops.tolist() if loo_drops is not None else None
            task_results["marginal"] = marginal.tolist()
            task_results["shapley"] = shapley.tolist()

            # Rank correlations between importance definitions
            from scipy.stats import spearmanr
            standalone_rank = np.argsort(standalone)[::-1]
            loo_rank = np.argsort(loo_drops)[::-1] if loo_drops is not None else None

            if loo_drops is not None:
                rho_loo, p_loo = spearmanr(standalone, loo_drops)
                task_results["rho_standalone_loo"] = float(rho_loo)
                task_results["p_standalone_loo"] = float(p_loo)

            if np.sum(np.abs(marginal)) > 0:
                rho_marginal, p_marginal = spearmanr(standalone, marginal)
                task_results["rho_standalone_marginal"] = float(rho_marginal)
                task_results["p_standalone_marginal"] = float(p_marginal)

            if args.n_shapley_perms > 0 and np.sum(np.abs(shapley)) > 0:
                rho_shapley, p_shapley = spearmanr(standalone, shapley)
                task_results["rho_standalone_shapley"] = float(rho_shapley)
                task_results["p_standalone_shapley"] = float(p_shapley)

            # Entropy for each definition
            for name, scores in [("standalone", standalone),
                                  ("loo", loo_drops),
                                  ("marginal", marginal),
                                  ("shapley", shapley)]:
                if scores is not None and np.sum(np.abs(scores)) > 0:
                    p = np.abs(scores) / np.sum(np.abs(scores))
                    p = p[p > 0]
                    entropy = -np.sum(p * np.log2(p)) / np.log2(len(scores))
                    task_results[f"entropy_{name}"] = float(entropy)

            # Top-k selection comparison
            n_select = args.budget // args.win_size
            rankings = {}
            for name, scores in [("standalone", standalone),
                                  ("loo", loo_drops),
                                  ("marginal", marginal),
                                  ("shapley", shapley)]:
                if scores is not None:
                    rankings[name] = np.argsort(scores)[::-1][:n_select].tolist()

            # Jaccard overlap between rankings
            ranking_names = list(rankings.keys())
            for i in range(len(ranking_names)):
                for j in range(i+1, len(ranking_names)):
                    n1, n2 = ranking_names[i], ranking_names[j]
                    s1, s2 = set(rankings[n1]), set(rankings[n2])
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    task_results[f"jaccard_{n1}_{n2}"] = float(jaccard)

            results["tasks"][task_name] = task_results

            # Print summary
            print(f"    Full score: {full_score:.2f}" if full_score else "    Full score: FAILED")
            if "rho_standalone_loo" in task_results:
                print(f"    Standalone-LOO rho: {task_results['rho_standalone_loo']:.3f}")
            if "rho_standalone_marginal" in task_results:
                print(f"    Standalone-Marginal rho: {task_results['rho_standalone_marginal']:.3f}")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"loo_marginal_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
