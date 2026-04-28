#!/usr/bin/env python3
"""
C. Non-contiguous selection: compare contiguous chunk oracle vs
   greedy forward selection vs learned sparse mask.

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/non_contiguous_selection.py --models gte-large-en-v1.5 stella_en_400M_v5
    CUDA_VISIBLE_DEVICES=2 python src/non_contiguous_selection.py --models roberta-large roberta-large-InBedder
"""

import os
import sys
import argparse
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import mteb
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances

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


class ChunkedModel:
    """Model wrapper supporting arbitrary chunk selection."""

    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}

    def get_dim(self):
        return self.model.encode("hello", convert_to_tensor=True).shape[-1]

    def encode_full(self, texts, **kwargs):
        """Get full-dim embeddings."""
        if "convert_to_tensor" not in kwargs:
            kwargs["convert_to_tensor"] = True
        if "device" not in kwargs:
            kwargs["device"] = self.device

        results = []
        texts_to_encode = []
        idx_to_encode = []

        for i, text in enumerate(texts):
            if text in self.cache:
                results.append(self.cache[text])
                idx_to_encode.append(-1)
            else:
                results.append(None)
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            new_embs = self.model.encode(texts_to_encode, max_len=1024, **kwargs)
            for j, (text, emb) in enumerate(zip(texts_to_encode, new_embs)):
                self.cache[text] = emb.detach()
                # Find where to put it
                for i, idx in enumerate(idx_to_encode):
                    if idx == j + sum(1 for x in idx_to_encode[:i] if x == -1):
                        break

        # Rebuild in order
        final = []
        enc_idx = 0
        for i, text in enumerate(texts):
            if text in self.cache:
                final.append(self.cache[text])
            else:
                final.append(None)

        if any(x is None for x in final):
            # Re-encode missing
            for i, text in enumerate(texts):
                if final[i] is None:
                    emb = self.model.encode([text], max_len=1024, **kwargs)
                    self.cache[text] = emb[0].detach()
                    final[i] = emb[0].detach()

        return torch.stack(final, dim=0)

    def select_dims(self, embeddings, dim_indices, dim=None):
        """Select specific dimensions from embeddings."""
        if isinstance(embeddings, torch.Tensor):
            return embeddings[:, dim_indices]
        return np.array(embeddings)[:, dim_indices]

    def clear_cache(self):
        self.cache = {}


class SimpleMTEBModel:
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
            embs = embs.detach().cpu().float()
        else:
            embs = torch.tensor(embs, dtype=torch.float32)

        if self.dim_indices is not None:
            embs = embs[:, self.dim_indices]

        return embs.numpy()


def evaluate_with_dims(model, task, dim_indices, device='cuda:0'):
    """Evaluate model on task using only selected dimensions."""
    wrapped = SimpleMTEBModel(model, dim_indices=dim_indices, device=device)
    evaluation = mteb.MTEB(tasks=[task])
    try:
        results = evaluation.run(wrapped, output_folder=None, verbosity=0)
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Eval error: {e}")
        return None


def evaluate_baseline(model, task, device='cuda:0'):
    """Evaluate with full dimensions."""
    wrapped = SimpleMTEBModel(model, dim_indices=None, device=device)
    evaluation = mteb.MTEB(tasks=[task])
    try:
        results = evaluation.run(wrapped, output_folder=None, verbosity=0)
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Baseline error: {e}")
        return None


def contiguous_chunk_oracle(model, task, model_dim, budget, win_size=4, device='cuda:0'):
    """Original method: score chunks standalone, select top-k contiguous chunks."""
    n_chunks = model_dim // win_size
    n_select = budget // win_size

    # Score each chunk
    chunk_scores = []
    for cid in tqdm(range(n_chunks), desc="ContiguousOracle", leave=False):
        dim_indices = list(range(cid * win_size, (cid + 1) * win_size))
        score = evaluate_with_dims(model, task, dim_indices, device)
        chunk_scores.append(score if score else 0.0)

    # Select top-k chunks
    ranked = np.argsort(chunk_scores)[::-1]
    selected_dims = []
    for cid in ranked[:n_select]:
        selected_dims.extend(range(cid * win_size, (cid + 1) * win_size))

    return selected_dims, chunk_scores


def greedy_forward_selection(model, task, model_dim, budget, win_size=4,
                              n_eval_limit=200, device='cuda:0'):
    """Greedy forward selection: add chunks one at a time, pick best each step."""
    n_chunks = model_dim // win_size
    n_select = budget // win_size

    selected = []
    remaining = list(range(n_chunks))
    scores_history = []

    for step in range(n_select):
        best_score = -float('inf')
        best_chunk = None

        # Evaluate each remaining chunk
        candidates = remaining[:min(len(remaining), n_eval_limit // max(step + 1, 1))]
        for cid in tqdm(candidates, desc=f"Greedy step {step+1}/{n_select}", leave=False):
            trial = selected + [cid]
            dims = []
            for c in trial:
                dims.extend(range(c * win_size, (c + 1) * win_size))

            score = evaluate_with_dims(model, task, dims, device)
            if score is not None and score > best_score:
                best_score = score
                best_chunk = cid

        if best_chunk is not None:
            selected.append(best_chunk)
            remaining.remove(best_chunk)
            scores_history.append(best_score)

    # Build dim list
    selected_dims = []
    for cid in selected:
        selected_dims.extend(range(cid * win_size, (cid + 1) * win_size))

    return selected_dims, scores_history


def random_selection(model_dim, budget, seed=42):
    """Random dimension selection."""
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(model_dim, size=budget, replace=False).tolist())


def load_tasks(task_names):
    all_tasks = mteb.get_tasks(tasks=task_names)
    return {t.metadata.name: t for t in all_tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+',
                        default=['gte-large-en-v1.5', 'stella_en_400M_v5',
                                 'roberta-large', 'roberta-large-InBedder'])
    parser.add_argument('--win-size', type=int, default=4)
    parser.add_argument('--budgets', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--n-random', type=int, default=5)
    parser.add_argument('--output-dir', default='data/experiment_results')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    MODEL_PATHS = {
        'gte-large-en-v1.5': '/home/linkco/exa/models/gte-large-en-v1.5',
        'stella_en_400M_v5': '/home/linkco/exa/models/stella_en_400M_v5',
        'roberta-large': '/home/linkco/exa/models/roberta-large',
        'roberta-large-InBedder': '/home/linkco/exa/models/roberta-large-InBedder',
    }

    print("Loading tasks...")
    task_map = load_tasks(REPRESENTATIVE_TASKS)
    print(f"Loaded {len(task_map)} tasks")

    for model_name in args.models:
        model_path = MODEL_PATHS.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"Skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        model = ChunkedModel(model_path, device=args.device)
        model_dim = model.get_dim()
        print(f"Dimension: {model_dim}")

        results = {
            "model": model_name,
            "model_dim": model_dim,
            "win_size": args.win_size,
            "methods": {},
        }

        for task_name in REPRESENTATIVE_TASKS:
            if task_name not in task_map:
                continue

            task = task_map[task_name]
            print(f"\n  Task: {task_name}")

            # Baseline
            baseline = evaluate_baseline(model, task, args.device)
            if baseline is None:
                continue
            print(f"    Baseline: {baseline:.2f}")

            task_results = {"baseline": baseline, "budgets": {}}

            for budget in args.budgets:
                print(f"    Budget: {budget}")

                b_results = {}

                # 1. Contiguous chunk oracle
                print("      Contiguous oracle...")
                cont_dims, chunk_scores = contiguous_chunk_oracle(
                    model, task, model_dim, budget, args.win_size, args.device)
                cont_score = evaluate_with_dims(model, task, cont_dims, args.device)
                b_results["contiguous_oracle"] = {
                    "score": cont_score,
                    "retention": cont_score / baseline * 100 if cont_score and baseline else None,
                }

                # 2. Greedy forward selection (non-contiguous)
                print("      Greedy forward...")
                greedy_dims, greedy_history = greedy_forward_selection(
                    model, task, model_dim, budget, args.win_size, device=args.device)
                greedy_score = evaluate_with_dims(model, task, greedy_dims, args.device)
                b_results["greedy_forward"] = {
                    "score": greedy_score,
                    "retention": greedy_score / baseline * 100 if greedy_score and baseline else None,
                    "n_steps": len(greedy_history),
                }

                # 3. Random selection (average of n_random seeds)
                random_scores = []
                for seed in range(args.n_random):
                    rand_dims = random_selection(model_dim, budget, seed=seed + 100)
                    rs = evaluate_with_dims(model, task, rand_dims, args.device)
                    if rs:
                        random_scores.append(rs)
                random_mean = np.mean(random_scores) if random_scores else None
                b_results["random"] = {
                    "mean_score": random_mean,
                    "retention": random_mean / baseline * 100 if random_mean and baseline else None,
                    "n_seeds": len(random_scores),
                }

                # 4. Jaccard overlap between contiguous and greedy selections
                cont_set = set(cont_dims)
                greedy_set = set(greedy_dims)
                if cont_set and greedy_set:
                    jaccard = len(cont_set & greedy_set) / len(cont_set | greedy_set)
                    b_results["selection_overlap_jaccard"] = float(jaccard)

                # Print summary
                cont_ret = b_results["contiguous_oracle"]["retention"]
                greedy_ret = b_results["greedy_forward"]["retention"]
                random_ret = b_results["random"]["retention"]

                print(f"      Contiguous: {cont_ret:.1f}%  Greedy: {greedy_ret:.1f}%  "
                      f"Random: {random_ret:.1f}%  Overlap: {b_results.get('selection_overlap_jaccard', 'N/A')}")

                task_results["budgets"][str(budget)] = b_results

            results["methods"][task_name] = task_results
            model.clear_cache()

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"non_contiguous_{model_name}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
