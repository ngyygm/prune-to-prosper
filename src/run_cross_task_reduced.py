"""
Cross-task dimension importance transfer at REDUCED dimensions.

This script extends the existing task_similar analysis to test whether
donor task rankings transfer at various compression levels (64, 128, 256, 512 dims).

Input: analyze/*.json (per-model chunk importance rankings)
Output: cross_task_reduced/{model}.json (transfer matrix at each dimension)
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

import mteb
from sentence_transformers import SentenceTransformer


class MyModel:
    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, trust_remote_code=True).to(device)
        self.device = device
        self.cache = {}
        self.model_card_data = {
            "model_name": "MyModel",
            "description": "Custom embedding model",
            "version": "1.0"
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
            embeddings, self.get_dim(),
            emb_len=self.emb_len, seed=self.seed,
            win_size=self.win_size, chunk_ids=self.chunk_ids,
            dtype=self.dtype, device=self.device
        )
        return embeddings.detach().to(torch.float32).cpu().numpy()


def get_texts_embeddings(embeddings, model_dim, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def', device="cuda:0"):
    if dtype == 'def':
        return embeddings
    elif dtype == 'random':
        g = torch.Generator(device=device).manual_seed(seed)
        indices = torch.randperm(model_dim, generator=g, device=device)[:emb_len]
        return embeddings.index_select(1, indices)
    elif dtype == 'chunk':
        if not chunk_ids:
            return torch.empty((embeddings.shape[0], 0), device=embeddings.device)
        chunks = [embeddings[:, i*win_size:(i+1)*win_size] for i in chunk_ids]
        return torch.cat(chunks, dim=1)


def run_task(model, task_name, win_size=2, chunk_ids=[0], batch_size=8):
    """Run a single MTEB task with specific chunk configuration."""
    model.set_config(batch_size=batch_size, win_size=win_size, chunk_ids=chunk_ids, dtype='chunk')
    tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
    evaluation = mteb.MTEB(tasks=tasks)
    try:
        results = evaluation.run(
            model, verbosity=0, overwrite_results=True,
            output_folder="/tmp/mteb_cross_task",
            encode_kwargs={'batch_size': batch_size},
            save_corpus_embeddings=False,
        )
        return results[0].scores['test'][0]['main_score'] * 100
    except Exception as e:
        print(f"  Error running {task_name}: {e}")
        return None


def load_chunk_rankings(analyze_path, model_name, win_size=2):
    """Load chunk importance rankings from analyze data."""
    filepath = os.path.join(analyze_path, f"{model_name}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        data = json.load(f)
    rankings = {}
    for task_name, task_data in data["task_name"].items():
        if str(win_size) in task_data.get("split_win_size", {}):
            chunk_scores = task_data["split_win_size"][str(win_size)]["chunk_result"]
            sorted_indices = [idx for idx, _ in sorted(enumerate(chunk_scores), key=lambda x: x[1], reverse=True)]
            rankings[task_name] = sorted_indices
    return rankings


def main():
    parser = argparse.ArgumentParser(description="Cross-task transfer at reduced dimensions")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--analyze_dir", type=str, default="/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/analyze")
    parser.add_argument("--output_dir", type=str, default="/home/linkco/exa/llm-usefulEeb/experiments/cross_task_reduced")
    parser.add_argument("--win_size", type=int, default=2, help="Chunk window size")
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 512], help="Target dimensions to test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k_donors", type=int, default=5, help="Top K donor tasks to test")
    args = parser.parse_args()

    model_path = args.model_path
    model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Target dims: {args.dims}")

    # Load chunk rankings
    rankings = load_chunk_rankings(args.analyze_dir, model_name, args.win_size)
    if rankings is None:
        print(f"No analyze data found for {model_name}")
        return
    print(f"Loaded rankings for {len(rankings)} tasks")

    # Load existing full-dim task_similar for reference
    task_similar_path = f"/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/task_similar/{model_name}.json"
    existing_data = {}
    if os.path.exists(task_similar_path):
        with open(task_similar_path, "r") as f:
            existing_data = json.load(f)

    # Output structure
    output = {}

    # Identify top/bottom donor tasks based on existing self-transfer scores
    # "Weak tasks are better donors" — verify at reduced dims
    all_tasks = list(rankings.keys())

    # Compute donor quality scores from existing data (if available)
    donor_scores = {}
    if existing_data:
        for donor_task in all_tasks:
            if donor_task in existing_data and donor_task in existing_data[donor_task]:
                self_score = existing_data[donor_task][donor_task]
                donor_scores[donor_task] = self_score

    # Select donors: top_k weakest + top_k strongest + random sample
    sorted_donors = sorted(donor_scores.keys(), key=lambda t: donor_scores[t])
    weak_donors = sorted_donors[:args.top_k_donors]
    strong_donors = sorted_donors[-args.top_k_donors:]
    # Also include a few middle donors
    mid_start = len(sorted_donors) // 2 - args.top_k_donors // 2
    mid_donors = sorted_donors[mid_start:mid_start + args.top_k_donors]
    selected_donors = list(set(weak_donors + strong_donors + mid_donors))
    print(f"\nSelected {len(selected_donors)} donors:")
    print(f"  Weak: {weak_donors[:3]}...")
    print(f"  Strong: {strong_donors[-3:]}...")

    # Initialize model
    model = MyModel(model_path, device)
    model_dim = model.get_dim()
    print(f"Model dim: {model_dim}")

    # For each target dimension
    for target_dim in args.dims:
        n_chunks = target_dim // args.win_size
        if n_chunks < 1:
            continue
        print(f"\n{'='*60}")
        print(f"Target dimension: {target_dim} ({n_chunks} chunks)")

        output[str(target_dim)] = {}

        # For each donor task, use its ranking to prune and evaluate on all tasks
        for donor_task in selected_donors:
            if donor_task not in rankings:
                continue

            donor_chunks = rankings[donor_task][:n_chunks]
            print(f"\n  Donor: {donor_task} (top {n_chunks} chunks)")

            output[str(target_dim)][donor_task] = {}

            for target_task in all_tasks:
                if target_task == donor_task and str(target_dim) in output:
                    # Self-transfer — run it
                    pass

                score = run_task(model, target_task, win_size=args.win_size, chunk_ids=donor_chunks, batch_size=args.batch_size)
                if score is not None:
                    output[str(target_dim)][donor_task][target_task] = score
                    print(f"    {target_task}: {score:.2f}")

                # Save intermediate results
                os.makedirs(args.output_dir, exist_ok=True)
                with open(os.path.join(args.output_dir, f"{model_name}.json"), "w") as f:
                    json.dump(output, f, indent=2)

    # Cleanup
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
