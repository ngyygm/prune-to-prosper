"""
Run missing tasks for incomplete models.
Usage: python run_missing_tasks.py --device cuda:0 [--model <name>]
If --model is not specified, runs all missing tasks for the given device.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import mteb

os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

task_categories = {
    "Classification": [
        'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
        'Banking77Classification', 'EmotionClassification',
        'ImdbClassification', 'MTOPDomainClassification',
        'MTOPIntentClassification', 'MassiveIntentClassification',
        'MassiveScenarioClassification', 'ToxicConversationsClassification',
        'TweetSentimentExtractionClassification'
    ],
    "Clustering": ['BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering'],
    "PairClassification": ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus'],
    "Reranking": ['AskUbuntuDupQuestions', 'SciDocsRR', 'StackOverflowDupQuestions', 'SummEval', 'MindSmallReranking'],
    "STS": ['BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STSBenchmark'],
    "Retrieval": ['ArguAna', 'CQADupstackEnglishRetrieval', 'NFCorpus', 'SCIDOCS', 'SciFact'],
}

ALL_TASKS = []
for cat_tasks in task_categories.values():
    ALL_TASKS.extend(cat_tasks)

KEY_DIMS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024]
N_RANDOM_SEEDS = 10  # Match existing data (10 seeds)
WIN_SIZE = 2
BATCH_SIZE = 16  # Reduced from 32 to avoid OOM on 24GB GPUs

# Missing tasks per model (MindSmallReranking skipped - too large: 2.6M texts)
MISSING = {
    "Qwen3-Embedding-0.6B": ["NFCorpus", "SCIDOCS", "SciFact"],
    "mxbai-embed-large-v1": ["CQADupstackEnglishRetrieval", "NFCorpus", "SCIDOCS", "SciFact"],
}


class MyModel:
    def __init__(self, modelpath, device='cuda:0'):
        self.model = SentenceTransformer(modelpath, device=device, trust_remote_code=True)
        self.device = device
        self.cache = {}
        self.model_card_data = {"model_name": "MyModel"}
        self.emb_len = 64
        self.seed = 42
        self.win_size = 64
        self.chunk_ids = [0]
        self.dtype = 'def'
        self.batch_size = BATCH_SIZE

    def get_dim(self):
        return self.model.encode("hello", convert_to_tensor=True, device=self.device).shape[-1]

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
                results[i] = self.cache[text].to(self.device)
            else:
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            kwargs["batch_size"] = self.batch_size
            new_embs = self.model.encode(texts_to_encode, **kwargs)
            for i, (orig_idx, text) in enumerate(zip(idx_to_encode, texts_to_encode)):
                # Store cache on CPU to avoid OOM on large retrieval tasks
                self.cache[text] = new_embs[i].cpu()
                results[orig_idx] = new_embs[i]

        all_embs = torch.stack(results).float()

        if self.dtype == 'def':
            return all_embs.cpu().numpy()
        elif self.dtype == 'random':
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            indices = torch.randperm(all_embs.shape[-1], generator=g, device=self.device)[:self.emb_len]
            return all_embs[:, indices].cpu().numpy()
        elif self.dtype == 'chunk':
            chunks = [all_embs[:, i*self.win_size:(i+1)*self.win_size] for i in self.chunk_ids]
            return torch.cat(chunks, dim=1).cpu().numpy()
        return all_embs.cpu().numpy()

    def clear_cache(self):
        self.cache = {}


def run_task(model, task_name, emb_len=64, seed=42, win_size=64, chunk_ids=[0], dtype='def'):
    model.emb_len = emb_len
    model.seed = seed
    model.win_size = win_size
    model.chunk_ids = chunk_ids
    model.dtype = dtype

    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=None, verbosity=0,
                             overwrite_results=True, save_corpus_embeddings=False)
    main_score = results[0].scores['test'][0]['main_score'] * 100
    return main_score


def process_task(model, task_name, model_dim):
    valid_dims = [d for d in KEY_DIMS if d < model_dim]
    n_chunks = model_dim // WIN_SIZE

    td = {"defult_score": 0, "random_score": {}, "sort_score": {}, "split_win_size": {}}

    # 1. Baseline
    td["defult_score"] = run_task(model, task_name, dtype="def")
    print(f"  [{task_name}] Baseline: {td['defult_score']:.2f}")
    model.clear_cache()
    torch.cuda.empty_cache()

    # 2. Random (10 seeds)
    for dim in valid_dims:
        scores = []
        for seed in range(N_RANDOM_SEEDS):
            s = run_task(model, task_name, emb_len=dim, seed=seed, dtype="random")
            scores.append(s)
        td["random_score"][str(dim)] = scores
        print(f"  [{task_name}] Random dim={dim}: mean={np.mean(scores):.2f}")
    model.clear_cache()
    torch.cuda.empty_cache()

    # 3. Sequential
    for dim in valid_dims:
        score = run_task(model, task_name, emb_len=dim, win_size=1,
                        chunk_ids=list(range(dim)), dtype='chunk')
        td["sort_score"][str(dim)] = score
        print(f"  [{task_name}] Seq dim={dim}: {score:.2f}")
    model.clear_cache()
    torch.cuda.empty_cache()

    # 4. Chunk (win_size=2)
    ws = str(WIN_SIZE)
    chunk_scores = []
    for chunk_id in tqdm(range(n_chunks), desc=f"  [{task_name}] Chunks"):
        s = run_task(model, task_name, win_size=WIN_SIZE,
                    chunk_ids=[chunk_id], dtype='chunk')
        chunk_scores.append(s)
    td["split_win_size"][ws] = {"chunk_result": chunk_scores}
    print(f"  [{task_name}] Chunks done: min={min(chunk_scores):.1f}, max={max(chunk_scores):.1f}")

    return td


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None, help="Specific model to run (e.g. bart-base)")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names to override MISSING dict")
    parser.add_argument("--output-file", default=None, help="Override output JSON file path (for parallel runs)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    analyze_path = "/home/linkco/exa/llm-usefulEeb/data/analyze"
    models_path = "/home/linkco/exa/models"

    if args.model and args.tasks:
        models_to_run = {args.model: [t.strip() for t in args.tasks.split(",")]}
    elif args.model:
        models_to_run = {args.model: MISSING[args.model]}
    else:
        models_to_run = MISSING

    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    for model_name, missing_tasks in models_to_run.items():
        model_dir = os.path.join(models_path, model_name)
        json_file = args.output_file if (args.output_file and os.path.exists(args.output_file)) else os.path.join(analyze_path, f"{model_name}.json")

        if not os.path.isdir(model_dir):
            print(f"SKIP {model_name}: model dir not found")
            continue

        # Load existing data
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                out = json.load(f)
        else:
            # Create from source file if using override output
            src_file = os.path.join(analyze_path, f"{model_name}.json")
            if os.path.exists(src_file):
                import shutil
                shutil.copy2(src_file, json_file)
                with open(json_file, "r") as f:
                    out = json.load(f)
            else:
                out = {"model_name": model_name, "model_dim": 0, "task_name": {}}

        print(f"\n=== {model_name}: {len(missing_tasks)} tasks to run ===")

        # Load model
        model = MyModel(model_dir, args.device)
        if args.batch_size:
            model.batch_size = args.batch_size
        model_dim = model.get_dim()
        print(f"dim={model_dim}")

        for task_name in missing_tasks:
            print(f"\n--- {task_name} ---")
            model.clear_cache()
            td = process_task(model, task_name, model_dim)
            out["task_name"][task_name] = td

            # Save after each task
            with open(json_file, 'w') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"  Saved ({len(out['task_name'])} tasks total)")

        print(f"\n=== DONE: {model_name} ===")

    print("\nAll models complete!")


if __name__ == "__main__":
    main()
