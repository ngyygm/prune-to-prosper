"""
Fast chunk analysis for all models - reduced scope.
Only runs win_size=2 chunk evaluation + random/sequential at key dims.
~2-4 hours per model (vs ~50 days for full run).
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

KEY_DIMS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 384, 512, 768, 1024]
N_RANDOM_SEEDS = 5
WIN_SIZE = 2
BATCH_SIZE = 32


class MyModel:
    """Model wrapper with embedding cache and dimension selection."""
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
                results[i] = self.cache[text]
            else:
                texts_to_encode.append(text)
                idx_to_encode.append(i)

        if texts_to_encode:
            kwargs["batch_size"] = BATCH_SIZE
            new_embs = self.model.encode(texts_to_encode, **kwargs)
            for i, (orig_idx, text) in enumerate(zip(idx_to_encode, texts_to_encode)):
                self.cache[text] = new_embs[i]
                results[orig_idx] = new_embs[i]

        all_embs = torch.stack(results)

        # Apply dimension selection
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
    """Run MTEB evaluation with given dimension selection."""
    model.emb_len = emb_len
    model.seed = seed
    model.win_size = win_size
    model.chunk_ids = chunk_ids
    model.dtype = dtype

    tasks = mteb.get_tasks(tasks=[task_name])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=None, verbosity=0,
                             overwrite_results=True,
                             save_corpus_embeddings=False)
    main_score = results[0].scores['test'][0]['main_score'] * 100
    return main_score


def main():
    model_path = sys.argv[1]
    model_name = model_path.replace('/home/linkco/exa/models/', '').replace('/', '-')
    analyze_path = "/home/linkco/exa/llm-usefulEeb/data/analyze"
    output_file = os.path.join(analyze_path, f"{model_name}.json")

    print(f"=== Fast Analysis: {model_name} ===")

    # Load or create output
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            out = json.load(f)
        print(f"Resuming: {len(out.get('task_name', {}))} tasks")
    else:
        out = {"model_name": model_name, "model_dim": 0, "task_name": {}}

    # Load model ONCE
    device = "cuda:0"
    model = MyModel(model_path, device)
    model_dim = model.get_dim()
    out["model_dim"] = model_dim

    valid_dims = [d for d in KEY_DIMS if d < model_dim]
    n_chunks = model_dim // WIN_SIZE
    print(f"dim={model_dim}, valid_dims={valid_dims}, n_chunks={n_chunks}")

    for category, task_names in task_categories.items():
        for task_name in task_names:
            print(f"\n--- {task_name} ---")

            if task_name not in out["task_name"]:
                out["task_name"][task_name] = {
                    "defult_score": 0, "random_score": {}, "sort_score": {}, "split_win_size": {}
                }
            td = out["task_name"][task_name]

            # Clear cache between tasks to save memory
            model.clear_cache()

            # 1. Baseline
            if td["defult_score"] == 0:
                score = run_task(model, task_name, dtype="def")
                td["defult_score"] = score
                print(f"  Baseline: {score:.2f}")

            # 2. Random (5 seeds × valid_dims)
            for dim in valid_dims:
                if str(dim) not in td["random_score"]:
                    scores = []
                    for seed in range(N_RANDOM_SEEDS):
                        s = run_task(model, task_name, emb_len=dim, seed=seed, dtype="random")
                        scores.append(s)
                    td["random_score"][str(dim)] = scores
                    print(f"  Random dim={dim}: mean={np.mean(scores):.2f}")

            # 3. Sequential
            for dim in valid_dims:
                if str(dim) not in td["sort_score"]:
                    score = run_task(model, task_name, emb_len=dim, win_size=1,
                                    chunk_ids=list(range(dim)), dtype='chunk')
                    td["sort_score"][str(dim)] = score
                    print(f"  Seq dim={dim}: {score:.2f}")

            # 4. Chunk evaluation (win_size=2 only)
            ws = str(WIN_SIZE)
            if ws not in td["split_win_size"]:
                td["split_win_size"][ws] = {}
            ws_data = td["split_win_size"][ws]

            if "chunk_result" not in ws_data or len(ws_data.get("chunk_result", [])) == 0:
                print(f"  Chunks (ws={WIN_SIZE}, {n_chunks} chunks)...")
                chunk_scores = []
                for chunk_id in tqdm(range(n_chunks), desc="  Chunks"):
                    s = run_task(model, task_name, win_size=WIN_SIZE,
                                chunk_ids=[chunk_id], dtype='chunk')
                    chunk_scores.append(s)
                ws_data["chunk_result"] = chunk_scores
                print(f"  Done: min={min(chunk_scores):.1f}, max={max(chunk_scores):.1f}")

            # Save after each task
            with open(output_file, 'w') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"  Saved ({len(out['task_name'])} tasks)")

    print(f"\n=== COMPLETE: {model_name}, {len(out['task_name'])} tasks ===")


if __name__ == "__main__":
    main()
