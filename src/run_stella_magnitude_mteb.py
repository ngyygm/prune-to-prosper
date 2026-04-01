"""
Run stella magnitude MTEB evaluation on GPU 2.
Uses separate output directory to avoid conflicts with gte-large run.
"""

import os
import json
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

import mteb
from sentence_transformers import SentenceTransformer

MODEL_PATH = "/home/linkco/exa/models/stella_en_400M_v5"
TARGET_DIM = 256
WIN_SIZE = 2
OUTPUT_DIR = "/tmp/mteb_magnitude_stella"
OUTPUT_JSON = "/home/linkco/exa/llm-usefulEeb/experiments/analysis_output/magnitude_stella_mteb.json"
BATCH_SIZE = 16


def compute_magnitude_ranking(model_path, win_size=2):
    model = SentenceTransformer(model_path, trust_remote_code=True)
    embed_weights = None
    for name, param in model.named_parameters():
        if 'word_embedding' in name.lower() or 'embed_tokens' in name.lower() or 'embedding' in name.lower():
            if param.dim() >= 2:
                embed_weights = param.data.cpu().float()
                print(f"  Found embedding: {name}, shape={embed_weights.shape}")
                break
    if embed_weights is None:
        for module in model.modules():
            if hasattr(module, 'weight') and module.weight.dim() >= 2:
                embed_weights = module.weight.data.cpu().float()
                break
    if embed_weights is None:
        raise RuntimeError("Could not find embedding weights")

    if embed_weights.dim() == 2:
        dim_norms = torch.norm(embed_weights, dim=0)
    else:
        dim_norms = torch.norm(embed_weights.flatten(1), dim=0)

    model_dim = len(dim_norms)
    n_chunks = model_dim // win_size
    chunk_importance = []
    for i in range(n_chunks):
        chunk_norm = dim_norms[i * win_size:(i + 1) * win_size].sum().item()
        chunk_importance.append(chunk_norm)
    chunk_importance = np.array(chunk_importance)
    ranking = np.argsort(chunk_importance)[::-1]

    del model
    import gc
    gc.collect()
    return ranking


def main():
    print("=" * 60)
    print("Stella Magnitude MTEB Evaluation (GPU 2)")
    print("=" * 60)

    print("\n[1/2] Computing magnitude ranking...")
    mag_ranking = compute_magnitude_ranking(MODEL_PATH, WIN_SIZE)
    n_chunks = TARGET_DIM // WIN_SIZE
    top_chunks = mag_ranking[:n_chunks]
    print(f"  Using top-{n_chunks} chunks (dim={TARGET_DIM})")

    print("\n[2/2] Running MTEB evaluation...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    class MagnitudeModel:
        def __init__(self, model_path, device='cuda'):
            self.model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
            self.device = device
            self.cache = {}
            self.model_card_data = {"model_name": "MagnitudeModel", "description": "", "version": "1.0"}

        def get_dim(self):
            return self.model.encode("hello", convert_to_tensor=True).shape[-1]

        def encode(self, texts, **kwargs):
            if "convert_to_tensor" not in kwargs:
                kwargs["convert_to_tensor"] = True
            if "device" not in kwargs:
                kwargs["device"] = self.device

            results = [None] * len(texts)
            texts_to_encode = []
            idx_to_encode = []
            for i, text in enumerate(texts):
                if text in self.cache:
                    results[i] = self.cache[text]
                else:
                    texts_to_encode.append(text)
                    idx_to_encode.append(i)

            if texts_to_encode:
                new_embs = self.model.encode(texts_to_encode, max_len=1024, **kwargs)
                for i, text, emb in zip(idx_to_encode, texts_to_encode, new_embs):
                    self.cache[text] = emb.detach()
                    results[i] = emb.detach()

            embeddings = torch.stack(results, dim=0)
            chunks = []
            for chunk_id in top_chunks:
                chunks.append(embeddings[:, chunk_id * WIN_SIZE:(chunk_id + 1) * WIN_SIZE])
            embeddings = torch.cat(chunks, dim=1)
            return embeddings.detach().to(torch.float32).cpu().numpy()

    wrapper = MagnitudeModel(MODEL_PATH, device)

    all_tasks = [
        'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
        'Banking77Classification', 'EmotionClassification', 'ImdbClassification',
        'MTOPDomainClassification', 'MTOPIntentClassification',
        'MassiveIntentClassification', 'MassiveScenarioClassification',
        'ToxicConversationsClassification', 'TweetSentimentExtractionClassification',
        'BiorxivClusteringS2S', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering',
        'SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus',
        'AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions',
        'ArguAna', 'CQADupstackEnglishRetrieval', 'NFCorpus', 'SCIDOCS', 'SciFact',
        'BIOSSES', 'SICK-R', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17',
        'STSBenchmark', 'SummEval',
    ]

    scores = {}
    for i, task_name in enumerate(all_tasks):
        try:
            tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
            if not tasks:
                continue
            evaluation = mteb.MTEB(tasks=tasks)
            result = evaluation.run(
                wrapper, verbosity=0, overwrite_results=True,
                output_folder=OUTPUT_DIR,
                encode_kwargs={'batch_size': BATCH_SIZE},
                save_corpus_embeddings=False,
            )
            score = result[0].scores['test'][0]['main_score'] * 100
            scores[task_name] = score
            print(f"  [{i+1}/{len(all_tasks)}] {task_name}: {score:.2f}")
        except Exception as e:
            print(f"  [{i+1}/{len(all_tasks)}] {task_name}: ERROR - {e}")

    del wrapper
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    output = {
        'model': 'stella_en_400M_v5',
        'target_dim': TARGET_DIM,
        'n_tasks': len(scores),
        'scores': scores
    }
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")
    print(f"Completed {len(scores)}/{len(all_tasks)} tasks")


if __name__ == "__main__":
    main()
