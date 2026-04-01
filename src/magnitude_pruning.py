"""
Magnitude-based pruning baseline analysis.

Two parts:
1. Compute magnitude rankings from model weights, compare with task-specific rankings (no GPU)
2. Run MTEB evaluation with magnitude-selected dimensions (needs GPU)

Input: models from /home/linkco/exa/models/, analyze data from data/analyze/
Output: magnitude_analysis.json, magnitude_mteb_results.json
"""

import os
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from scipy.stats import spearmanr
from itertools import combinations


def compute_magnitude_ranking(model_path, win_size=2):
    """
    Compute per-dimension magnitude from model embedding weights.
    Returns chunk-level importance ranking grouped by win_size.
    """
    # Load model to get embedding weights
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model from {model_path}...")
    model = SentenceTransformer(model_path, trust_remote_code=True)

    # Get embedding weights
    # For different architectures, the embedding layer has different names
    embed_weights = None
    for name, param in model.named_parameters():
        if 'word_embedding' in name.lower() or 'embed_tokens' in name.lower() or 'embedding' in name.lower():
            if param.dim() >= 2:
                embed_weights = param.data.cpu().float()
                print(f"  Found embedding: {name}, shape={embed_weights.shape}")
                break

    if embed_weights is None:
        # Try to get it from the model directly
        if hasattr(model, 'modules'):
            for module in model.modules():
                if hasattr(module, 'weight') and module.weight.dim() >= 2:
                    embed_weights = module.weight.data.cpu().float()
                    print(f"  Found weight: {module.__class__.__name__}, shape={embed_weights.shape}")
                    break

    if embed_weights is None:
        print(f"  WARNING: Could not find embedding weights, using random")
        return None

    # Compute per-dimension L2 norm (importance)
    # embed_weights shape: (vocab_size, hidden_dim) or similar
    if embed_weights.dim() == 2:
        dim_norms = torch.norm(embed_weights, dim=0)  # (hidden_dim,)
    else:
        dim_norms = torch.norm(embed_weights.flatten(1), dim=0)

    print(f"  Dimension norms: shape={dim_norms.shape}, mean={dim_norms.mean():.4f}, std={dim_norms.std():.4f}")

    # Group into chunks and compute chunk importance
    model_dim = len(dim_norms)
    n_chunks = model_dim // win_size
    chunk_importance = []
    for i in range(n_chunks):
        chunk_norm = dim_norms[i * win_size:(i + 1) * win_size].sum().item()
        chunk_importance.append(chunk_norm)

    chunk_importance = np.array(chunk_importance)
    ranking = np.argsort(chunk_importance)[::-1]  # descending (highest magnitude first)

    # Also return per-dimension ranking
    dim_ranking = np.argsort(dim_norms.numpy())[::-1]

    del model
    import gc
    gc.collect()

    return {
        "model_dim": model_dim,
        "n_chunks": n_chunks,
        "win_size": win_size,
        "per_dim_norms": dim_norms.numpy().tolist(),
        "chunk_importance": chunk_importance.tolist(),
        "chunk_ranking": ranking.tolist(),
        "dim_ranking": dim_ranking.tolist(),
    }


def compare_magnitude_with_task_rankings(magnitude_data, analyze_data, model_name):
    """
    Compare magnitude-based ranking with task-specific chunk rankings.
    """
    if magnitude_data is None:
        return None

    model_analyze = analyze_data.get(model_name)
    if model_analyze is None:
        return None

    win_size = magnitude_data["win_size"]
    mag_ranking = np.array(magnitude_data["chunk_ranking"])

    results = {"task_correlations": {}, "summary": {}}

    for task_name, task_data in model_analyze["task_name"].items():
        if str(win_size) not in task_data.get("split_win_size", {}):
            continue

        chunk_scores = task_data["split_win_size"][str(win_size)]["chunk_result"]
        task_ranking = np.argsort(chunk_scores)[::-1]

        # Spearman correlation between magnitude ranking and task ranking
        rho, p_val = spearmanr(mag_ranking, task_ranking)

        results["task_correlations"][task_name] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
        }

    # Summary statistics
    rhos = [v["spearman_rho"] for v in results["task_correlations"].values()]
    p_vals = [v["p_value"] for v in results["task_correlations"].values()]
    n_sig = sum(1 for p in p_vals if p < 0.05)

    results["summary"] = {
        "n_tasks": len(rhos),
        "mean_rho": float(np.mean(rhos)) if rhos else None,
        "std_rho": float(np.std(rhos)) if rhos else None,
        "min_rho": float(np.min(rhos)) if rhos else None,
        "max_rho": float(np.max(rhos)) if rhos else None,
        "n_significant": n_sig,
        "interpretation": (
            "Low correlation means magnitude is a poor predictor of task-specific importance"
        ),
    }

    return results


def run_magnitude_mteb(model_path, magnitude_data, target_dim=256, win_size=2, batch_size=8):
    """
    Run MTEB evaluation using magnitude-selected dimensions.
    Returns scores for all tasks.
    """
    os.environ["HTTP_PROXY"] = "http://219.223.187.254:7890"
    os.environ["HTTPS_PROXY"] = "http://219.223.187.254:7890"

    import mteb
    from sentence_transformers import SentenceTransformer

    if magnitude_data is None:
        print("  No magnitude data, skipping MTEB evaluation")
        return None

    n_chunks = target_dim // win_size
    top_chunks = magnitude_data["chunk_ranking"][:n_chunks]

    print(f"  Running MTEB with top-{n_chunks} magnitude chunks (dim={target_dim})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build a simple wrapper
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
            # Select chunks by magnitude ranking
            chunks = []
            for chunk_id in top_chunks:
                chunks.append(embeddings[:, chunk_id * win_size:(chunk_id + 1) * win_size])
            embeddings = torch.cat(chunks, dim=1)
            return embeddings.detach().to(torch.float32).cpu().numpy()

    try:
        wrapper = MagnitudeModel(model_path, device)

        # Get all tasks from analyze data
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
        for task_name in all_tasks:
            try:
                tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
                if not tasks:
                    continue
                evaluation = mteb.MTEB(tasks=tasks)
                result = evaluation.run(
                    wrapper, verbosity=0, overwrite_results=True,
                    output_folder="/tmp/mteb_magnitude",
                    encode_kwargs={'batch_size': batch_size},
                    save_corpus_embeddings=False,
                )
                score = result[0].scores['test'][0]['main_score'] * 100
                scores[task_name] = score
                print(f"    {task_name}: {score:.2f}")
            except Exception as e:
                print(f"    {task_name}: ERROR - {e}")

        del wrapper
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return scores
    except Exception as e:
        print(f"  MTEB evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Magnitude-based pruning baseline")
    parser.add_argument("--analyze_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/Useful-Embedding/Useful-Embedding/data/analyze")
    parser.add_argument("--output_dir", type=str,
                        default="/home/linkco/exa/llm-usefulEeb/experiments/analysis_output")
    parser.add_argument("--models_dir", type=str,
                        default="/home/linkco/exa/models")
    parser.add_argument("--run_mteb", action="store_true",
                        help="Run full MTEB evaluation (needs GPU)")
    parser.add_argument("--target_dim", type=int, default=256,
                        help="Target dimension for MTEB evaluation")
    parser.add_argument("--win_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU device ID for MTEB evaluation")
    args = parser.parse_args()

    # Models to analyze (local paths)
    model_configs = {
        "stella_en_400M_v5": "stella_en_400M_v5",
        "gte-large-en-v1.5": "gte-large-en-v1.5",
        "roberta-large-InBedder": "roberta-large-InBedder",
        "bge-m3": "bge-m3",
        "gte-base": "gte-base",
        "instructor-large": "instructor-large",
        "gtr-t5-large": "gtr-t5-large",
        "mxbai-embed-large-v1": "mxbai-embed-large-v1",
        "bart-base": "bart-base",
        "roberta-large": "roberta-large",
    }

    # Load existing analyze data
    print("Loading analyze data...")
    analyze_data = {}
    for fname in os.listdir(args.analyze_dir):
        if fname.endswith('.json'):
            model_name = fname.replace('.json', '')
            with open(os.path.join(args.analyze_dir, fname), "r") as f:
                analyze_data[model_name] = json.load(f)
    print(f"  Loaded {len(analyze_data)} models")

    all_results = {}

    # Part 1: Compute magnitude rankings and compare with task rankings (no GPU needed for MTEB)
    print("\n" + "=" * 60)
    print("[Part 1] Magnitude ranking analysis (no MTEB)")
    print("=" * 60)

    magnitude_results = {}
    comparison_results = {}

    for model_name, model_subdir in model_configs.items():
        model_path = os.path.join(args.models_dir, model_subdir)
        if not os.path.exists(model_path):
            print(f"\n  Skipping {model_name}: path not found")
            continue

        print(f"\n--- {model_name} ---")

        # Compute magnitude ranking
        mag_data = compute_magnitude_ranking(model_path, args.win_size)
        if mag_data is None:
            continue

        magnitude_results[model_name] = {
            "model_dim": mag_data["model_dim"],
            "n_chunks": mag_data["n_chunks"],
            "norm_mean": float(np.mean(mag_data["per_dim_norms"])),
            "norm_std": float(np.std(mag_data["per_dim_norms"])),
            "norm_cv": float(np.std(mag_data["per_dim_norms"]) / np.mean(mag_data["per_dim_norms"])),
            "chunk_ranking": mag_data["chunk_ranking"],
        }

        # Compare with task rankings
        if model_name in analyze_data:
            comp = compare_magnitude_with_task_rankings(mag_data, analyze_data, model_name)
            if comp:
                comparison_results[model_name] = comp
                print(f"  Magnitude vs Task ranking correlation:")
                print(f"    Mean rho = {comp['summary']['mean_rho']:.4f} ± {comp['summary']['std_rho']:.4f}")
                print(f"    Range: [{comp['summary']['min_rho']:.4f}, {comp['summary']['max_rho']:.4f}]")
                print(f"    Significant (p<0.05): {comp['summary']['n_significant']}/{comp['summary']['n_tasks']}")
        else:
            print(f"  No analyze data for {model_name}, skipping comparison")

    all_results["magnitude_rankings"] = magnitude_results
    all_results["magnitude_vs_task_correlation"] = comparison_results

    # Part 2: Run MTEB with magnitude-selected dimensions (needs GPU)
    if args.run_mteb:
        print("\n" + "=" * 60)
        print(f"[Part 2] MTEB evaluation with magnitude pruning (GPU {args.gpu})")
        print("=" * 60)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        mteb_results = {}
        for model_name in ["stella_en_400M_v5", "gte-large-en-v1.5"]:
            model_path = os.path.join(args.models_dir, model_configs[model_name])
            if not os.path.exists(model_path):
                continue

            print(f"\n--- {model_name} (dim={args.target_dim}) ---")
            mag_data = compute_magnitude_ranking(model_path, args.win_size)
            scores = run_magnitude_mteb(model_path, mag_data, args.target_dim, args.win_size, args.batch_size)
            if scores:
                mteb_results[model_name] = {
                    "target_dim": args.target_dim,
                    "scores": scores,
                }

        all_results["magnitude_mteb"] = mteb_results

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "magnitude_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nMagnitude vs Task-Specific Ranking Correlation:")
    print(f"{'Model':<30} {'Mean rho':>10} {'Std':>8} {'Significant':>12}")
    print("-" * 62)
    for model_name, comp in comparison_results.items():
        s = comp["summary"]
        print(f"{model_name:<30} {s['mean_rho']:>10.4f} {s['std_rho']:>8.4f} "
              f"{s['n_significant']:>5}/{s['n_tasks']:<5}")


if __name__ == "__main__":
    main()
