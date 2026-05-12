"""
Magnitude-based pruning baseline analysis (paper Sec 3.2 Eq.(3)).

Two parts:
1. Compute per-dimension magnitude rankings from model token-embedding weights
   and compare with task-specific chunk rankings (Spearman). No GPU needed.
2. Run MTEB evaluation while retaining the Top-k highest-magnitude *dimensions*
   (per the paper, NOT chunks). GPU recommended.

Inputs (defaults are tuned for this machine):
- Models: loaded from /data/fengdm/models/<key> by default (all 11 paper
  models are present there). Pass --models_dir "" to instead load by
  HuggingFace ID via SentenceTransformer (HF cache at ~/.cache/huggingface/hub).
- Analyze data: /home/fengdm/Useful-Embedding/data/analyze_new (11 models,
  matching the paper's model list).

Output:
- {output_dir}/magnitude_analysis.json

GPU selection:
- Use the environment variable CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=3
  python magnitude_pruning.py --run_mteb). The --gpu flag is also honoured by
  binding the model to cuda:{gpu} when CUDA is visible, but CUDA_VISIBLE_DEVICES
  is the recommended way to pick a device because it has to be set BEFORE torch
  initializes the CUDA context.
"""

import os
import json
import argparse
import gc
import numpy as np
import torch
from scipy.stats import spearmanr


# Token-embedding parameter name patterns, ordered from most specific to most
# generic. The first non-empty hit on a 2-D weight with V >> D wins.
_EMBED_NAME_PATTERNS_PRIMARY = (
    "word_embeddings",   # BERT/RoBERTa
    "embed_tokens",      # BART/T5/LLaMA/Qwen
    "shared",            # BART encoder/decoder shared embedding
    "wte",               # GPT-2 style
)
_EMBED_NAME_PATTERNS_FALLBACK = ("embedding",)


def _is_likely_token_embedding(param):
    """Heuristic: token embedding is 2-D with V (rows) much larger than D (cols)."""
    if param.dim() != 2:
        return False
    rows, cols = param.shape
    # Vocab is typically >= 1000 and at least 2x the hidden dim.
    return rows >= 1000 and rows > 2 * cols


def _find_token_embedding(model):
    """Return (name, weight) of the token-embedding matrix, or (None, None)."""
    # Pass 1: specific names that almost always denote the token embedding.
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(p in lname for p in _EMBED_NAME_PATTERNS_PRIMARY):
            if _is_likely_token_embedding(param):
                return name, param.data.cpu().float()

    # Pass 2: broad "embedding" match but with the V>>D shape guard so we don't
    # pick up positional / token-type / LayerNorm tensors.
    for name, param in model.named_parameters():
        if any(p in name.lower() for p in _EMBED_NAME_PATTERNS_FALLBACK):
            if _is_likely_token_embedding(param):
                return name, param.data.cpu().float()

    # Pass 3: any 2-D parameter satisfying the V>>D heuristic.
    for name, param in model.named_parameters():
        if _is_likely_token_embedding(param):
            return name, param.data.cpu().float()

    return None, None


def compute_magnitude_ranking(model_path, win_size=2):
    """Compute per-dimension magnitude ranking from the token-embedding matrix.

    Per paper Sec 3.2 Eq.(3):
        Importance(d) = || w_d ||_2,   w_d = W[:, d] in R^V
    Top-k dimensions with the largest magnitudes are retained.

    Returns a dict with both per-dim ranking (used for MTEB pruning) and
    chunk-level aggregation (used only for Spearman comparison with the
    chunk-level task rankings).
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model from {model_path}...")
    model = SentenceTransformer(model_path, trust_remote_code=True)

    name, embed_weights = _find_token_embedding(model)
    if embed_weights is None:
        print("  WARNING: Could not find token-embedding weights, skipping")
        del model
        gc.collect()
        return None
    print(f"  Found embedding: {name}, shape={tuple(embed_weights.shape)}")

    # Per paper: w_d = W[:, d] in R^V, Importance(d) = ||w_d||_2.
    dim_norms = torch.norm(embed_weights, dim=0)  # shape: (D,)

    print(
        f"  Dimension norms: shape={tuple(dim_norms.shape)}, "
        f"mean={dim_norms.mean():.4f}, std={dim_norms.std():.4f}"
    )

    model_dim = int(dim_norms.shape[0])
    # Per-dimension ranking, descending by magnitude (this is what the paper uses).
    dim_ranking = np.argsort(dim_norms.numpy())[::-1]

    # Chunk-level aggregation -- ONLY for comparison with the chunk-level task
    # rankings produced by the rest of the project. NOT used for MTEB pruning.
    n_chunks = model_dim // win_size
    chunk_importance = np.array(
        [
            dim_norms[i * win_size : (i + 1) * win_size].sum().item()
            for i in range(n_chunks)
        ]
    )
    chunk_ranking = np.argsort(chunk_importance)[::-1]

    del model
    gc.collect()

    return {
        "model_dim": model_dim,
        "n_chunks": n_chunks,
        "win_size": win_size,
        "per_dim_norms": dim_norms.numpy().tolist(),
        "dim_ranking": dim_ranking.tolist(),
        "chunk_importance": chunk_importance.tolist(),
        "chunk_ranking": chunk_ranking.tolist(),
    }


def compare_magnitude_with_task_rankings(magnitude_data, analyze_data, model_name):
    """Spearman correlation between magnitude (chunk-aggregated) and task
    chunk rankings. Chunk granularity is mandatory here because the per-task
    rankings in analyze data are already chunk-level (split_win_size).
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

        if len(task_ranking) != len(mag_ranking):
            continue

        rho, p_val = spearmanr(mag_ranking, task_ranking)

        results["task_correlations"][task_name] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
        }

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
            "Low correlation means magnitude is a poor predictor of "
            "task-specific importance"
        ),
    }

    return results


def run_magnitude_mteb(
    model_path,
    magnitude_data,
    target_dim=256,
    batch_size=8,
    gpu=0,
    output_folder="./mteb_magnitude_tmp",
):
    """Run MTEB while retaining the Top-`target_dim` magnitude *dimensions*.

    Strictly follows paper Sec 3.2 Eq.(3): rank dimensions by ||w_d||_2 and
    keep the largest k. No chunk constraint.
    """
    import mteb
    from sentence_transformers import SentenceTransformer

    if magnitude_data is None:
        print("  No magnitude data, skipping MTEB evaluation")
        return None

    dim_ranking = np.asarray(magnitude_data["dim_ranking"], dtype=np.int64)
    top_dims_np = dim_ranking[:target_dim]
    print(f"  Running MTEB with top-{target_dim} magnitude dimensions...")

    if torch.cuda.is_available():
        device = f"cuda:{gpu}" if gpu is not None else "cuda"
    else:
        device = "cpu"
    print(f"  Using device: {device}")

    class MagnitudeModel:
        def __init__(self, model_path, device="cuda"):
            self.model = SentenceTransformer(model_path, trust_remote_code=True).to(device)
            #self.model.max_seq_length = 1024
            self.device = device
            self.cache = {}
            self.top_dims = torch.as_tensor(top_dims_np, dtype=torch.long, device=device)
            self.model_card_data = {
                "model_name": "MagnitudeModel",
                "description": "Magnitude top-k dimensions",
                "version": "1.0",
            }

        def get_dim(self):
            return self.top_dims.numel()

        def encode(self, texts, **kwargs):
            # Strip MTEB-specific kwargs that SentenceTransformer does not accept.
            for k in ("task_name", "prompt_type", "prompt_name"):
                kwargs.pop(k, None)
            kwargs.setdefault("convert_to_tensor", True)
            kwargs.setdefault("device", self.device)

            results = [None] * len(texts)
            texts_to_encode, idx_to_encode = [], []
            for i, text in enumerate(texts):
                if text in self.cache:
                    results[i] = self.cache[text]
                else:
                    texts_to_encode.append(text)
                    idx_to_encode.append(i)

            if texts_to_encode:
                new_embs = self.model.encode(texts_to_encode, **kwargs)
                for i, text, emb in zip(idx_to_encode, texts_to_encode, new_embs):
                    self.cache[text] = emb.detach()
                    results[i] = emb.detach()

            embeddings = torch.stack(results, dim=0)

            # Per-dimension Top-k selection (paper Eq.3).
            if self.top_dims.device != embeddings.device:
                self.top_dims = self.top_dims.to(embeddings.device)
            if embeddings.shape[1] != int(magnitude_data["model_dim"]):
                # Output dim differs from token-embedding dim (e.g. models with
                # a projection head): the magnitude ranking is not applicable.
                raise RuntimeError(
                    f"Output dim {embeddings.shape[1]} does not match "
                    f"token-embedding dim {magnitude_data['model_dim']}; "
                    "magnitude pruning cannot be applied to this model."
                )
            embeddings = embeddings.index_select(1, self.top_dims)
            return embeddings.detach().to(torch.float32).cpu().numpy()

    try:
        wrapper = MagnitudeModel(model_path, device)

        all_tasks = [
            "AmazonCounterfactualClassification", "AmazonReviewsClassification",
            "Banking77Classification", "EmotionClassification", "ImdbClassification",
            "MTOPDomainClassification", "MTOPIntentClassification",
            "MassiveIntentClassification", "MassiveScenarioClassification",
            "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
            "BiorxivClusteringS2S", "MedrxivClusteringS2S", "TwentyNewsgroupsClustering",
            "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
            "AskUbuntuDupQuestions", "SciDocsRR", "StackOverflowDupQuestions",
            "ArguAna", "CQADupstackEnglishRetrieval", "NFCorpus", "SCIDOCS", "SciFact",
            "BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16",
            "STSBenchmark", "SummEval",
        ]

        scores = {}
        for task_name in all_tasks:
            try:
                tasks = mteb.get_tasks(tasks=[task_name], languages=["eng"])
                if not tasks:
                    continue
                evaluation = mteb.MTEB(tasks=tasks)
                result = evaluation.run(
                    wrapper,
                    verbosity=0,
                    overwrite_results=True,
                    output_folder=output_folder,
                    encode_kwargs={"batch_size": batch_size},
                    save_corpus_embeddings=False,
                )
                score = result[0].scores["test"][0]["main_score"] * 100
                scores[task_name] = score
                print(f"    {task_name}: {score:.2f}")
            except Exception as e:
                print(f"    {task_name}: ERROR - {e}")

        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return scores
    except Exception as e:
        print(f"  MTEB evaluation failed: {e}")
        return None


def _resolve_model_path(models_dir, hf_id):
    """If models_dir is empty/None, return the HF model ID (loaded from HF cache).
    Otherwise, treat it as a local prefix and join with the basename of hf_id.
    """
    if not models_dir:
        return hf_id
    # e.g. "Alibaba-NLP/gte-large-en-v1.5" -> "gte-large-en-v1.5"
    local_subdir = hf_id.split("/")[-1]
    return os.path.join(models_dir, local_subdir)


def main():
    parser = argparse.ArgumentParser(description="Magnitude-based pruning baseline")
    parser.add_argument(
        "--analyze_dir",
        type=str,
        default="/home/fengdm/Useful-Embedding/data/analyze_new",
        help="Folder of per-model analyze JSON files (chunk-level task rankings).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/fengdm/Useful-Embedding/data/experiment_results",
        help="Where to write magnitude_analysis.json.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="/data/fengdm/models",
        help=(
            "Local prefix for model directories. Default '/data/fengdm/models' "
            "matches the layout on this machine (e.g. /data/fengdm/models/"
            "bart-base). Pass '' to load by HuggingFace ID via "
            "SentenceTransformer instead (uses ~/.cache/huggingface/hub)."
        ),
    )
    parser.add_argument(
        "--run_mteb",
        action="store_true",
        help="Run full MTEB evaluation (needs GPU).",
    )
    parser.add_argument(
        "--target_dim", type=int, default=256,
        help="Number of dimensions to retain (paper Table 4 uses 256).",
    )
    parser.add_argument(
        "--win_size", type=int, default=2,
        help="Chunk size used ONLY for Spearman correlation against task chunk rankings.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--gpu", type=int, default=0,
        help=(
            "CUDA device index (used as cuda:{gpu}). For full isolation, "
            "set CUDA_VISIBLE_DEVICES before launching this script."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help=(
            "Subset of models to process, space- or comma-separated. "
            "Accepted values are model keys (e.g. 'stella_en_400M_v5 "
            "gte-large-en-v1.5') or raw HuggingFace IDs / local paths. "
            "Empty (default) means: Part 1 runs all 11 models, Part 2 runs "
            "stella_en_400M_v5 + gte-large-en-v1.5 (paper Figure 4)."
        ),
    )
    args = parser.parse_args()

    # HuggingFace model IDs matching the paper's 11-model setup.
    model_configs = {
        "stella_en_400M_v5": "NovaSearch/stella_en_400M_v5",
        "gte-large-en-v1.5": "Alibaba-NLP/gte-large-en-v1.5",
        "roberta-large-InBedder": "BrandonZYW/roberta-large-InBedder",
        "bge-m3": "BAAI/bge-m3",
        "gte-base": "thenlper/gte-base",
        "instructor-large": "hkunlp/instructor-large",
        "gtr-t5-large": "sentence-transformers/gtr-t5-large",
        "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
        "bart-base": "facebook/bart-base",
        "roberta-large": "FacebookAI/roberta-large",
        "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
    }

    # Parse --models: split on commas and whitespace, accept keys or raw IDs.
    # If a token is not a known key, register it on the fly using its basename
    # as the display name (mirrors the `sys.argv[1]` style of `10 rank chunk mteb.py`).
    selected_models = []
    if args.models:
        raw_tokens = args.models.replace(",", " ").split()
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok in model_configs:
                selected_models.append(tok)
            else:
                display = tok.split("/")[-1].rstrip(os.sep)
                model_configs[display] = tok
                selected_models.append(display)
                print(f"  Registered ad-hoc model '{display}' -> {tok}")

    print("Loading analyze data...")
    analyze_data = {}
    if os.path.isdir(args.analyze_dir):
        for fname in os.listdir(args.analyze_dir):
            if fname.endswith(".json"):
                model_name = fname.replace(".json", "")
                with open(os.path.join(args.analyze_dir, fname), "r") as f:
                    analyze_data[model_name] = json.load(f)
    print(f"  Loaded {len(analyze_data)} models from {args.analyze_dir}")

    all_results = {}

    print("\n" + "=" * 60)
    print("[Part 1] Magnitude ranking analysis (no MTEB)")
    print("=" * 60)

    magnitude_results = {}
    comparison_results = {}
    magnitude_cache = {}  # model_name -> mag_data (re-used in Part 2)

    part1_models = (
        [(m, model_configs[m]) for m in selected_models]
        if selected_models
        else list(model_configs.items())
    )
    if selected_models:
        print(f"  Restricting to --models: {selected_models}")

    for model_name, hf_id in part1_models:
        model_path = _resolve_model_path(args.models_dir, hf_id)
        # When using HF IDs, skip the os.path.exists guard.
        if args.models_dir and not os.path.exists(model_path):
            print(f"\n  Skipping {model_name}: local path not found ({model_path})")
            continue

        print(f"\n--- {model_name} ---")

        try:
            mag_data = compute_magnitude_ranking(model_path, args.win_size)
        except Exception as e:
            print(f"  ERROR computing magnitude for {model_name}: {e}")
            continue
        if mag_data is None:
            continue

        magnitude_cache[model_name] = mag_data
        magnitude_results[model_name] = {
            "model_dim": mag_data["model_dim"],
            "n_chunks": mag_data["n_chunks"],
            "norm_mean": float(np.mean(mag_data["per_dim_norms"])),
            "norm_std": float(np.std(mag_data["per_dim_norms"])),
            "norm_cv": float(
                np.std(mag_data["per_dim_norms"])
                / max(np.mean(mag_data["per_dim_norms"]), 1e-12)
            ),
            "dim_ranking": mag_data["dim_ranking"],
            "chunk_ranking": mag_data["chunk_ranking"],
        }

        if model_name in analyze_data:
            model_analyze = analyze_data[model_name]
            analyze_dim = model_analyze.get("model_dim", 0)
            mag_dim = mag_data["model_dim"]
            if analyze_dim != mag_dim:
                print(
                    f"  Skipping comparison: dimension mismatch "
                    f"(magnitude={mag_dim}, analyze={analyze_dim})"
                )
            else:
                comp = compare_magnitude_with_task_rankings(
                    mag_data, analyze_data, model_name
                )
                if comp:
                    comparison_results[model_name] = comp
                    s = comp["summary"]
                    print("  Magnitude vs Task ranking correlation:")
                    print(
                        f"    Mean rho = {s['mean_rho']:.4f} ± {s['std_rho']:.4f}"
                    )
                    print(
                        f"    Range: [{s['min_rho']:.4f}, {s['max_rho']:.4f}]"
                    )
                    print(
                        f"    Significant (p<0.05): "
                        f"{s['n_significant']}/{s['n_tasks']}"
                    )
        else:
            print(f"  No analyze data for {model_name}, skipping comparison")

    all_results["magnitude_rankings"] = magnitude_results
    all_results["magnitude_vs_task_correlation"] = comparison_results

    if args.run_mteb:
        cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print("\n" + "=" * 60)
        print(
            f"[Part 2] MTEB evaluation with magnitude pruning "
            f"(cuda:{args.gpu}, CUDA_VISIBLE_DEVICES={cuda_devs})"
        )
        print("=" * 60)

        os.makedirs(args.output_dir, exist_ok=True)
        mteb_tmp = os.path.join(args.output_dir, "mteb_magnitude_tmp")

        mteb_results = {}
        # Paper Figure 4 caption: magnitude evaluated on GTE-Large and Stella.
        # If --models was provided, restrict Part 2 to that list instead.
        default_mteb_models = ["stella_en_400M_v5", "gte-large-en-v1.5"]
        mteb_model_names = selected_models if selected_models else default_mteb_models
        for model_name in mteb_model_names:
            if model_name not in model_configs:
                print(f"  Unknown model '{model_name}', skipping.")
                continue
            model_path = _resolve_model_path(args.models_dir, model_configs[model_name])
            if args.models_dir and not os.path.exists(model_path):
                continue

            print(f"\n--- {model_name} (dim={args.target_dim}) ---")

            mag_data = magnitude_cache.get(model_name)
            if mag_data is None:
                try:
                    mag_data = compute_magnitude_ranking(model_path, args.win_size)
                except Exception as e:
                    print(f"  ERROR computing magnitude: {e}")
                    continue
            if mag_data is None:
                continue

            # Guard: per-dim selection requires output_dim == token-embed dim.
            if args.target_dim > mag_data["model_dim"]:
                print(
                    f"  target_dim={args.target_dim} > model_dim={mag_data['model_dim']}, "
                    "skipping."
                )
                continue

            scores = run_magnitude_mteb(
                model_path,
                mag_data,
                target_dim=args.target_dim,
                batch_size=args.batch_size,
                gpu=args.gpu,
                output_folder=mteb_tmp,
            )
            if scores:
                mteb_results[model_name] = {
                    "target_dim": args.target_dim,
                    "scores": scores,
                }

        all_results["magnitude_mteb"] = mteb_results

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "magnitude_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nMagnitude vs Task-Specific Ranking Correlation:")
    print(f"{'Model':<30} {'Mean rho':>10} {'Std':>8} {'Significant':>12}")
    print("-" * 62)
    for model_name, comp in comparison_results.items():
        s = comp["summary"]
        if s["mean_rho"] is None:
            continue
        print(
            f"{model_name:<30} {s['mean_rho']:>10.4f} {s['std_rho']:>8.4f} "
            f"{s['n_significant']:>5}/{s['n_tasks']:<5}"
        )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models bart-base --output_dir data/experiment_results/mag_bart
# CUDA_VISIBLE_DEVICES=1 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models bge-m3 --output_dir data/experiment_results/mag_bge-m3
# CUDA_VISIBLE_DEVICES=2 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models gte-base --output_dir data/experiment_results/mag_gte-base
# CUDA_VISIBLE_DEVICES=5 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models roberta-large-InBedder --output_dir data/experiment_results/mag_roberta-large-InBedder
# CUDA_VISIBLE_DEVICES=6 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models mxbai-embed-large-v1 --output_dir data/experiment_results/mag_mxbai-embed-large-v1
# CUDA_VISIBLE_DEVICES=7 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models Qwen3-Embedding-0.6B --output_dir data/experiment_results/mag_Qwen3-Embedding
# CUDA_VISIBLE_DEVICES=3 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models stella_en_400M_v5 --output_dir data/experiment_results/mag_stella
# CUDA_VISIBLE_DEVICES=4 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models gte-large-en-v1.5 --output_dir data/experiment_results/mag_gte-large-en-v1.5
# CUDA_VISIBLE_DEVICES=0 /data/fengdm/anaconda3/envs/embedding/bin/python "./magnitude_pruning.py" --run_mteb --models roberta-large --output_dir data/experiment_results/mag_roberta-large