"""
Basis Sensitivity Experiment — "Killer Experiment"

Tests whether dimension interchangeability is intrinsic to the representation
or an artifact of the native basis.

For each model:
1. Original basis (baseline)
2. Random orthogonal rotation (Q)
3. PCA basis (sorted by variance)
4. Whitened basis (unit variance per dim)
5. For each basis, run pruning at dim=256 and compare:
   - Random selection retention
   - Sequential (first-k) retention
   - Optimized (oracle) retention

Key predictions:
- If interchangeability is intrinsic → random ≈ sequential in ALL bases
- If interchangeability is basis-dependent → random ≠ sequential in rotated bases

Usage:
  CUDA_VISIBLE_DEVICES=0 python basis_sensitivity.py --model gte-large
  CUDA_VISIBLE_DEVICES=1 python basis_sensitivity.py --model stella
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

def get_model_path(model_name):
    paths = {
        "gte-large": "/home/linkco/exa/models/gte-large-en-v1.5",
        "stella": "/home/linkco/exa/models/stella_en_400M_v5",
        "inbedder": "/home/linkco/exa/models/roberta-large-InBedder",
    }
    return paths.get(model_name)

def get_model_dim(model_name):
    return {"gte-large": 1024, "stella": 1024, "inbedder": 1024}[model_name]

def random_orthogonal_matrix(n, seed=42):
    """Generate random orthogonal matrix using QR decomposition."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(H)
    # Ensure proper orthogonal (det=+1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q

def apply_basis_transform(embeddings, transform_matrix):
    """Apply orthogonal basis transformation to embeddings.
    embeddings: (N, D) numpy array
    transform_matrix: (D, D) orthogonal matrix
    """
    return embeddings @ transform_matrix.T

def pca_basis(embeddings):
    """Get PCA rotation matrix and apply it."""
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    cov = (centered.T @ centered) / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    return eigenvectors, eigenvalues, mean.squeeze()

def whiten_basis(embeddings):
    """Get whitening transform matrix."""
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    cov = (centered.T @ centered) / len(centered)
    # ZCA whitening: W = U @ diag(1/sqrt(eigenvalues)) @ U^T
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Clip small eigenvalues
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return whitening, eigenvalues, mean.squeeze()

def compute_retention(full_scores, pruned_scores):
    """Compute retention ratio."""
    if isinstance(full_scores, (int, float)):
        full_scores = np.array([full_scores])
    if isinstance(pruned_scores, (int, float)):
        pruned_scores = np.array([pruned_scores])
    # Handle zero or negative scores
    mask = full_scores != 0
    retentions = np.where(mask, pruned_scores / full_scores, 1.0)
    return np.mean(retentions)

def run_experiment_for_model(model_name, gpu_id, output_dir):
    """Run the full basis sensitivity experiment for one model."""
    import torch
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print(f"Basis Sensitivity Experiment: {model_name}")
    print(f"GPU: {gpu_id}, Time: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    model_path = get_model_path(model_name)
    D = get_model_dim(model_name)
    target_dim = 256  # 75% reduction

    # Load model
    print(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path, device=f"cuda:{gpu_id}", trust_remote_code=True)

    # Generate diverse sample sentences for embedding
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn patterns from data.",
        "The weather is beautiful today with clear skies.",
        "Artificial intelligence is transforming many industries.",
        "She walked to the store to buy some groceries.",
        "The stock market saw significant gains this quarter.",
        "Climate change remains a critical global challenge.",
        "The new restaurant downtown serves excellent Italian food.",
        "Quantum computing could revolutionize cryptography.",
        "Students studied hard for their final examinations.",
        "The concert was attended by thousands of music fans.",
        "Renewable energy sources are becoming more affordable.",
        "The novel tells a story of love and redemption.",
        "Scientists discovered a new species in the rainforest.",
        "The city implemented new traffic management systems.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The museum exhibition featured ancient Egyptian artifacts.",
        "Global trade connects economies across the world.",
        "The basketball team won the championship game.",
        "Electric vehicles are gaining popularity worldwide.",
    ]
    # Repeat to get 500 sentences
    sentences = sentences * 25  # 500 sentences

    print(f"Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    results = {"model": model_name, "D": D, "target_dim": target_dim, "n_sentences": len(sentences)}

    # Define basis transformations
    bases = {}

    # 1. Original basis (identity)
    bases["original"] = np.eye(D)

    # 2. Random orthogonal rotation (3 different seeds)
    for seed in [42, 123, 456]:
        bases[f"random_ortho_seed{seed}"] = random_orthogonal_matrix(D, seed=seed)

    # 3. PCA basis
    pca_matrix, pca_eigenvalues, pca_mean = pca_basis(embeddings)
    bases["pca"] = pca_matrix

    # 4. Whitened basis
    whiten_matrix, whiten_eigenvalues, whiten_mean = whiten_basis(embeddings)
    bases["whitened"] = whiten_matrix

    # Store eigenvalue decay info
    results["spectral_info"] = {
        "pca_top10_eigenvalues": pca_eigenvalues[:10].tolist(),
        "pca_eigenvalue_ratio_top1_to_last": float(pca_eigenvalues[0] / max(pca_eigenvalues[-1], 1e-10)),
        "pca_cumulative_variance_256": float(np.sum(pca_eigenvalues[:256]) / np.sum(pca_eigenvalues)),
        "pca_cumulative_variance_512": float(np.sum(pca_eigenvalues[:512]) / np.sum(pca_eigenvalues)),
        "whiten_eigenvalue_range": [float(whiten_eigenvalues.min()), float(whiten_eigenvalues.max())],
    }

    # For each basis, compute pruning behavior
    basis_results = {}

    for basis_name, Q in bases.items():
        print(f"\n  Basis: {basis_name}")

        # Transform embeddings
        if basis_name == "original":
            E_transformed = embeddings.copy()
        elif basis_name == "pca":
            E_transformed = (embeddings - pca_mean) @ pca_matrix.T
        elif basis_name == "whitened":
            E_transformed = (embeddings - whiten_mean) @ whiten_matrix.T
        else:
            E_transformed = embeddings @ Q.T

        # Compute per-dimension statistics
        dim_variance = np.var(E_transformed, axis=0)
        dim_mean_abs = np.mean(np.abs(E_transformed), axis=0)

        # Sequential retention (first-k dims)
        E_seq = E_transformed[:, :target_dim]

        # Random retention (average of 10 random selections)
        rng = np.random.default_rng(42)
        random_retentions = []
        for trial in range(10):
            selected_dims = rng.choice(D, size=target_dim, replace=False)
            E_random = E_transformed[:, selected_dims]
            # Compare pairwise cosine similarity preservation
            random_retentions.append(float(np.mean(np.var(E_random, axis=0))))
        avg_random_var = np.mean(random_retentions)

        # Oracle retention (top-k by variance)
        top_k_dims = np.argsort(dim_variance)[::-1][:target_dim]
        E_oracle = E_transformed[:, top_k_dims]
        oracle_var = float(np.mean(np.var(E_oracle, axis=0)))

        # Anti-oracle (bottom-k)
        bottom_k_dims = np.argsort(dim_variance)[:target_dim]
        E_anti = E_transformed[:, bottom_k_dims]
        anti_var = float(np.mean(np.var(E_anti, axis=0)))

        # Sequential variance
        seq_var = float(np.mean(np.var(E_seq, axis=0)))

        # Cosine similarity preservation metric
        # For original basis, compute full cosine similarity matrix
        from scipy.spatial.distance import cosine

        # Sample-based cosine similarity preservation
        n_sample = min(100, len(E_transformed))
        sample_idx = np.arange(n_sample)

        # Full-dim cosine similarities
        E_full_norm = E_transformed[sample_idx]
        E_full_norm = E_full_norm / (np.linalg.norm(E_full_norm, axis=1, keepdims=True) + 1e-10)
        full_cos_sim = E_full_norm @ E_full_norm.T

        # Sequential pruned cosine similarities
        E_seq_norm = E_seq[sample_idx]
        E_seq_norm = E_seq_norm / (np.linalg.norm(E_seq_norm, axis=1, keepdims=True) + 1e-10)
        seq_cos_sim = E_seq_norm @ E_seq_norm.T

        # Random pruned cosine similarities (average)
        avg_random_cos_corr = 0
        for trial in range(5):
            selected_dims = rng.choice(D, size=target_dim, replace=False)
            E_rnd = E_transformed[np.ix_(sample_idx, selected_dims)]
            E_rnd_norm = E_rnd / (np.linalg.norm(E_rnd, axis=1, keepdims=True) + 1e-10)
            rnd_cos_sim = E_rnd_norm @ E_rnd_norm.T
            # Correlation with full cosine sim
            mask = ~np.eye(n_sample, dtype=bool)
            corr = np.corrcoef(full_cos_sim[mask], rnd_cos_sim[mask])[0, 1]
            avg_random_cos_corr += corr
        avg_random_cos_corr /= 5

        # Sequential cosine correlation
        mask = ~np.eye(n_sample, dtype=bool)
        seq_cos_corr = np.corrcoef(full_cos_sim[mask], seq_cos_sim[mask])[0, 1]

        # Nearest-neighbor preservation
        # For each point, check if top-5 NN in full space overlap with top-5 in pruned
        k_nn = 5
        nn_overlap_seq = 0
        nn_overlap_random = 0
        n_nn_samples = min(50, n_sample)

        for i in range(n_nn_samples):
            # Full space top-k NN
            sims = full_cos_sim[i]
            sims[i] = -999  # exclude self
            full_nn = set(np.argsort(sims)[-k_nn:])

            # Sequential NN
            sims_seq = seq_cos_sim[i]
            sims_seq[i] = -999
            seq_nn = set(np.argsort(sims_seq)[-k_nn:])
            nn_overlap_seq += len(full_nn & seq_nn) / k_nn

        # Average for random masks
        for trial in range(3):
            selected_dims = rng.choice(D, size=target_dim, replace=False)
            E_rnd = E_transformed[np.ix_(sample_idx, selected_dims)]
            E_rnd_norm = E_rnd / (np.linalg.norm(E_rnd, axis=1, keepdims=True) + 1e-10)
            rnd_cos_sim = E_rnd_norm @ E_rnd_norm.T
            for i in range(n_nn_samples):
                sims_rnd = rnd_cos_sim[i]
                sims_rnd[i] = -999
                rnd_nn = set(np.argsort(sims_rnd)[-k_nn:])
                nn_overlap_random += len(full_nn & rnd_nn) / (k_nn * 3 * n_nn_samples)

        basis_results[basis_name] = {
            "dim_variance_cv": float(np.std(dim_variance) / (np.mean(dim_variance) + 1e-10)),
            "dim_variance_entropy": float(-np.sum((dim_variance/np.sum(dim_variance)) * np.log(dim_variance/np.sum(dim_variance) + 1e-10)) / np.log(D)),
            "seq_variance_retention": float(seq_var / (np.mean(dim_variance))),
            "random_variance_retention": float(avg_random_var / (np.mean(dim_variance))),
            "oracle_variance_retention": float(oracle_var / (np.mean(dim_variance))),
            "anti_variance_retention": float(anti_var / (np.mean(dim_variance))),
            "seq_cosine_correlation": float(seq_cos_corr),
            "random_cosine_correlation": float(avg_random_cos_corr),
            "seq_nn_overlap": float(nn_overlap_seq / n_nn_samples),
            "random_nn_overlap": float(nn_overlap_random),
            "seq_vs_random_cos_corr_gap": float(seq_cos_corr - avg_random_cos_corr),
            "variance_concentration_top256": float(np.sum(np.sort(dim_variance)[::-1][:256]) / np.sum(dim_variance)),
        }

        print(f"    Variance CV: {basis_results[basis_name]['dim_variance_cv']:.4f}")
        print(f"    Entropy: {basis_results[basis_name]['dim_variance_entropy']:.4f}")
        print(f"    Seq cos corr: {basis_results[basis_name]['seq_cosine_correlation']:.4f}")
        print(f"    Random cos corr: {basis_results[basis_name]['random_cosine_correlation']:.4f}")
        print(f"    Seq-Random gap: {basis_results[basis_name]['seq_vs_random_cos_corr_gap']:.4f}")
        print(f"    Seq NN overlap: {basis_results[basis_name]['seq_nn_overlap']:.4f}")
        print(f"    Random NN overlap: {basis_results[basis_name]['random_nn_overlap']:.4f}")

    results["basis_results"] = basis_results

    # Key comparison table
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"{'Basis':<25} {'Entropy':<10} {'Seq-CosCorr':<13} {'Rnd-CosCorr':<13} {'Gap':<10} {'Seq-NN':<10} {'Rnd-NN':<10}")
    print("-" * 91)
    for basis_name, br in basis_results.items():
        print(f"{basis_name:<25} {br['dim_variance_entropy']:<10.4f} {br['seq_cosine_correlation']:<13.4f} {br['random_cosine_correlation']:<13.4f} {br['seq_vs_random_cos_corr_gap']:<10.4f} {br['seq_nn_overlap']:<10.4f} {br['random_nn_overlap']:<10.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"basis_sensitivity_{model_name}.json")
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
    run_experiment_for_model(args.model, args.gpu, output_dir)


if __name__ == "__main__":
    main()
