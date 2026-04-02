---
license: mit
task_categories:
  - feature-extraction
language:
  - en
tags:
  - embeddings
  - pruning
  - dimensionality-reduction
  - mteb
  - sentence-transformers
pretty_name: Prune to Prosper - Embedding Dimension Analysis Data
size_categories:
  - 1K<n<10K
---

# Prune to Prosper - Embedding Dimension Analysis Data

This dataset contains experimental data for the paper **"Dimensions Are Interchangeable: Evidence That Task-Aware Embedding Pruning Does Not Outperform Random Selection"**.

## Dataset Structure

### `analyze/` — Per-Model Chunk Importance Analysis

Chunk-level importance scores for 3 models evaluated with win_size=2 (512 chunks for 1024-dim models):

| File | Model | Size |
|------|-------|------|
| `gte-large-en-v1.5.json` | GTE-Large | 4.7 MB |
| `stella_en_400M_v5.json` | Stella EN 400M | 4.7 MB |
| `roberta-large-InBedder.json` | Roberta-Large-InBedder | 4.7 MB |

Each file contains per-task chunk importance scores, including:
- `task_name` → task → `split_win_size` → win_size → `chunk_result` (512 scores)
- `defult_score`, `random_score`, `sort_score` at task level

### `task_similar/` — Cross-Task Dimension Ranking Transfer

Dimension ranking transfer data for 12 models, showing retention when using task A's ranking to prune for task B.

Each JSON file contains task pairs with:
- Source task dimension ranking
- Target task retention ratio
- Spearman rank correlation between rankings

### `mteb/` — MTEB Evaluation Results

Full MTEB benchmark results for 13 embedding models:

- `gte-large-en-v1.5/`, `stella_en_400M_v5/`, `roberta-large-InBedder/` (detailed models)
- `bge-m3/`, `gte-base/`, `gtr-t5-large/`, `instructor-large/` (additional models)
- `mxbai-embed-large-v1/`, `Qwen3-Embedding-0.6B/` (recent models)
- `roberta-large/`, `bart-base/` (non-contrastive models)
- `gte-Qwen2-1.5B-instruct/`, `jina-embeddings-v3/` (instruction-tuned models)

### `experiment_results/` — Analysis Outputs

Key experimental analysis results:

| File | Description |
|------|-------------|
| `analysis_results.json` (906K) | Main chunk analysis results |
| `near_optimal_mask_analysis.json` (1.7M) | Near-optimal mask degeneracy analysis |
| `universal_mask_analysis.json` | Universal mask transfer experiment |
| `basis_sensitivity_gte-large.json` | Basis independence for GTE-Large |
| `basis_sensitivity_stella.json` | Basis independence for Stella |
| `magnitude_analysis.json` | Magnitude pruning analysis |
| `reviewer_response_analysis.json` | Reviewer response experiments |
| `all_methods_comparison.json` | All 5 methods comparison |

## Key Findings (from this data)

1. **Normalized entropy = 0.988–0.993**: Dimension importance is nearly uniform
2. **Optimized-Random gap = +2.2–5.0%**: Task-aware pruning barely helps
3. **Cross-task retention = 95–100%**: Despite ρ ≈ 0.001 ranking correlation
4. **Basis independence**: Sequential-Random gap < 1% under all tested rotations
5. **31.6% of random masks within 1% of oracle**: Near-optimal mask degeneracy

## Usage

```python
import json

# Load chunk importance for GTE-Large
with open("analyze/gte-large-en-v1.5.json") as f:
    data = json.load(f)

# Get chunk importance for a specific task
task = "Banking77Classification"
scores = data["task_name"][task]["split_win_size"]["2"]["chunk_result"]
print(f"Number of chunks: {len(scores)}")
print(f"Top-10 most important chunks: {sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]}")
```

## Related

- Paper: [GitHub - ngyygm/prune-to-prosper](https://github.com/ngyygm/prune-to-prosper)
- MTEB Benchmark: https://github.com/embeddings-benchmark/mteb
