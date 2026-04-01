# Dimensions Are Interchangeable: Evidence That Task-Aware Embedding Pruning Does Not Outperform Random Selection

This repository contains the paper and code for our study on embedding dimension interchangeability.

## Paper

**Title**: Dimensions Are Interchangeable: Evidence That Task-Aware Embedding Pruning Does Not Outperform Random Selection

**Target Venue**: ACL/EMNLP 2026

**PDF**: [paper/paper.pdf](paper/paper.pdf)

### Key Findings

1. **Dimension importance is nearly uniform** (normalized entropy = 0.988–0.993) across 13 embedding models and 35 MTEB tasks
2. **Task-optimized selection provides only +2–5% improvement over random** at 75% dimension reduction for general-purpose models
3. **Magnitude-based pruning fails** — it performs equal to or worse than random selection
4. **Cross-task transfer retains 95–100% performance** despite zero ranking correlation (ρ ≈ 0.001)
5. **Basis independence**: interchangeability holds under orthogonal rotations, PCA, and whitening

### Paper Structure

```
paper/
├── main.tex          # LaTeX source
├── paper.pdf         # Compiled PDF
├── references.bib    # Bibliography
├── acl_natbib.bst    # Style file
├── eacl2021.sty      # EACL style
└── figures/          # All paper figures (19 images)
```

## Code

All experiment and analysis scripts:

```
src/
├── rank_chunk_mteb.py           # Main chunk-level MTEB evaluation
├── rank_chunk.py                # Core chunk ranking analysis
├── task_similar_mteb.py         # Cross-task similarity analysis
├── basis_sensitivity.py         # Basis independence experiment
├── universal_mask_experiment.py # Universal mask experiment
├── near_optimal_mask_analysis.py # Near-optimal mask degeneracy
├── magnitude_pruning.py         # Magnitude-based pruning analysis
├── redesign_figures.py          # Paper figure generation
├── generate_figures.py          # Additional figure generation
├── MTEBconverter.py             # MTEB data conversion utility
├── analyze_results.py           # Results analysis
├── ...                          # Additional scripts and notebooks
```

### Requirements

```
sentence-transformers
mteb
torch
numpy
scipy
matplotlib
seaborn
```

## Data

Experimental data is available on HuggingFace: [link pending]

```
data/
├── analyze/              # Per-model chunk importance analysis (3 models)
│   ├── gte-large-en-v1.5.json
│   ├── stella_en_400M_v5.json
│   └── roberta-large-InBedder.json
├── task_similar/         # Cross-task dimension ranking transfer (12 models)
├── mteb/                 # MTEB evaluation results (13 models)
│   ├── gte-large-en-v1.5/
│   ├── stella_en_400M_v5/
│   └── ...
└── experiment_results/   # Analysis outputs
    ├── basis_sensitivity_gte-large.json
    ├── basis_sensitivity_stella.json
    ├── universal_mask_analysis.json
    ├── near_optimal_mask_analysis.json
    └── ...
```

### Models Evaluated

| Model | Dimensions | Type |
|-------|-----------|------|
| GTE-Large | 1024 | General-purpose |
| Stella EN 400M | 1024 | General-purpose |
| Roberta-Large-InBedder | 1024 | Task-specific (retrieval) |
| BGE-M3 | 1024 | General-purpose |
| GTE-Qwen2 | 1536 | General-purpose |
| GTE-Base | 768 | General-purpose |
| GTR-T5-Large | 768 | General-purpose |
| Instructor-Large | 768 | General-purpose |
| MxBai-Embed-Large | 1024 | General-purpose |
| Qwen3-Embedding | 1024 | General-purpose |
| Roberta-Large | 1024 | Encoder-only LM |
| BART-Base | 768 | Encoder-decoder LM |

## Citation

```bibtex
@inproceedings{dimensions-interchangeable-2026,
  title={Dimensions Are Interchangeable: Evidence That Task-Aware Embedding Pruning Does Not Outperform Random Selection},
  author={Anonymous},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year={2026}
}
```

## License

MIT
