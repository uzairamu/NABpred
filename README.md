# NABpred

**NABpred: a dedicated nucleic acid-binding protein predictor reveals systematic confounding in LLPS models**


---

## Overview

NABpred is a lightweight classifier for predicting liquid–liquid phase separation (LLPS) propensity in nucleic acid-binding proteins (NABs). It uses mean+max-pooled [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D) embeddings as input to a shallow MLP trained on 711 human NABs (400 LLPS-positive, 311 non-LLPS). NABpred achieves strong performance on held-out human NABs and generalises across five non-human species, while existing generic LLPS predictors trained without NAB-specific curation show inflated performance attributable to nucleic acid-binding bias.

---

## Repository structure

```
NABpred/
├── model/
│   ├── model.py          # ShallowMLP architecture
│   ├── train.py          # 5-fold CV + full training pipeline
│   ├── predict.py        # Inference on FASTA / CSV / precomputed embeddings
│   └── embeddings.py     # ESM2-650M embedding generation
├── data/
│   ├── README.md         # Data dictionary
│   ├── training_labels.csv
│   ├── holdout/
│   ├── cross_species/
│   └── phasepdb/
├── evaluation/
│   ├── holdout_benchmark.py
│   ├── cross_species_benchmark.py
│   ├── ablation_study.py
│   └── phasepdb_benchmark.py
├── figures/
│   ├── generate_Fig3B_loss_curves.py
│   ├── generate_Fig4_holdout_benchmark.py
│   ├── generate_Fig5_cross_species.py
│   ├── generate_Fig6_proteome_screening.py
│   └── generate_supplementary_figures.py
├── screening/
│   └── screen_human_NABs.py
└── results/
    ├── novel_LLPS_candidates.csv
    └── human_NAB_proteome_scored.csv
```

---

## Quick start

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate nabpred
```

or with pip:

```bash
pip install -r requirements.txt
```

### 2. Predict on your own sequences (FASTA)

```bash
python model/predict.py \
  --fasta my_proteins.fasta \
  --checkpoint checkpoints/final_mlp_model.pt \
  --output results/my_predictions.csv
```

### 3. Predict from precomputed embeddings

Download `X_train.npy` and `X_human_NAB_proteome.npy` from [Zenodo](https://doi.org/10.5281/zenodo.XXXXXXX).

```bash
python model/predict.py \
  --embeddings data/X_human_NAB_proteome.npy \
  --ids results/human_NAB_proteome_scored.csv \
  --checkpoint checkpoints/final_mlp_model.pt \
  --output results/my_predictions.csv
```

### 4. Retrain the model

```bash
python model/train.py \
  --embeddings data/X_train.npy \
  --labels data/training_labels.csv \
  --output_dir checkpoints/
```

---

## Benchmarks

```bash
# Holdout benchmark (Fig. 4)
python evaluation/holdout_benchmark.py

# Cross-species benchmark (Fig. 5)
python evaluation/cross_species_benchmark.py

# PhaSepDB independent benchmark
python evaluation/phasepdb_benchmark.py

# Ablation study
python evaluation/ablation_study.py
```

---

---

## Large data files (Zenodo)

The precomputed ESM2-650M embedding arrays are too large for GitHub and are hosted on Zenodo:

| File | Size | Description |
|------|------|-------------|
| `X_train.npy` | 7.3 MB | Embeddings for 711 training proteins (711 × 2560) |
| `X_human_NAB_proteome.npy` | 137 MB | Embeddings for 13,349 human NABs (13349 × 2560) |

Download: [Zenodo DOI](https://doi.org/10.5281/zenodo.XXXXXXX)

---



---

## License

MIT — see [LICENSE](LICENSE).
