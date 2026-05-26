"""
Ablation study: compare embedding strategies and model variants on the
holdout benchmark.

Strategies tested (all use ESM2-650M):
  1. mean-only pooling (dim 1280)
  2. max-only pooling (dim 1280)
  3. mean+max concat — NABpred default (dim 2560)

Requires X_train.npy (download from Zenodo) and the holdout CSV.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import ShallowMLP

HOLDOUT_PATH = (Path(__file__).parent.parent / "data" / "holdout"
                / "final_dataset_corrected.csv")
TRAIN_LABELS = (Path(__file__).parent.parent / "data" / "training_labels.csv")

LR = 1e-5; EPOCHS = 80; BATCH_SIZE = 32; DROPOUT = 0.2
N_FOLDS = 5; SEED = 42


def train_cv(X, y, device, input_dim):
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aurocs = []
    for tr_idx, val_idx in skf.split(X, y):
        model = ShallowMLP(input_dim=input_dim, dropout=DROPOUT).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        crit  = nn.BCEWithLogitsLoss()
        dl    = DataLoader(
            TensorDataset(torch.tensor(X[tr_idx], dtype=torch.float32),
                          torch.tensor(y[tr_idx], dtype=torch.float32)),
            batch_size=BATCH_SIZE, shuffle=True)
        for _ in range(EPOCHS):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(
                X[val_idx], dtype=torch.float32).to(device)).cpu().numpy()
        aurocs.append(roc_auc_score(y[val_idx], 1/(1+np.exp(-logits))))
    return np.mean(aurocs), np.std(aurocs)


def main():
    parser = argparse.ArgumentParser(description="Ablation study.")
    parser.add_argument("--embeddings", required=True,
                        help="Path to X_train.npy (mean+max, shape N×2560)")
    parser.add_argument("--labels", default=str(TRAIN_LABELS))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_full = np.load(args.embeddings)   # (N, 2560)
    df = pd.read_csv(args.labels)
    y  = df["Label"].values.astype(int)

    assert X_full.shape[1] == 2560, "Expected mean+max embeddings of dim 2560"
    X_mean = X_full[:, :1280]
    X_max  = X_full[:, 1280:]

    variants = {
        "mean-only  (1280-d)":    X_mean,
        "max-only   (1280-d)":    X_max,
        "mean+max   (2560-d) *":  X_full,
    }

    print(f"Ablation study | {N_FOLDS}-fold CV | device: {device}\n")
    print(f"{'Variant':<30} {'CV AUROC':>10}  {'±':>6}")
    print("─" * 52)
    for name, X in variants.items():
        torch.manual_seed(SEED)
        mu, sd = train_cv(X, y, device, input_dim=X.shape[1])
        print(f"  {name:<28} {mu:.4f}     {sd:.4f}")
    print("\n  * NABpred default")


if __name__ == "__main__":
    main()
