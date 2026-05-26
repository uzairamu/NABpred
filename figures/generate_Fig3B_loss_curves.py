"""
Fig. 3B — Training and validation loss curves.

Re-trains the 5-fold CV to capture per-epoch losses; requires X_train.npy.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import ShallowMLP

TRAIN_LABELS = Path(__file__).parent.parent / "data" / "training_labels.csv"
OUT_DIR = Path(__file__).parent.parent / "figures"

LR = 1e-5; EPOCHS = 80; BATCH_SIZE = 32; N_FOLDS = 5; SEED = 42


def run_fold(X_tr, y_tr, X_val, y_val, device):
    model = ShallowMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.BCEWithLogitsLoss()
    dl    = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED))
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0; n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            epoch_loss += loss.item() * len(xb); n += len(xb)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_losses.append(crit(model(X_v), y_v).item())

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", default=str(TRAIN_LABELS))
    parser.add_argument("--output", default=str(OUT_DIR / "Fig3B_loss_curves.pdf"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(args.embeddings)
    y = pd.read_csv(args.labels)["Label"].values.astype(int)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_train, all_val = [], []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{N_FOLDS}...")
        torch.manual_seed(SEED)
        tl, vl = run_fold(X[tr], y[tr], X[va], y[va], device)
        all_train.append(tl); all_val.append(vl)

    tr_mean = np.mean(all_train, axis=0)
    tr_std  = np.std(all_train, axis=0)
    va_mean = np.mean(all_val,   axis=0)
    va_std  = np.std(all_val,    axis=0)
    epochs  = np.arange(1, EPOCHS + 1)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, tr_mean, label="Train", color="#2166ac")
    ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                    alpha=0.15, color="#2166ac")
    ax.plot(epochs, va_mean, label="Validation", color="#d6604d")
    ax.fill_between(epochs, va_mean - va_std, va_mean + va_std,
                    alpha=0.15, color="#d6604d")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training and validation loss (5-fold CV)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
