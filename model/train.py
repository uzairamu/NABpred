
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import ShallowMLP

# ── Hyperparameters (match original training exactly) ─────────────────────
LR          = 1e-5
EPOCHS      = 80
BATCH_SIZE  = 32
DROPOUT     = 0.2
N_FOLDS     = 5
RANDOM_SEED = 42


def train_one_fold(X_tr, y_tr, X_val, y_val, device):
    model     = ShallowMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loader    = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            nn.BCEWithLogitsLoss()(model(xb), yb).backward()
            optimiser.step()

    # Evaluate on validation fold
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    auroc = roc_auc_score(y_val, probs)
    return model, auroc


def train_full(X, y, device):
    
    model     = ShallowMLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loader    = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED))

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()
    return model


def main():
    parser = argparse.ArgumentParser(description='Train NABpred.')
    parser.add_argument('--embeddings', required=True,
                        help='Path to X_train.npy')
    parser.add_argument('--labels', required=True,
                        help='CSV with Entry and Label columns')
    parser.add_argument('--label_col', default='Label')
    parser.add_argument('--output_dir', default='../checkpoints/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X = np.load(args.embeddings)
    df = pd.read_csv(args.labels)
    label_col = args.label_col
    y = df[label_col].values.astype(int)

    assert len(X) == len(y), f"Embedding/label mismatch: {len(X)} vs {len(y)}"
    print(f"Training set: {len(y)} proteins | "
          f"{y.sum()} positive | {(1-y).sum()} negative")

    # ── 5-fold cross-validation ───────────────────────────────────────────
    print(f"\nRunning {N_FOLDS}-fold stratified cross-validation...")
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_SEED)
    aurocs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        _, auroc = train_one_fold(X[tr_idx], y[tr_idx],
                                  X[val_idx], y[val_idx], device)
        aurocs.append(auroc)
        print(f"  Fold {fold+1}: AUROC = {auroc:.4f}")

    mean_auc = np.mean(aurocs)
    std_auc  = np.std(aurocs)
    print(f"\nCV AUROC: {mean_auc:.4f} ± {std_auc:.4f}")

    # ── Train final model on full dataset ─────────────────────────────────
    print("\nTraining final model on full dataset...")
    torch.manual_seed(RANDOM_SEED)
    final_model = train_full(X, y, device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.output_dir) / 'final_mlp_model.pt'
    torch.save({'model_state_dict': final_model.state_dict(),
                'input_dim': X.shape[1],
                'cv_auroc_mean': mean_auc,
                'cv_auroc_std': std_auc,
                'hyperparameters': {
                    'lr': LR, 'epochs': EPOCHS,
                    'batch_size': BATCH_SIZE, 'dropout': DROPOUT,
                }}, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == '__main__':
    main()
