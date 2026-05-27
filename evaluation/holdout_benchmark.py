# NOTE: Before running, download required files from Zenodo:
#   python download_weights.py          # model checkpoint
#   # Download X_train.npy from https://doi.org/10.5281/zenodo.20407029
#
# Zenodo DOI: 10.5281/zenodo.20407029
# GitHub: https://github.com/uzairamu/NABpred
"""
Holdout benchmark (Fig. 4): NABpred vs catGRANULE 2.0 on the corrected
human NAB holdout set (data/holdout/final_dataset_corrected.csv).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "holdout" / "final_dataset_corrected.csv"
THRESHOLD = 0.5


def classification_report_dict(y_true, y_score, threshold=THRESHOLD):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    mcc_denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc  = (tp*tn - fp*fn) / mcc_denom if mcc_denom > 0 else 0.0
    return dict(AUROC=roc_auc_score(y_true, y_score),
                AUPRC=average_precision_score(y_true, y_score),
                Sensitivity=sens, Specificity=spec,
                Precision=prec, F1=f1, MCC=mcc,
                TP=tp, TN=tn, FP=fp, FN=fn)


def bootstrap_auroc(y_true, y_score, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(aurocs, [2.5, 97.5])
    return lo, hi


def main():
    parser = argparse.ArgumentParser(description="Holdout benchmark.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y  = df["Label"].values

    print(f"Holdout set: {len(df)} proteins | "
          f"{y.sum()} positive | {(1-y).sum()} negative\n")

    methods = {"NABpred": "ModelScore", "catGRANULE 2.0": "LLPS_Score"}

    for name, col in methods.items():
        scores = df[col].values
        m = classification_report_dict(y, scores, args.threshold)
        lo, hi = bootstrap_auroc(y, scores)
        print(f"{'─'*50}")
        print(f"  {name}")
        print(f"  AUROC: {m['AUROC']:.4f}  (95% CI {lo:.4f}–{hi:.4f})")
        print(f"  AUPRC: {m['AUPRC']:.4f}")
        print(f"  Sensitivity: {m['Sensitivity']:.4f}  Specificity: {m['Specificity']:.4f}")
        print(f"  Precision:   {m['Precision']:.4f}  F1: {m['F1']:.4f}  MCC: {m['MCC']:.4f}")
        print(f"  TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
