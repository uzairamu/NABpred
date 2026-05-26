"""
Fig. 4 — Holdout benchmark: ROC and PR curves for NABpred vs catGRANULE 2.0.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

DATA_PATH = (Path(__file__).parent.parent / "data" / "holdout"
             / "final_dataset_corrected.csv")
OUT_DIR = Path(__file__).parent.parent / "figures"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(DATA_PATH))
    parser.add_argument("--output", default=str(OUT_DIR / "Fig4_holdout_benchmark.pdf"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y  = df["Label"].values

    methods = {
        "NABpred":        ("ModelScore", "#2166ac", "-"),
        "catGRANULE 2.0": ("LLPS_Score", "#d6604d", "--"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for name, (col, color, ls) in methods.items():
        scores = df[col].values

        # ROC
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, ls=ls,
                     label=f"{name}  (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        axes[1].plot(rec, prec, color=color, ls=ls,
                     label=f"{name}  (AP={ap:.3f})")

    # ROC panel
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve — holdout benchmark")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # PR panel
    baseline = y.mean()
    axes[1].axhline(baseline, color="k", ls=":", lw=0.8, label=f"Baseline ({baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curve")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
