"""
Supplementary figures:
  S1 — Score distributions per class (holdout)
  S2 — Calibration curve (holdout)
  S3 — PhaSepDB score distributions (NABpred vs catGRANULE 2.0)
  S4 — Top-50 novel LLPS candidate scores
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

HOLDOUT_PATH = (Path(__file__).parent.parent / "data" / "holdout"
                / "final_dataset_corrected.csv")
CATG_PATH    = (Path(__file__).parent.parent / "data" / "phasepdb"
                / "catgranule_phasepdb_results.csv")
NOVEL_PATH   = Path(__file__).parent.parent / "results" / "novel_LLPS_candidates.csv"
OUT_DIR      = Path(__file__).parent.parent / "figures"


def fig_s1(df, out):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, color, name in [(1, "#2166ac", "LLPS-positive"),
                                (0, "#d6604d", "Non-LLPS")]:
        scores = df[df["Label"] == label]["ModelScore"]
        ax.hist(scores, bins=30, alpha=0.6, color=color, label=name,
                density=True, edgecolor="white", lw=0.3)
    ax.set_xlabel("NABpred score")
    ax.set_ylabel("Density")
    ax.set_title("S1 — Score distributions by class (holdout)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"Saved: {out}")


def fig_s2(df, out):
    y, scores = df["Label"].values, df["ModelScore"].values
    frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(mean_pred, frac_pos, "o-", color="#2166ac", label="NABpred")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("S2 — Calibration curve (holdout)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"Saved: {out}")


def fig_s3(cg_df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, (col, title, color) in zip(axes, [
        ("NABpred_score",   "NABpred",        "#2166ac"),
        ("catGRANULE_score", "catGRANULE 2.0", "#d6604d"),
    ]):
        ax.hist(cg_df[col], bins=30, color=color, alpha=0.8,
                edgecolor="white", lw=0.3, density=True)
        ax.axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
        ax.set_xlabel("Score"); ax.set_title(f"S3 — {title}\n(PhaSepDB positives)")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Density")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print(f"Saved: {out}")


def fig_s4(novel_df, out, top_n=50):
    sub = novel_df.nlargest(top_n, "NABpred_score").reset_index(drop=True)
    id_col = "Entry" if "Entry" in sub.columns else sub.columns[0]
    fig, ax = plt.subplots(figsize=(5, max(4, top_n * 0.18)))
    ax.barh(range(len(sub)), sub["NABpred_score"][::-1],
            color="#2166ac", edgecolor="white", lw=0.3)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub[id_col][::-1].tolist(), fontsize=6)
    ax.set_xlabel("NABpred score")
    ax.set_title(f"S4 — Top {top_n} novel LLPS candidates")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout",    default=str(HOLDOUT_PATH))
    parser.add_argument("--catgranule", default=str(CATG_PATH))
    parser.add_argument("--novel",      default=str(NOVEL_PATH))
    parser.add_argument("--outdir",     default=str(OUT_DIR))
    args = parser.parse_args()

    out = Path(args.outdir)
    ho  = pd.read_csv(args.holdout)
    cg  = pd.read_csv(args.catgranule)
    nov = pd.read_csv(args.novel)

    fig_s1(ho,  out / "FigS1_score_distributions.pdf")
    fig_s2(ho,  out / "FigS2_calibration.pdf")
    fig_s3(cg,  out / "FigS3_phasepdb_distributions.pdf")
    fig_s4(nov, out / "FigS4_top50_candidates.pdf")


if __name__ == "__main__":
    main()
