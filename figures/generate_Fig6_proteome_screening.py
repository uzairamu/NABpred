"""
Fig. 6 — Human NAB proteome screening results.

Score distribution, top-candidate annotations, and novel vs known breakdown.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

SCORED_PATH  = Path(__file__).parent.parent / "results" / "human_NAB_proteome_scored.csv"
NOVEL_PATH   = Path(__file__).parent.parent / "results" / "novel_LLPS_candidates.csv"
OUT_DIR      = Path(__file__).parent.parent / "figures"
THRESHOLD    = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored",  default=str(SCORED_PATH))
    parser.add_argument("--novel",   default=str(NOVEL_PATH))
    parser.add_argument("--output",  default=str(OUT_DIR / "Fig6_proteome_screening.pdf"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    scored = pd.read_csv(args.scored)
    novel  = pd.read_csv(args.novel)

    score_col = "NABpred_score"
    preds = (scored[score_col] >= args.threshold).astype(int)
    n_pos = preds.sum()
    n_neg = len(preds) - n_pos

    print(f"Human NAB proteome: {len(scored)} proteins")
    print(f"  LLPS-positive (>= {args.threshold}): {n_pos} ({n_pos/len(scored)*100:.1f}%)")
    print(f"  Non-LLPS:                            {n_neg} ({n_neg/len(scored)*100:.1f}%)")
    print(f"  Novel candidates:                    {len(novel)}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: score distribution
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(scored[score_col], bins=bins, color="#2166ac", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvline(args.threshold, color="#d6604d", ls="--", lw=1.5,
               label=f"Threshold ({args.threshold})")
    ax.set_xlabel("NABpred score")
    ax.set_ylabel("Number of proteins")
    ax.set_title(f"Score distribution\n(n = {len(scored)} human NABs)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: LLPS-status breakdown pie
    ax2 = axes[1]
    if "LLPS_status" in scored.columns:
        counts = scored["LLPS_status"].value_counts()
    else:
        counts = pd.Series({"LLPS-positive": int(n_pos), "Non-LLPS": int(n_neg)})
    ax2.pie(counts.values, labels=counts.index,
            colors=["#2166ac", "#d6604d", "#4dac26"][:len(counts)],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.2))
    ax2.set_title(f"Human NAB proteome\nscreening outcome")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
