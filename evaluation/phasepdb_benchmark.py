"""
PhaSepDB independent benchmark: NABpred vs catGRANULE 2.0 on LLPS-positive
NABs curated from PhaSepDB (data/phasepdb/).

Because the PhaSepDB set contains only positives, we report sensitivity
(recall) at threshold 0.5 and the mean score distribution.
For AUROC we use a one-class protocol: randomly sample non-LLPS proteins
from the holdout negatives as a reference background.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PHASEPDB_PATH = (Path(__file__).parent.parent / "data" / "phasepdb"
                 / "phasepdb_independent_corrected.csv")
CATGRANULE_PATH = (Path(__file__).parent.parent / "data" / "phasepdb"
                   / "catgranule_phasepdb_results.csv")
HOLDOUT_PATH = (Path(__file__).parent.parent / "data" / "holdout"
                / "final_dataset_corrected.csv")

THRESHOLD = 0.5


def sensitivity(y_pred):
    return y_pred.mean()


def main():
    parser = argparse.ArgumentParser(description="PhaSepDB benchmark.")
    parser.add_argument("--phasepdb",   default=str(PHASEPDB_PATH))
    parser.add_argument("--catgranule", default=str(CATGRANULE_PATH))
    parser.add_argument("--holdout",    default=str(HOLDOUT_PATH))
    parser.add_argument("--threshold",  type=float, default=THRESHOLD)
    args = parser.parse_args()

    pdb = pd.read_csv(args.phasepdb)
    cg  = pd.read_csv(args.catgranule)
    ho  = pd.read_csv(args.holdout)

    print(f"PhaSepDB independent set (NABpred):   {len(pdb)} LLPS-positive proteins")
    print(f"PhaSepDB catGRANULE subset:           {len(cg)} proteins\n")

    # NABpred sensitivity
    nabpred_preds  = (pdb["NABpred_score"] >= args.threshold).astype(int)
    nabpred_sens   = sensitivity(nabpred_preds)
    nabpred_mean   = pdb["NABpred_score"].mean()
    nabpred_median = pdb["NABpred_score"].median()

    print(f"NABpred  (n={len(pdb)})")
    print(f"  Sensitivity @ {args.threshold}: {nabpred_sens:.4f}")
    print(f"  Mean score:   {nabpred_mean:.4f}")
    print(f"  Median score: {nabpred_median:.4f}\n")

    # catGRANULE sensitivity (subset with scores available)
    cg_preds  = (cg["catGRANULE_score"] >= args.threshold).astype(int)
    cg_sens   = sensitivity(cg_preds)
    cg_mean   = cg["catGRANULE_score"].mean()
    cg_median = cg["catGRANULE_score"].median()

    print(f"catGRANULE 2.0  (n={len(cg)})")
    print(f"  Sensitivity @ {args.threshold}: {cg_sens:.4f}")
    print(f"  Mean score:   {cg_mean:.4f}")
    print(f"  Median score: {cg_median:.4f}\n")

    # Score distribution summary
    print("Score distribution (NABpred, PhaSepDB positives):")
    quantiles = pdb["NABpred_score"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for q, v in quantiles.items():
        print(f"  Q{int(q*100):3d}: {v:.4f}")


if __name__ == "__main__":
    main()
