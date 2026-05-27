# NOTE: Before running, download required files from Zenodo:
#   python download_weights.py          # model checkpoint
#   # Download X_train.npy from https://doi.org/10.5281/zenodo.20407029
#
# Zenodo DOI: 10.5281/zenodo.20407029
# GitHub: https://github.com/uzairamu/NABpred
"""
Cross-species benchmark (Fig. 5): per-organism sensitivity for NABpred,
PScore, PICNIC, and PSPire on data/cross_species/cross_species_fiveway_final.csv.

All entries are LLPS-positive NABs, so sensitivity = fraction predicted positive.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_PATH = (Path(__file__).parent.parent / "data" / "cross_species"
             / "cross_species_fiveway_final.csv")

METHODS = {
    "NABpred":  ("YourScore",    "YourPred"),
    "PScore":   ("PScore",       "PScorePred"),
    "PICNIC":   ("PICNIC_score", "PICNIC_pred"),
    "PSPire":   ("PSPire_score", "PSPire_pred"),
}

ORGANISMS = {
    "Saccharomyces cerevisiae": "S. cerevisiae",
    "Caenorhabditis elegans":   "C. elegans",
    "Drosophila melanogaster":  "D. melanogaster",
    "Danio rerio":              "D. rerio (zebrafish)",
    "Mus musculus":             "M. musculus (mouse)",
}


def main():
    parser = argparse.ArgumentParser(description="Cross-species benchmark.")
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Cross-species set: {len(df)} proteins (all LLPS-positive)\n")

    # Overall sensitivity
    print(f"{'Method':<15} {'Overall sensitivity':>20}  {'AUROC*':>8}")
    print("─" * 50)
    for name, (score_col, pred_col) in METHODS.items():
        if score_col not in df.columns:
            print(f"  {name}: column '{score_col}' not found — skipping")
            continue
        preds  = df[pred_col].values.astype(int)
        scores = df[score_col].values
        sens   = preds.mean()
        # All labels are 1, so AUROC is not meaningful globally;
        # report per-organism sensitivity instead
        print(f"  {name:<13} {sens:>20.4f}")

    # Per-organism breakdown (if Organism column present)
    if "Organism" in df.columns:
        orgs = df["Organism"].unique()
        print(f"\nPer-organism sensitivity:")
        header = f"{'Organism':<30}" + "".join(f"{n:>12}" for n in METHODS)
        print(header)
        print("─" * len(header))
        for org in sorted(orgs):
            sub   = df[df["Organism"] == org]
            label = ORGANISMS.get(org, org)
            row   = f"  {label:<28}"
            for name, (_, pred_col) in METHODS.items():
                if pred_col in sub.columns:
                    row += f"{sub[pred_col].mean():>12.4f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
    else:
        print("\n  (No 'Organism' column found — skipping per-organism breakdown)")


if __name__ == "__main__":
    main()
