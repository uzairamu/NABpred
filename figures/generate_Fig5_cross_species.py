"""
Fig. 5 — Cross-species benchmark: grouped bar chart of per-organism
sensitivity for each method.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

DATA_PATH = (Path(__file__).parent.parent / "data" / "cross_species"
             / "cross_species_fiveway_final.csv")
OUT_DIR = Path(__file__).parent.parent / "figures"

METHODS = {
    "NABpred":  "YourPred",
    "PScore":   "PScorePred",
    "PICNIC":   "PICNIC_pred",
    "PSPire":   "PSPire_pred",
}

COLORS = ["#2166ac", "#4dac26", "#d6604d", "#b2abd2"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default=str(DATA_PATH))
    parser.add_argument("--output", default=str(OUT_DIR / "Fig5_cross_species.pdf"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    if "Organism" not in df.columns:
        print("No 'Organism' column found — plotting overall sensitivity only.")
        orgs = ["All"]
        def get_sub(org): return df
    else:
        orgs = sorted(df["Organism"].unique())
        def get_sub(org): return df[df["Organism"] == org]

    n_orgs    = len(orgs)
    n_methods = len(METHODS)
    x         = np.arange(n_orgs)
    width     = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, n_orgs * 1.5), 4.5))

    for i, (name, col) in enumerate(METHODS.items()):
        if col not in df.columns:
            continue
        sensitivities = [get_sub(o)[col].mean() for o in orgs]
        ax.bar(x + i * width - (n_methods - 1) * width / 2,
               sensitivities, width * 0.9,
               label=name, color=COLORS[i % len(COLORS)])

    short_names = [o.split()[-1] if o != "All" else "All" for o in orgs]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("Sensitivity (fraction predicted positive)")
    ax.set_title("Cross-species benchmark — LLPS-positive NABs")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
