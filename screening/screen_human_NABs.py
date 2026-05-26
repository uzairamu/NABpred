"""
Screen the human NAB proteome with NABpred.

Input:  precomputed ESM2-650M embeddings (X_human_NAB_proteome.npy — Zenodo)
        + a CSV with UniProt metadata for those proteins
Output: results/human_NAB_proteome_scored.csv
        results/novel_LLPS_candidates.csv  (high-confidence unknowns)

Usage:
  python screening/screen_human_NABs.py \\
    --embeddings data/X_human_NAB_proteome.npy \\
    --metadata   <metadata_csv> \\
    --checkpoint checkpoints/final_mlp_model.pt
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import ShallowMLP

CHECKPOINT   = Path(__file__).parent.parent / "checkpoints" / "final_mlp_model.pt"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
THRESHOLD    = 0.5
HIGH_CONF    = 0.7   # threshold for "novel candidate" list


def predict(embeddings: np.ndarray, checkpoint: str) -> np.ndarray:
    model = ShallowMLP.load(checkpoint)
    X = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    parser = argparse.ArgumentParser(description="Screen human NABs with NABpred.")
    parser.add_argument("--embeddings",  required=True,
                        help="Path to X_human_NAB_proteome.npy")
    parser.add_argument("--metadata",    required=True,
                        help="CSV with protein metadata (must have an ID column)")
    parser.add_argument("--checkpoint",  default=str(CHECKPOINT))
    parser.add_argument("--id_col",      default="Entry")
    parser.add_argument("--label_col",   default="LLPS_status",
                        help="Optional column marking known LLPS status")
    parser.add_argument("--threshold",   type=float, default=THRESHOLD)
    parser.add_argument("--high_conf",   type=float, default=HIGH_CONF)
    parser.add_argument("--output_dir",  default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    X = np.load(args.embeddings)
    meta = pd.read_csv(args.metadata)
    assert len(X) == len(meta), (
        f"Embedding rows ({len(X)}) != metadata rows ({len(meta)})")

    print(f"Scoring {len(X)} proteins...")
    scores = predict(X, args.checkpoint)
    preds  = (scores >= args.threshold).astype(int)

    meta = meta.copy()
    meta["NABpred_score"] = scores
    meta["NABpred_pred"]  = preds

    # Save full scored proteome
    full_out = out_dir / "human_NAB_proteome_scored.csv"
    meta.sort_values("NABpred_score", ascending=False).to_csv(full_out, index=False)
    print(f"Full results: {full_out}  ({len(meta)} proteins)")

    # Novel candidates: predicted positive AND not already annotated as LLPS-positive
    if args.label_col in meta.columns:
        novel_mask = (scores >= args.high_conf) & (
            ~meta[args.label_col].str.contains("positive|known", case=False, na=False))
        novel = meta[novel_mask].sort_values("NABpred_score", ascending=False)
    else:
        novel = meta[scores >= args.high_conf].sort_values("NABpred_score", ascending=False)

    novel_out = out_dir / "novel_LLPS_candidates.csv"
    novel.to_csv(novel_out, index=False)
    print(f"Novel candidates (score >= {args.high_conf}): {len(novel)}")
    print(f"  Saved: {novel_out}")

    print(f"\nSummary:")
    print(f"  Total screened:   {len(meta)}")
    print(f"  LLPS-positive:    {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"  Non-LLPS:         {(1-preds).sum()}")


if __name__ == "__main__":
    main()
