

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from model import ShallowMLP

THRESHOLD = 0.5


def predict_from_embeddings(embeddings: np.ndarray,
                            checkpoint: str) -> np.ndarray:
    
    model  = ShallowMLP.load(checkpoint)
    X      = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    parser = argparse.ArgumentParser(
        description='NABpred: predict LLPS propensity for NAB sequences.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--fasta',      help='Input FASTA file')
    src.add_argument('--csv',        help='Input CSV file')
    src.add_argument('--embeddings', help='Precomputed .npy embeddings')

    parser.add_argument('--ids',        help='ID file (one per line) for --embeddings')
    parser.add_argument('--seq_col',    default='Sequence')
    parser.add_argument('--id_col',     default='Entry')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to final_mlp_model.pt')
    parser.add_argument('--output',     required=True,
                        help='Output CSV path')
    parser.add_argument('--threshold',  type=float, default=THRESHOLD,
                        help=f'Classification threshold (default {THRESHOLD})')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    # ── Load sequences / embeddings ────────────────────────────────────────
    if args.embeddings:
        embeddings = np.load(args.embeddings)
        ids = (open(args.ids).read().splitlines()
               if args.ids else [str(i) for i in range(len(embeddings))])
        sequences = None
    else:
        # Generate embeddings on-the-fly
        from embeddings import load_esm2, embed_sequences, read_fasta
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading ESM2-650M...")
        tokeniser, esm_model = load_esm2(device)

        if args.fasta:
            ids, sequences = read_fasta(args.fasta)
        else:
            df        = pd.read_csv(args.csv)
            ids       = df[args.id_col].tolist()
            sequences = df[args.seq_col].tolist()

        print(f"Generating embeddings for {len(sequences)} sequences...")
        embeddings = embed_sequences(sequences, tokeniser, esm_model, device,
                                     batch_size=args.batch_size)

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running NABpred inference...")
    probs = predict_from_embeddings(embeddings, args.checkpoint)
    preds = (probs >= args.threshold).astype(int)

    # ── Build output DataFrame ─────────────────────────────────────────────
    records = {'ID': ids, 'NABpred_score': probs, 'NABpred_pred': preds}
    if sequences is not None:
        records['Sequence_length'] = [len(s) for s in sequences]
    out = pd.DataFrame(records)
    out['LLPS_label'] = out['NABpred_pred'].map({1: 'LLPS-positive', 0: 'Non-LLPS'})
    out = out.sort_values('NABpred_score', ascending=False).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nResults saved: {args.output}")
    print(f"  Total:          {len(out)}")
    print(f"  LLPS-positive:  {preds.sum()} ({preds.mean()*100:.1f}%)")
    print(f"  Non-LLPS:       {(1-preds).sum()}")
    print(f"  Threshold:      {args.threshold}")


if __name__ == '__main__':
    main()
