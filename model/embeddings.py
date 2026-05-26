

import argparse
import numpy as np
import torch
from pathlib import Path


ESM2_MODEL  = "facebook/esm2_t33_650M_UR50D"
MAX_LENGTH  = 1022   # ESM2-650M positional embedding limit
BATCH_SIZE  = 8


def load_esm2(device: torch.device):
    
    from transformers import EsmTokenizer, EsmModel
    tokeniser = EsmTokenizer.from_pretrained(ESM2_MODEL)
    model     = EsmModel.from_pretrained(ESM2_MODEL).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokeniser, model


def embed_sequences(sequences: list, tokeniser, model,
                    device: torch.device,
                    batch_size: int = BATCH_SIZE) -> np.ndarray:
    
    sequences = [s[:MAX_LENGTH] for s in sequences]
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokeniser(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=MAX_LENGTH + 2)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden  = outputs.last_hidden_state   # (B, L, 1280)
        mask    = inputs['attention_mask']    # (B, L)

        # Exclude [CLS] and [EOS] tokens (positions 0 and last non-pad)
        mask[:, 0] = 0
        for b in range(mask.shape[0]):
            last = mask[b].sum().item() - 1
            mask[b, int(last)] = 0

        mask_f = mask.unsqueeze(-1).float()
        denom  = mask_f.sum(dim=1).clamp(min=1e-9)

        e_mean = (hidden * mask_f).sum(dim=1) / denom         # (B, 1280)
        e_max  = (hidden + (1 - mask_f) * -1e9).max(dim=1).values  # (B, 1280)

        concat = torch.cat([e_mean, e_max], dim=-1)           # (B, 2560)
        all_embeddings.append(concat.cpu().float().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"  Embedded {min(i + batch_size, len(sequences))}"
                  f"/{len(sequences)} sequences")

    return np.vstack(all_embeddings)


def read_fasta(path: str) -> tuple[list, list]:
    
    ids, seqs, current_seq = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    seqs.append(''.join(current_seq))
                    current_seq = []
                ids.append(line[1:].split()[0])
            else:
                current_seq.append(line)
    if current_seq:
        seqs.append(''.join(current_seq))
    return ids, seqs


def main():
    parser = argparse.ArgumentParser(
        description='Generate ESM2-650M embeddings for NABpred.')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--fasta',  help='Input FASTA file')
    src.add_argument('--csv',    help='Input CSV file')
    parser.add_argument('--seq_col', default='Sequence',
                        help='Column name for sequences (CSV input)')
    parser.add_argument('--id_col',  default='Entry',
                        help='Column name for IDs (CSV input)')
    parser.add_argument('--output', required=True,
                        help='Output .npy file path')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.fasta:
        ids, sequences = read_fasta(args.fasta)
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        ids       = df[args.id_col].tolist()
        sequences = df[args.seq_col].tolist()

    print(f"Loaded {len(sequences)} sequences")
    print("Loading ESM2-650M (first run downloads ~2.5 GB)...")
    tokeniser, model = load_esm2(device)

    print("Generating embeddings...")
    embeddings = embed_sequences(sequences, tokeniser, model, device,
                                 batch_size=args.batch_size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    print(f"Saved: {args.output}  shape={embeddings.shape}")

    # Optionally save ID mapping
    id_out = args.output.replace('.npy', '_ids.txt')
    with open(id_out, 'w') as f:
        f.write('\n'.join(ids))
    print(f"IDs saved: {id_out}")


if __name__ == '__main__':
    main()
