"""
download_weights.py — Download NABpred model weights from Zenodo.

Usage:
    python download_weights.py
    python download_weights.py --output checkpoints/final_mlp_model.pt
"""

import argparse
import urllib.request
import os
from pathlib import Path

ZENODO_DOI = "10.5281/zenodo.20407029"
ZENODO_URL = "https://zenodo.org/records/20407029/files/final_mlp_model.pt"
DEFAULT_PATH = "checkpoints/final_mlp_model.pt"

def download(url: str, dest: str):
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model weights from Zenodo...")
    print(f"  Source: {url}")
    print(f"  Destination: {dest}")

    def progress(count, block_size, total_size):
        pct = count * block_size / total_size * 100
        print(f"\r  {min(pct, 100):.1f}%", end='', flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\nSaved: {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Download NABpred model weights from Zenodo.')
    parser.add_argument('--output', default=DEFAULT_PATH,
                        help=f'Save path (default: {DEFAULT_PATH})')
    args = parser.parse_args()

    if Path(args.output).exists():
        print(f"Already exists: {args.output}  (delete to re-download)")
        return

    download(ZENODO_URL, args.output)
    print(f"\nDOI: {ZENODO_DOI}")
    print("Run inference with:")
    print(f"  python model/predict.py --fasta input.fasta "
          f"--checkpoint {args.output} --output predictions.csv")


if __name__ == '__main__':
    main()
