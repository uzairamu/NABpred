#!/bin/bash

set -e

echo "=== NABpred GitHub Upload ==="

# Initialize git
git init
git config user.name "Muhammad Uzair Ashraf"
git config user.email "uzairamu@myamu.ac.in"

# Add remote
git remote add origin https://github.com/uzairamu/NABpred.git

# Stage all files (gitignore excludes large .npy files)
git add .
git status

# Initial commit
git commit -m "Initial release: NABpred v1.0.0

NABpred: a dedicated nucleic acid-binding protein predictor
reveals systematic confounding in LLPS models.

- ShallowMLP trained on frozen ESM2-650M embeddings
- Training, inference, and evaluation scripts
- All benchmark datasets
- Figure generation scripts
- Precomputed screening results

Manuscript: Ashraf MU, Ahmad O, Khan RH (2025) NAR Methods"

# Push to main branch
git branch -M main
git push -u origin main

echo ""
echo "=== Upload complete ==="
echo "Repo: https://github.com/uzairamu/NABpred"
echo ""
echo "Next steps:"
echo "  1. Go to https://zenodo.org"
echo "  2. Connect GitHub account -> enable NABpred repo"
echo "  3. Create a GitHub Release (v1.0.0)"
echo "  4. Zenodo auto-generates DOI"
echo "  5. Upload large files separately to Zenodo:"
echo "     - X_train.npy (7.3 MB)"
echo "     - X_human_NAB_proteome.npy (137 MB)"
