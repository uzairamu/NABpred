# NABpred — Data Directory

## Files in this repository

| File | N | Description |
|------|---|-------------|
| `training_labels.csv` | 711 | UniProt accessions + LLPS labels for training set (400 positive, 311 negative) |
| `holdout/final_dataset_corrected.csv` | 471 | Corrected human NAB holdout benchmark (332 positive, 139 negative) with NABpred and catGRANULE 2.0 scores |
| `cross_species/cross_species_fiveway_final.csv` | 745 | Five-way cross-species benchmark (positive-only) with scores for NABpred, PScore, PICNIC, and PSPire |
| `phasepdb/phasepdb_independent_corrected.csv` | 572 | PhaSepDB-curated LLPS-positive NABs after accession- and sequence-level overlap removal |
| `phasepdb/catgranule_phasepdb_results.csv` | 545 | Subset of PhaSepDB proteins with catGRANULE 2.0 scores available |

## Files on Zenodo (too large for GitHub)

| File | Size | Description |
|------|------|-------------|
| `X_train.npy` | 7.3 MB | Precomputed ESM2-650M embeddings for 711 training proteins (shape: 711 × 2560) |
| `X_human_NAB_proteome.npy` | 137 MB | Precomputed ESM2-650M embeddings for 13,349 human NABs (shape: 13349 × 2560) |

Download from: [Zenodo DOI link]

## Column descriptions

### training_labels.csv
- `Entry`: UniProt accession
- `Label`: 1 = LLPS-positive NAB, 0 = non-LLPS NAB

### final_dataset_corrected.csv
- `ID`: UniProt accession
- `Label`: corrected LLPS label (1/0)
- `Sequence`: full protein sequence
- `ModelScore`: NABpred LLPS propensity score (sigmoid output)
- `LLPS_Score`: catGRANULE 2.0 score

### cross_species_fiveway_final.csv
- `Entry_norm`: UniProt accession
- `class`: LLPS label (1 = positive)
- `YourScore`: NABpred propensity score
- `PScore`: PScore value
- `PICNIC_score`: PICNIC score
- `PSPire_score`: PSPire score

### phasepdb_independent_corrected.csv
- `Entry`: UniProt accession
- `Sequence`: full protein sequence
- `NABpred_score`: NABpred propensity score
- `NABpred_pred`: binary prediction (threshold 0.5)

### catgranule_phasepdb_results.csv
- `Entry`: UniProt accession
- `Sequence`: full protein sequence
- `catGRANULE_score`: catGRANULE 2.0 score
- `catGRANULE_pred`: binary prediction
- `NABpred_score`: NABpred propensity score
- `NABpred_pred`: binary prediction
