# Polymer Carrier Mobility Prediction

Contrastive prediction of polymer carrier mobility based on D-MPNN and Bayesian Personalized Ranking (BPR). Given a pair of polymers, the model simultaneously predicts the relative order of electron mobility (mu_e) and hole mobility (mu_h).

## Pipeline Overview

```
MonomerA + MonomerB (with * markers)
        │
        ▼
  ① Monomer concatenation (A+B → repeat unit, handles asymmetric monomers)
        │
        ▼
  ② Repeat unit cyclization → cyclic model compound (connection points marked as CP)
        │
        ▼
  ③ chemprop v2 D-MPNN encoding (multiple structures averaged) + extra feature concatenation
        │
        ▼
  ④ BPR ranking loss + delta regression loss
        │
        ▼
  ⑤ Multi-task prediction of mu_e / mu_h
```

## Modules

| Module | Description |
|--------|-------------|
| `config.py` | Constants and configuration dataclasses (`ModelConfig`, `TrainingConfig`, `FinetuneConfig`, `PredictConfig`) |
| `chemistry.py` | Chemical preprocessing: SMILES canonicalization, monomer concatenation, cyclization, extra feature extraction, DataFrame preprocessing |
| `featurizer.py` | Custom atom featurizer (chemprop `MultiHotAtomFeaturizer` + CP binary feature) and factory function |
| `dataset.py` | `PairDataset` / `CachedPairDataset` and collate functions |
| `model.py` | `DMPNNEncoder` (chemprop v2 `BondMessagePassing`) + `PolymerRankingModel` (siamese network + FFN) |
| `loss.py` | `MultiTaskBayesianRankingLoss`: weighted BPR ranking loss + delta regression loss |
| `training.py` | Training logic: `EarlyStopping`, epoch runner, metric computation, `train()` / `finetune()` |
| `predict.py` | Inference: checkpoint loading, single-pair prediction (`predict_pair`), batch prediction (`predict_batch`) |
| `cli.py` | Command line entry point with `train` / `finetune` / `predict` modes |

## Requirements

- Python >= 3.9
- PyTorch
- chemprop >= 2.0.0
- RDKit
- scikit-learn
- scipy
- pandas
- numpy

## Data Format

The training CSV must contain the following columns (`_1` / `_2` suffixes distinguish the two polymers):

| Column | Description |
|--------|-------------|
| `Materials_1` / `Materials_2` | Polymer name |
| `MonomerA_1` / `MonomerA_2` | SMILES of monomer A (with 2 `*` markers) |
| `MonomerB_1` / `MonomerB_2` | SMILES of monomer B (with 2 `*` markers) |
| `conjugation_{s}` | Conjugation (linear=1, otherwise=0) |
| `Isomer_{s}` | Isomer flag |
| `CentroSymmetry_{s}` | Centrosymmetry flag |
| `E_LUMO (eV)_{s}` | LUMO level |
| `E_HOMO (eV)_{s}` | HOMO level |
| `mu_e_{s}` | Electron mobility (raw value; log10 is applied internally) |
| `mu_h_{s}` | Hole mobility (raw value; log10 is applied internally) |

`{s}` is either `1` or `2`.

### Data Preprocessing

`preprocess.py` groups the raw monomer data by DOI and generates paired data:

```bash
python preprocess.py
# reads contrastive_monomer.csv → writes contrastive_monomer_paired.csv
```

## Usage

### Training

```bash
python polymer_ranking.py --mode train --csv contrastive_monomer_paired.csv
```

Optional arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--hidden_size` | 256 | D-MPNN hidden dimension |
| `--depth` | 6 | Message passing depth |
| `--dropout` | 0.1 | Dropout rate |
| `--ffn_hidden` | 128 | FFN hidden dimension |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-3 | Weight decay |
| `--rank_weight` | 0.8 | Weight alpha of the BPR ranking term |
| `--reg_weight` | 0.2 | Weight beta of the delta regression term |
| `--delta_scale` | auto | Scale of the regression target (std of the measured log10 differences) |
| `--early_stop_metric` | pair_acc | Quantity monitored by early stopping (`pair_acc` / `loss`) |
| `--scheduler` | plateau | LR scheduler (`plateau` / `cosine`) |
| `--patience` | 30 | Early stopping patience |
| `--val_ratio` | 0.1 | Validation set ratio |
| `--test_ratio` | 0.1 | Test set ratio |
| `--seed` | 42 | Random seed |
| `--save_dir` | checkpoints | Checkpoint directory |

### Fine-tuning

Fine-tune an existing checkpoint on the full dataset:

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_monomer_paired.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### Prediction

Batch prediction for new polymer pairs:

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

Prediction output columns:

| Column | Description |
|--------|-------------|
| `score_mu_e_1` / `score_mu_e_2` | mu_e scores of the two polymers |
| `score_mu_h_1` / `score_mu_h_2` | mu_h scores of the two polymers |
| `prob_mu_e` / `prob_mu_h` | Ranking confidence (sigmoid of the score difference) |
| `preferred_mu_e` / `preferred_mu_h` | Polymer with the higher mobility |

## Model Architecture

```
SMILES (multiple structures) ──► D-MPNN ──► averaged embedding ──► mol_emb [H]
                                                    │
extra_features [5] ────────────────────────────────┤
                                                    ▼
                                              Concatenate
                                                    │
                                                    ▼
                                    Linear → LayerNorm → SiLU → Dropout
                                                    │
                                                    ▼
                                    Linear → SiLU → Dropout
                                                    │
                                                    ▼
                                         Linear → [score_e, score_h]
```

The two polymers share one set of parameters (siamese network). A polymer may yield multiple cyclized structures when its monomers are asymmetric; their D-MPNN embeddings are averaged before being fed to the FFN.

## Loss Function

For each task t in {mu_e, mu_h}:

- **BPR ranking loss**: `L_bpr = -log(sigma(sign(y1 - y2) · (s1 - s2)))`, maximizing the log-likelihood of the correct ranking.
- **Delta regression loss**: `L_reg = MSE(s1 - s2, y1 - y2)`, forcing the score difference to match the true difference.

Total loss = sum_t lambda_t · (alpha · L_bpr_t + beta · L_reg_t), with alpha=0.6 and beta=0.4 by default.

## Metrics

| Metric | Description |
|--------|-------------|
| `{task}_pair_acc` | Pairwise ranking accuracy |
| `{task}_spearman` | Spearman rank correlation coefficient |
| `{task}_avg_prob` | Average ranking confidence |
