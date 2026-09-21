# Polymer Carrier Mobility Prediction via D-MPNN

Ranking model for polymer carrier mobility based on a D-MPNN (Directed Message Passing Neural Network). The model is trained with a Bayesian Personalized Ranking (BPR) loss on polymer pairs and jointly predicts electron mobility (mu_e) and hole mobility (mu_h).

## Workflow

```
Monomer A + Monomer B ──► monomer concatenation (repeat unit) ──► cyclization (cyclic model compound) ──► D-MPNN encoding ──► FFN ──► ranking score
                                                                        │
                                            extra features (conjugation, isomer, symmetry, LUMO, HOMO) ──┘
                                              │
                                              ▼
                                    edge weights (sp3 C down-weighted to 0.2, others 1.0)
```

1. **Monomer concatenation**: two monomers (containing `*` markers) are joined through their `*` atoms to build the polymer repeat unit. For asymmetric monomers all possible connection orientations are generated automatically.
2. **Cyclization**: the remaining `*` atoms of the repeat unit are joined to form a cyclic model compound, and the connection points (CP) are marked.
3. **D-MPNN encoding (with edge weights)**: molecular graphs are encoded with chemprop v2 `WeightedBondMessagePassing`. During message passing, edges that touch a **non-ring** sp3-hybridized carbon (non-conjugated side-chain fragments such as alkyl chains) receive weight `sp3_weight=0.2`, while ring sp3 carbons and all other edges keep weight 1.0. This reduces the influence of non-conjugated side chains on mobility learning. Multiple cyclized structures of the same polymer are averaged.
4. **Ranking prediction**: extra features are concatenated and fed through an FFN to produce a score. A siamese network with shared parameters compares the mobility of two polymers.

### Note

- **The candidate space is derived from the known chemical space, not entirely new chemistry.** The new structures to be evaluated are basically variants of alkyl chains, substituents, and functional groups with known skeletons, sharing the same chemical family as the training materials. If it is a completely new molecule different from the known system, the model performance will inevitably drop.
- **Training data comes from literature values with high noise.** Due to the generally high noise in experimental material data, it is recommended to repeat the hold-out method multiple times and then average the results.

## Project Structure

```
ofet_monomer_dmpnn/
├── polymer_ranking.py           # Main entry script
├── preprocess.py                # Data preprocessing: generate contrastive pairs
├── preprocess_new.py            # SMILES validation and canonicalization
├── check_smiles.py              # SMILES connection correctness check
├── polymer_ranking/             # Core package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Model / training / prediction configuration
│   ├── model.py                 # WeightedBondMessagePassing encoder + ranking model
│   ├── dataset.py               # Datasets and collate functions
│   ├── chemistry.py             # Chemical preprocessing (concatenation, cyclization, features)
│   ├── featurizer.py            # Custom atom featurizer + sp3 edge weight computation
│   ├── loss.py                  # Multi-task BPR loss
│   ├── training.py              # Training / fine-tuning / early stopping
│   ├── predict.py               # Prediction logic
│   └── cli.py                   # Command line interface
├── contrastive_monomer.csv      # Raw data
├── contrastive_monomer_paired.csv # Preprocessed paired data
├── new_mol.csv                  # New molecules for prediction
└── checkpoints/                 # Model checkpoints
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- [chemprop >= 2.0.0](https://github.com/chemprop/chemprop)
- RDKit
- pandas, numpy, scikit-learn, scipy

```bash
pip install torch chemprop rdkit pandas numpy scikit-learn scipy
```

## Usage

### 1. Data Preprocessing

Group the raw CSV data by DOI and generate contrastive pairs:

```bash
python preprocess.py
```

Reads `contrastive_monomer.csv` and writes `contrastive_monomer_paired.csv`.

### 2. Training

```bash
python polymer_ranking.py --mode train --csv contrastive_monomer_paired.csv
```

Optional arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 200 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size |
| `--lr` | float | 1e-3 | Learning rate |
| `--hidden_size` | int | 256 | Hidden dimension |
| `--depth` | int | 6 | D-MPNN depth |
| `--dropout` | float | 0.1 | Dropout rate |
| `--ffn_hidden` | int | 128 | FFN hidden dimension |
| `--sp3_weight` | float | 0.2 | Message passing weight for edges touching non-ring sp3 carbons |
| `--weight_decay` | float | 1e-3 | AdamW weight decay |
| `--rank_weight` | float | 0.8 | Weight alpha of the BPR ranking term |
| `--reg_weight` | float | 0.2 | Weight beta of the delta regression term |
| `--delta_scale` | float | auto | Scale of the regression target (std of the measured log10 differences) |
| `--early_stop_metric` | str | pair_acc | Quantity monitored by early stopping (`pair_acc` / `loss`) |
| `--scheduler` | str | plateau | LR scheduler (`plateau` / `cosine`) |
| `--patience` | int | 30 | Early stopping patience |
| `--seed` | int | 42 | Random seed |
| `--save_dir` | str | checkpoints | Directory for saving models |

Recommended values come from `hyperparam_search.py --stage all`; see
[Hyper-parameter search](#hyper-parameter-search). |

### 3. Fine-tuning

Fine-tune the best model on the full dataset:

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_monomer_paired.csv \
    --checkpoint checkpoints/best_model.pt \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

### 4. Prediction

Predict mobility ranking for new molecule pairs:

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.pt \
    --output predictions.csv
```

### 5. SMILES Validation

Check whether monomer concatenation correctly produces the target polymer:

```bash
python check_smiles.py
```

### 6. New Data Preprocessing

Canonicalize SMILES in the new molecule CSV:

```bash
python preprocess_new.py
```

## Model Architecture

```
PolymerRankingModel (siamese)
├── DMPNNEncoder
│   ├── WeightedBondMessagePassing (subclasses chemprop v2 BondMessagePassing)
│   │   ├── initial hidden state h0 = W_i(atom_feat || bond_feat)
│   │   ├── message passing m_vw = Σ w_uv · h_uv   (w_uv = 0.2 if the edge touches an sp3 C, else 1.0)
│   │   └── final aggregation M_v = Σ w_wv · h_wv
│   └── Aggregation (mean/sum/norm)
└── FFN
    ├── Linear(hidden + extra_dim → ffn_hidden)
    ├── LayerNorm + SiLU + Dropout
    ├── Linear(ffn_hidden → ffn_hidden/2)
    ├── SiLU + Dropout
    └── Linear(ffn_hidden/2 → num_tasks)
```

- **Edge weight mechanism**: the repeat unit is cyclized into a model compound approximating an infinite chain and encoded with a D-MPNN. **Non-ring** sp3-hybridized carbon atoms (non-conjugated fragments with little influence on mobility, mainly side chains) get a reduced message passing weight. If either endpoint of an edge is a non-ring sp3 carbon, that edge is weighted `sp3_weight` (default 0.2); ring sp3 carbons and all other edges keep weight 1.0. The weight is applied at every message passing iteration and in the final atom-level aggregation.
- **Extra features** (5 dims): conjugation type, isomer, centrosymmetry, E_LUMO, E_HOMO.
- **Loss**: `MultiTaskBayesianRankingLoss` = alpha · BPR Loss + beta · Delta Regression Loss.
- **Optimizer**: AdamW + CosineAnnealingLR.
- **Metrics**: Pairwise Accuracy, Spearman correlation, average ranking probability.

## Input Data Format

### Training / Fine-tuning Data (paired CSV)

The following columns are required (`_1` and `_2` suffixes refer to the two polymers):

| Column | Description |
|--------|-------------|
| `MonomerA_1`, `MonomerB_1` | Monomer SMILES of polymer 1 |
| `MonomerA_2`, `MonomerB_2` | Monomer SMILES of polymer 2 |
| `conjugation_1`, `conjugation_2` | Conjugation type (linear=1, otherwise=0) |
| `Isomer_1`, `Isomer_2` | Isomer flag |
| `CentroSymmetry_1`, `CentroSymmetry_2` | Centrosymmetry flag |
| `E_LUMO (eV)_1`, `E_LUMO (eV)_2` | LUMO level (eV) |
| `E_HOMO (eV)_1`, `E_HOMO (eV)_2` | HOMO level (eV) |
| `mu_e_1`, `mu_e_2` | Electron mobility targets |
| `mu_h_1`, `mu_h_2` | Hole mobility targets |

### Prediction Data (new_mol.csv)

Same format as the training data; the `mu_e` and `mu_h` target columns may be omitted.

### Censored Labels (mobility = 0)

A mobility of `0` means **below the detection limit** (left-censored), i.e. a very
low value, not a missing measurement. The two parts of the loss therefore treat
it differently:

| Situation | Ranking (BPR) | Delta regression |
|-----------|---------------|------------------|
| both values measured (`> 0`) | yes | yes |
| one value censored (`= 0`) | yes — a measured value is always above the detection limit (weighted by `--censored_weight`, default 0.5) | no — the numeric difference is unknown |
| both values censored | no — the ordering is undecidable | no |

Pairwise accuracy is reported on all pairs with a decidable ordering,
while Spearman rho is computed only on pairs with
two measured values.

The detection limit is around 1e-6 cm2/Vs for this data.

## Hyper-parameter Search

`hyperparam_search.py` analyses the data and searches for a good configuration:

```bash
D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage analyze
D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage all --n_trials 10 --max_epochs 40
```

| Stage | What it does |
|-------|--------------|
| `analyze` | Label availability, scale of the log10 differences, class balance, material reuse, plus descriptor-only baselines (LUMO/HOMO heuristic and a logistic regression on the 5 extra features) |
| `search` | Random search over `lr`, `hidden_size`, `depth`, `dropout`, `weight_decay`, `batch_size` and the `rank_weight`/`reg_weight` pair, each trial early-stopped on validation pairwise accuracy |
| `ablation` | `sp3_weight` sweep (0.2 → 1.0) with everything else fixed |
| `all` | All of the above, then prints the recommended configuration |

Results are written to `hyperparam_search.csv` and `best_hyperparams.json`.

## Output

- **Training**: `checkpoints/best_model.pt` — best model on the validation set.
- **Fine-tuning**: `checkpoints/final_model.pt` — model fine-tuned on the full dataset.
- **Prediction**: `predictions.csv` — predicted scores, ranking probabilities and preferred polymer for each pair.
