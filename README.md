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
3. **D-MPNN encoding (with edge weights)**: molecular graphs are encoded with chemprop v2 `WeightedBondMessagePassing`. During message passing, edges that touch a **non-ring** sp3-hybridized carbon (non-conjugated side-chain fragments such as alkyl chains) receive weight `sp3_weight=0.2`, while ring sp3 carbons, halogen-bearing sp3 carbons (CF3/OCF3/SCF3/CH2Cl — strong inductive HOMO/LUMO-shifting groups) and all other edges keep weight 1.0. This reduces the influence of non-conjugated side chains on mobility learning. Multiple cyclized structures of the same polymer are averaged.
4. **Ranking prediction**: extra features are concatenated and fed through an FFN to produce a score. A siamese network with shared parameters compares the mobility of two polymers.

### Note

- **The candidate space is derived from the known chemical space, not entirely new chemistry.** The new structures to be evaluated are basically variants of alkyl chains, substituents, and functional groups with known skeletons, sharing the same chemical family as the training materials. If it is a completely new molecule different from the known system, the model performance will inevitably drop.
- **The literature data are extremely noisy.** *Treat every reported performance with great caution.*

### ⚠ About the default split (read before quoting any metric)

`--split_method` defaults to **`random`**, i.e. pairs are split randomly. Because the same material
appears in several pairs (on average **2.64** pairs per material in `contrastive_full_paired.csv`),
a random pair-level split re-uses materials across train/val/test — the same polymer (and often the
same paper) is seen during training and then evaluated. Every number produced under this split is an
**interpolation / in-distribution** score and is optimistically biased.

For an honest estimate of generalization to *unseen papers*, always train and report with the
group split:

```bash
python polymer_ranking.py --mode train --csv contrastive_full_paired.csv --split_method group
# -> checkpoints/best_model_group.safetensors
```

Under the leakage-free group split (all pairs from one `doi` stay in the same split) the measured
metrics are much lower — only slightly better than a random ordering. This is the number to quote;
the random-split number is not.

The default is kept as `random` for backwards compatibility with the existing checkpoints and
`best_hyperparams.json`, which were also produced under this split (`hyperparam_search.py` defaults
to `--split_method random` as well; pass `--split_method group` to tune on the honest split).

Effective sample size is also much smaller than the row count: with `mu_e`, **641 / 1663** pairs are
censored on *both* sides (below the detection limit) and therefore carry no ranking signal at all.

## Project Structure

```
ofet_sidechain_mask/
├── polymer_ranking.py           # Main entry script
├── hyperparam_search.py         # Data analysis / random search / sp3 ablation
├── draw_molecules.py            # Backbone vs side-chain visualization of polymers
├── polymer_ranking/             # Core package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Model / training / prediction configuration
│   ├── checkpoint.py            # safetensors checkpoint I/O (no pickle)
│   ├── model.py                 # WeightedBondMessagePassing encoder + ranking model
│   ├── dataset.py               # Datasets and collate functions
│   ├── chemistry.py             # Chemical preprocessing (concatenation, cyclization, features)
│   ├── featurizer.py            # Custom atom featurizer + sp3 edge weight computation
│   ├── loss.py                  # Multi-task BPR loss
│   ├── training.py              # Training / fine-tuning / early stopping
│   ├── predict.py               # Prediction logic
│   └── cli.py                   # Command line interface
├── tests/                       # pytest suite
├── contrastive_full_paired.csv  # Paired training data
├── new_mol.csv                  # New molecules for prediction
└── checkpoints/                 # Model checkpoints (git-ignored)
```

The data-preparation helpers mentioned in older revisions (`preprocess.py`, `preprocess_new.py`,
`check_smiles.py`, `convert_checkpoints.py`) no longer exist; the paired CSV is consumed directly.

## Requirements

- Python >= 3.10 (verified on 3.13)
- PyTorch >= 2.0 (verified on 2.10)
- [chemprop >= 2.0.0](https://github.com/chemprop/chemprop) — verified against **2.3.1**; the
  weighted message passing re-implements parts of chemprop's internal `BondMessagePassing.forward`,
  so a major version bump should be re-checked with `pytest`
- RDKit, pandas, numpy, scikit-learn, scipy, safetensors, pillow

```bash
pip install -r requirements.txt
```

`environment.yaml` holds the full pinned conda environment (`chemprop2`) used for development.

## Usage

### 1. Training

```bash
# random pair-level split (default, leaky -- see the note above)
python polymer_ranking.py --mode train --csv contrastive_full_paired.csv

# group split by paper (doi): the honest setting
python polymer_ranking.py --mode train --csv contrastive_full_paired.csv --split_method group
```

No separate preprocessing step is required — monomer concatenation and cyclization happen
inside `load_and_preprocess` when the CSV is read.

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
| `--sp3_weight` | float | 0.2 | Message passing weight for edges touching non-ring, halogen-free sp3 carbons |
| `--weight_decay` | float | 1e-3 | AdamW weight decay |
| `--rank_weight` | float | 0.8 | Weight alpha of the BPR ranking term |
| `--reg_weight` | float | 0.2 | Weight beta of the delta regression term |
| `--delta_scale` | float(s) | auto | Scale of the regression target. One value (broadcast) or two in `mu_e mu_h` order; default: std of the measured log10 differences, **estimated per task** |
| `--censored_weight` | float | 0.5 | BPR weight of pairs containing a censored value (mobility = 0) |
| `--early_stop_metric` | str | pair_acc | Quantity monitored by early stopping (`pair_acc` / `loss`) |
| `--scheduler` | str | plateau | LR scheduler (`plateau` / `cosine`) |
| `--aggregation` | str | mean | Atom aggregation (`mean` / `sum` / `norm`) |
| `--split_method` | str | random | `random` (leaky, default) or `group` (split by `doi`) |
| `--val_ratio` / `--test_ratio` | float | 0.1 / 0.1 | Validation / test proportions |
| `--patience` | int | 30 | Early stopping patience |
| `--seed` | int | 42 | Random seed |
| `--save_dir` | str | checkpoints | Directory for saving models |
| `--hyperparams` | str | best_hyperparams.json | JSON with tuned hyper-parameters; loaded automatically when present, explicit CLI args still win |
| `--log_file` | str | result.log | Log file path |

Recommended values come from `hyperparam_search.py --stage all`; see
[Hyper-parameter search](#hyper-parameter-search).

### 2. Fine-tuning

Fine-tune the best model on the full dataset:

```bash
python polymer_ranking.py --mode finetune \
    --csv contrastive_full_paired.csv \
    --checkpoint checkpoints/best_model.safetensors \
    --finetune_epochs 10 \
    --finetune_lr 1e-5
```

The `rank_weight` / `reg_weight` / `censored_weight` used when the checkpoint was trained are read
back from its metadata; pass them explicitly to override (a warning is logged when they differ).

### 3. Prediction

Predict mobility ranking for new molecule pairs:

```bash
python polymer_ranking.py --mode predict \
    --predict_csv new_mol.csv \
    --checkpoint checkpoints/final_model.safetensors \
    --output predictions.csv
```

`Materials_1` / `Materials_2` are optional; without them the `preferred_*` columns fall back to
`polymer_1` / `polymer_2`.

## Model Architecture

```
PolymerRankingModel (siamese)
├── DMPNNEncoder
│   ├── WeightedBondMessagePassing (subclasses chemprop v2 BondMessagePassing)
│   │   ├── initial hidden state h0 = W_i(atom_feat || bond_feat)
│   │   ├── message passing m_vw = Σ w_uv · h_uv   (w_uv = 0.2 if the edge touches a non-ring, halogen-free sp3 C, else 1.0)
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
- **Optimizer / scheduler**: AdamW + `ReduceLROnPlateau` (default `--scheduler plateau`;
  `--scheduler cosine` selects `CosineAnnealingLR`).
- **Metrics**: Pairwise Accuracy, Spearman correlation of the predicted vs true *pairwise
  difference*, and the average ranking probability. `mean_pair_acc` is the sample-weighted mean
  over `mu_e` / `mu_h` and is what early stopping monitors.

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
while Spearman rho is computed only on pairs with two measured values
(on the predicted vs true *difference* `s1-s2` / `y1-y2`).

The detection limit is around 1e-6 cm2/Vs for this data.

## Hyper-parameter Search

`hyperparam_search.py` analyses the data and searches for a good configuration:

```bash
python hyperparam_search.py --stage analyze
python hyperparam_search.py --stage all --n_trials 10 --max_epochs 40

# tune on the honest (leakage-free) split instead
python hyperparam_search.py --stage all --split_method group --n_trials 10 --max_epochs 40
```

| Stage | What it does |
|-------|--------------|
| `analyze` | Label availability, per-task scale of the log10 differences, class balance, material reuse, plus descriptor-only baselines (LUMO/HOMO heuristic and a logistic regression on the 5 extra features) |
| `search` | Random search over `lr`, `hidden_size`, `depth`, `dropout`, `weight_decay`, `batch_size` and the `rank_weight`/`reg_weight` pair, each trial early-stopped on validation pairwise accuracy |
| `ablation` | `sp3_weight` sweep (0.2 → 1.0) with everything else fixed |
| `all` | All of the above, then prints the recommended configuration |

Results are written to `hyperparam_search.csv` (search), `hyperparam_ablation.csv` (sp3 ablation)
and `best_hyperparams.json` (recommended configuration, loaded automatically by
`polymer_ranking.py --mode train`).

> The split used by the search is `--split_method` (default `random`, i.e. leaky). The
> `val_mean_acc_from_search` recorded in `best_hyperparams.json` is therefore only comparable to
> other `random`-split runs — see [the note above](#-about-the-default-split-read-before-quoting-any-metric).

## Output

- **Training**: `checkpoints/best_model.safetensors` — best model on the validation set.
- **Fine-tuning**: `checkpoints/final_model.safetensors` — model fine-tuned on the full dataset.
- **Prediction**: `predictions.csv` — predicted scores, ranking probabilities and preferred polymer for each pair.

Checkpoints are `.safetensors` only (no pickle). The legacy `.pt` pickle format
and the one-time `convert_checkpoints.py` migration script have been removed;
load only safetensors checkpoints.
