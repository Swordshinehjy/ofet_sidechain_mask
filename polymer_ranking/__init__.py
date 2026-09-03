"""
Polymer Carrier Mobility Prediction
=====================================
Pipeline:
  1. Monomer concatenation (A+B -> repeat unit, handles asymmetric monomers)
  2. Polymer repeat unit cyclization -> cyclic model compound
  3. chemprop v2 D-MPNN encoding (multiple structures averaged) + additional feature concatenation
  4. Bayesian Personalized Ranking (BPR) Loss + Delta Regression
  5. Multi-task simultaneous prediction of electron mobility (mu_e) and hole mobility (mu_h)
"""

import logging

from .config import EXTRA_COLS, EXTRA_DIM, TASK_NAMES, NUM_TASKS, CP_FEATURE_DIM, ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig
from .featurizer import CustomMultiHotAtomFeaturizer, create_featurizer, get_feature_dims
from .chemistry import (
    normalize_smiles, get_star_info, connect_at_stars, generate_all_connections,
    cyclize_polymer_with_cp_marking, generate_cyclized_from_monomers,
    cyclize_df, extra_feat, load_and_preprocess,
)
from .dataset import PairDataset, CachedPairDataset, collate_fn, collate_cached_batch
from .model import DMPNNEncoder, PolymerRankingModel
from .loss import MultiTaskBayesianRankingLoss
from .training import EarlyStopping, train, finetune
from .predict import load_checkpoint, predict_pair, predict_batch
from .cli import main

logger = logging.getLogger(__name__)

try:
    from chemprop.data.molgraph import MolGraph
    from chemprop.data.collate import BatchMolGraph, collate_batch
    logger.info("chemprop v2 loaded.")
except ImportError:
    logger.warning("chemprop not installed. Run: pip install chemprop>=2.0.0")
