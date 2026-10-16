"""Shared pytest fixtures for the polymer_ranking test suite."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from polymer_ranking.chemistry import cyclize_df, extra_feat
from polymer_ranking.config import ModelConfig
from polymer_ranking.model import PolymerRankingModel

from samples import make_raw_pair_df

# small enough for fast tests, still exercises the full architecture
TINY_MODEL_KWARGS = dict(
    hidden_size=16,
    depth=2,
    dropout=0.0,
    ffn_hidden=8,
)


@pytest.fixture(scope="session")
def raw_pair_df() -> pd.DataFrame:
    return make_raw_pair_df()


@pytest.fixture(scope="session")
def paired_df(raw_pair_df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame after cyclization (mol_1/mol_2 columns added)."""
    return cyclize_df(raw_pair_df)


@pytest.fixture(scope="session")
def pair_csv(tmp_path_factory) -> str:
    """Training CSV in the same format as contrastive_full_paired.csv."""
    path = tmp_path_factory.mktemp("data") / "pairs.csv"
    make_raw_pair_df().to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def prepped_df(pair_csv) -> pd.DataFrame:
    """Fully preprocessed DataFrame (log targets, ok flags, mol lists)."""
    from polymer_ranking.chemistry import load_and_preprocess

    return load_and_preprocess(pair_csv)


@pytest.fixture(scope="session")
def tiny_model() -> PolymerRankingModel:
    model = PolymerRankingModel(**TINY_MODEL_KWARGS)
    model.eval()
    return model


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory, prepped_df):
    """A saved .safetensors checkpoint (tiny model + fitted scaler).

    The model lives on DEVICE (same as production load_checkpoint).

    Returns (checkpoint_path, model_config, scaler, model).
    """
    from polymer_ranking.checkpoint import save_checkpoint, DEVICE

    cfg = ModelConfig(**TINY_MODEL_KWARGS)
    model = PolymerRankingModel(**cfg.to_dict()).to(DEVICE)
    model.eval()

    ef = np.vstack([extra_feat(prepped_df, "1"), extra_feat(prepped_df, "2")])
    scaler = StandardScaler().fit(ef)

    path = save_checkpoint(
        tmp_path_factory.mktemp("ckpt") / "best_model",
        model_state=model.state_dict(),
        scaler=scaler,
        config=cfg.to_dict(),
        extra={"delta_scale": 1.5, "split_method": "random"},
    )
    return path, cfg, scaler, model
