"""Prediction logic: checkpoint loading, single pair prediction, batch prediction."""

import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch

try:
    from chemprop.data.collate import BatchMolGraph
except ImportError as e:
    raise ImportError(
        "chemprop>=2.0.0 is required. Install with: pip install chemprop>=2.0.0"
    ) from e

from .config import ModelConfig, TASK_NAMES
from .model import PolymerRankingModel
from .featurizer import create_featurizer
from .chemistry import (
    generate_cyclized_from_monomers,
    cyclize_df,
    extra_feat,
)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(
    checkpoint_path: str,
) -> Tuple[ModelConfig, Any, PolymerRankingModel]:
    """
    Load checkpoint, returns (model_config, scaler, model).
    """
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model_config = ModelConfig.from_dict(ckpt["config"])
    scaler = ckpt["scaler"]

    model = PolymerRankingModel(**model_config.to_dict()).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model_config, scaler, model


@torch.no_grad()
def predict_pair(
    monomerA_1: str,
    monomerB_1: str,
    monomerA_2: str,
    monomerB_2: str,
    extra_raw_1: np.ndarray,
    extra_raw_2: np.ndarray,
    checkpoint_path: str,
    model: Optional[PolymerRankingModel] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Predict electron/hole mobility ranking for a single new structure pair.

    Parameters
    ----------
    monomerA_1/2, monomerB_1/2 : str
        Monomer SMILES with '*' markers.
    extra_raw_1/2 : np.ndarray shape [5]
        [conjugation, isomer, centrosymmetry, E_LUMO(eV), E_HOMO(eV)]
    model, scaler : optional
        Pass an already loaded model/scaler to avoid reloading the checkpoint on
        every call (important when predicting many pairs one by one).
    """
    if model is None or scaler is None:
        _, scaler, model = load_checkpoint(checkpoint_path)
    featurizer = create_featurizer()

    results1 = generate_cyclized_from_monomers(monomerA_1, monomerB_1)
    results2 = generate_cyclized_from_monomers(monomerA_2, monomerB_2)
    if not results1 or not results2:
        raise ValueError(
            "Monomer concatenation/cyclization failed, check SMILES format and * count")

    g1s = [featurizer(mol) for _, mol in results1]
    g2s = [featurizer(mol) for _, mol in results2]

    mg1 = BatchMolGraph(g1s)
    mg2 = BatchMolGraph(g2s)
    n1 = [len(g1s)]
    n2 = [len(g2s)]

    ef1 = torch.tensor(scaler.transform(extra_raw_1.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)
    ef2 = torch.tensor(scaler.transform(extra_raw_2.reshape(1, -1)),
                       dtype=torch.float32).to(DEVICE)

    s1, s2 = model(mg1, n1, ef1, mg2, n2, ef2)
    s1 = s1.cpu().numpy()[0]
    s2 = s2.cpu().numpy()[0]

    scores_1 = {n: float(s1[i]) for i, n in enumerate(TASK_NAMES)}
    scores_2 = {n: float(s2[i]) for i, n in enumerate(TASK_NAMES)}
    ranking = {
        n: ("polymer_1" if s1[i] > s2[i] else "polymer_2")
        for i, n in enumerate(TASK_NAMES)
    }
    probability = {
        n: float(1.0 / (1.0 + np.exp(-abs(s1[i] - s2[i]))))
        for i, n in enumerate(TASK_NAMES)
    }

    cyc_smiles_1 = [r[0] for r in results1]
    cyc_smiles_2 = [r[0] for r in results2]

    return {
        "scores_1": scores_1,
        "scores_2": scores_2,
        "ranking": ranking,
        "probability": probability,
        "cyc_smiles_1": cyc_smiles_1,
        "cyc_smiles_2": cyc_smiles_2,
    }


@torch.no_grad()
def predict_batch(
    df_new: pd.DataFrame,
    checkpoint_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Batch predict all new polymer pairs in DataFrame.
    df_new must contain MonomerA_1, MonomerB_1, MonomerA_2, MonomerB_2
    and 5 extra feature columns per side (same format as training set).
    """
    model_config, scaler, model = load_checkpoint(checkpoint_path)
    featurizer = create_featurizer()

    df = cyclize_df(df_new)

    valid_mask = df["mol_1"].apply(len) > 0
    valid_mask &= df["mol_2"].apply(len) > 0
    valid = df[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Valid pairs: {len(valid)}/{len(df)}")

    ef1 = scaler.transform(extra_feat(valid, "1"))
    ef2 = scaler.transform(extra_feat(valid, "2"))

    all_s1, all_s2 = [], []
    for i in range(0, len(valid), batch_size):
        end_idx = min(i + batch_size, len(valid))

        all_g1s = []
        all_g2s = []
        n1s = []
        n2s = []

        for j in range(i, end_idx):
            mols1 = valid.at[j, "mol_1"]
            mols2 = valid.at[j, "mol_2"]
            g1s = [featurizer(m) for m in mols1]
            g2s = [featurizer(m) for m in mols2]
            all_g1s.extend(g1s)
            all_g2s.extend(g2s)
            n1s.append(len(g1s))
            n2s.append(len(g2s))

        mg1 = BatchMolGraph(all_g1s)
        mg2 = BatchMolGraph(all_g2s)

        t1 = torch.tensor(ef1[i:end_idx], dtype=torch.float32).to(DEVICE)
        t2 = torch.tensor(ef2[i:end_idx], dtype=torch.float32).to(DEVICE)
        s1, s2 = model(mg1, n1s, t1, mg2, n2s, t2)
        all_s1.append(s1.cpu().numpy())
        all_s2.append(s2.cpu().numpy())

    S1 = np.concatenate(all_s1, axis=0)
    S2 = np.concatenate(all_s2, axis=0)

    for i, n in enumerate(TASK_NAMES):
        valid[f"score_{n}_1"] = S1[:, i]
        valid[f"score_{n}_2"] = S2[:, i]
        valid[f"prob_{n}"] = 1.0 / (1.0 + np.exp(-np.abs(S1[:, i] - S2[:, i])))
        valid[f"preferred_{n}"] = np.where(S1[:, i] > S2[:, i],
                                           valid["Materials_1"],
                                           valid["Materials_2"])

    output_cols = [c for c in valid.columns if c not in ("mol_1", "mol_2")]
    if output_path:
        valid[output_cols].to_csv(output_path, index=False)
        logger.info(f"Predictions saved -> {output_path}")
    return valid[output_cols]
