"""Tests for prediction: single pair, batch prediction, invalid input handling."""

import numpy as np
import pandas as pd
import pytest

from polymer_ranking.predict import predict_pair, predict_batch
from polymer_ranking.config import EXTRA_COLS, TASK_NAMES

from samples import (
    THIOPHENE,
    BENZENE,
    INVALID_MONOMER,
    EXTRA_VALUES_1,
    EXTRA_VALUES_2,
)


def _extra_row(values):
    return np.array(values, dtype=np.float64)


# ── predict_pair ────────────────────────────────────────────────────────────

def test_predict_pair_returns_complete_result(trained_checkpoint):
    path, cfg, scaler, model = trained_checkpoint
    result = predict_pair(
        THIOPHENE, THIOPHENE,
        THIOPHENE, BENZENE,
        _extra_row(EXTRA_VALUES_1), _extra_row(EXTRA_VALUES_2),
        str(path), model=model, scaler=scaler,
    )

    assert set(result) == {"scores_1", "scores_2", "ranking", "probability",
                           "cyc_smiles_1", "cyc_smiles_2"}
    for task in TASK_NAMES:
        assert task in result["scores_1"]
        assert result["ranking"][task] in ("polymer_1", "polymer_2")
        # probability is sigmoid(|score difference|) -> always > 0.5
        assert 0.5 < result["probability"][task] <= 1.0
    assert len(result["cyc_smiles_1"]) >= 1
    assert len(result["cyc_smiles_2"]) >= 1


def test_predict_pair_homocoupling_single_structure(trained_checkpoint):
    path, cfg, scaler, model = trained_checkpoint
    result = predict_pair(
        THIOPHENE, THIOPHENE, THIOPHENE, THIOPHENE,
        _extra_row(EXTRA_VALUES_1), _extra_row(EXTRA_VALUES_1),
        str(path), model=model, scaler=scaler,
    )
    # homocoupling cyclizes to exactly one structure
    assert result["cyc_smiles_1"] == ["c1ccsc1"]


def test_predict_pair_identical_inputs_symmetric(trained_checkpoint):
    """Same polymer on both sides -> identical scores (siamese network)."""
    path, cfg, scaler, model = trained_checkpoint
    result = predict_pair(
        THIOPHENE, BENZENE, THIOPHENE, BENZENE,
        _extra_row(EXTRA_VALUES_1), _extra_row(EXTRA_VALUES_1),
        str(path), model=model, scaler=scaler,
    )
    # not bit-exact: scatter-reduce sums on CUDA are order-dependent (~1e-8)
    for task in TASK_NAMES:
        assert result["scores_1"][task] == pytest.approx(
            result["scores_2"][task], abs=1e-5)
    assert result["cyc_smiles_1"] == result["cyc_smiles_2"]


def test_predict_pair_invalid_monomer_raises(trained_checkpoint):
    path, cfg, scaler, model = trained_checkpoint
    with pytest.raises(ValueError, match="cyclization"):
        predict_pair(
            INVALID_MONOMER, THIOPHENE, THIOPHENE, THIOPHENE,
            _extra_row(EXTRA_VALUES_1), _extra_row(EXTRA_VALUES_2),
            str(path), model=model, scaler=scaler,
        )


# ── predict_batch ───────────────────────────────────────────────────────────

def _predict_df(n_rows=2, extra_invalid=False):
    rows = []
    for i in range(n_rows + (1 if extra_invalid else 0)):
        row = {
            "Materials_1": f"matA{i}",
            "Materials_2": f"matB{i}",
            "MonomerA_1": INVALID_MONOMER if (extra_invalid and i == n_rows) else THIOPHENE,
            "MonomerB_1": BENZENE,
            "MonomerA_2": BENZENE,
            "MonomerB_2": THIOPHENE,
        }
        for s, vals in (("1", EXTRA_VALUES_1), ("2", EXTRA_VALUES_2)):
            for col, v in zip([c.format(s=s) for c in EXTRA_COLS], vals):
                row[col] = v
        rows.append(row)
    return pd.DataFrame(rows)


def test_predict_batch_adds_score_columns(trained_checkpoint, tmp_path):
    path, cfg, scaler, model = trained_checkpoint
    df = _predict_df(2)
    out = predict_batch(df, str(path), output_path=str(tmp_path / "pred.csv"))

    assert len(out) == 2
    for task in TASK_NAMES:
        assert f"score_{task}_1" in out.columns
        assert f"score_{task}_2" in out.columns
        assert f"prob_{task}" in out.columns
        assert f"preferred_{task}" in out.columns
        assert out[f"prob_{task}"].between(0.5, 1.0).all()
    # output CSV was written without the internal mol columns
    saved = pd.read_csv(tmp_path / "pred.csv")
    assert "mol_1" not in saved.columns and "mol_2" not in saved.columns
    assert len(saved) == 2


def test_predict_batch_prefers_material_names(trained_checkpoint):
    path, cfg, scaler, model = trained_checkpoint
    df = _predict_df(3)
    out = predict_batch(df, str(path))
    for task in TASK_NAMES:
        assert out[f"preferred_{task}"].isin(
            df["Materials_1"].tolist() + df["Materials_2"].tolist()).all()


def test_predict_batch_drops_invalid_rows(trained_checkpoint):
    path, cfg, scaler, model = trained_checkpoint
    df = _predict_df(2, extra_invalid=True)
    out = predict_batch(df, str(path))
    assert len(out) == 2  # the invalid row is dropped


def test_predict_batch_smaller_than_batch_size_chunks(trained_checkpoint):
    """batch_size=1 path must produce the same scores as one big batch."""
    path, cfg, scaler, model = trained_checkpoint
    df = _predict_df(3)
    out_one = predict_batch(df, str(path), batch_size=1)
    out_all = predict_batch(df, str(path), batch_size=10)
    np.testing.assert_allclose(
        out_one["score_mu_e_1"].values, out_all["score_mu_e_1"].values,
        rtol=1e-5, atol=1e-6)
