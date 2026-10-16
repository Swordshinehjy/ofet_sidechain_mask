"""Tests for safe checkpoint I/O (.safetensors round-trip)."""

import json

import numpy as np
import pytest
import torch
from safetensors import safe_open
from sklearn.preprocessing import StandardScaler

from polymer_ranking.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_dict,
    _json_default,
)
from polymer_ranking.config import ModelConfig
from polymer_ranking.model import PolymerRankingModel

from conftest import TINY_MODEL_KWARGS


def _dummy_scaler():
    scaler = StandardScaler()
    scaler.mean_ = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler.var_ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    scaler.scale_ = np.sqrt(scaler.var_)
    scaler.n_features_in_ = 5
    scaler.n_samples_seen_ = 10
    return scaler


def test_save_checkpoint_appends_safetensors_suffix(tmp_path):
    model = PolymerRankingModel(**TINY_MODEL_KWARGS)
    path = save_checkpoint(tmp_path / "ckpt", model.state_dict())
    assert path.name == "ckpt.safetensors"
    assert path.exists()


def test_save_checkpoint_creates_parent_dirs(tmp_path):
    model = PolymerRankingModel(**TINY_MODEL_KWARGS)
    path = save_checkpoint(tmp_path / "deep" / "nested" / "m", model.state_dict())
    assert path.exists()


def test_roundtrip_model_state(tmp_path):
    torch.manual_seed(0)
    model = PolymerRankingModel(**TINY_MODEL_KWARGS)
    state = model.state_dict()
    path = save_checkpoint(tmp_path / "m", state)
    loaded = load_checkpoint_dict(str(path))
    assert set(loaded["model_state"]) == set(state)
    for k, v in state.items():
        torch.testing.assert_close(loaded["model_state"][k], v)


def test_roundtrip_config_and_extra(tmp_path):
    cfg = ModelConfig(hidden_size=32, depth=3)
    model = PolymerRankingModel(**cfg.to_dict())
    path = save_checkpoint(tmp_path / "m", model.state_dict(),
                           config=cfg.to_dict(),
                           extra={"delta_scale": 1.25, "split_method": "group"})
    loaded = load_checkpoint_dict(str(path))
    assert loaded["config"]["hidden_size"] == 32
    assert loaded["delta_scale"] == 1.25
    assert loaded["split_method"] == "group"


def test_roundtrip_scaler(tmp_path):
    model = PolymerRankingModel(**TINY_MODEL_KWARGS)
    scaler = _dummy_scaler()
    path = save_checkpoint(tmp_path / "m", model.state_dict(), scaler=scaler)
    loaded = load_checkpoint_dict(str(path))
    restored = loaded["scaler"]
    np.testing.assert_allclose(restored.mean_, scaler.mean_)
    np.testing.assert_allclose(restored.var_, scaler.var_)
    np.testing.assert_allclose(restored.scale_, scaler.scale_)
    assert restored.n_features_in_ == 5
    assert restored.n_samples_seen_ == 10
    # restored scaler actually transforms
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    np.testing.assert_allclose(restored.transform(x), [[0.0, 0.0, 0.0, 0.0, 0.0]])


def test_load_checkpoint_rebuilds_model(trained_checkpoint):
    path, cfg, scaler, model = trained_checkpoint
    loaded_cfg, loaded_scaler, loaded_model = load_checkpoint(str(path))

    assert loaded_cfg == cfg
    assert isinstance(loaded_model, PolymerRankingModel)
    assert not loaded_model.training  # eval mode
    np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_)


def test_load_checkpoint_outputs_match_original(trained_checkpoint):
    from chemprop.data.collate import BatchMolGraph
    from polymer_ranking.featurizer import create_featurizer

    path, _, _, model = trained_checkpoint
    _, _, loaded_model = load_checkpoint(str(path))

    feat = create_featurizer()
    from rdkit import Chem
    from samples import THIOPHENE, BENZENE

    graphs = [feat(Chem.MolFromSmiles(THIOPHENE)), feat(Chem.MolFromSmiles(BENZENE))]
    bmg = BatchMolGraph(graphs)
    device = next(loaded_model.parameters()).device
    ef = torch.zeros(2, 5, device=device)

    with torch.no_grad():
        s_ref = model.encode(bmg, [1, 1], ef)
        s_new = loaded_model.encode(bmg, [1, 1], ef)
    torch.testing.assert_close(s_ref.cpu(), s_new.cpu(), rtol=1e-5, atol=1e-6)


def test_metadata_is_plain_json(tmp_path):
    """safetensors metadata must not rely on pickle (safe I/O guarantee)."""
    model = PolymerRankingModel(**TINY_MODEL_KWARGS)
    path = save_checkpoint(tmp_path / "m", model.state_dict(),
                           config={"hidden_size": 16},
                           extra={"delta_scale": 2.0})
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    assert json.loads(meta["config"])["hidden_size"] == 16
    assert json.loads(meta["extra.delta_scale"]) == 2.0


def test_json_default_unserializable_type_raises():
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=_json_default)


def test_json_default_handles_numpy_scalars_and_arrays():
    payload = {"i": np.int64(3), "f": np.float32(0.5), "a": np.array([1, 2])}
    parsed = json.loads(json.dumps(payload, default=_json_default))
    assert parsed == {"i": 3, "f": 0.5, "a": [1, 2]}
