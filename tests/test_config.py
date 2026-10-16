"""Tests for constants and configuration dataclasses."""

import pytest

from polymer_ranking.config import (
    EXTRA_COLS,
    EXTRA_DIM,
    NUM_TASKS,
    TASK_NAMES,
    CP_FEATURE_DIM,
    ModelConfig,
    TrainingConfig,
    FinetuneConfig,
    PredictConfig,
)


def test_constants_consistent():
    assert EXTRA_DIM == len(EXTRA_COLS) == 5
    assert NUM_TASKS == len(TASK_NAMES) == 2
    assert TASK_NAMES == ["mu_e", "mu_h"]
    assert CP_FEATURE_DIM == 1


# ── ModelConfig ─────────────────────────────────────────────────────────────

def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.hidden_size == 256
    assert cfg.depth == 6
    assert cfg.extra_dim == EXTRA_DIM
    assert cfg.num_tasks == NUM_TASKS


def test_model_config_to_dict_roundtrip():
    cfg = ModelConfig(hidden_size=64, depth=3, dropout=0.2)
    restored = ModelConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_model_config_from_dict_ignores_unknown_keys():
    cfg = ModelConfig.from_dict({"hidden_size": 32, "not_a_field": 123})
    assert cfg.hidden_size == 32
    assert not hasattr(cfg, "not_a_field")


# ── TrainingConfig validation ───────────────────────────────────────────────

def test_training_config_valid_defaults():
    cfg = TrainingConfig()
    assert cfg.split_method == "random"
    assert cfg.early_stop_metric == "pair_acc"
    assert cfg.scheduler == "plateau"


@pytest.mark.parametrize("test_ratio", [0.0, 1.0, -0.1, 1.5])
def test_training_config_rejects_bad_test_ratio(test_ratio):
    with pytest.raises(ValueError, match="test_ratio"):
        TrainingConfig(test_ratio=test_ratio)


@pytest.mark.parametrize("val_ratio", [0.95, 1.0, -0.1])
def test_training_config_rejects_bad_val_ratio(val_ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        TrainingConfig(val_ratio=val_ratio)


def test_training_config_val_ratio_must_leave_room_for_test():
    # val_ratio must be < 1 - test_ratio
    with pytest.raises(ValueError):
        TrainingConfig(test_ratio=0.5, val_ratio=0.5)
    TrainingConfig(test_ratio=0.5, val_ratio=0.49)  # boundary is valid


@pytest.mark.parametrize("split_method", ["by_paper", "", "RANDOM"])
def test_training_config_rejects_bad_split_method(split_method):
    with pytest.raises(ValueError, match="split_method"):
        TrainingConfig(split_method=split_method)


@pytest.mark.parametrize("metric", ["auc", "rmse", ""])
def test_training_config_rejects_bad_early_stop_metric(metric):
    with pytest.raises(ValueError, match="early_stop_metric"):
        TrainingConfig(early_stop_metric=metric)


@pytest.mark.parametrize("scheduler", ["step", "onecycle", ""])
def test_training_config_rejects_bad_scheduler(scheduler):
    with pytest.raises(ValueError, match="scheduler"):
        TrainingConfig(scheduler=scheduler)


# ── other configs ───────────────────────────────────────────────────────────

def test_finetune_config_defaults():
    cfg = FinetuneConfig()
    assert cfg.finetune_epochs > 0
    assert cfg.val_ratio == 0.1
    assert cfg.checkpoint_path.endswith(".safetensors")


def test_predict_config_defaults():
    cfg = PredictConfig()
    assert cfg.predict_csv == ""
    assert cfg.batch_size == 32
