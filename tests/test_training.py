"""Tests for training utilities: EarlyStopping, delta scale, metrics, splits,
and an end-to-end smoke training run."""

import numpy as np
import pytest
import torch
from torch import nn

from polymer_ranking.config import ModelConfig, TrainingConfig, TASK_NAMES
from polymer_ranking.training import (
    EarlyStopping,
    _monitor_value,
    compute_delta_scale,
    compute_delta_scales,
    compute_metrics,
    prepare_splits,
    run_experiment,
)

from conftest import TINY_MODEL_KWARGS


# ── EarlyStopping ───────────────────────────────────────────────────────────

class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)


def test_early_stopping_min_mode():
    stop = EarlyStopping(patience=2, mode="min")
    model = _DummyModel()
    assert not stop.step(1.0, model)   # improves
    assert not stop.step(0.9, model)   # improves
    assert stop.counter == 0
    assert not stop.step(0.95, model)  # worse -> counter 1
    assert stop.step(0.96, model)      # worse -> counter 2 == patience -> stop
    assert stop.best_score == pytest.approx(0.9)


def test_early_stopping_max_mode_and_reset():
    stop = EarlyStopping(patience=1, mode="max")
    model = _DummyModel()
    assert not stop.step(0.5, model)
    assert not stop.step(0.6, model)
    assert not stop.step(0.9, model)   # keeps improving, counter stays 0
    assert stop.best_score == pytest.approx(0.9)
    assert stop.step(0.1, model)       # counter 1 >= patience -> stop
    assert stop.best_state is not None


def test_early_stopping_captures_best_state():
    stop = EarlyStopping(patience=5, mode="min", delta=0.0)
    model = _DummyModel()
    with torch.no_grad():
        model.lin.weight.fill_(1.0)
    stop.step(1.0, model)
    with torch.no_grad():
        model.lin.weight.fill_(2.0)
    stop.step(2.0, model)
    assert torch.all(stop.best_state["lin.weight"] == 1.0)


def test_early_stopping_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(mode="average")


def test_early_stopping_delta_prevents_micro_improvements():
    stop = EarlyStopping(patience=1, delta=0.1, mode="min")
    model = _DummyModel()
    stop.step(1.0, model)
    stop.step(0.95, model)   # improvement < delta -> counts as no improvement
    assert stop.counter == 1


# ── compute_delta_scale ─────────────────────────────────────────────────────

def test_compute_delta_scale_std_of_diffs():
    y1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    y2 = np.array([[0.0, 1.0], [1.0, 2.0]])
    ok = np.ones((2, 2), dtype=bool)
    # diffs: [1, 2, 1, 2] -> std 0.5
    assert compute_delta_scale(y1, y2, ok) == pytest.approx(0.5)


def test_compute_delta_scale_ignores_unmeasured():
    y1 = np.array([[1.0, 5.0]])
    y2 = np.array([[0.0, 5.0]])
    ok = np.array([[True, False]])  # task 1 unmeasured
    assert compute_delta_scale(y1, y2, ok) == pytest.approx(1.0)  # single diff


def test_compute_delta_scale_no_data_returns_one():
    y = np.zeros((2, 2))
    ok = np.zeros((2, 2), dtype=bool)
    assert compute_delta_scale(y, y, ok) == 1.0


def test_compute_delta_scales_is_per_task():
    y1 = np.array([[1.0, 4.0], [3.0, 4.0]])
    y2 = np.array([[0.0, 1.0], [1.0, 1.0]])
    ok = np.ones((2, 2), dtype=bool)
    # mu_e diffs [1, 2] -> std 0.5 ; mu_h diffs [3, 3] -> std 0 -> fallback 1.0
    assert compute_delta_scales(y1, y2, ok) == pytest.approx([0.5, 1.0])


def test_compute_delta_scales_handles_missing_task():
    y1 = np.array([[1.0, 4.0]])
    y2 = np.array([[0.0, 1.0]])
    ok = np.array([[True, False]])
    assert compute_delta_scales(y1, y2, ok) == pytest.approx([1.0, 1.0])


def test_compute_delta_scale_constant_diffs_returns_one():
    y1 = np.array([[2.0], [2.0]])
    y2 = np.array([[1.0], [1.0]])
    ok = np.ones((2, 1), dtype=bool)
    assert compute_delta_scale(y1, y2, ok) == 1.0  # std ~ 0


# ── compute_metrics ─────────────────────────────────────────────────────────

def test_compute_metrics_perfect_ranking():
    # scores proportional to targets (x10) -> all metrics optimal
    y1 = np.array([[3.0, 2.0], [2.0, 3.0], [1.0, 1.0], [0.5, 0.2]])
    y2 = np.array([[0.0, 0.0], [0.5, 0.5], [1.5, 2.0], [2.0, 1.0]])
    s1, s2 = 10 * y1, 10 * y2
    valid = np.ones_like(y1, dtype=bool)
    rank_valid = np.ones_like(y1, dtype=bool)

    m = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
    assert m["mu_e_pair_acc"] == 1.0
    assert m["mu_h_pair_acc"] == 1.0
    assert m["mean_pair_acc"] == 1.0
    assert m["mu_e_spearman"] == pytest.approx(1.0)
    assert m["mu_e_avg_prob"] >= 0.99   # correct direction, high confidence
    assert m["mu_e_n_rank"] == 4.0
    assert m["mu_e_n_reg"] == 4.0


def test_compute_metrics_inverted_ranking():
    # scores are exactly the negated targets -> perfect anti-correlation.
    # dy must vary across pairs, otherwise the difference-level correlation is
    # undefined (spearman returns 0.0 for a constant input).
    y1 = np.array([[3.0], [2.0], [1.0]])
    y2 = np.array([[1.0], [0.5], [2.0]])   # dy = [2.0, 1.5, -1.0]
    s1, s2 = -y1, -y2                      # dp = -dy for every pair
    valid = np.ones_like(y1, dtype=bool)
    m = compute_metrics(s1, s2, y1, y2, valid, valid)
    assert m["mu_e_pair_acc"] == 0.0
    assert m["mu_e_spearman"] == pytest.approx(-1.0)


def test_compute_metrics_excludes_undecidable_pairs():
    # pair 1 has y1 == y2 -> excluded from accuracy
    y1 = np.array([[2.0], [1.0]])
    y2 = np.array([[0.0], [1.0]])
    s1 = np.array([[5.0], [0.0]])
    s2 = np.array([[0.0], [5.0]])   # pair 1 wrongly ranked but undecidable
    m = compute_metrics(s1, s2, y1, y2)
    assert m["mu_e_n_rank"] == 1.0
    assert m["mu_e_pair_acc"] == 1.0


def test_compute_metrics_censored_masking():
    # valid=False -> excluded from spearman; rank_valid=True -> in accuracy
    y1 = np.array([[2.0, 1.0]])
    y2 = np.array([[0.0, 1.0]])
    s1 = np.array([[1.0, 0.0]])
    s2 = np.array([[0.0, 0.0]])
    valid = np.array([[False, True]])
    rank_valid = np.array([[True, True]])
    m = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
    assert m["mu_e_n_reg"] == 0.0     # censored: no spearman
    assert m["mu_e_n_rank"] == 1.0    # but still ranked
    assert m["mu_h_n_rank"] == 0.0    # dy == 0 -> undecidable


# ── prepare_splits ──────────────────────────────────────────────────────────

def test_monitor_value_is_higher_is_better_for_both_metrics():
    """Regression guard: early stopping runs in mode='max', so the monitored
    value must be larger for a *better* model for BOTH metrics."""
    metrics = {"mean_pair_acc": 0.8, "loss": 2.0}
    better = {"mean_pair_acc": 0.9, "loss": 1.0}

    cfg_acc = TrainingConfig(early_stop_metric="pair_acc")
    cfg_loss = TrainingConfig(early_stop_metric="loss")
    for cfg in (cfg_acc, cfg_loss):
        assert _monitor_value(better, cfg) > _monitor_value(metrics, cfg)


def test_early_stopping_keeps_lowest_loss_model():
    """--early_stop_metric loss must keep the model with the *lowest* loss.

    Previously `_monitor_value` returned `-loss` while the stopper used
    mode="min", which kept the worst epoch instead.
    """
    stop = EarlyStopping(patience=2, mode="max")
    model = _DummyModel()
    monitor = {"mean_pair_acc": 0.5}

    for loss, weight in ((5.0, 1.0), (0.1, 2.0), (9.0, 3.0)):
        with torch.no_grad():
            model.lin.weight.fill_(weight)
        stop.step(_monitor_value({"loss": loss}, TrainingConfig(
            early_stop_metric="loss")), model)

    # best monitored value == -0.1 -> the 2.0-filled weights must be kept
    assert float(stop.best_state["lin.weight"].mean()) == pytest.approx(2.0)


def test_prepare_splits_random(prepped_df):
    cfg = TrainingConfig(test_ratio=0.25, val_ratio=0.25, batch_size=2)
    splits = prepare_splits(prepped_df, cfg)

    n_tr, n_va, n_te = splits["sizes"]
    assert n_tr + n_va + n_te == len(prepped_df)
    assert n_tr > 0 and n_va > 0 and n_te > 0
    # train scaler is fitted; val/test reuse it
    assert splits["scaler"].n_features_in_ == 5
    assert len(splits["delta_scale"]) == len(TASK_NAMES)
    assert all(v > 0 for v in splits["delta_scale"])
    # loaders yield batches
    mg1, n1s, mg2, n2s, ef1, ef2, y1, y2, ok1, ok2 = next(iter(splits["tr_loader"]))
    assert y1.shape[1] == 2


def test_prepare_splits_group_no_paper_leakage(prepped_df):
    cfg = TrainingConfig(test_ratio=1 / 3, val_ratio=1 / 3, batch_size=2,
                         split_method="group")
    splits = prepare_splits(prepped_df, cfg)

    doi_tr = set(splits["tr_ds"].df["doi"])
    doi_va = set(splits["va_ds"].df["doi"])
    doi_te = set(splits["te_ds"].df["doi"])
    assert not (doi_tr & doi_va or doi_tr & doi_te or doi_va & doi_te)
    assert doi_tr | doi_va | doi_te == set(prepped_df["doi"])


def test_prepare_splits_group_requires_doi_column(prepped_df):
    cfg = TrainingConfig(split_method="group")
    with pytest.raises(ValueError, match="doi"):
        prepare_splits(prepped_df.drop(columns=["doi"]), cfg)


# ── end-to-end smoke run ────────────────────────────────────────────────────

def test_run_experiment_smoke(prepped_df):
    model_config = ModelConfig(**TINY_MODEL_KWARGS)
    train_config = TrainingConfig(
        epochs=2, batch_size=2, patience=5,
        test_ratio=0.25, val_ratio=0.25, seed=0,
    )
    result = run_experiment(prepped_df, model_config, train_config)

    assert "val_metrics" in result and "mu_e_pair_acc" in result["val_metrics"]
    assert result["best_epoch"] >= 1
    assert result["n_params"] > 0
    assert sum(result["sizes"]) == len(prepped_df)
    assert all(v > 0 for v in result["delta_scale"])
    # metrics well-formed
    vm = result["val_metrics"]
    assert 0.0 <= vm["mu_e_pair_acc"] <= 1.0
    assert -1.0 <= vm["mu_e_spearman"] <= 1.0


def test_train_saves_checkpoint(prepped_df, tmp_path, monkeypatch):
    """Full train() writes a usable .safetensors checkpoint."""
    from polymer_ranking import training
    from polymer_ranking.checkpoint import load_checkpoint_dict

    # patch load_and_preprocess to use the fixture df instead of reading CSV
    monkeypatch.setattr(training, "load_and_preprocess", lambda csv: prepped_df.copy())

    model_config = ModelConfig(**TINY_MODEL_KWARGS)
    train_config = TrainingConfig(
        epochs=1, batch_size=2, patience=5, save_dir=str(tmp_path),
        test_ratio=0.25, val_ratio=0.25, seed=0,
    )
    result = training.train(model_config, train_config)

    ckpt_path = result["checkpoint"]
    assert ckpt_path.name == "best_model.safetensors"
    assert ckpt_path.exists()
    loaded = load_checkpoint_dict(str(ckpt_path))
    assert loaded["config"]["hidden_size"] == TINY_MODEL_KWARGS["hidden_size"]
    assert loaded["scaler"] is not None
    assert "delta_scale" in loaded

    tm = result["test_metrics"]
    assert 0.0 <= tm["mean_pair_acc"] <= 1.0
    assert result["model"] is not None
