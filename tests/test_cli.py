"""Tests for CLI helpers: argument merging, hyper-parameter loading, config building."""

import argparse
import json

import pytest

from polymer_ranking.cli import merge_args, load_hyperparams, build_config
from polymer_ranking.config import ModelConfig, TrainingConfig


def _args(**kwargs):
    defaults = {"csv": "data.csv", "epochs": None, "lr": None,
                "hidden_size": None, "batch_size": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ── merge_args ──────────────────────────────────────────────────────────────

def test_merge_args_skips_none():
    cfg = merge_args(TrainingConfig, _args(), {
        "csv_path": "csv", "epochs": "epochs", "lr": "lr",
    })
    assert cfg.csv_path == "data.csv"
    assert cfg.epochs == TrainingConfig.epochs      # default kept
    assert cfg.lr == TrainingConfig.lr


def test_merge_args_applies_explicit_values():
    cfg = merge_args(TrainingConfig, _args(epochs=7, lr=0.01), {
        "csv_path": "csv", "epochs": "epochs", "lr": "lr",
    })
    assert cfg.epochs == 7
    assert cfg.lr == 0.01


# ── load_hyperparams ────────────────────────────────────────────────────────

def test_load_hyperparams_none_path():
    assert load_hyperparams(None) == {}
    assert load_hyperparams("") == {}


def test_load_hyperparams_missing_file(tmp_path):
    assert load_hyperparams(str(tmp_path / "nope.json")) == {}


def test_load_hyperparams_valid_json(tmp_path):
    p = tmp_path / "hp.json"
    p.write_text(json.dumps({"hidden_size": 128, "lr": 0.005}), encoding="utf-8")
    assert load_hyperparams(str(p)) == {"hidden_size": 128, "lr": 0.005}


def test_load_hyperparams_invalid_json_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert load_hyperparams(str(p)) == {}


# ── build_config ────────────────────────────────────────────────────────────

def test_build_config_defaults_when_nothing_given():
    cfg = build_config(ModelConfig, _args(), {"hidden_size": "hidden_size"}, {})
    assert cfg == ModelConfig()


def test_build_config_hp_overrides_beat_defaults(tmp_path):
    hp = {"hidden_size": 128, "unknown_field": 999}
    cfg = build_config(ModelConfig, _args(), {"hidden_size": "hidden_size"}, hp)
    assert cfg.hidden_size == 128
    assert not hasattr(cfg, "unknown_field")


def test_build_config_cli_beats_hyperparams():
    hp = {"hidden_size": 128, "lr": 0.005}
    args = _args(hidden_size=64)  # explicit CLI arg
    mapping = {"hidden_size": "hidden_size", "lr": "lr"}
    model_cfg = build_config(ModelConfig, args, mapping, hp)
    assert model_cfg.hidden_size == 64


def test_build_config_cli_none_falls_back_to_hp():
    hp = {"hidden_size": 128}
    args = _args(hidden_size=None)
    cfg = build_config(ModelConfig, args, {"hidden_size": "hidden_size"}, hp)
    assert cfg.hidden_size == 128


def test_build_config_train_config_from_cli():
    hp = {"epochs": 3}
    args = _args(csv="my.csv", epochs=50, batch_size=8)
    mapping = {"csv_path": "csv", "epochs": "epochs", "batch_size": "batch_size"}
    cfg = build_config(TrainingConfig, args, mapping, hp)
    assert cfg.csv_path == "my.csv"       # CLI wins
    assert cfg.epochs == 50               # CLI wins over hp
    assert cfg.batch_size == 8
    assert cfg.lr == TrainingConfig.lr    # untouched -> default
