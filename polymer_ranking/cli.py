"""CLI entry point."""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional

import pandas as pd

from .config import ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig
from .training import train, finetune
from .predict import predict_batch

logger = logging.getLogger(__name__)


def merge_args(config_cls, args, arg_mapping: dict):
    """Merge non-None CLI arguments into config dataclass."""
    kwargs = {}
    for config_field, arg_name in arg_mapping.items():
        arg_val = getattr(args, arg_name)
        if arg_val is not None:
            kwargs[config_field] = arg_val
    return config_cls(**kwargs)


def load_hyperparams(path: Optional[str]) -> Dict[str, Any]:
    """Load best-hyperparameter overrides from a JSON file if it exists.

    Returns a dict of overrides (callers only apply the keys that are valid
    fields of the target config). Returns ``{}`` when the file is missing or
    unreadable, so the caller falls back to dataclass defaults + CLI args.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        logger.info("Hyper-parameter file %s not found; using defaults.", path)
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not parse hyper-parameter file %s: %s", path, e)
        return {}
    logger.info("Loaded hyper-parameters from %s", path)
    return data


def build_config(
    config_cls,
    args,
    arg_mapping: dict,
    hp_overrides: Dict[str, Any],
):
    """Build a config dataclass.

    Precedence (highest first): explicit CLI args > best_hyperparams.json >
    dataclass defaults.
    """
    valid = set(config_cls.__dataclass_fields__)
    merged = {k: v for k, v in hp_overrides.items() if k in valid}
    for config_field, arg_name in arg_mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            merged[config_field] = val
    return config_cls(**merged)


def _format_config(cfg) -> str:
    return json.dumps(asdict(cfg), indent=2, default=str)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler("result.log"),
            logging.StreamHandler()
        ]
    )

    p = argparse.ArgumentParser(
        description="Polymer Mobility Bayesian Personalized Ranking (BPR) via D-MPNN")
    p.add_argument("--mode",
                   choices=["train", "predict", "finetune"],
                   default="train")
    p.add_argument("--csv", type=str, default=None, help="Training CSV path")
    p.add_argument("--predict_csv", type=str, default=None, help="Prediction CSV path")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--ffn_hidden", type=int, default=None)
    p.add_argument("--sp3_weight", type=float, default=None,
                   help="Edge weight for bonds touching sp3 carbons (default 0.2)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--rank_weight", type=float, default=None,
                   help="Weight alpha of the BPR ranking term (default 0.8)")
    p.add_argument("--reg_weight", type=float, default=None,
                   help="Weight beta of the delta regression term (default 0.2)")
    p.add_argument("--censored_weight", type=float, default=None,
                   help="BPR weight of pairs with a censored value (mobility=0, "
                        "below the detection limit); default 0.5")
    p.add_argument("--delta_scale", type=float, default=None,
                   help="Scale of the regression target; default: std of the measured "
                        "log10 differences")
    p.add_argument("--early_stop_metric", choices=["pair_acc", "loss"], default=None,
                   help="Quantity monitored by early stopping / LR scheduling")
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default=None,
                   help="Learning rate scheduler (default plateau)")
    p.add_argument("--aggregation", choices=["mean", "sum", "norm"], default=None,
                   help="D-MPNN atom aggregation")
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--split_method", choices=["random", "group"], default=None,
                   help="random: pair-level split; group: split by paper (doi) "
                        "so pairs from the same study never cross train/test "
                        "(model saved as best_model_group.safetensors)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--finetune_epochs",
                   type=int,
                   default=None,
                   help="Fine-tuning epoch count")
    p.add_argument("--finetune_lr", type=float, default=None, help="Fine-tuning learning rate")
    p.add_argument("--hyperparams", type=str, default="best_hyperparams.json",
                   help="Best hyper-parameter JSON from hyperparam_search.py; used "
                        "automatically when present, explicit CLI args still override it")
    args = p.parse_args()

    if args.mode == "train":
        hp = load_hyperparams(args.hyperparams)
        model_config = build_config(
            ModelConfig, args, {
                "hidden_size": "hidden_size",
                "depth": "depth",
                "dropout": "dropout",
                "ffn_hidden": "ffn_hidden",
                "sp3_weight": "sp3_weight",
                "aggregation": "aggregation",
            }, hp)
        train_config = build_config(
            TrainingConfig, args, {
                "csv_path": "csv",
                "save_dir": "save_dir",
                "epochs": "epochs",
                "batch_size": "batch_size",
                "lr": "lr",
                "weight_decay": "weight_decay",
                "patience": "patience",
                "val_ratio": "val_ratio",
                "test_ratio": "test_ratio",
                "split_method": "split_method",
                "seed": "seed",
                "rank_weight": "rank_weight",
                "reg_weight": "reg_weight",
                "censored_weight": "censored_weight",
                "delta_scale": "delta_scale",
                "early_stop_metric": "early_stop_metric",
                "scheduler": "scheduler",
            }, hp)
        logger.info("Model configuration:\n%s", _format_config(model_config))
        logger.info("Training configuration:\n%s", _format_config(train_config))
        results = train(model_config, train_config)
        logger.info("\n===== Final Test Metrics =====")
        for k, v in results["test_metrics"].items():
            logger.info(f"  {k:30s}: {v:.4f}")

    elif args.mode == "finetune":
        finetune_config = merge_args(
            FinetuneConfig, args, {
                "csv_path": "csv",
                "checkpoint_path": "checkpoint",
                "save_dir": "save_dir",
                "finetune_epochs": "finetune_epochs",
                "batch_size": "batch_size",
                "lr": "finetune_lr",
                "seed": "seed",
                "val_ratio": "val_ratio",
                "patience": "patience",
            })
        results = finetune(finetune_config)
        logger.info(
            f"\n===== Final model saved to {results['final_checkpoint']} ====="
        )

    elif args.mode == "predict":
        if not args.predict_csv:
            p.error("--predict_csv is required for prediction mode")
        predict_config = merge_args(
            PredictConfig, args, {
                "predict_csv": "predict_csv",
                "checkpoint_path": "checkpoint",
                "output_path": "output",
            })
        df_new = pd.read_csv(predict_config.predict_csv)
        result = predict_batch(
            df_new=df_new,
            checkpoint_path=predict_config.checkpoint_path,
            output_path=predict_config.output_path,
            batch_size=predict_config.batch_size,
        )
        logger.info(result[["Materials_1", "Materials_2", "preferred_mu_e", "prob_mu_e", "preferred_mu_h", "prob_mu_h"]])
