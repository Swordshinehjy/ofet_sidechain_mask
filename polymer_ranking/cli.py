"""CLI entry point."""

import logging
import argparse

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
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--test_ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--finetune_epochs",
                   type=int,
                   default=None,
                   help="Fine-tuning epoch count")
    p.add_argument("--finetune_lr", type=float, default=None, help="Fine-tuning learning rate")
    args = p.parse_args()

    if args.mode == "train":
        model_config = merge_args(
            ModelConfig, args, {
                "hidden_size": "hidden_size",
                "depth": "depth",
                "dropout": "dropout",
                "ffn_hidden": "ffn_hidden",
                "sp3_weight": "sp3_weight",
            })
        train_config = merge_args(
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
                "seed": "seed",
            })
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
