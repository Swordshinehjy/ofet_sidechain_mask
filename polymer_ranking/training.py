"""Training logic: EarlyStopping, epoch runner, metric computation, train, finetune."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from .config import ModelConfig, TrainingConfig, FinetuneConfig, TASK_NAMES
from .model import PolymerRankingModel
from .loss import MultiTaskBayesianRankingLoss
from .dataset import PairDataset, CachedPairDataset, collate_fn, collate_cached_batch
from .chemistry import load_and_preprocess

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:

    def __init__(self, patience: int = 20, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience


def _run_epoch(
    model: PolymerRankingModel,
    loader: DataLoader,
    criterion: MultiTaskBayesianRankingLoss,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute one epoch (training or validation), returns (loss, s1, s2, y1, y2)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_s1, all_s2, all_y1, all_y2 = [], [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for mg1, n1, mg2, n2, ef1, ef2, y1, y2 in loader:
            ef1, ef2 = ef1.to(DEVICE), ef2.to(DEVICE)
            y1, y2 = y1.to(DEVICE), y2.to(DEVICE)

            s1, s2 = model(mg1, n1, ef1, mg2, n2, ef2)
            loss, _ = criterion(s1, s2, y1, y2)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = y1.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_s1.append(s1.detach().cpu())
            all_s2.append(s2.detach().cpu())
            all_y1.append(y1.cpu())
            all_y2.append(y2.cpu())

    def _cat(lst):
        return torch.cat(lst).numpy()

    return (
        total_loss / total_samples,
        _cat(all_s1),
        _cat(all_s2),
        _cat(all_y1),
        _cat(all_y2),
    )


def compute_metrics(s1, s2, y1, y2) -> Dict[str, float]:
    """Compute Pairwise Accuracy, Spearman rho, and average ranking probability."""
    out = {}
    for t, name in enumerate(TASK_NAMES):
        dp = s1[:, t] - s2[:, t]
        dy = y1[:, t] - y2[:, t]
        mask = dy != 0
        acc = ((np.sign(dp[mask]) == np.sign(
            dy[mask]))).mean() if mask.any() else 0.0
        scores = np.concatenate([s1[:, t], s2[:, t]])
        targets = np.concatenate([y1[:, t], y2[:, t]])
        rho, _ = spearmanr(scores, targets)
        rho = 0.0 if np.isnan(rho) else rho
        avg_prob = float(1.0 / (1.0 + np.exp(-np.abs(dp))).mean())
        out[f"{name}_pair_acc"] = float(acc)
        out[f"{name}_spearman"] = float(rho)
        out[f"{name}_avg_prob"] = avg_prob
    return out


def train(
    model_config: ModelConfig,
    train_config: TrainingConfig,
) -> Dict[str, Any]:
    """Complete training pipeline, returns dict with test_metrics / history / model / scalers."""
    cfg = train_config
    mcfg = model_config

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = load_and_preprocess(cfg.csv_path)
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx,
                                      test_size=cfg.test_ratio,
                                      random_state=cfg.seed)
    tr_idx, va_idx = train_test_split(tr_idx,
                                      test_size=cfg.val_ratio /
                                      (1 - cfg.test_ratio),
                                      random_state=cfg.seed)

    tr_ds = PairDataset(df.iloc[tr_idx], fit_scaler=True)
    va_ds = CachedPairDataset(df.iloc[va_idx],
                              cfg.batch_size,
                              scaler=tr_ds.scaler)
    te_ds = CachedPairDataset(df.iloc[te_idx],
                              cfg.batch_size,
                              scaler=tr_ds.scaler)

    tr_loader = DataLoader(tr_ds,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           collate_fn=collate_fn,
                           num_workers=0)
    va_loader = DataLoader(va_ds,
                           batch_size=1,
                           shuffle=False,
                           collate_fn=collate_cached_batch)
    te_loader = DataLoader(te_ds,
                           batch_size=1,
                           shuffle=False,
                           collate_fn=collate_cached_batch)

    logger.info(
        f"Train/Val/Test: {len(tr_ds)}/{len(va_ds.df)}/{len(te_ds.df)}")

    model = PolymerRankingModel(
        hidden_size=mcfg.hidden_size,
        depth=mcfg.depth,
        dropout=mcfg.dropout,
        ffn_hidden=mcfg.ffn_hidden,
        extra_dim=mcfg.extra_dim,
        num_tasks=mcfg.num_tasks,
        aggregation=mcfg.aggregation,
        sp3_weight=mcfg.sp3_weight,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    criterion = MultiTaskBayesianRankingLoss()
    optimizer = AdamW(model.parameters(),
                      lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=cfg.epochs,
                                  eta_min=cfg.lr * 1e-2)
    stopper = EarlyStopping(patience=cfg.patience)

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, *_ = _run_epoch(model, tr_loader, criterion, optimizer)
        va_loss, s1, s2, y1, y2 = _run_epoch(model, va_loader, criterion)
        va_met = compute_metrics(s1, s2, y1, y2)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_metrics"].append(va_met)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Ep {epoch:4d} | "
                f"tr={tr_loss:.4f}  va={va_loss:.4f} | "
                f"mu_e acc={va_met['mu_e_pair_acc']:.3f} rho={va_met['mu_e_spearman']:.3f} p={va_met['mu_e_avg_prob']:.3f} | "
                f"mu_h acc={va_met['mu_h_pair_acc']:.3f} rho={va_met['mu_h_spearman']:.3f} p={va_met['mu_h_avg_prob']:.3f}"
            )

        if stopper.step(va_loss, model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if stopper.best_state:
        model.load_state_dict({
            k: v.to(DEVICE)
            for k, v in stopper.best_state.items()
        })

    te_loss, s1, s2, y1, y2 = _run_epoch(model, te_loader, criterion)
    te_met = compute_metrics(s1, s2, y1, y2)
    logger.info("\n========== Test Results ==========")
    for k, v in te_met.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    ckpt_path = Path(cfg.save_dir) / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": tr_ds.scaler,
            "config": mcfg.to_dict(),
        },
        ckpt_path,
    )
    logger.info(f"Checkpoint saved -> {ckpt_path}")

    return {
        "test_metrics": te_met,
        "history": history,
        "model": model,
        "scaler": tr_ds.scaler,
        "checkpoint": ckpt_path,
    }


def finetune(config: FinetuneConfig) -> Dict[str, Any]:
    """Fine-tuning mode: load best model weights, train on full data for a few epochs."""
    from .predict import load_checkpoint

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model_config, scaler, model = load_checkpoint(config.checkpoint_path)

    df = load_and_preprocess(config.csv_path)
    logger.info(f"Full dataset size: {len(df)} pairs")

    full_ds = PairDataset(df, scaler=scaler, fit_scaler=False)
    full_loader = DataLoader(
        full_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    criterion = MultiTaskBayesianRankingLoss()
    optimizer = AdamW(model.parameters(),
                      lr=config.lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config.finetune_epochs,
                                  eta_min=config.lr * 0.1)

    history = {"train_loss": [], "train_metrics": []}

    logger.info(
        f"Starting fine-tuning for {config.finetune_epochs} epochs with lr={config.lr}"
    )
    for epoch in range(1, config.finetune_epochs + 1):
        tr_loss, s1, s2, y1, y2 = _run_epoch(model, full_loader, criterion,
                                             optimizer)
        tr_met = compute_metrics(s1, s2, y1, y2)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_metrics"].append(tr_met)

        logger.info(
            f"Ep {epoch:4d} | loss={tr_loss:.4f} | "
            f"mu_e acc={tr_met['mu_e_pair_acc']:.3f} rho={tr_met['mu_e_spearman']:.3f} p={tr_met['mu_e_avg_prob']:.3f} | "
            f"mu_h acc={tr_met['mu_h_pair_acc']:.3f} rho={tr_met['mu_h_spearman']:.3f} p={tr_met['mu_h_avg_prob']:.3f}"
        )

    final_ckpt_path = Path(config.save_dir) / "final_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "config": model_config.to_dict(),
            "finetune_history": history,
        },
        final_ckpt_path,
    )
    logger.info(f"Final model saved -> {final_ckpt_path}")

    return {
        "final_checkpoint": final_ckpt_path,
        "history": history,
        "model": model,
    }
