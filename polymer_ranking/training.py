"""Training logic: EarlyStopping, epoch runner, metric computation, train, finetune."""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
    """Stop training when the monitored value stops improving.

    mode="min" -> lower is better (e.g. validation loss)
    mode="max" -> higher is better (e.g. pairwise accuracy)
    """

    def __init__(self, patience: int = 20, delta: float = 1e-4, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_loss = self.best_score
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, value: float, model: nn.Module) -> bool:
        if self.mode == "min":
            improved = value < self.best_score - self.delta
        else:
            improved = value > self.best_score + self.delta

        if improved:
            self.best_score = value
            self.best_loss = value
            self.counter = 0
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience


def compute_delta_scale(y1: np.ndarray, y2: np.ndarray, ok: np.ndarray) -> float:
    """Standard deviation of the measured log10 mobility differences.

    Used to normalize the delta regression target so that it stays in the same
    order of magnitude as the BPR term. Samples with a missing mobility are
    excluded.
    """
    diffs = []
    for t in range(ok.shape[1]):
        m = ok[:, t]
        if m.any():
            diffs.append(y1[m, t] - y2[m, t])
    if not diffs:
        return 1.0
    s = float(np.std(np.concatenate(diffs)))
    return s if s > 1e-6 else 1.0


def prepare_splits(df, cfg: TrainingConfig) -> Dict[str, Any]:
    """Build train/val/test datasets and loaders for a preprocessed DataFrame."""
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx,
                                      test_size=cfg.test_ratio,
                                      random_state=cfg.seed)
    tr_idx, va_idx = train_test_split(tr_idx,
                                      test_size=cfg.val_ratio / (1 - cfg.test_ratio),
                                      random_state=cfg.seed)

    tr_ds = PairDataset(df.iloc[tr_idx], fit_scaler=True)
    va_ds = CachedPairDataset(df.iloc[va_idx], cfg.batch_size, scaler=tr_ds.scaler)
    te_ds = CachedPairDataset(df.iloc[te_idx], cfg.batch_size, scaler=tr_ds.scaler)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                           collate_fn=collate_cached_batch)
    te_loader = DataLoader(te_ds, batch_size=1, shuffle=False,
                           collate_fn=collate_cached_batch)

    ok = tr_ds.ok1 & tr_ds.ok2
    delta_scale = cfg.delta_scale or compute_delta_scale(tr_ds.y1, tr_ds.y2, ok)

    return {
        "tr_ds": tr_ds, "va_ds": va_ds, "te_ds": te_ds,
        "tr_loader": tr_loader, "va_loader": va_loader, "te_loader": te_loader,
        "scaler": tr_ds.scaler, "delta_scale": float(delta_scale),
        "sizes": (len(tr_ds), len(va_ds.df), len(te_ds.df)),
    }


def _run_epoch(
    model: PolymerRankingModel,
    loader: DataLoader,
    criterion: MultiTaskBayesianRankingLoss,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute one epoch (training or validation).

    Returns (loss, s1, s2, y1, y2, valid, rank_valid):

    * ``valid`` [N, T] — both polymers were measured (``> 0``), so the numeric
      difference is known and the delta regression can be applied.
    * ``rank_valid`` [N, T] — the ordering is decidable. A mobility of 0 is
      left-censored (below the detection limit), hence ``measured > censored`` is
      decidable, while two censored values are not comparable.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_s1, all_s2, all_y1, all_y2, all_valid, all_rank = [], [], [], [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for mg1, n1, mg2, n2, ef1, ef2, y1, y2, ok1, ok2 in loader:
            ef1, ef2 = ef1.to(DEVICE), ef2.to(DEVICE)
            y1, y2 = y1.to(DEVICE), y2.to(DEVICE)
            ok1, ok2 = ok1.to(DEVICE), ok2.to(DEVICE)
            valid = ok1 & ok2                       # both measured -> regression
            rank_valid = (ok1 | ok2) & (y1 != y2)   # ordering decidable -> BPR

            s1, s2 = model(mg1, n1, ef1, mg2, n2, ef2)
            loss, _ = criterion(s1, s2, y1, y2,
                                valid=valid, rank_valid=rank_valid)

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
            all_valid.append(valid.cpu())
            all_rank.append(rank_valid.cpu())

    def _cat(lst):
        return torch.cat(lst).numpy()

    return (
        total_loss / total_samples,
        _cat(all_s1),
        _cat(all_s2),
        _cat(all_y1),
        _cat(all_y2),
        _cat(all_valid),
        _cat(all_rank),
    )


def compute_metrics(s1, s2, y1, y2, valid=None, rank_valid=None) -> Dict[str, float]:
    """Compute Pairwise Accuracy, Spearman rho and correct-direction probability.

    ``rank_valid`` [N, T] (optional) marks pairs whose ordering is decidable —
    accuracy and ``avg_prob`` are computed on those (this includes pairs with one
    censored value, because a measured mobility is always above the detection
    limit).

    ``valid`` [N, T] (optional) marks pairs where both mobilities were measured;
    only those carry a numeric difference, so Spearman rho is computed on them.

    ``avg_prob`` is the mean probability assigned to the *correct* direction,
    i.e. it is high only when the ranking is right.
    """
    out = {}
    n_tasks = s1.shape[1]
    for t in range(n_tasks):
        name = TASK_NAMES[t] if t < len(TASK_NAMES) else f"task{t}"
        dp = s1[:, t] - s2[:, t]
        dy = y1[:, t] - y2[:, t]

        # ordering-decidable pairs (accuracy / probability)
        if rank_valid is None:
            mask = dy != 0
        else:
            mask = rank_valid[:, t].astype(bool) & (dy != 0)

        # pairs with two measured values (spearman)
        if valid is None:
            v = np.ones_like(dp, dtype=bool)
        else:
            v = valid[:, t].astype(bool)

        acc = float((np.sign(dp[mask]) == np.sign(dy[mask])).mean()) if mask.any() else 0.0

        if v.sum() >= 3:
            scores = np.concatenate([s1[v, t], s2[v, t]])
            targets = np.concatenate([y1[v, t], y2[v, t]])
            rho, _ = spearmanr(scores, targets)
            rho = 0.0 if np.isnan(rho) else float(rho)
        else:
            rho = 0.0

        prob = float((1.0 / (1.0 + np.exp(-np.sign(dy[mask]) * dp[mask]))).mean()) \
            if mask.any() else 0.0

        out[f"{name}_pair_acc"] = acc
        out[f"{name}_spearman"] = rho
        out[f"{name}_avg_prob"] = prob
        out[f"{name}_n_rank"] = float(mask.sum())
        out[f"{name}_n_reg"] = float(v.sum())
    out["mean_pair_acc"] = float(np.mean(
        [out[f"{TASK_NAMES[t]}_pair_acc"] for t in range(min(n_tasks, len(TASK_NAMES)))]))
    return out


def _monitor_value(metrics: Dict[str, float], cfg: TrainingConfig) -> float:
    """Value used for early stopping / LR scheduling."""
    if cfg.early_stop_metric == "pair_acc":
        return metrics["mean_pair_acc"]
    return -metrics.get("loss", 0.0)


def _build_model(mcfg: ModelConfig) -> PolymerRankingModel:
    return PolymerRankingModel(
        hidden_size=mcfg.hidden_size,
        depth=mcfg.depth,
        dropout=mcfg.dropout,
        ffn_hidden=mcfg.ffn_hidden,
        extra_dim=mcfg.extra_dim,
        num_tasks=mcfg.num_tasks,
        aggregation=mcfg.aggregation,
        sp3_weight=mcfg.sp3_weight,
    ).to(DEVICE)


def _make_scheduler(cfg: TrainingConfig, optimizer):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2)
    return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6)


def run_experiment(
    df,
    model_config: ModelConfig,
    train_config: TrainingConfig,
    max_epochs: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train one configuration and return its validation metrics.

    Lightweight variant of :func:`train` used for hyper-parameter search: no
    checkpoint is written and only the best validation metrics are kept.
    """
    cfg = copy.deepcopy(train_config)
    if max_epochs:
        cfg.epochs = max_epochs

    splits = prepare_splits(df, cfg)
    model = _build_model(model_config)
    criterion = MultiTaskBayesianRankingLoss(
        rank_weight=cfg.rank_weight,
        reg_weight=cfg.reg_weight,
        delta_scale=splits["delta_scale"],
        censored_weight=cfg.censored_weight,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(cfg, optimizer)
    stopper = EarlyStopping(
        patience=cfg.patience,
        mode="max" if cfg.early_stop_metric == "pair_acc" else "min",
    )

    best_metrics: Dict[str, float] = {}
    best_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, *_ = _run_epoch(model, splits["tr_loader"], criterion, optimizer)
        va_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(model, splits["va_loader"], criterion)
        va_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
        va_met["loss"] = va_loss

        monitor = _monitor_value(va_met, cfg)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(monitor)
        else:
            scheduler.step()

        if stopper.step(monitor, model):
            best_epoch = epoch
            best_metrics = va_met
            break
        if stopper.counter == 0:
            best_epoch = epoch
            best_metrics = va_met

        if verbose and (epoch % 10 == 0 or epoch == 1):
            logger.info(
                f"Ep {epoch:4d} | tr={tr_loss:.4f} va={va_loss:.4f} | "
                f"acc_e={va_met['mu_e_pair_acc']:.3f} acc_h={va_met['mu_h_pair_acc']:.3f}"
            )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "val_metrics": best_metrics,
        "best_epoch": best_epoch,
        "delta_scale": splits["delta_scale"],
        "n_params": n_params,
        "sizes": splits["sizes"],
    }


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
    splits = prepare_splits(df, cfg)
    tr_ds, tr_loader = splits["tr_ds"], splits["tr_loader"]
    va_loader, te_loader = splits["va_loader"], splits["te_loader"]

    logger.info(f"Train/Val/Test: {splits['sizes'][0]}/{splits['sizes'][1]}/{splits['sizes'][2]}")
    logger.info(f"delta_scale (std of measured log10 differences): {splits['delta_scale']:.3f}")

    model = _build_model(mcfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    criterion = MultiTaskBayesianRankingLoss(
        rank_weight=cfg.rank_weight,
        reg_weight=cfg.reg_weight,
        delta_scale=splits["delta_scale"],
        censored_weight=cfg.censored_weight,
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(cfg, optimizer)
    stopper = EarlyStopping(
        patience=cfg.patience,
        mode="max" if cfg.early_stop_metric == "pair_acc" else "min",
    )

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, *_ = _run_epoch(model, tr_loader, criterion, optimizer)
        va_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(model, va_loader, criterion)
        va_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
        va_met["loss"] = va_loss

        monitor = _monitor_value(va_met, cfg)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(monitor)
        else:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_metrics"].append(va_met)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Ep {epoch:4d} | "
                f"tr={tr_loss:.4f}  va={va_loss:.4f} | "
                f"mu_e acc={va_met['mu_e_pair_acc']:.3f} rho={va_met['mu_e_spearman']:.3f} n={int(va_met['mu_e_n_rank'])} | "
                f"mu_h acc={va_met['mu_h_pair_acc']:.3f} rho={va_met['mu_h_spearman']:.3f} n={int(va_met['mu_h_n_rank'])}"
            )

        if stopper.step(monitor, model):
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(best {cfg.early_stop_metric}={stopper.best_score:.4f})")
            break

    if stopper.best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in stopper.best_state.items()})

    te_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(model, te_loader, criterion)
    te_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
    logger.info("\n========== Test Results ==========")
    for k, v in te_met.items():
        logger.info(f"  {k:25s}: {v:.4f}")

    ckpt_path = Path(cfg.save_dir) / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": splits["scaler"],
            "config": mcfg.to_dict(),
            "delta_scale": splits["delta_scale"],
            "train_config": {
                "rank_weight": cfg.rank_weight,
                "reg_weight": cfg.reg_weight,
            },
        },
        ckpt_path,
    )
    logger.info(f"Checkpoint saved -> {ckpt_path}")

    return {
        "test_metrics": te_met,
        "history": history,
        "model": model,
        "scaler": splits["scaler"],
        "checkpoint": ckpt_path,
    }


def finetune(config: FinetuneConfig) -> Dict[str, Any]:
    """Fine-tuning mode: load best weights, continue on the full data with monitoring."""
    from .predict import load_checkpoint

    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model_config, scaler, model = load_checkpoint(config.checkpoint_path)

    df = load_and_preprocess(config.csv_path)
    logger.info(f"Full dataset size: {len(df)} pairs")

    idx = np.arange(len(df))
    if config.val_ratio > 0:
        tr_idx, va_idx = train_test_split(idx, test_size=config.val_ratio,
                                          random_state=config.seed)
    else:
        tr_idx, va_idx = idx, np.array([], dtype=int)

    full_ds = PairDataset(df.iloc[tr_idx], scaler=scaler, fit_scaler=False)
    full_loader = DataLoader(full_ds, batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)

    va_loader = None
    if len(va_idx):
        va_ds = CachedPairDataset(df.iloc[va_idx], config.batch_size,
                                  scaler=scaler, fit_scaler=False)
        va_loader = DataLoader(va_ds, batch_size=1, shuffle=False,
                               collate_fn=collate_cached_batch)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    ok = full_ds.ok1 & full_ds.ok2
    delta_scale = compute_delta_scale(full_ds.y1, full_ds.y2, ok)

    criterion = MultiTaskBayesianRankingLoss(
        rank_weight=0.8, reg_weight=0.2, delta_scale=delta_scale,
        censored_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=config.lr,
                      weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.finetune_epochs,
                                  eta_min=config.lr * 0.1)
    stopper = EarlyStopping(patience=config.patience, mode="max")

    history = {"train_loss": [], "train_metrics": [], "val_metrics": []}

    logger.info(f"Starting fine-tuning for {config.finetune_epochs} epochs with lr={config.lr}")
    for epoch in range(1, config.finetune_epochs + 1):
        tr_loss, s1, s2, y1, y2, valid, rank_valid = _run_epoch(model, full_loader, criterion,
                                                    optimizer)
        tr_met = compute_metrics(s1, s2, y1, y2, valid, rank_valid)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_metrics"].append(tr_met)

        msg = (f"Ep {epoch:4d} | loss={tr_loss:.4f} | "
               f"mu_e acc={tr_met['mu_e_pair_acc']:.3f} rho={tr_met['mu_e_spearman']:.3f} | "
               f"mu_h acc={tr_met['mu_h_pair_acc']:.3f} rho={tr_met['mu_h_spearman']:.3f}")

        monitor = tr_met["mean_pair_acc"]
        if va_loader is not None:
            _, v1, v2, vy1, vy2, vvalid, vrank = _run_epoch(model, va_loader, criterion)
            va_met = compute_metrics(v1, v2, vy1, vy2, vvalid, vrank)
            history["val_metrics"].append(va_met)
            monitor = va_met["mean_pair_acc"]
            msg += (f" | val acc_e={va_met['mu_e_pair_acc']:.3f} "
                    f"acc_h={va_met['mu_h_pair_acc']:.3f}")
            if stopper.step(monitor, model):
                logger.info(msg)
                logger.info(f"Fine-tuning early stop at epoch {epoch}")
                break

        logger.info(msg)

    if stopper.best_state:
        model.load_state_dict({k: v.to(DEVICE) for k, v in stopper.best_state.items()})

    final_ckpt_path = Path(config.save_dir) / "final_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": scaler,
            "config": model_config.to_dict(),
            "delta_scale": delta_scale,
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
