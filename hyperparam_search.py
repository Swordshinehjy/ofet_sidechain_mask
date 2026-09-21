"""Hyper-parameter analysis and search for the polymer ranking model.

Stages
------
``analyze``
    Data health check: how many pairs actually carry a measured mobility, the
    scale of the log10 differences, class balance, material reuse (leakage
    indicator), plus reference baselines (descriptor-only logistic regression
    and LUMO/HOMO heuristics).
``search``
    Random search over lr / hidden_size / depth / dropout / weight_decay /
    batch_size / rank-reg weighting. Each trial is trained with early stopping
    on the validation pairwise accuracy.
``ablation``
    ``sp3_weight`` sweep (0.2 ... 1.0), i.e. how much the side-chain
    down-weighting actually helps.
``all``
    analyze -> search -> ablation, then print the recommended configuration.

Usage
-----
::

    D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage analyze
    D:/anaconda3/envs/chemprop2/python.exe hyperparam_search.py --stage all --n_trials 12 --max_epochs 40

Results are written to ``hyperparam_search.csv`` and ``best_hyperparams.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from polymer_ranking.chemistry import load_and_preprocess
from polymer_ranking.config import ModelConfig, TrainingConfig
from polymer_ranking.training import run_experiment, compute_delta_scale

EXTRA_RAW = [
    "conjugation_{s}",
    "Isomer_{s}",
    "CentroSymmetry_{s}",
    "E_LUMO (eV)_{s}",
    "E_HOMO (eV)_{s}",
]
TASKS = ["mu_e", "mu_h"]


# ── stage 1: data analysis ──────────────────────────────────────────────────

def analyze(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Data health check + descriptor-only baselines."""
    report: Dict[str, Any] = {"n_pairs": len(df)}

    mats = pd.concat([df["Materials_1"], df["Materials_2"]]) \
        if "Materials_1" in df.columns else None
    if mats is not None:
        report["n_materials"] = int(mats.nunique())
        report["material_reuse_mean"] = float(mats.value_counts().mean())

    # label availability: mobility == 0 is left-censored (below detection limit)
    for t in TASKS:
        ok1, ok2 = df[f"ok_{t}_1"], df[f"ok_{t}_2"]
        both = ok1 & ok2
        either = ok1 | ok2
        dy_all = df[f"log_{t}_1"] - df[f"log_{t}_2"]
        report[f"{t}_both_measured"] = int(both.sum())          # regression target known
        report[f"{t}_both_measured_frac"] = round(float(both.mean()), 3)
        report[f"{t}_rank_decidable"] = int((either & (dy_all != 0)).sum())
        report[f"{t}_one_censored"] = int((ok1 ^ ok2).sum())
        report[f"{t}_both_censored"] = int((~ok1 & ~ok2).sum())
        dy = dy_all[both]
        report[f"{t}_delta_std"] = round(float(dy.std()), 3)
        report[f"{t}_delta_abs_median"] = round(float(dy.abs().median()), 3)
        report[f"{t}_pos_frac"] = round(float((dy > 0).mean()), 3)

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=seed)
    report.update(_baselines(df, tr_idx, te_idx))
    return report


def _rank_mask(df: pd.DataFrame, t: str) -> np.ndarray:
    """Pairs whose ordering is decidable: at least one measured value and no tie.

    A mobility of 0 is left-censored (below the detection limit), so a measured
    value is always ranked above a censored one.
    """
    ok1 = df[f"ok_{t}_1"].values.astype(bool)
    ok2 = df[f"ok_{t}_2"].values.astype(bool)
    dy = (df[f"log_{t}_1"] - df[f"log_{t}_2"]).values
    return (ok1 | ok2) & (dy != 0)


def _baselines(df: pd.DataFrame, tr_idx, te_idx) -> Dict[str, float]:
    """Reference scores: heuristic rules and a descriptor-only classifier."""
    out: Dict[str, float] = {}

    # heuristic: lower LUMO -> higher mu_e ; higher HOMO -> higher mu_h
    for t, sign in (("mu_e", -1.0), ("mu_h", 1.0)):
        col = "E_LUMO (eV)" if t == "mu_e" else "E_HOMO (eV)"
        d = df[f"{col}_1"] - df[f"{col}_2"]
        dy = df[f"log_{t}_1"] - df[f"log_{t}_2"]
        m = _rank_mask(df, t)
        te = np.isin(np.arange(len(df)), te_idx) & m
        if te.sum():
            pred = sign * np.sign(d.values[te])
            out[f"{t}_baseline_heuristic_acc"] = round(
                float((pred == np.sign(dy.values[te])).mean()), 3)

    # descriptor-only logistic regression on the 5 extra features
    for t in TASKS:
        cols1 = [c.format(s="1") for c in EXTRA_RAW]
        cols2 = [c.format(s="2") for c in EXTRA_RAW]
        X = (df[cols1].values - df[cols2].values).astype(float)
        dy = (df[f"log_{t}_1"] - df[f"log_{t}_2"]).values
        m = _rank_mask(df, t)
        if m.sum() < 50:
            continue
        y = np.sign(dy[m])
        Xm = X[m]
        tr_m = np.isin(np.where(m)[0], tr_idx)
        te_m = np.isin(np.where(m)[0], te_idx)
        if tr_m.sum() < 20 or te_m.sum() < 10:
            continue
        sc = StandardScaler().fit(Xm[tr_m])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xm[tr_m]), y[tr_m])
        pred = clf.predict(sc.transform(Xm[te_m]))
        out[f"{t}_baseline_logreg_acc"] = round(float((pred == y[te_m]).mean()), 3)
    return out


def print_report(report: Dict[str, Any]) -> None:
    print("\n========== Data Analysis ==========")
    for k, v in report.items():
        print(f"  {k:34s}: {v}")

    print("\n--- Interpretation ---")
    for t in TASKS:
        frac = report.get(f"{t}_both_measured_frac")
        n_rank = report.get(f"{t}_rank_decidable")
        n_cens = report.get(f"{t}_one_censored")
        if frac is not None:
            print(f"  {t}: {frac:.1%} of pairs have two measured values (regression target "
                  f"known); {n_rank} pairs have a decidable ordering (ranking target known, "
                  f"including {n_cens} pairs with one censored value). "
                  f"{report.get(f'{t}_both_censored')} pairs are censored on both sides and "
                  f"carry no ordering information.")
        std = report.get(f"{t}_delta_std")
        if std:
            print(f"  {t}: std of log10 difference = {std} -> use delta_scale={std} "
                  f"so the regression term stays comparable to the BPR term.")
    reuse = report.get("material_reuse_mean")
    if reuse:
        print(f"  Each material appears in ~{reuse:.2f} pairs on average; random pair-level "
              f"splitting therefore re-uses materials across train/test (interpolation setting).")


# ── stage 2: random search ──────────────────────────────────────────────────

SEARCH_SPACE: Dict[str, List[Any]] = {
    "lr": [1e-4, 3e-4, 1e-3],
    "hidden_size": [128, 256, 300],
    "depth": [3, 4, 6],
    "dropout": [0.1, 0.2, 0.3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64],
    "loss_weights": [(0.8, 0.2), (0.6, 0.4), (1.0, 0.0)],
}


def _sample_config(rng: random.Random) -> Dict[str, Any]:
    cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    cfg["ffn_hidden"] = max(64, cfg["hidden_size"] // 2)
    return cfg


def search(
    df: pd.DataFrame,
    n_trials: int = 12,
    max_epochs: int = 40,
    patience: int = 15,
    seed: int = 42,
    sp3_weight: float = 0.2,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Random search; returns one record per trial sorted by validation accuracy."""
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    for i in range(1, n_trials + 1):
        cand = _sample_config(rng)
        alpha, beta = cand.pop("loss_weights")
        mcfg = ModelConfig(
            hidden_size=cand["hidden_size"],
            depth=cand["depth"],
            dropout=cand["dropout"],
            ffn_hidden=cand["ffn_hidden"],
            sp3_weight=sp3_weight,
        )
        tcfg = TrainingConfig(
            batch_size=cand["batch_size"],
            lr=cand["lr"],
            weight_decay=cand["weight_decay"],
            patience=patience,
            epochs=max_epochs,
            seed=seed,
            rank_weight=alpha,
            reg_weight=beta,
        )
        out = run_experiment(df, mcfg, tcfg, max_epochs=max_epochs, verbose=verbose)
        vm = out["val_metrics"]
        rec = {
            "trial": i,
            "lr": cand["lr"],
            "hidden_size": cand["hidden_size"],
            "depth": cand["depth"],
            "dropout": cand["dropout"],
            "ffn_hidden": cand["ffn_hidden"],
            "weight_decay": cand["weight_decay"],
            "batch_size": cand["batch_size"],
            "rank_weight": alpha,
            "reg_weight": beta,
            "val_mean_acc": round(vm.get("mean_pair_acc", 0.0), 4),
            "val_mu_e_acc": round(vm.get("mu_e_pair_acc", 0.0), 4),
            "val_mu_h_acc": round(vm.get("mu_h_pair_acc", 0.0), 4),
            "val_mu_e_rho": round(vm.get("mu_e_spearman", 0.0), 4),
            "val_mu_h_rho": round(vm.get("mu_h_spearman", 0.0), 4),
            "best_epoch": out["best_epoch"],
            "n_params": out["n_params"],
            "delta_scale": round(out["delta_scale"], 3),
        }
        results.append(rec)
        print(f"[trial {i:2d}/{n_trials}] val_acc={rec['val_mean_acc']:.3f} "
              f"(e={rec['val_mu_e_acc']:.3f} h={rec['val_mu_h_acc']:.3f}) "
              f"ep={rec['best_epoch']:3d} | lr={rec['lr']:g} h={rec['hidden_size']} "
              f"d={rec['depth']} do={rec['dropout']} wd={rec['weight_decay']:g} "
              f"bs={rec['batch_size']} a/b={alpha}/{beta}")

    results.sort(key=lambda r: r["val_mean_acc"], reverse=True)
    return results


# ── stage 3: sp3_weight ablation ────────────────────────────────────────────

def ablation(
    df: pd.DataFrame,
    weights: List[float],
    base: Dict[str, Any],
    max_epochs: int = 40,
    patience: int = 15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sweep sp3_weight with everything else fixed."""
    out: List[Dict[str, Any]] = []
    for w in weights:
        mcfg = ModelConfig(
            hidden_size=base["hidden_size"],
            depth=base["depth"],
            dropout=base["dropout"],
            ffn_hidden=base["ffn_hidden"],
            sp3_weight=w,
        )
        tcfg = TrainingConfig(
            batch_size=base["batch_size"],
            lr=base["lr"],
            weight_decay=base["weight_decay"],
            patience=patience,
            epochs=max_epochs,
            seed=seed,
            rank_weight=base["rank_weight"],
            reg_weight=base["reg_weight"],
        )
        res = run_experiment(df, mcfg, tcfg, max_epochs=max_epochs)
        vm = res["val_metrics"]
        rec = {
            "sp3_weight": w,
            "val_mean_acc": round(vm.get("mean_pair_acc", 0.0), 4),
            "val_mu_e_acc": round(vm.get("mu_e_pair_acc", 0.0), 4),
            "val_mu_h_acc": round(vm.get("mu_h_pair_acc", 0.0), 4),
            "best_epoch": res["best_epoch"],
        }
        out.append(rec)
        print(f"[sp3={w}] val_acc={rec['val_mean_acc']:.3f} "
              f"(e={rec['val_mu_e_acc']:.3f} h={rec['val_mu_h_acc']:.3f}) "
              f"ep={rec['best_epoch']}")
    return out


# ── recommendation ──────────────────────────────────────────────────────────

def recommend(
    report: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    ablation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine search / ablation output into one recommended configuration."""
    best = dict(search_results[0]) if search_results else {}
    delta_scale = max(
        report.get("mu_e_delta_std", 1.0), report.get("mu_h_delta_std", 1.0), 1e-3)

    sp3 = 0.2
    if ablation_results:
        sp3 = max(ablation_results, key=lambda r: r["val_mean_acc"])["sp3_weight"]

    rec = {
        "hidden_size": best.get("hidden_size", 256),
        "depth": best.get("depth", 4),
        "dropout": best.get("dropout", 0.2),
        "ffn_hidden": best.get("ffn_hidden", 128),
        "sp3_weight": sp3,
        "lr": best.get("lr", 3e-4),
        "weight_decay": best.get("weight_decay", 1e-4),
        "batch_size": best.get("batch_size", 32),
        "rank_weight": best.get("rank_weight", 0.8),
        "reg_weight": best.get("reg_weight", 0.2),
        "delta_scale": round(float(delta_scale), 3),
        "epochs": 200,
        "patience": 30,
        "early_stop_metric": "pair_acc",
        "scheduler": "plateau",
        "val_mean_acc_from_search": best.get("val_mean_acc"),
    }
    return rec


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyper-parameter analysis and search")
    p.add_argument("--csv", type=str, default="contrastive_full_paired.csv")
    p.add_argument("--stage", choices=["analyze", "search", "ablation", "all"],
                   default="all")
    p.add_argument("--n_trials", type=int, default=12)
    p.add_argument("--max_epochs", type=int, default=40,
                   help="Epoch budget per trial (early stopping may stop earlier)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablation_weights", type=float, nargs="*",
                   default=[0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--out_csv", type=str, default="hyperparam_search.csv")
    p.add_argument("--out_json", type=str, default="best_hyperparams.json")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("Preprocessing data (monomer concatenation + cyclization)...")
    df = load_and_preprocess(args.csv)
    print(f"Loaded {len(df)} valid pairs\n")

    report = analyze(df, seed=args.seed)
    print_report(report)

    search_results: List[Dict[str, Any]] = []
    ablation_results: List[Dict[str, Any]] = []

    if args.stage in ("search", "all"):
        print(f"\n========== Random search ({args.n_trials} trials) ==========")
        search_results = search(
            df, n_trials=args.n_trials, max_epochs=args.max_epochs,
            patience=args.patience, seed=args.seed, verbose=args.verbose,
        )
        pd.DataFrame(search_results).to_csv(args.out_csv, index=False)
        print(f"\nSearch results -> {args.out_csv}")
        print(pd.DataFrame(search_results).head().to_string(index=False))

    if args.stage in ("ablation", "all"):
        base = dict(search_results[0]) if search_results else {
            "hidden_size": 256, "depth": 4, "dropout": 0.2, "ffn_hidden": 128,
            "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 32,
            "rank_weight": 0.8, "reg_weight": 0.2,
        }
        print(f"\n========== sp3_weight ablation ==========")
        ablation_results = ablation(
            df, list(args.ablation_weights), base,
            max_epochs=args.max_epochs, patience=args.patience, seed=args.seed,
        )
        print(pd.DataFrame(ablation_results).to_string(index=False))

    if args.stage == "analyze":
        return 0

    rec = recommend(report, search_results, ablation_results)
    Path(args.out_json).write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print("\n========== Recommended Hyper-parameters ==========")
    for k, v in rec.items():
        print(f"  {k:28s}: {v}")
    print(f"\nSaved -> {args.out_json}")
    print("\nTrain with:")
    print(f"  python polymer_ranking.py --mode train --csv {args.csv} \\")
    print(f"      --hidden_size {rec['hidden_size']} --depth {rec['depth']} "
          f"--dropout {rec['dropout']} --ffn_hidden {rec['ffn_hidden']} \\")
    print(f"      --sp3_weight {rec['sp3_weight']} --lr {rec['lr']} "
          f"--batch_size {rec['batch_size']} \\")
    print(f"      --rank_weight {rec['rank_weight']} --reg_weight {rec['reg_weight']} "
          f"--delta_scale {rec['delta_scale']} \\")
    print(f"      --epochs {rec['epochs']} --patience {rec['patience']} "
          f"--early_stop_metric {rec['early_stop_metric']} --scheduler {rec['scheduler']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
