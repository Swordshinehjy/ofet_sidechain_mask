"""Tests for PairDataset, CachedPairDataset, and collate functions.

The datasets consume the *preprocessed* DataFrame produced by
load_and_preprocess (log targets, ok flags, mol lists).
"""

import numpy as np
import pandas as pd
import pytest
import torch
from chemprop.data.molgraph import MolGraph
from sklearn.preprocessing import StandardScaler

from polymer_ranking.dataset import (
    PairDataset,
    CachedPairDataset,
    collate_fn,
    collate_cached_batch,
)
from polymer_ranking.config import EXTRA_DIM, NUM_TASKS, TASK_NAMES


# ── PairDataset ─────────────────────────────────────────────────────────────

def test_pair_dataset_len(prepped_df):
    ds = PairDataset(prepped_df)
    assert len(ds) == len(prepped_df)


def test_pair_dataset_drops_rows_without_structures(prepped_df):
    bad = prepped_df.iloc[[0]].copy()
    bad["mol_1"] = pd.Series([[]] * len(bad), index=bad.index)
    df = pd.concat([prepped_df, bad], ignore_index=True)
    ds = PairDataset(df)
    assert len(ds) == len(prepped_df)


def test_pair_dataset_item_shapes_and_dtypes(prepped_df):
    ds = PairDataset(prepped_df)
    g1s, g2s, ef1, ef2, y1, y2, ok1, ok2 = ds[0]

    assert isinstance(g1s, list) and len(g1s) >= 1
    assert all(isinstance(g, MolGraph) for g in g1s)
    assert ef1.shape == (EXTRA_DIM,) and ef1.dtype == torch.float32
    assert y1.shape == (NUM_TASKS,) and y1.dtype == torch.float32
    assert ok1.dtype == torch.bool
    np.testing.assert_array_equal(ok1.numpy(), ds.ok1[0])


def test_pair_dataset_lazy_featurization(prepped_df):
    ds = PairDataset(prepped_df)
    assert ds._graphs1 is None          # not featurized yet
    _ = ds.graphs1
    assert ds._graphs1 is not None      # computed and cached
    assert len(ds.graphs1) == len(ds)
    assert len(ds.graphs2) == len(ds)


def test_pair_dataset_fit_scaler_standardizes(prepped_df):
    ds = PairDataset(prepped_df, fit_scaler=True)
    assert isinstance(ds.scaler, StandardScaler)
    # concatenated ef1/ef2 was standardized -> mean ~ 0 per feature
    stacked = np.vstack([ds.ef1, ds.ef2])
    np.testing.assert_allclose(stacked.mean(axis=0), 0.0, atol=1e-5)


def test_pair_dataset_reuses_passed_scaler(prepped_df):
    ref = PairDataset(prepped_df, fit_scaler=True)
    other = PairDataset(prepped_df, scaler=ref.scaler)
    np.testing.assert_allclose(other.ef1, ref.ef1)


def test_pair_dataset_targets_are_log10(prepped_df):
    ds = PairDataset(prepped_df)
    assert np.allclose(ds.y1[:, 0], prepped_df["log_mu_e_1"].values)
    assert np.allclose(ds.y2[:, 1], prepped_df["log_mu_h_2"].values)


def test_pair_dataset_ok_flags(prepped_df):
    ds = PairDataset(prepped_df)
    assert ds.ok1.dtype == bool and ds.ok2.dtype == bool
    assert ds.ok1.shape == (len(ds), NUM_TASKS)
    # the censored benzene pair has ok_mu_e_2 = False
    assert not ds.ok2[:, 0].all()


# ── collate_fn ──────────────────────────────────────────────────────────────

def test_collate_fn_batches_graphs_and_targets(prepped_df):
    ds = PairDataset(prepped_df)
    batch = collate_fn([ds[0], ds[1]])
    bmg1, n1s, bmg2, n2s, ef1, ef2, y1, y2, ok1, ok2 = batch

    assert n1s == [len(ds.graphs1[0]), len(ds.graphs1[1])]
    # all atoms of all flattened structures are in the batched graph
    n_atoms = sum(g.V.shape[0] for g1s in ds.graphs1[:2] for g in g1s)
    assert bmg1.V.shape[0] == n_atoms
    assert ef1.shape == (2, EXTRA_DIM)
    assert y1.shape == (2, NUM_TASKS)
    assert ok1.dtype == torch.bool


def test_collate_fn_empty_graphs_raises(prepped_df):
    ds = PairDataset(prepped_df)
    item = ds[0]
    with pytest.raises(ValueError, match="empty graph"):
        collate_fn([([], item[1], item[2], item[3],
                     item[4], item[5], item[6], item[7])])


# ── CachedPairDataset ───────────────────────────────────────────────────────

def test_cached_pair_dataset_batches(prepped_df):
    ds = CachedPairDataset(prepped_df, batch_size=2)
    assert len(ds) == (len(prepped_df) + 1) // 2  # ceil split
    bmg1, n1s, bmg2, n2s, ef1, ef2, y1, y2, ok1, ok2 = ds[0]
    # all atoms of the batch's structures are in the batched graph
    n_atoms = sum(m.GetNumAtoms() for _, row in prepped_df.iloc[:2].iterrows()
                  for m in row["mol_1"])
    assert bmg1.V.shape[0] == n_atoms
    assert ef1.shape[0] == 2
    # memory is released after batching
    assert ds._graphs1 is None and ds._mols1 == []


def test_cached_pair_dataset_last_partial_batch(prepped_df):
    # 3 valid rows, batch_size=2 -> first batch 2 rows, last batch 1 row
    ds = CachedPairDataset(prepped_df, batch_size=2)
    assert len(ds) == 2
    _, _, _, _, ef1_first, _, _, _, _, _ = ds[0]
    _, _, _, _, ef1_last, _, _, _, _, _ = ds[1]
    assert ef1_first.shape[0] == 2
    assert ef1_last.shape[0] == 1


def test_cached_pair_dataset_scaler_sharing(prepped_df):
    ref = PairDataset(prepped_df, fit_scaler=True)
    cached = CachedPairDataset(prepped_df, batch_size=4, scaler=ref.scaler)
    np.testing.assert_allclose(cached.ef1, ref.ef1)


def test_collate_cached_batch_passthrough(prepped_df):
    ds = CachedPairDataset(prepped_df, batch_size=2)
    batch = collate_cached_batch([ds[0]])
    assert batch is ds[0]


def test_collate_cached_batch_rejects_multi(prepped_df):
    ds = CachedPairDataset(prepped_df, batch_size=2)
    with pytest.raises(ValueError, match="batch_size=1"):
        collate_cached_batch([ds[0], ds[1]])


# ── DataLoader integration ──────────────────────────────────────────────────

def test_pair_dataset_works_with_dataloader(prepped_df):
    from torch.utils.data import DataLoader

    ds = PairDataset(prepped_df, fit_scaler=True)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    mg1, n1s, mg2, n2s, ef1, ef2, y1, y2, ok1, ok2 = next(iter(loader))
    assert y1.shape[0] == 2
    assert TASK_NAMES[0] == "mu_e"
