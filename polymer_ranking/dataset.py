"""Dataset classes and collate functions."""

from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

try:
    from chemprop.data.molgraph import MolGraph
    from chemprop.data.collate import BatchMolGraph
except ImportError as e:
    raise ImportError(
        "chemprop>=2.0.0 is required. Install with: pip install chemprop>=2.0.0"
    ) from e

from .featurizer import create_featurizer
from .chemistry import extra_feat
from .config import TASK_NAMES


class _BasePairDataset(Dataset):
    """Base class encapsulating featurization / standardization / target extraction logic.

    Each polymer may have multiple cyclized structures (from monomer concatenation),
    stored as a list of MolGraphs. During encoding, embeddings from all structures
    of the same polymer are averaged.

    Featurization is lazy: MolGraphs are computed on first access and cached,
    avoiding upfront memory spike for large datasets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self._featurizer = create_featurizer()

        self._mols1: List[list] = []
        self._mols2: List[list] = []
        valid_indices: List[int] = []
        for idx in range(len(self.df)):
            mols1 = self.df.loc[idx, "mol_1"]
            mols2 = self.df.loc[idx, "mol_2"]
            if mols1 and mols2:
                self._mols1.append(mols1)
                self._mols2.append(mols2)
                valid_indices.append(idx)

        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        self._graphs1: Optional[List[List[MolGraph]]] = None
        self._graphs2: Optional[List[List[MolGraph]]] = None

        ef1 = extra_feat(self.df, "1")
        ef2 = extra_feat(self.df, "2")

        if fit_scaler:
            ef_all = np.vstack([ef1, ef2])
            self.scaler = StandardScaler().fit(ef_all)
        else:
            self.scaler = scaler

        self.ef1 = self.scaler.transform(ef1) if self.scaler else ef1
        self.ef2 = self.scaler.transform(ef2) if self.scaler else ef2

        self.y1 = self.df[[f"log_{t}_1" for t in TASK_NAMES]].values.astype(np.float32)
        self.y2 = self.df[[f"log_{t}_2" for t in TASK_NAMES]].values.astype(np.float32)

    def _ensure_featurized(self):
        if self._graphs1 is not None:
            return
        self._graphs1 = []
        self._graphs2 = []
        for mols1, mols2 in zip(self._mols1, self._mols2):
            self._graphs1.append([self._featurizer(m) for m in mols1])
            self._graphs2.append([self._featurizer(m) for m in mols2])

    @property
    def graphs1(self) -> List[List[MolGraph]]:
        self._ensure_featurized()
        return self._graphs1

    @property
    def graphs2(self) -> List[List[MolGraph]]:
        self._ensure_featurized()
        return self._graphs2

    def __len__(self):
        return len(self.df)


class PairDataset(_BasePairDataset):
    """Each sample = a pair of polymers (mol1, mol2), each with potentially multiple structures."""

    def __init__(
        self,
        df: pd.DataFrame,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        super().__init__(df, scaler, fit_scaler)

    def __getitem__(self, idx):
        return (
            self.graphs1[idx],
            self.graphs2[idx],
            torch.tensor(self.ef1[idx]),
            torch.tensor(self.ef2[idx]),
            torch.tensor(self.y1[idx]),
            torch.tensor(self.y2[idx]),
        )


def collate_fn(batch):
    """
    Collate function for PairDataset.

    Flattens all MolGraphs from all samples into a single BatchMolGraph per side,
    and tracks the number of structures per sample (n1s, n2s) for embedding averaging.
    """
    g1s_list, g2s_list, ef1s, ef2s, y1s, y2s = zip(*batch)

    all_g1s = []
    all_g2s = []
    n1s = []
    n2s = []

    for g1_list in g1s_list:
        all_g1s.extend(g1_list)
        n1s.append(len(g1_list))

    for g2_list in g2s_list:
        all_g2s.extend(g2_list)
        n2s.append(len(g2_list))

    if not all_g1s or not all_g2s:
        raise ValueError("Found empty graph list in batch")

    return (
        BatchMolGraph(all_g1s),
        n1s,
        BatchMolGraph(all_g2s),
        n2s,
        torch.stack(ef1s),
        torch.stack(ef2s),
        torch.stack(y1s),
        torch.stack(y2s),
    )


class CachedPairDataset(_BasePairDataset):
    """Dataset with pre-built BatchMolGraph, suitable for shuffle=False scenarios."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.batch_size = batch_size
        super().__init__(df, scaler, fit_scaler)
        self._build_cached_batches()

    def _build_cached_batches(self):
        self._cached_batches = []
        n = len(self.df)
        for i in range(0, n, self.batch_size):
            end_idx = min(i + self.batch_size, n)
            batch_indices = list(range(i, end_idx))

            all_g1s = []
            all_g2s = []
            n1s = []
            n2s = []

            for idx in batch_indices:
                all_g1s.extend(self.graphs1[idx])
                n1s.append(len(self.graphs1[idx]))
                all_g2s.extend(self.graphs2[idx])
                n2s.append(len(self.graphs2[idx]))

            bmg1 = BatchMolGraph(all_g1s)
            bmg2 = BatchMolGraph(all_g2s)

            ef1_batch = torch.tensor(self.ef1[batch_indices])
            ef2_batch = torch.tensor(self.ef2[batch_indices])
            y1_batch = torch.tensor(self.y1[batch_indices])
            y2_batch = torch.tensor(self.y2[batch_indices])

            self._cached_batches.append(
                (bmg1, n1s, bmg2, n2s, ef1_batch, ef2_batch, y1_batch, y2_batch))

        self._graphs1 = None
        self._graphs2 = None
        self._mols1 = []
        self._mols2 = []

    def __len__(self):
        return len(self._cached_batches)

    def __getitem__(self, idx):
        return self._cached_batches[idx]


def collate_cached_batch(batch):
    """Collate function for CachedPairDataset, returns pre-built batch directly."""
    if len(batch) == 1:
        return batch[0]
    raise ValueError(
        "CachedPairDataset should be used with DataLoader batch_size=1. "
        "Got batch_size > 1, which requires complex merging of group sizes."
    )
