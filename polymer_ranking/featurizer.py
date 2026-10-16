"""Featurizer factory and custom atom featurizer."""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType

try:
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.featurizers.atom import MultiHotAtomFeaturizer
    from chemprop.featurizers.bond import MultiHotBondFeaturizer
    from chemprop.conf import DEFAULT_BOND_FDIM
except ImportError as e:
    raise ImportError(
        "chemprop>=2.0.0 is required. Install with: pip install chemprop>=2.0.0"
    ) from e

from .config import CP_FEATURE_DIM


class CustomMultiHotAtomFeaturizer(MultiHotAtomFeaturizer):
    """
    chemprop MultiHotAtomFeaturizer + 1 connection point (CP) binary feature.

    Atom feature vector = chemprop default features || [is_cp]
    Requires atom.SetBoolProp("is_cp", True/False) to be set beforehand.
    """

    def __init__(self, atomic_nums=None):
        if atomic_nums is None:
            atomic_nums = list(range(1, 37)) + [52, 53]

        super().__init__(
            atomic_nums=atomic_nums,
            degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0],
            chiral_tags=list(range(4)),
            num_Hs=list(range(5)),
            hybridizations=[
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
        )

    def __call__(self, atom: Chem.Atom) -> np.ndarray:
        base = super().__call__(atom)
        is_cp = float(atom.HasProp("is_cp") and atom.GetBoolProp("is_cp"))
        return np.append(base, is_cp).astype(np.float32)

    def __len__(self) -> int:
        return super().__len__() + CP_FEATURE_DIM


def create_featurizer() -> SimpleMoleculeMolGraphFeaturizer:
    """Create unified molecule graph featurizer (eliminates duplicate creation)."""
    return SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=CustomMultiHotAtomFeaturizer(),
        bond_featurizer=MultiHotBondFeaturizer(),
    )


def get_feature_dims():
    """Return (atom_fdim, bond_fdim) based on CustomMultiHotAtomFeaturizer."""
    atom_fdim = len(CustomMultiHotAtomFeaturizer())
    bond_fdim = DEFAULT_BOND_FDIM
    return atom_fdim, bond_fdim


# ── Edge weights for non-ring sp3-carbon down-weighting ─────────────────────


def _compute_sp3_carbon_indices():
    """Probe CustomMultiHotAtomFeaturizer to find feature indices for carbon
    (atomic_num=6) and SP3 hybridization in the atom feature vector.

    The feature vector layout is a concatenation of one-hot blocks:
      [atomic_nums | degree | formal_charge | chiral_tag | num_Hs | hybridization
       | aromaticity | mass | is_cp]

    Each one-hot block has length len(choices) + 1 (the +1 is an unknown-pad slot).

    NOTE: This function accesses ``MultiHotAtomFeaturizer._subfeats``, a private
    attribute. If chemprop changes its internal layout, the indices must be
    recomputed. A public API for feature-index queries does not exist in chemprop
    v2, so this is the pragmatic approach.
    """
    feat = CustomMultiHotAtomFeaturizer()
    # _subfeats order: atomic_nums, degrees, formal_charges, chiral_tags, num_Hs, hybridizations
    offset = 0
    carbon_idx = None
    sp3_idx = None
    for i, sf in enumerate(feat._subfeats):
        if i == 0:  # atomic_nums
            carbon_idx = offset + sf.get(6, -1)  # 6 = carbon atomic number
        if i == 5:  # hybridizations
            sp3_idx = offset + sf.get(HybridizationType.SP3, -1)
        offset += len(sf) + 1  # +1 for unknown-pad slot
    if carbon_idx is None or sp3_idx is None or carbon_idx < 0 or sp3_idx < 0:
        raise RuntimeError("Could not determine carbon/sp3 feature indices")
    return carbon_idx, sp3_idx


def _compute_bond_in_ring_idx():
    """Probe MultiHotBondFeaturizer to find the in-ring feature index.

    Bond feature layout: [null(1) | bond_type(len) | conjugated(1) | in_ring(1) | stereo(len+1)]
    so in_ring index = 2 + len(bond_types).

    Uses the same default MultiHotBondFeaturizer as create_featurizer() to
    guarantee consistency.
    """
    bf = MultiHotBondFeaturizer()
    return 2 + len(bf.bond_types)


def _compute_halogen_indices():
    """Probe CustomMultiHotAtomFeaturizer for the F/Cl/Br/I atomic-number slots.

    Used by compute_edge_weights to detect halogen-substituted sp3 carbons
    (CF3, OCF3, SCF3, CH2Cl, ...), which must stay on the main chain.
    """
    feat = CustomMultiHotAtomFeaturizer()
    offset = 0
    idxs = []
    for i, sf in enumerate(feat._subfeats):
        if i == 0:  # atomic_nums
            for z in (9, 17, 35, 53):  # F, Cl, Br, I
                pos = sf.get(z, -1)
                if pos >= 0:
                    idxs.append(offset + pos)
        offset += len(sf) + 1  # +1 for unknown-pad slot
    if not idxs:
        raise RuntimeError("Could not determine halogen feature indices")
    return idxs


_ATOM_CARBON_IDX, _HYBRID_SP3_IDX = _compute_sp3_carbon_indices()
_BOND_IN_RING_IDX = _compute_bond_in_ring_idx()
_ATOM_HALOGEN_IDXS = _compute_halogen_indices()


def compute_edge_weights(
    V: torch.Tensor,
    edge_index: torch.Tensor,
    E: torch.Tensor,
    sp3_weight: float = 0.2,
) -> torch.Tensor:
    """Compute per-edge message-passing weights.

    Edges (bonds) touching **non-ring** sp3-hybridized carbons — i.e. non-conjugated
    side chains (e.g. alkyl groups) that have minimal influence on charge mobility —
    receive weight ``sp3_weight`` (default 0.2). sp3 carbons that are part of a ring
    (e.g. cyclohexane fused to an aromatic system) are NOT down-weighted, since they
    belong to the main structural backbone. sp3 carbons bonded to a halogen
    (CF3, OCF3, SCF3, CH2Cl, ...) are also NOT down-weighted: their C–X bonds
    carry a strong inductive effect on the frontier energy levels (HOMO/LUMO),
    so these substituents must stay on the main chain. All other edges receive
    weight 1.0. Both directed edges of a given bond share the same weight.

    Parameters
    ----------
    V : Tensor [num_atoms, feat_dim]
        Atom feature matrix (``BatchMolGraph.V``).
    edge_index : Tensor [2, num_edges]
        COO-format edge index (``BatchMolGraph.edge_index``).
    E : Tensor [num_edges, bond_fdim]
        Bond feature matrix (``BatchMolGraph.E``); used to detect ring membership.
    sp3_weight : float, default 0.2
        Weight applied to edges whose source or destination atom is a non-ring
        sp3 carbon without halogen substituents.

    Returns
    -------
    Tensor [num_edges]
        Per-edge weights.
    """
    # sp3 carbon detection
    is_carbon = V[:, _ATOM_CARBON_IDX] == 1.0
    is_sp3 = V[:, _HYBRID_SP3_IDX] == 1.0
    is_sp3_carbon = is_carbon & is_sp3

    # Determine which atoms are in a ring: an atom is in a ring if any of its
    # bonds has the in-ring bit set.
    edge_in_ring = E[:, _BOND_IN_RING_IDX] == 1.0  # [num_edges]
    num_atoms = V.shape[0]
    atom_in_ring = torch.zeros(num_atoms, dtype=torch.bool, device=V.device)
    atom_in_ring[edge_index[0][edge_in_ring]] = True
    atom_in_ring[edge_index[1][edge_in_ring]] = True

    # sp3 carbons with a halogen neighbor (CF3, OCF3, SCF3, CH2Cl, ...) are
    # energy-level-relevant substituents, not inert side chains -> keep them
    # on the main chain. scatter_add_ over both edge directions accumulates
    # every neighbor (safe under duplicate indices, unlike buffered |=).
    is_halogen = (V[:, _ATOM_HALOGEN_IDXS] == 1.0).any(dim=1)
    has_halog_nbr = torch.zeros(num_atoms, dtype=V.dtype, device=V.device)
    has_halog_nbr.scatter_add_(0, edge_index[0],
                               is_halogen[edge_index[1]].to(V.dtype))
    has_halog_nbr.scatter_add_(0, edge_index[1],
                               is_halogen[edge_index[0]].to(V.dtype))

    # Only non-ring, halogen-free sp3 carbons (alkyl side chains) are down-weighted
    is_sidechain_sp3 = is_sp3_carbon & ~atom_in_ring & (has_halog_nbr == 0)

    src = edge_index[0]
    dst = edge_index[1]
    touches_sidechain_sp3 = is_sidechain_sp3[src] | is_sidechain_sp3[dst]

    weights = torch.ones(edge_index.shape[1], device=V.device, dtype=V.dtype)
    weights[touches_sidechain_sp3] = sp3_weight
    return weights
