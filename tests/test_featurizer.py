"""Tests for the custom atom featurizer, feature dims, and edge weights."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from chemprop.conf import DEFAULT_BOND_FDIM
from chemprop.data.collate import BatchMolGraph
from rdkit import Chem

from polymer_ranking.featurizer import (
    CustomMultiHotAtomFeaturizer,
    create_featurizer,
    get_feature_dims,
    compute_edge_weights,
)
from polymer_ranking.chemistry import cyclize_polymer_with_cp_marking

from samples import ALKYL_THIOPHENE, THIOPHENE


# ── CustomMultiHotAtomFeaturizer ────────────────────────────────────────────

def test_feature_dims_consistent():
    atom_fdim, bond_fdim = get_feature_dims()
    assert atom_fdim == len(CustomMultiHotAtomFeaturizer())
    assert bond_fdim == DEFAULT_BOND_FDIM
    assert atom_fdim > 1 and bond_fdim > 1


def test_cp_feature_adds_one_slot():
    # custom featurizer = chemprop default features + 1 connection-point bit
    feat = CustomMultiHotAtomFeaturizer()
    atom = Chem.MolFromSmiles(THIOPHENE).GetAtomWithIdx(0)  # a '*' atom
    vec = feat(atom)
    assert vec.shape[0] == len(feat)
    assert feat.__len__() >= 2  # base features + is_cp slot


def test_is_cp_flag_is_last_feature():
    feat = CustomMultiHotAtomFeaturizer()
    atom = Chem.MolFromSmiles(THIOPHENE).GetAtomWithIdx(1)  # an aromatic carbon

    atom.SetBoolProp("is_cp", True)
    marked = feat(atom)
    atom.ClearProp("is_cp")
    unmarked = feat(atom)

    assert marked[-1] == 1.0
    assert unmarked[-1] == 0.0
    # base features identical, only the CP bit differs
    np.testing.assert_array_equal(marked[:-1], unmarked[:-1])
    assert marked.dtype == np.float32


def test_create_featurizer_returns_configured_instance():
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    feat = create_featurizer()
    assert isinstance(feat, SimpleMoleculeMolGraphFeaturizer)
    vec = feat.atom_featurizer(Chem.MolFromSmiles(THIOPHENE).GetAtomWithIdx(0))
    assert vec.shape[0] == get_feature_dims()[0]


# ── compute_edge_weights ────────────────────────────────────────────────────

def _edge_weights_for(smiles: str, sp3_weight: float = 0.2) -> torch.Tensor:
    feat = create_featurizer()
    graph = feat(Chem.MolFromSmiles(smiles))
    bmg = BatchMolGraph([graph])
    return compute_edge_weights(bmg.V, bmg.edge_index, bmg.E, sp3_weight)


def test_edge_weights_conjugated_backbone_all_one():
    # aromatic thiophene ring: no non-ring sp3 carbons
    w = _edge_weights_for(THIOPHENE)
    assert torch.all(w == 1.0)


def test_edge_weights_downweight_sp3_sidechain():
    w = _edge_weights_for(ALKYL_THIOPHENE)
    assert torch.any(w == 0.2)    # alkyl chain edges are down-weighted
    assert torch.any(w == 1.0)    # aromatic backbone edges are not
    # the number of down-weighted directed edges matches the butyl chain
    # (4 chain carbons -> 4 bonds, each appearing twice as directed edges)
    assert int((w == 0.2).sum()) == 8


def test_edge_weights_ring_sp3_not_downweighted():
    # distar cyclohexane: sp3 carbons but inside a ring -> weight stays 1.0
    w = _edge_weights_for("*C1CCCCC1*")
    assert torch.all(w == 1.0)


def test_edge_weights_full_sp3_chain_all_downweighted():
    w = _edge_weights_for("*CCCC*")
    assert torch.all(w == 0.2)


def test_edge_weights_custom_weight():
    w = _edge_weights_for("*CCCC*", sp3_weight=0.5)
    assert torch.all(w == 0.5)


def test_edge_weights_shape_matches_edge_index():
    feat = create_featurizer()
    graph = feat(Chem.MolFromSmiles(ALKYL_THIOPHENE))
    bmg = BatchMolGraph([graph])
    w = compute_edge_weights(bmg.V, bmg.edge_index, bmg.E)
    assert w.shape == (bmg.edge_index.shape[1],)
    assert w.dtype == bmg.V.dtype
    assert w.device == bmg.V.device


# ── halogenated sp3 substituents stay on the main chain ─────────────────────
#
# CF3 / OCF3 / SCF3 / CH2Cl are strong inductive (HOMO/LUMO-shifting) groups,
# so sp3 carbons bearing a halogen are exempt from side-chain down-weighting.
# Pure hydrocarbon sp3 carbons (alkyl chains) are still down-weighted.

def test_halogen_probe_indices_exist():
    from polymer_ranking.featurizer import _ATOM_HALOGEN_IDXS
    assert len(_ATOM_HALOGEN_IDXS) == 4  # F, Cl, Br, I slots


def test_edge_weights_cf3_not_downweighted():
    # butyl chain masked, CF3 fully on the main chain
    w = _edge_weights_for("*c1cc(CCCC)cc(C(F)(F)F)c1*")
    # only the butyl chain (4 bonds x 2 directed edges) is down-weighted
    assert int((w == 0.2).sum()) == 8
    assert torch.any(w == 1.0)


def test_edge_weights_scf3_not_downweighted():
    w = _edge_weights_for("*c1cc(SC(F)(F)F)ccc1*")
    assert torch.all(w == 1.0)


def test_edge_weights_ocf3_not_downweighted():
    w = _edge_weights_for("*c1cc(OC(F)(F)F)ccc1*")
    assert torch.all(w == 1.0)


def test_edge_weights_chloromethyl_not_downweighted():
    w = _edge_weights_for("*c1cc(CCl)ccc1*")
    assert torch.all(w == 1.0)


def test_edge_weights_methoxy_methyl_still_downweighted():
    # only halogen-bearing sp3 carbons are exempt: the methoxy methyl remains
    # a side chain, while the Ar-O bond (the electronic-effect carrier) does not
    w = _edge_weights_for("*c1cc(OC)ccc1*")
    # exactly the O-CH3 bond (1 bond x 2 directed edges) is down-weighted
    assert int((w == 0.2).sum()) == 2


# ── real dataset monomers: energy-level substituents on the main chain ──────

DATASET_CSV = Path(__file__).resolve().parents[1] / "contrastive_full_paired.csv"

# (name, SMARTS). Matches include the backbone attachment atom so that the
# assertion below covers the attachment bond as well.
DATASET_SUBSTITUENT_CASES = [
    ("F_aryl", "[#6;!X4]~[F]"),
    ("Cl_aryl", "[#6;!X4]~[Cl]"),
    ("nitro", "[$([#6]-[NX3](=O)=O),$([#6]-[NX3+](=O)[O-])]"),
    ("nitrile", "[#6;!X4]~[#6]#[NX1]"),
    ("CF3",
     "[$([#6;!X4]-[CX4]([F])([F])[F]),"
     "$([OX2]-[CX4]([F])([F])[F]),"
     "$([SX2]-[CX4]([F])([F])[F])]"),
    ("SCF3", "[#6;a]-[SX2]-[CX4]([F])([F])[F]"),
]


@lru_cache(maxsize=None)
def _dataset_monomers() -> Tuple[str, ...]:
    df = pd.read_csv(DATASET_CSV)
    smis = []
    for col in ("MonomerA_1", "MonomerB_1", "MonomerA_2", "MonomerB_2"):
        smis += df[col].dropna().astype(str).str.strip().tolist()
    return tuple(smis)


def _first_cyclizable_monomer(smarts: str) -> Optional[Tuple[str, Chem.Mol]]:
    patt = Chem.MolFromSmarts(smarts)
    for smi in _dataset_monomers():
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not mol.HasSubstructMatch(patt):
            continue
        _, cyc_mol = cyclize_polymer_with_cp_marking(smi)
        if cyc_mol is not None and cyc_mol.GetSubstructMatch(patt):
            return smi, cyc_mol
    return None


def _undirected_edge_weights(mol: Chem.Mol) -> Dict[Tuple[int, int], float]:
    """Run the real featurizer + compute_edge_weights; map undirected bond -> weight.

    Takes the Mol object itself (NOT a re-parsed SMILES) so atom indices stay
    aligned with SMARTS matches on the same molecule.
    """
    feat = create_featurizer()
    bmg = BatchMolGraph([feat(mol)])
    w = compute_edge_weights(bmg.V, bmg.edge_index, bmg.E)
    ei = bmg.edge_index
    und: Dict[Tuple[int, int], float] = {}
    for k in range(ei.shape[1]):
        key = tuple(sorted((int(ei[0, k]), int(ei[1, k]))))
        und[key] = float(w[k])
    return und


@pytest.mark.skipif(not DATASET_CSV.exists(), reason="dataset CSV not present")
@pytest.mark.parametrize("name,smarts", DATASET_SUBSTITUENT_CASES,
                         ids=[c[0] for c in DATASET_SUBSTITUENT_CASES])
def test_real_dataset_substituent_on_main_chain(name, smarts):
    """Every bond inside (and attaching) the substituent must keep weight 1.0
    on a real monomer taken from contrastive_full_paired.csv."""
    found = _first_cyclizable_monomer(smarts)
    assert found is not None, f"no cyclizable monomer with {name} in dataset"
    smi, cyc_mol = found

    patt = Chem.MolFromSmarts(smarts)
    matches = cyc_mol.GetSubstructMatches(patt)
    assert matches

    cyc_smi = Chem.MolToSmiles(cyc_mol)
    und = _undirected_edge_weights(cyc_mol)
    for match in matches:
        mset = set(match)
        for (i, j), w in und.items():
            if i in mset and j in mset:
                assert w == 1.0, (
                    f"{name}: bond {i}-{j} down-weighted to {w} inside the "
                    f"substituent of {smi} (cyclized: {cyc_smi})")
