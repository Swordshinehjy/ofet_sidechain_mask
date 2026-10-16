"""Tests for chemistry preprocessing: SMILES helpers, monomer concatenation,
cyclization, DataFrame operations."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from polymer_ranking.chemistry import (
    normalize_smiles,
    is_homocoupling,
    get_star_info,
    connect_at_stars,
    generate_all_connections,
    cyclize_polymer_with_cp_marking,
    generate_cyclized_from_monomers,
    cyclize_df,
    extra_feat,
    load_and_preprocess,
)
from polymer_ranking.config import EXTRA_COLS, TASK_NAMES

from samples import (
    THIOPHENE,
    BENZENE,
    VINYLENE,
    INVALID_MONOMER,
    BAD_SMILES,
)


# ── normalize_smiles ────────────────────────────────────────────────────────

def test_normalize_smiles_canonicalizes():
    assert normalize_smiles("C(C)C") == "CCC"


def test_normalize_smiles_is_idempotent():
    once = normalize_smiles("F/C=C/F")
    assert normalize_smiles(once) == once


def test_normalize_smiles_aromatizes_kekule_input():
    assert normalize_smiles("C1=CC=CC=C1") == "c1ccccc1"


def test_normalize_smiles_invalid_returns_none():
    assert normalize_smiles(BAD_SMILES) is None


def test_normalize_smiles_empty_returns_none():
    assert normalize_smiles("") is None
    assert normalize_smiles(None) is None


# ── is_homocoupling ─────────────────────────────────────────────────────────

def test_is_homocoupling_same_molecule():
    assert is_homocoupling(THIOPHENE, THIOPHENE)


def test_is_homocoupling_different_notation():
    assert is_homocoupling(BENZENE, "*C1=CC=CC=C1*")


def test_is_homocoupling_different_molecules():
    assert not is_homocoupling(THIOPHENE, BENZENE)


def test_is_homocoupling_invalid_smiles():
    assert not is_homocoupling(BAD_SMILES, THIOPHENE)


# ── get_star_info ───────────────────────────────────────────────────────────

def test_get_star_info_finds_two_connection_points():
    mol = Chem.MolFromSmiles(THIOPHENE)
    stars = get_star_info(mol)
    assert len(stars) == 2
    for star_idx, nbr_idx in stars:
        assert mol.GetAtomWithIdx(star_idx).GetSymbol() == "*"
        assert mol.GetAtomWithIdx(nbr_idx).GetSymbol() != "*"


def test_get_star_info_ignores_bare_dummies():
    # a lone '*' has no neighbor -> not a usable connection point
    mol = Chem.MolFromSmiles("[*]")
    assert get_star_info(mol) == []


# ── connect_at_stars ────────────────────────────────────────────────────────

def test_connect_at_stars_thiophene_pair():
    molA = Chem.MolFromSmiles(THIOPHENE)
    molB = Chem.MolFromSmiles(THIOPHENE)
    starA, nbrA = get_star_info(molA)[0]
    starB, nbrB = get_star_info(molB)[0]

    smi = connect_at_stars(molA, molB, starA, nbrA, starB, nbrB)

    assert smi is not None
    product = Chem.MolFromSmiles(smi)
    assert product is not None
    # 2 x 7 atoms, minus the 2 removed dummy atoms
    assert product.GetNumAtoms() == 12
    assert sum(1 for a in product.GetAtoms() if a.GetSymbol() == "*") == 2


# ── generate_all_connections ────────────────────────────────────────────────

def test_generate_all_connections_alternating():
    results = generate_all_connections(THIOPHENE, BENZENE)
    assert len(results) == 2          # two connection orientations
    assert results == sorted(results)  # deterministic order
    for smi in results:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        assert sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "*") == 2


def test_generate_all_connections_preserves_double_bond_stereo():
    results = generate_all_connections(VINYLENE, THIOPHENE)
    # E and Z orientations of the vinylene linkage both appear
    assert any("\\/C" in smi or "/C=C\\" in smi for smi in results)
    assert any("/C=C/" in smi for smi in results)


def test_generate_all_connections_invalid_smiles():
    assert generate_all_connections(BAD_SMILES, THIOPHENE) == []


# ── cyclize_polymer_with_cp_marking ─────────────────────────────────────────

def test_cyclize_thiophene_repeat_unit():
    cyc_smi, cyc_mol = cyclize_polymer_with_cp_marking(THIOPHENE)
    assert cyc_smi == "c1ccsc1"
    cp_atoms = [a.GetIdx() for a in cyc_mol.GetAtoms()
                if a.HasProp("is_cp") and a.GetBoolProp("is_cp")]
    assert len(cp_atoms) == 2


def test_cyclize_vinylene_already_bonded():
    # both '*' neighbors are already double-bonded to each other
    cyc_smi, cyc_mol = cyclize_polymer_with_cp_marking(VINYLENE)
    assert cyc_smi == "C=C"
    assert cyc_mol is not None


def test_cyclize_wrong_star_count_returns_none():
    assert cyclize_polymer_with_cp_marking("CCC") == (None, None)   # 0 stars
    assert cyclize_polymer_with_cp_marking("*c1ccsc1") == (None, None)  # 1 star


def test_cyclize_invalid_smiles_returns_none():
    assert cyclize_polymer_with_cp_marking(BAD_SMILES) == (None, None)


# ── generate_cyclized_from_monomers ─────────────────────────────────────────

def test_homocoupling_gives_single_structure():
    results = generate_cyclized_from_monomers(THIOPHENE, THIOPHENE)
    assert len(results) == 1
    cyc_smi, cyc_mol = results[0]
    assert cyc_smi == "c1ccsc1"
    cp_atoms = [a for a in cyc_mol.GetAtoms()
                if a.HasProp("is_cp") and a.GetBoolProp("is_cp")]
    assert len(cp_atoms) == 2


def test_alternating_gives_multiple_structures():
    results = generate_cyclized_from_monomers(THIOPHENE, BENZENE)
    assert len(results) >= 1
    for cyc_smi, cyc_mol in results:
        assert isinstance(cyc_smi, str)
        assert Chem.MolFromSmiles(cyc_smi) is not None
        assert cyc_mol.GetNumAtoms() > 0


def test_invalid_monomers_give_empty_list():
    assert generate_cyclized_from_monomers(INVALID_MONOMER, THIOPHENE) == []
    assert generate_cyclized_from_monomers(BAD_SMILES, BENZENE) == []


# ── cyclize_df ──────────────────────────────────────────────────────────────

def test_cyclize_df_adds_structure_columns(paired_df):
    for s in ("1", "2"):
        assert f"cyc_{s}" in paired_df.columns
        assert f"mol_{s}" in paired_df.columns
    # valid monomers yield at least one structure per side
    assert all(len(lst) > 0 for lst in paired_df["mol_1"][:4])
    assert all(len(lst) > 0 for lst in paired_df["mol_2"])


def test_cyclize_df_does_not_mutate_input(raw_pair_df):
    before = raw_pair_df.copy()
    cyclize_df(raw_pair_df)
    pd.testing.assert_frame_equal(raw_pair_df, before)


def test_cyclize_df_invalid_monomer_gives_empty_list(raw_pair_df):
    out = cyclize_df(raw_pair_df)
    # row 4 has an invalid monomer A_1
    assert len(out.loc[4, "mol_1"]) == 0


# ── extra_feat ──────────────────────────────────────────────────────────────

def test_extra_feat_shape_and_dtype(paired_df):
    ef = extra_feat(paired_df, "1")
    assert ef.shape == (len(paired_df), len(EXTRA_COLS))
    assert ef.dtype == np.float32


def test_extra_feat_fills_nan_with_zero():
    df = pd.DataFrame({"conjugation_1": [np.nan], "Isomer_1": [1.0],
                       "CentroSymmetry_1": [0.0], "E_LUMO (eV)_1": [np.nan],
                       "E_HOMO (eV)_1": [-5.0]})
    ef = extra_feat(df, "1")
    assert np.isnan(ef).sum() == 0
    assert ef[0, 0] == 0.0 and ef[0, 3] == 0.0


# ── load_and_preprocess ─────────────────────────────────────────────────────

def test_load_and_preprocess_drops_invalid_rows(prepped_df, raw_pair_df):
    # 5 raw rows; row 3 has a NaN target, row 4 an invalid monomer
    assert len(raw_pair_df) == 5
    assert len(prepped_df) == 3


def test_load_and_preprocess_adds_log_targets(prepped_df):
    for t in TASK_NAMES:
        for s in ("1", "2"):
            assert f"log_{t}_{s}" in prepped_df.columns
            assert f"ok_{t}_{s}" in prepped_df.columns


def test_load_and_preprocess_log10_values(prepped_df, raw_pair_df):
    kept = raw_pair_df.iloc[[0, 1, 2]].reset_index(drop=True)
    expected = np.log10(kept["mu_e_1"].clip(lower=1e-12))
    assert np.allclose(prepped_df["log_mu_e_1"].values, expected.values)


def test_load_and_preprocess_ok_flags(prepped_df):
    # row 2 (benzene homocoupling) has mu_e_2 = 0 -> censored
    ok_e2 = prepped_df["ok_mu_e_2"].values
    assert ok_e2.tolist() == [True, True, False]
    assert prepped_df["ok_mu_h_1"].all()
