"""Chemistry preprocessing: monomer concatenation, cyclization, extra feature extraction, DataFrame preprocessing."""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import EXTRA_COLS, TASK_NAMES

logger = logging.getLogger(__name__)


# ── SMILES helpers ──────────────────────────────────────────────────────────

def normalize_smiles(smi: str) -> Optional[str]:
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None
    except (ValueError, RuntimeError):
        return None


def get_star_info(mol) -> List[Tuple[int, int]]:
    """
    Return [(star_idx, neighbor_idx), ...] for every '*' atom that has
    exactly one neighbor.
    """
    stars = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            nbrs = atom.GetNeighbors()
            if len(nbrs) == 1:
                stars.append((atom.GetIdx(), nbrs[0].GetIdx()))
    return stars


# ── Monomer concatenation ──────────────────────────────────────────────────

def connect_at_stars(molA, molB, starA: int, nbrA: int, starB: int, nbrB: int) -> Optional[str]:
    """
    Connect one '*' from A to one '*' from B, preserving double bond stereochemistry.
    Returns SMILES of the connected molecule (still has 2 '*' atoms), or None on failure.
    """
    try:
        def get_stereo_bonds(mol, star_idx):
            stereo_bonds = []
            for bond in mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    st = bond.GetStereo()
                    if st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
                        st_atoms = list(bond.GetStereoAtoms())
                        if star_idx in st_atoms:
                            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            stereo_bonds.append((u, v, st, st_atoms))
            return stereo_bonds

        stereoA = get_stereo_bonds(molA, starA)
        stereoB = get_stereo_bonds(molB, starB)

        bondA = molA.GetBondBetweenAtoms(starA, nbrA)
        dirA = bondA.GetBondDir() if bondA else Chem.BondDir.NONE
        typeA = bondA.GetBondType() if bondA else Chem.BondType.SINGLE

        bondB = molB.GetBondBetweenAtoms(starB, nbrB)
        dirB = bondB.GetBondDir() if bondB else Chem.BondDir.NONE
        typeB = bondB.GetBondType() if bondB else Chem.BondType.SINGLE

        rw = Chem.RWMol(Chem.CombineMols(Chem.RWMol(molA), Chem.RWMol(molB)))
        offset = molA.GetNumAtoms()
        nbrB_g = nbrB + offset
        starB_g = starB + offset

        new_bond_type = Chem.BondType.SINGLE
        if typeA == Chem.BondType.DOUBLE and typeB == Chem.BondType.DOUBLE:
            new_bond_type = Chem.BondType.DOUBLE

        rw.AddBond(nbrA, nbrB_g, new_bond_type)
        new_bond = rw.GetBondBetweenAtoms(nbrA, nbrB_g)

        if new_bond and new_bond_type == Chem.BondType.SINGLE:
            if dirA != Chem.BondDir.NONE:
                new_bond.SetBondDir(dirA)
            elif dirB != Chem.BondDir.NONE:
                rev_dir = {
                    Chem.BondDir.ENDUPRIGHT: Chem.BondDir.ENDDOWNRIGHT,
                    Chem.BondDir.ENDDOWNRIGHT: Chem.BondDir.ENDUPRIGHT
                }
                new_bond.SetBondDir(rev_dir.get(dirB, dirB))

        for u, v, st, st_atoms in stereoA:
            b = rw.GetBondBetweenAtoms(u, v)
            new_st_atoms = [nbrB_g if x == starA else x for x in st_atoms]
            b.SetStereoAtoms(*new_st_atoms)
            b.SetStereo(st)

        for u, v, st, st_atoms in stereoB:
            b = rw.GetBondBetweenAtoms(u + offset, v + offset)
            new_st_atoms = [nbrA if x == starB else (x + offset) for x in st_atoms]
            b.SetStereoAtoms(*new_st_atoms)
            b.SetStereo(st)

        for idx in sorted([starA, starB_g], reverse=True):
            rw.RemoveAtom(idx)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=False)

        return Chem.MolToSmiles(mol, isomericSmiles=True)

    except (ValueError, RuntimeError, Chem.KekulizeException) as e:
        logger.debug(f"connect_at_stars failed: {e}")
        return None


def generate_all_connections(smilesA: str, smilesB: str) -> List[str]:
    """
    Each of A and B has (at least) 2 '*' atoms.
    Try every combination of one '*' from A x one '*' from B -> up to 4 products.
    Returns a deduplicated list of canonical SMILES (each still has 2 '*').
    For symmetric monomers (A==B) this typically yields 1 unique structure;
    for asymmetric monomers (A!=B) this typically yields 2 unique structures.
    """
    molA = Chem.MolFromSmiles(smilesA)
    molB = Chem.MolFromSmiles(smilesB)
    if molA is None or molB is None:
        return []

    starsA = get_star_info(molA)
    starsB = get_star_info(molB)

    results = set()
    for starA, nbrA in starsA:
        for starB, nbrB in starsB:
            smi = connect_at_stars(molA, molB, starA, nbrA, starB, nbrB)
            if smi:
                norm = normalize_smiles(smi)
                if norm:
                    results.add(norm)

    return list(results)


# ── Cyclization ────────────────────────────────────────────────────────────

def cyclize_polymer_with_cp_marking(
        smiles: str) -> Tuple[Optional[str], Optional[Chem.Mol]]:
    """
    Cyclize polymer repeat unit and mark connection points (CP).

    Returns
    -------
    Tuple[smiles, mol]
        - smiles: Cyclized SMILES string
        - mol: Cyclized RDKit Mol object with connection point atoms marked with is_cp property
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    Chem.SanitizeMol(mol)

    dummy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*'
    ]
    if len(dummy_indices) != 2:
        return None, None
    try:
        idx1, idx2 = dummy_indices[0], dummy_indices[1]
        atom1 = mol.GetAtomWithIdx(idx1)
        atom2 = mol.GetAtomWithIdx(idx2)
        neighbors1 = atom1.GetNeighbors()
        neighbors2 = atom2.GetNeighbors()
        if not neighbors1 or not neighbors2:
            return None, None
        n1_idx = neighbors1[0].GetIdx()
        n2_idx = neighbors2[0].GetIdx()

        bond1 = mol.GetBondBetweenAtoms(idx1, n1_idx)
        bond2 = mol.GetBondBetweenAtoms(idx2, n2_idx)
        bond_type = Chem.BondType.SINGLE
        if bond1 is not None and bond1.GetBondType() != Chem.BondType.SINGLE:
            bond_type = bond1.GetBondType()
        elif bond2 is not None:
            bond_type = bond2.GetBondType()
        bond_dir = Chem.BondDir.NONE
        if bond1 is not None and bond1.GetBondDir() != Chem.BondDir.NONE:
            bond_dir = bond1.GetBondDir()
        elif bond2 is not None:
            bond_dir = bond2.GetBondDir()

        rw_mol = Chem.RWMol(mol)

        if n1_idx != n2_idx:
            existing_bond = rw_mol.GetBondBetweenAtoms(n1_idx, n2_idx)
            if existing_bond is None:
                rw_mol.AddBond(n1_idx, n2_idx, order=bond_type)
                new_bond = rw_mol.GetBondBetweenAtoms(n1_idx, n2_idx)
                if bond_dir != Chem.BondDir.NONE and new_bond is not None:
                    new_bond.SetBondDir(bond_dir)

        for idx in sorted(dummy_indices, reverse=True):
            if idx < n1_idx:
                n1_idx -= 1
            if idx < n2_idx:
                n2_idx -= 1
            rw_mol.RemoveAtom(idx)

        cp_atom_indices = [n1_idx, n2_idx]

        cyclic_mol = rw_mol.GetMol()
        Chem.SanitizeMol(cyclic_mol)
        Chem.AssignStereochemistry(cyclic_mol, force=True, cleanIt=True)

        for cp_idx in cp_atom_indices:
            atom = cyclic_mol.GetAtomWithIdx(cp_idx)
            atom.SetBoolProp("is_cp", True)

        return Chem.MolToSmiles(cyclic_mol), cyclic_mol

    except (ValueError, RuntimeError, Chem.KekulizeException) as e:
        logger.error(f"Error processing smiles {smiles}: {e}")
        return None, None


def generate_cyclized_from_monomers(
        monomerA_smiles: str,
        monomerB_smiles: str) -> List[Tuple[str, Chem.Mol]]:
    """
    Generate all possible cyclized structures from two monomers.

    Pipeline:
      1. Connect monomers A and B at their '*' atoms to form repeat units
         (using generate_all_connections, which tries all star pair combinations)
      2. Cyclize each repeat unit by connecting remaining '*' atoms
      3. Mark connection points

    For symmetric monomers (A==B), typically 1 unique cyclized structure.
    For asymmetric monomers (A!=B), typically 2 unique cyclized structures
    (corresponding to the two possible connection orientations).

    Returns list of (cyc_smiles, mol_with_cp) tuples.
    """
    repeat_units = generate_all_connections(monomerA_smiles, monomerB_smiles)

    results = []
    for smi in repeat_units:
        cyc_smi, cyc_mol = cyclize_polymer_with_cp_marking(smi)
        if cyc_smi is not None and cyc_mol is not None:
            results.append((cyc_smi, cyc_mol))

    return results


# ── DataFrame operations ───────────────────────────────────────────────────

def cyclize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cyclized structures from monomers, add cyc_1/2, mol_1/2 columns.

    Each polymer is built from MonomerA_{s} + MonomerB_{s} via
    generate_cyclized_from_monomers, which may produce multiple cyclized
    structures for asymmetric monomers.

    cyc_{s} and mol_{s} columns store lists (one element per unique structure).
    """
    df = df.copy()
    for s in ["1", "2"]:
        col_a = f"MonomerA_{s}"
        col_b = f"MonomerB_{s}"

        cyc_lists = []
        mol_lists = []
        smi_a_series = df[col_a].astype(str).str.strip()
        smi_b_series = df[col_b].astype(str).str.strip()
        for smi_a, smi_b in zip(smi_a_series, smi_b_series):
            results = generate_cyclized_from_monomers(smi_a, smi_b)
            cyc_lists.append([r[0] for r in results])
            mol_lists.append([r[1] for r in results])

        df[f"cyc_{s}"] = cyc_lists
        df[f"mol_{s}"] = mol_lists
    return df


def extra_feat(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Extract extra feature columns, fill NaN with 0."""
    cols = [c.format(s=suffix) for c in EXTRA_COLS]
    return df[cols].fillna(0.0).values.astype(np.float32)


def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and perform:
      - Monomer concatenation + polymer cyclization
      - Take log10 of mobility columns
      - Drop invalid rows
    """
    df = pd.read_csv(csv_path)
    df = cyclize_df(df)

    target_raw = [f"{t}_{s}" for t in TASK_NAMES for s in ("1", "2")]

    mask = df["mol_1"].apply(len) > 0
    mask &= df["mol_2"].apply(len) > 0
    for col in target_raw:
        mask &= df[col].notna()
    df = df[mask].reset_index(drop=True)

    for col in target_raw:
        df[f"log_{col}"] = np.log10(df[col].clip(lower=1e-12))

    logger.info(f"Preprocessed dataset: {len(df)} valid pairs")
    return df
