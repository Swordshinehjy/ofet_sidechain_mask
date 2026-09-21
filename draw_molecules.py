"""Grid visualization of unique polymer structures with backbone / side-chain coloring.

Each polymer is rendered as a 2D structure where

* the **backbone** (ring / conjugated part) is highlighted in one color, and
* the **side chain** (non-ring sp3 carbons, e.g. alkyl chains) is highlighted in
  another color.

The side-chain criterion is exactly the one used by
``polymer_ranking/featurizer.py::compute_edge_weights`` (non-ring sp3 carbon),
so the picture shows which part of the molecule is actually down-weighted by
``sp3_weight`` during message passing.

Structures are arranged into pages of ``cols x rows`` molecules (4x4 by default)
and written as PNG files.

Usage
-----
::

    # one repeat unit per unique polymer, dummy (*) atoms removed (default)
    python draw_molecules.py --limit 16

    # cyclized model compound (the input actually seen by the D-MPNN encoder)
    python draw_molecules.py --mode cyclized --cols 4 --rows 4

    # custom colors and page size
    python draw_molecules.py --main_color "#1f6fb2" --side_color "#f0803c" --cols 4 --rows 4

Run this with the environment that has rdkit installed, e.g.::

    D:/anaconda3/envs/chemprop2/python.exe draw_molecules.py --limit 16
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

DEFAULT_MAIN_COLOR = "#1f6fb2"   # backbone: blue
DEFAULT_SIDE_COLOR = "#f0803c"   # side chain: orange


# ── helpers ─────────────────────────────────────────────────────────────────

def hex_to_rgb255(color: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` to a 0-255 RGB tuple."""
    c = color.strip().lstrip("#")
    if len(c) != 6:
        raise ValueError(f"color must be in #rrggbb form, got {color!r}")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb01(color: str) -> Tuple[float, float, float]:
    """Convert ``#rrggbb`` to the 0-1 RGB tuple expected by RDKit."""
    r, g, b = hex_to_rgb255(color)
    return (r / 255.0, g / 255.0, b / 255.0)


def canonical_smiles(smi: str) -> str:
    """Canonicalize a SMILES string; fall back to the input if it cannot be parsed."""
    if not smi:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return str(smi).strip()
    return Chem.MolToSmiles(mol)


def strip_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Remove ``*`` dummy atoms from a repeat unit.

    A singly bonded dummy is first turned into an explicit H and then removed by
    ``RemoveHs``, so the neighbouring atom keeps a valid valence (no radicals).
    Any remaining dummy (e.g. attached by a non-single bond) is deleted directly.
    """
    if mol is None:
        return None

    rw = Chem.RWMol(mol)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummies:
        return mol

    for idx in dummies:
        atom = rw.GetAtomWithIdx(idx)
        bonds = list(atom.GetBonds())
        if len(bonds) == 1 and bonds[0].GetBondType() == Chem.BondType.SINGLE:
            atom.SetAtomicNum(1)  # dummy -> H

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        out = Chem.RemoveHs(out)
    except (ValueError, RuntimeError, Chem.KekulizeException):
        pass

    rw2 = Chem.RWMol(out)
    rest = [a.GetIdx() for a in rw2.GetAtoms() if a.GetAtomicNum() == 0]
    if rest:
        for idx in sorted(rest, reverse=True):
            rw2.RemoveAtom(idx)
        out = rw2.GetMol()
        try:
            Chem.SanitizeMol(out)
        except (ValueError, RuntimeError, Chem.KekulizeException):
            return None
    return out


def _set_opt(opts, name: str, value) -> None:
    """Set a MolDrawOptions attribute if the installed RDKit supports it."""
    if hasattr(opts, name):
        setattr(opts, name, value)


def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


# ── unique structure collection ─────────────────────────────────────────────

def collect_unique_polymers(df: pd.DataFrame) -> List[Dict]:
    """Collect unique polymers from the paired CSV.

    Uniqueness is based on the canonicalized (MonomerA, MonomerB) pair, gathered
    from both sides (``_1`` and ``_2``) of the paired table.
    """
    entries: Dict[Tuple[str, str], Dict] = {}

    for s in ("1", "2"):
        needed = [f"MonomerA_{s}", f"MonomerB_{s}"]
        if not all(c in df.columns for c in needed):
            continue
        mats = df[f"Materials_{s}"] if f"Materials_{s}" in df.columns else None
        raw_a = df[f"MonomerA_{s}"].astype(str).str.strip()
        raw_b = df[f"MonomerB_{s}"].astype(str).str.strip()
        polymer = (df[f"Polymer_{s}"].astype(str).str.strip()
                   if f"Polymer_{s}" in df.columns else None)

        for i in range(len(df)):
            smi_a, smi_b = raw_a.iloc[i], raw_b.iloc[i]
            key = (canonical_smiles(smi_a), canonical_smiles(smi_b))
            if key == ("", ""):
                continue
            entry = entries.setdefault(
                key,
                {"monomer_a": smi_a, "monomer_b": smi_b, "polymer": "", "names": set()},
            )
            if mats is not None:
                name = str(mats.iloc[i]).strip()
                if name and name != "nan":
                    entry["names"].add(name)
            if polymer is not None and not entry["polymer"]:
                entry["polymer"] = polymer.iloc[i]

    out = []
    for e in entries.values():
        names = sorted(e["names"])
        e["label"] = names[0] if names else (e["polymer"][:40] or "polymer")
        e["label_full"] = " / ".join(names) if names else ""
        out.append(e)

    out.sort(key=lambda e: e["label"].lower())
    return out


def _repeat_unit_from_monomers(entry: Dict) -> Optional[Chem.Mol]:
    """Fallback: concatenate A+B into a repeat unit (still carries two '*')."""
    from polymer_ranking.chemistry import generate_all_connections

    for smi in generate_all_connections(entry["monomer_a"], entry["monomer_b"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol
    return None


def build_mol(entry: Dict, mode: str) -> Optional[Chem.Mol]:
    """Build the RDKit Mol to draw for one unique polymer entry."""
    if mode == "cyclized":
        # cyclic model compound: reuse the exact chemistry used by the encoder
        from polymer_ranking.chemistry import generate_cyclized_from_monomers

        results = generate_cyclized_from_monomers(entry["monomer_a"], entry["monomer_b"])
        return results[0][1] if results else None

    # default: a single repeat unit, dummy atoms stripped
    smi = entry.get("polymer") or ""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        mol = _repeat_unit_from_monomers(entry)
    return strip_dummy_atoms(mol)


# ── backbone / side-chain classification ────────────────────────────────────

def classify(mol: Chem.Mol) -> Tuple[List[int], List[int], List[int]]:
    """Split atoms/bonds into side chain and backbone.

    Side chain = non-ring sp3 carbon and every bond touching it (identical to the
    ``touches_sidechain_sp3`` rule used for the D-MPNN edge weights).
    """
    side_atoms = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C"
        and a.GetHybridization() == Chem.HybridizationType.SP3
        and not a.IsInRing()
    ]
    side_set = set(side_atoms)

    side_bonds: List[int] = []
    main_bonds: List[int] = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in side_set or bond.GetEndAtomIdx() in side_set:
            side_bonds.append(bond.GetIdx())
        else:
            main_bonds.append(bond.GetIdx())

    return side_atoms, side_bonds, main_bonds


# ── drawing ─────────────────────────────────────────────────────────────────

def draw_molecule(
    mol: Chem.Mol,
    legend: str,
    size: int,
    main_color: str,
    side_color: str,
) -> Image.Image:
    """Render one molecule with two-color highlighting, returns a PIL image."""
    draw_mol = rdMolDraw2D.PrepareMolForDrawing(
        mol, kekulize=True, addChiralHs=False, wedgeBonds=True
    )
    side_atoms, side_bonds, main_bonds = classify(draw_mol)

    side_rgb = hex_to_rgb01(side_color)
    main_rgb = hex_to_rgb01(main_color)

    atom_colors = {idx: side_rgb for idx in side_atoms}
    bond_colors = {idx: side_rgb for idx in side_bonds}
    bond_colors.update({idx: main_rgb for idx in main_bonds})

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    _set_opt(opts, "highlightBondWidthMultiplier", 10)
    _set_opt(opts, "continuousHighlight", True)
    _set_opt(opts, "legendFontSize", 20)
    _set_opt(opts, "padding", 0.05)

    drawer.DrawMolecule(
        draw_mol,
        highlightAtoms=side_atoms,
        highlightBonds=list(bond_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        legend=legend,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def make_grid(
    tiles: Sequence[Image.Image],
    cols: int,
    rows: int,
    main_color: str,
    side_color: str,
    footer: bool = True,
) -> Image.Image:
    """Paste molecule tiles into a cols x rows grid and add a color legend."""
    tile_w, tile_h = tiles[0].size
    footer_h = 44 if footer else 0
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h + footer_h), "white")

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(tile, (c * tile_w, r * tile_h))

    if footer:
        pen = ImageDraw.Draw(grid)
        font = _load_font(16)
        y = rows * tile_h + 10
        main_rgb = hex_to_rgb255(main_color)
        side_rgb = hex_to_rgb255(side_color)

        pen.rectangle([12, y + 4, 32, y + 20], fill=main_rgb, outline=(0, 0, 0))
        pen.text((40, y + 2), "backbone (ring / conjugated)", fill=(0, 0, 0), font=font)

        x0 = 300
        pen.rectangle([x0, y + 4, x0 + 20, y + 20], fill=side_rgb, outline=(0, 0, 0))
        pen.text(
            (x0 + 28, y + 2),
            "side chain (non-ring sp3 C, down-weighted)",
            fill=(0, 0, 0),
            font=font,
        )
    return grid


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Draw unique polymer structures in a cols x rows grid with "
                    "backbone / side-chain coloring."
    )
    p.add_argument("--csv", type=str, default="contrastive_full_paired.csv",
                   help="Paired CSV containing MonomerA_*/MonomerB_* (and Polymer_*) columns")
    p.add_argument("--mode", choices=["repeat", "cyclized"], default="repeat",
                   help="repeat: one repeat unit with dummy (*) atoms removed (default); "
                        "cyclized: cyclic model compound used by the encoder")
    p.add_argument("--cols", type=int, default=4, help="Molecules per row")
    p.add_argument("--rows", type=int, default=4, help="Molecules per column")
    p.add_argument("--subimg", type=int, default=500, help="Size of a single molecule tile (px)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only draw the first N unique structures (useful for a quick check)")
    p.add_argument("--outdir", type=str, default="mol_grids", help="Output directory")
    p.add_argument("--prefix", type=str, default="mol_grid", help="Output filename prefix")
    p.add_argument("--main_color", type=str, default=DEFAULT_MAIN_COLOR,
                   help="Backbone highlight color (#rrggbb)")
    p.add_argument("--side_color", type=str, default=DEFAULT_SIDE_COLOR,
                   help="Side-chain highlight color (#rrggbb)")
    p.add_argument("--no_footer", action="store_true", help="Do not draw the color legend")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    entries = collect_unique_polymers(df)
    if args.limit:
        entries = entries[: args.limit]
    print(f"Unique structures to draw: {len(entries)} (mode={args.mode})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tiles: List[Image.Image] = []
    labels: List[str] = []
    skipped = 0
    for i, entry in enumerate(entries, 1):
        mol = build_mol(entry, args.mode)
        if mol is None:
            skipped += 1
            continue
        label = entry["label"]
        if len(label) > 42:
            label = label[:39] + "..."
        try:
            tiles.append(
                draw_molecule(mol, label, args.subimg, args.main_color, args.side_color)
            )
        except Exception as exc:  # keep drawing the remaining molecules
            print(f"[WARN] failed to draw {label}: {exc}")
            skipped += 1
            continue
        labels.append(label)
        if i % 20 == 0 or i == len(entries):
            print(f"  rendered {len(tiles)}/{len(entries)}")

    if not tiles:
        print("[ERROR] no molecule could be rendered")
        return 1

    per_page = args.cols * args.rows
    n_pages = (len(tiles) + per_page - 1) // per_page
    for page in range(n_pages):
        chunk = tiles[page * per_page: (page + 1) * per_page]
        grid = make_grid(
            chunk, args.cols, args.rows,
            args.main_color, args.side_color, footer=not args.no_footer,
        )
        out_path = outdir / f"{args.prefix}_{page + 1:03d}.png"
        grid.save(out_path)
        print(f"saved {out_path}  ({len(chunk)} molecules)")

    if skipped:
        print(f"skipped {skipped} structures (invalid SMILES / cyclization failure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
