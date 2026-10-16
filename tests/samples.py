"""Shared sample molecules and DataFrame builders for the test suite.

All monomer SMILES use ``*`` dummy atoms as polymer connection points,
matching the format of the training CSV (MonomerA_1, MonomerB_1, ...).

Fixture SMILES were validated against the real pipeline:
  - THIOPHENE  cyclizes to thiophene          (homocoupling -> 1 structure)
  - BENZENE    alternating with thiophene     (2 connection orientations)
  - VINYLENE   cyclizes to ethene (C=C)       (E/Z stereo preserved in connections)
  - ALKYL_THIOPHENE has a non-ring sp3 chain (edge down-weighting tests)
"""

import numpy as np
import pandas as pd

from polymer_ranking.config import EXTRA_COLS

THIOPHENE = "*c1ccsc1*"
BENZENE = "*c1ccccc1*"
VINYLENE = "*/C=C/*"
ALKYL_THIOPHENE = "*c1cc(CCCC)sc1*"   # hexylthiophene-like: sp3 side chain
INVALID_MONOMER = "CCC"               # parses, but has no '*' atoms
BAD_SMILES = "this_is_not_smiles"

EXTRA_VALUES_1 = [1.0, 0.0, 0.5, -3.8, -5.2]
EXTRA_VALUES_2 = [1.0, 1.0, 0.7, -3.2, -5.6]


def _extra_cols(suffix: str) -> list:
    return [c.format(s=suffix) for c in EXTRA_COLS]


def make_raw_pair_df() -> pd.DataFrame:
    """Raw (pre-cyclization) pair DataFrame with 5 rows:

    row 0: thiophene homocoupling, both mobilities measured
    row 1: thiophene/benzene alternating, both measured
    row 2: benzene homocoupling, mu_e_2 = 0 (censored, below detection limit)
    row 3: vinylene/thiophene, mu_e_1 = NaN   -> dropped by load_and_preprocess
    row 4: invalid monomer A                  -> dropped (no cyclized structure)

    After load_and_preprocess exactly rows 0-2 survive (3 valid pairs).
    """
    rows = []
    for i, (a1, b1, a2, b2) in enumerate([
        (THIOPHENE, THIOPHENE, THIOPHENE, THIOPHENE),
        (THIOPHENE, BENZENE, BENZENE, THIOPHENE),
        (BENZENE, BENZENE, BENZENE, BENZENE),
        (VINYLENE, THIOPHENE, THIOPHENE, VINYLENE),
        (INVALID_MONOMER, THIOPHENE, THIOPHENE, THIOPHENE),
    ]):
        row = {
            "Materials_1": f"polymer_{i}a",
            "Materials_2": f"polymer_{i}b",
            "doi": f"10.1000/{chr(ord('a') + i)}",
            "MonomerA_1": a1, "MonomerB_1": b1,
            "MonomerA_2": a2, "MonomerB_2": b2,
        }
        for s, vals in (("1", EXTRA_VALUES_1), ("2", EXTRA_VALUES_2)):
            for col, v in zip(_extra_cols(s), vals):
                row[col] = v
        # mobilities in cm^2/Vs scale
        row.update({
            "mu_e_1": 1e-3, "mu_e_2": 1e-5 if i != 1 else 1e-3,
            "mu_h_1": 1e-2, "mu_h_2": 1e-4,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.loc[2, "mu_e_2"] = 0.0        # censored: below detection limit
    df.loc[3, "mu_e_1"] = np.nan     # unmeasured: row must be dropped
    return df
