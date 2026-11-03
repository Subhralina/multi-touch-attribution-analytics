"""Bootstrap confidence intervals for the data-driven models.

Resampling journeys with replacement is equivalent to drawing new counts for
each (path, outcome) cell from a multinomial, which keeps this fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CHANNELS
from .markov import MarkovCells, removal_effects
from .shapley import shapley_from_sets


def bootstrap_data_driven(paths: pd.DataFrame, n_boot: int = 500, seed: int = 7) -> pd.DataFrame:
    """Long table: one row per (replicate, channel) with Markov share, removal effect, Shapley share."""
    rng = np.random.default_rng(seed)
    cells = MarkovCells(paths)
    total = cells.counts.sum()
    probs = cells.counts / total
    cell_sets = [frozenset(p) for p, _ in cells.cells]
    converted = np.array([c for _, c in cells.cells], dtype=bool)

    rows = []
    for b in range(n_boot):
        w = rng.multinomial(int(total), probs).astype(float)
        re = removal_effects(cells.transition_counts(w))
        share = re / re.sum()
        sets = pd.Series(w[converted], index=[s for s, c in zip(cell_sets, converted) if c]).groupby(level=0).sum()
        sh = shapley_from_sets(sets)
        for ch in CHANNELS:
            rows.append({"replicate": b, "channel": ch, "markov_share": share[ch],
                         "removal_effect": re[ch], "shapley_share": sh[ch]})
    return pd.DataFrame(rows)


def summarize(boot: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    stats = boot.groupby("channel")[["markov_share", "removal_effect", "shapley_share"]]
    return pd.concat({"low": stats.quantile(lo), "high": stats.quantile(hi)}, axis=1)
