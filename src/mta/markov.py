"""First-order Markov chain attribution via the removal effect.

Journeys become walks through states: start -> channel -> ... -> (conversion | null).
Transition probabilities are estimated from the data, and the probability of
reaching `conversion` from `start` is solved exactly with the absorbing-chain
formula B = (I - Q)^-1 R.

Removal effect of a channel = relative drop in that probability when every
transition into the channel is redirected to `null`. Normalising removal
effects across channels gives each channel's share of credit.

Transition counts are stored per (path, converted) cell, so the transition
matrix for any reweighting of journeys (e.g. a bootstrap resample) is a single
matrix product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CHANNELS

STATES = ["start", *CHANNELS, "conversion", "null"]
IDX = {s: i for i, s in enumerate(STATES)}
S = len(STATES)
TRANSIENT = [IDX["start"]] + [IDX[c] for c in CHANNELS]


class MarkovCells:
    """Per-cell transition counts. A cell is one unique path with one outcome."""

    def __init__(self, paths: pd.DataFrame):
        cells, counts = [], []
        for path, n, c in zip(paths["path"], paths["journeys"], paths["conversions"]):
            if c:
                cells.append((path, 1)); counts.append(c)
            if n - c:
                cells.append((path, 0)); counts.append(n - c)
        self.cells = cells
        self.counts = np.asarray(counts, dtype=float)
        M = np.zeros((len(cells), S * S))
        for row, (path, converted) in enumerate(cells):
            seq = ["start", *path, "conversion" if converted else "null"]
            for a, b in zip(seq[:-1], seq[1:]):
                M[row, IDX[a] * S + IDX[b]] += 1
        self.M = M

    def transition_counts(self, weights: np.ndarray | None = None) -> np.ndarray:
        w = self.counts if weights is None else weights
        return (w @ self.M).reshape(S, S)


def conversion_probability(T: np.ndarray, removed: str | None = None) -> float:
    T = T.copy()
    if removed is not None:
        r = IDX[removed]
        T[:, IDX["null"]] += T[:, r]  # anyone who would have reached it drops out
        T[:, r] = 0.0
        T[r, :] = 0.0
    rows = T.sum(axis=1, keepdims=True)
    P = np.divide(T, rows, out=np.zeros_like(T), where=rows > 0)
    Q = P[np.ix_(TRANSIENT, TRANSIENT)]
    R = P[np.ix_(TRANSIENT, [IDX["conversion"]])]
    B = np.linalg.solve(np.eye(len(TRANSIENT)) - Q, R)
    return float(B[0, 0])


def removal_effects(T: np.ndarray) -> pd.Series:
    base = conversion_probability(T)
    return pd.Series({c: 1 - conversion_probability(T, c) / base for c in CHANNELS})


def markov_attribution(paths: pd.DataFrame, cells: MarkovCells | None = None) -> tuple[pd.Series, pd.Series]:
    """Returns (credit share per channel, raw removal effect per channel)."""
    cells = cells or MarkovCells(paths)
    re = removal_effects(cells.transition_counts())
    return re / re.sum(), re
