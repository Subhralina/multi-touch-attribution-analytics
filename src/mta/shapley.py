"""Shapley-value attribution over channel sets.

Coalition value v(S) = conversions from journeys whose set of channels is a
subset of S. A channel's Shapley value is its average marginal contribution
over every order in which channels could join. Order inside a journey is
ignored, which is the main thing that separates it from the Markov model.
"""
from __future__ import annotations

import itertools
import math

import pandas as pd

from .data import CHANNELS


def _coalition_values(set_conversions: pd.Series) -> dict[frozenset, float]:
    values = {}
    for r in range(len(CHANNELS) + 1):
        for combo in itertools.combinations(CHANNELS, r):
            coal = frozenset(combo)
            values[coal] = float(sum(v for s, v in set_conversions.items() if s <= coal))
    return values


def shapley_from_sets(set_conversions: pd.Series) -> pd.Series:
    v = _coalition_values(set_conversions)
    n = len(CHANNELS)
    phi = dict.fromkeys(CHANNELS, 0.0)
    for ch in CHANNELS:
        others = [c for c in CHANNELS if c != ch]
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for combo in itertools.combinations(others, r):
                s = frozenset(combo)
                phi[ch] += w * (v[s | {ch}] - v[s])
    out = pd.Series(phi)
    return out / out.sum()


def channel_sets(paths: pd.DataFrame, weight: str = "conversions") -> pd.Series:
    return paths.assign(set=paths["path"].map(frozenset)).groupby("set")[weight].sum()


def shapley_attribution(paths: pd.DataFrame, weight: str = "conversions") -> pd.Series:
    return shapley_from_sets(channel_sets(paths, weight))
