"""Rule-based attribution: a fixed rule decides how credit is split along a path."""
from __future__ import annotations

import pandas as pd

from .data import CHANNELS

HEURISTIC_MODELS = ["First touch", "Last touch", "Linear", "Position-based"]


def _weights(model: str, k: int) -> list[float]:
    if model == "First touch":
        return [1.0] + [0.0] * (k - 1)
    if model == "Last touch":
        return [0.0] * (k - 1) + [1.0]
    if model == "Linear":
        return [1.0 / k] * k
    if model == "Position-based":  # 40% first, 40% last, 20% spread over the middle
        if k == 1:
            return [1.0]
        if k == 2:
            return [0.5, 0.5]
        return [0.4] + [0.2 / (k - 2)] * (k - 2) + [0.4]
    raise ValueError(f"Unknown model: {model}")


def heuristic_attribution(paths: pd.DataFrame, model: str, weight: str = "conversions") -> pd.Series:
    """Share of credit per channel (sums to 1). `weight` is 'conversions' or 'value'."""
    credit = dict.fromkeys(CHANNELS, 0.0)
    for path, w in zip(paths["path"], paths[weight]):
        if w == 0:
            continue
        for ch, share in zip(path, _weights(model, len(path))):
            credit[ch] += share * w
    s = pd.Series(credit)
    return s / s.sum()
