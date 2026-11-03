"""Descriptive checks that explain *why* the models disagree."""
from __future__ import annotations

import pandas as pd

from .data import CHANNELS


def assist_share(journeys: pd.DataFrame) -> pd.DataFrame:
    """For converting journeys that include a channel: how often was it NOT the final touch?

    A high number means last-touch reporting hides most of the channel's involvement.
    """
    conv = journeys[journeys["converted"] == 1]
    rows = []
    for ch in CHANNELS:
        has = conv["path"].map(lambda p: ch in p)
        closed = conv["path"].map(lambda p: p[-1] == ch)
        rows.append({
            "channel": ch,
            "in_converting_journeys": has.mean(),
            "closed": (has & closed).sum() / has.sum(),
            "assisted_only": (has & ~closed).sum() / has.sum(),
        })
    return pd.DataFrame(rows).set_index("channel")


def length_matched_lift(journeys: pd.DataFrame, max_len: int = 6) -> pd.DataFrame:
    """Conversion rate of journeys WITH a channel / journeys WITHOUT it, at the same length.

    Longer journeys convert more often, and some channels show up mostly in long
    journeys. Comparing within a length removes that confound (not every confound).
    """
    j = journeys.assign(len_bucket=journeys["length"].clip(upper=max_len))
    out = {}
    for ch in CHANNELS:
        has = j["path"].map(lambda p: ch in p)
        rates = j.assign(has=has).groupby(["len_bucket", "has"])["converted"].mean().unstack()
        out[ch] = rates[True] / rates[False]
    lift = pd.DataFrame(out)
    lift.index.name = "journey_length"
    return lift


def position_profile(journeys: pd.DataFrame) -> pd.DataFrame:
    """Share of each channel's touches that are the only / first / middle / last touch."""
    counts = {ch: dict.fromkeys(["only", "first", "middle", "last"], 0) for ch in CHANNELS}
    for path in journeys["path"]:
        k = len(path)
        for i, ch in enumerate(path):
            pos = "only" if k == 1 else "first" if i == 0 else "last" if i == k - 1 else "middle"
            counts[ch][pos] += 1
    df = pd.DataFrame(counts).T
    return df.div(df.sum(axis=1), axis=0)
