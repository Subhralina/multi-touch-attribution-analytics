"""Load raw touchpoints and turn them into user journeys.

Each row of the raw file is one interaction (an impression or a conversion)
tagged with a channel. A journey is every interaction a cookie had, in time
order. The conversion row carries its own channel, so it counts as the
journey's final touch.
"""
from __future__ import annotations

import pandas as pd

CHANNELS = ["Facebook", "Instagram", "Online Display", "Online Video", "Paid Search"]


def load_touchpoints(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    expected = {"cookie", "time", "interaction", "conversion", "conversion_value", "channel"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values(["cookie", "time"], kind="stable").reset_index(drop=True)


def build_journeys(touchpoints: pd.DataFrame) -> pd.DataFrame:
    """One row per cookie: ordered channel path, converted flag, conversion value."""
    g = touchpoints.groupby("cookie", sort=False)
    journeys = pd.DataFrame({
        "path": g["channel"].agg(tuple),
        "converted": g["conversion"].max().astype(int),
        "value": g["conversion_value"].sum(),
    })
    journeys["length"] = journeys["path"].str.len()
    return journeys


def aggregate_paths(journeys: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical paths. Every model below runs on this table.

    Columns: path, journeys (count), conversions, value.
    """
    return (
        journeys.groupby("path")
        .agg(journeys=("converted", "size"), conversions=("converted", "sum"), value=("value", "sum"))
        .reset_index()
    )
