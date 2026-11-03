"""Run the full attribution analysis and write every table and figure.

    python run_analysis.py                       # uses data/attribution_data.csv
    python run_analysis.py --data path/to.csv --boot 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mta import (  # noqa: E402
    CHANNELS, HEURISTIC_MODELS, MarkovCells, aggregate_paths, assist_share, bootstrap_data_driven,
    build_journeys, heuristic_attribution, length_matched_lift, load_touchpoints, markov_attribution,
    position_profile, shapley_attribution,
)
from mta.bootstrap import summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "attribution_data.csv"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    touch = load_touchpoints(args.data)
    journeys = build_journeys(touch)
    paths = aggregate_paths(journeys)
    cells = MarkovCells(paths)

    # 1. Credit share by model, weighted by conversions and by conversion value
    for weight, fname in [("conversions", "attribution_shares.csv"), ("value", "attribution_shares_value.csv")]:
        shares = {m: heuristic_attribution(paths, m, weight) for m in HEURISTIC_MODELS}
        if weight == "conversions":
            mk, removal = markov_attribution(paths, cells)
            shares["Markov"] = mk
        else:
            shares["Markov"] = markov_attribution(paths, cells)[0]  # Markov counts journeys, not value
        shares["Shapley"] = shapley_attribution(paths, weight)
        pd.DataFrame(shares).loc[CHANNELS].to_csv(out / fname, float_format="%.6f")

    # 2. Diagnostics: why the models disagree
    assist_share(journeys).to_csv(out / "assist_share.csv", float_format="%.6f")
    length_matched_lift(journeys).to_csv(out / "length_matched_lift.csv", float_format="%.6f")
    position_profile(journeys).to_csv(out / "position_profile.csv", float_format="%.6f")

    # 3. Uncertainty: bootstrap the data-driven models
    boot = bootstrap_data_driven(paths, n_boot=args.boot)
    ci = summarize(boot)
    ci.columns = [f"{metric}_{bound}" for bound, metric in ci.columns]
    ci.loc[CHANNELS].to_csv(out / "bootstrap_ci.csv", float_format="%.6f")
    wide = boot.pivot(index="replicate", columns="channel", values="markov_share")
    gap = wide["Instagram"] - wide["Online Display"]
    wide_sh = boot.pivot(index="replicate", columns="channel", values="shapley_share")
    gap_sh = wide_sh["Instagram"] - wide_sh["Online Display"]

    # 4. What removal would cost
    n_conv = int(journeys["converted"].sum())
    total_value = float(journeys["value"].sum())
    removal_tbl = pd.DataFrame({
        "removal_effect": removal,
        "removal_effect_low": ci.loc[CHANNELS, "removal_effect_low"],
        "removal_effect_high": ci.loc[CHANNELS, "removal_effect_high"],
        "conversions_at_risk": (removal * n_conv).round(0),
        "value_at_risk": (removal * total_value).round(0),
    }).loc[CHANNELS]
    removal_tbl.to_csv(out / "removal_effects.csv", float_format="%.6f")

    shares = pd.read_csv(out / "attribution_shares.csv", index_col=0)
    headline = {
        "touchpoints": int(len(touch)),
        "journeys": int(len(journeys)),
        "unique_paths": int(len(paths)),
        "conversions": n_conv,
        "conversion_rate": n_conv / len(journeys),
        "conversion_value": total_value,
        "date_min": str(touch["time"].min().date()),
        "date_max": str(touch["time"].max().date()),
        "single_touch_share": float((journeys["length"] == 1).mean()),
        "median_path_length": float(journeys["length"].median()),
        "last_touch_gap_ig_minus_display": float(shares.loc["Instagram", "Last touch"] - shares.loc["Online Display", "Last touch"]),
        "markov_gap_ig_minus_display": float(shares.loc["Instagram", "Markov"] - shares.loc["Online Display", "Markov"]),
        "markov_gap_ci": [float(gap.quantile(0.025)), float(gap.quantile(0.975))],
        "markov_gap_prob_positive": float((gap > 0).mean()),
        "shapley_gap_ig_minus_display": float(shares.loc["Instagram", "Shapley"] - shares.loc["Online Display", "Shapley"]),
        "shapley_gap_ci": [float(gap_sh.quantile(0.025)), float(gap_sh.quantile(0.975))],
        "bootstrap_replicates": args.boot,
    }
    (out / "headline.json").write_text(json.dumps(headline, indent=2))

    try:
        from mta.plots import make_all
        make_all(out, Path(args.figures))
    except ImportError:
        pass
    print(f"Done in {time.time() - t0:.0f}s. Results in {out}")


if __name__ == "__main__":
    main()
