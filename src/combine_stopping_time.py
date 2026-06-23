import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


OUTDIR = Path("stopping_results")

files = glob.glob(str(OUTDIR / "results_chunk_*.csv"))

results_df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True,
)

summary_df = (
    results_df
    .dropna(subset=["stopping_time", "value"])
    .groupby(["theta", "N", "alpha", "beta"], as_index=False)
    .agg(
        mean_value=("value", "mean"),
        sd_value=("value", "std"),

        mean_stopping_time=("stopping_time", "mean"),
        sd_stopping_time=("stopping_time", "std"),

        mean_stopping_fraction=("stopping_fraction", "mean"),
        sd_stopping_fraction=("stopping_fraction", "std"),

        early_stop_prob=("stopped_early", "mean"),
        n_success=("stopping_time", "size"),
    )
)

z_alpha = norm.ppf(1 - summary_df["alpha"])
z_beta = norm.ppf(summary_df["beta"])

summary_df["upper_bound_fraction"] = (
    (z_alpha + z_beta)
    / (summary_df["theta"] * np.sqrt(summary_df["N"]))
)

summary_df["upper_bound_time"] = (
    summary_df["upper_bound_fraction"]
    * summary_df["N"]
)

summary_df["relative_gap"] = (
    summary_df["upper_bound_fraction"]
    - summary_df["mean_stopping_fraction"]
) / summary_df["upper_bound_fraction"]

summary_df = summary_df[
    [
        "theta",
        "N",
        "alpha",
        "beta",
        "mean_value",
        "sd_value",
        "mean_stopping_time",
        "sd_stopping_time",
        "mean_stopping_fraction",
        "sd_stopping_fraction",
        "upper_bound_time",
        "upper_bound_fraction",
        "relative_gap",
        "early_stop_prob",
        "n_success",
    ]
]

summary_df.to_csv(
    OUTDIR / "relative_gap_summary.csv",
    index=False,
)