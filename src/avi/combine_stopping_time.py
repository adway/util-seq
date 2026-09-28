"""Combine anytime-valid stopping-time chunks and compare Wald bounds."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from avi.martingale import wald_upper_bound

OUTDIR = Path(__file__).resolve().parent / "stopping_results"
files = sorted(glob.glob(str(OUTDIR / "results_chunk_*.csv")))
if not files:
    raise FileNotFoundError(f"No chunk files found in {OUTDIR}")

results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
clean = results.dropna(subset=["stopping_time", "martingale_at_stop"]).copy()
summary = (clean.groupby(["theta", "N", "alpha", "rho"], as_index=False)
           .agg(mean_stopping_time=("stopping_time", "mean"),
                sd_stopping_time=("stopping_time", "std"),
                mean_stopping_fraction=("stopping_fraction", "mean"),
                early_stop_prob=("stopped_before_N", "mean"),
                anytime_power=("power", "mean"),
                n_success=("stopping_time", "size")))
summary["wald_upper_bound_time"] = [wald_upper_bound(n, t, a, r)
                                     for n, t, a, r in zip(summary.N,
                                                            summary.theta,
                                                            summary.alpha,
                                                            summary.rho)]
summary["wald_upper_bound_fraction"] = (summary["wald_upper_bound_time"]
                                         / summary["N"])
summary["stopping_time_minus_wald_bound"] = (
    summary["mean_stopping_time"] - summary["wald_upper_bound_time"]
)
summary.to_csv(OUTDIR / "stopping_time_summary.csv", index=False)
