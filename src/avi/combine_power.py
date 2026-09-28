"""Combine anytime-valid power, stopping-time, and utility chunks."""

import glob
from pathlib import Path

import pandas as pd

from avi.martingale import terminal_power_lower_bound

OUTDIR = Path(__file__).resolve().parent / "power_results"
files = sorted(glob.glob(str(OUTDIR / "results_chunk_*.csv")))
if not files:
    raise FileNotFoundError(f"No chunk files found in {OUTDIR}")

results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
clean = results.dropna(subset=["seq_value", "seq_power", "seq_stopping_time",
                               "std_reject", "utility_difference"]).copy()
summary = (clean.groupby(["theta", "N", "alpha", "rho"], as_index=False)
           .agg(anytime_power=("seq_power", "mean"),
                deterministic_rejection_rate=("seq_reject", "mean"),
                mean_stopping_time=("seq_stopping_time", "mean"),
                mean_stopping_fraction=("seq_stopping_fraction", "mean"),
                std_power_empirical=("std_reject", "mean"),
                std_power_theoretical=("std_power_theoretical", "mean"),
                utility=("utility", "mean"),
                standard_utility=("std_utility", "mean"),
                utility_difference=("utility_difference", "mean"),
                early_stop_prob=("seq_power", "mean"),
                wald_upper_bound_time=("wald_upper_bound_time", "mean"),
                n_success=("seq_power", "size")))
summary["power_difference"] = (summary["std_power_empirical"]
                                - summary["anytime_power"])
summary["wald_upper_bound_fraction"] = (summary["wald_upper_bound_time"]
                                         / summary["N"])
summary["terminal_power_lower_bound"] = [terminal_power_lower_bound(
    n, t, a, r
) for n, t, a, r in zip(summary.N, summary.theta, summary.alpha, summary.rho)]
summary["oracle_utility_difference_upper_bound"] = 0.5 * (
    summary["wald_upper_bound_fraction"]
    - summary["terminal_power_lower_bound"]
    - 1.0
    + summary["std_power_theoretical"]
)
summary.to_csv(OUTDIR / "power_summary.csv", index=False)
