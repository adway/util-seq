import glob
from pathlib import Path

import pandas as pd


OUTDIR = Path("sample_split_results")

files = glob.glob(str(OUTDIR / "results_chunk_*.csv"))

results_df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True,
)

clean_df = results_df.dropna(
    subset=[
        "theta_hat",
        "b_hat",
        "inner_power",
        "inner_seq_stopping_time",
        "inner_total_stopping_time",
        "inner_total_stopping_fraction",
        "inner_early_stop_prob",
    ]
).copy()

summary_df = (
    clean_df
    .groupby(["theta", "N", "alpha", "lambda", "psi"], as_index=False)
    .agg(
        mean_theta_hat=("theta_hat", "mean"),
        sd_theta_hat=("theta_hat", "std"),
        mean_b_hat=("b_hat", "mean"),
        sd_b_hat=("b_hat", "std"),
        power=("inner_power", "mean"),
        sd_outer_power=("inner_power", "std"),
        mean_seq_stopping_time=("inner_seq_stopping_time", "mean"),
        mean_total_stopping_time=("inner_total_stopping_time", "mean"),
        mean_total_stopping_fraction=("inner_total_stopping_fraction", "mean"),
        early_stop_prob=("inner_early_stop_prob", "mean"),
        n_outer_success=("inner_power", "size"),
        n_inner_success=("inner_n_success", "sum"),
    )
)

summary_df.to_csv(OUTDIR / "sample_split_summary.csv", index=False)
