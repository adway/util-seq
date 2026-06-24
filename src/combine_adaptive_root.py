import glob
from pathlib import Path

import pandas as pd


OUTDIR = Path("adaptive_root_results")

files = glob.glob(str(OUTDIR / "results_chunk_*.csv"))

results_df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True,
)

clean_df = results_df.dropna(
    subset=[
        "adaptive_value",
        "adaptive_reject",
        "adaptive_stopping_time",
        "adaptive_stopping_fraction",
        "adaptive_stopped_early",
        "valid_root_looks",
        "invalid_root_looks",
    ]
).copy()

summary_df = (
    clean_df
    .groupby(["theta", "N", "alpha", "lambda"], as_index=False)
    .agg(
        adaptive_power=("adaptive_reject", "mean"),
        sd_adaptive_power=("adaptive_reject", "std"),
        mean_adaptive_value=("adaptive_value", "mean"),
        mean_stopping_time=("adaptive_stopping_time", "mean"),
        mean_stopping_fraction=("adaptive_stopping_fraction", "mean"),
        early_stop_prob=("adaptive_stopped_early", "mean"),
        mean_theta_hat_at_stop=("theta_hat_at_stop", "mean"),
        mean_b_hat_at_stop=("b_hat_at_stop", "mean"),
        mean_last_b_hat=("last_b_hat", "mean"),
        mean_valid_root_looks=("valid_root_looks", "mean"),
        mean_invalid_root_looks=("invalid_root_looks", "mean"),
        n_success=("adaptive_reject", "size"),
    )
)

summary_df.to_csv(OUTDIR / "adaptive_root_summary.csv", index=False)
