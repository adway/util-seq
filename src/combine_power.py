import glob
from pathlib import Path

import pandas as pd

OUTDIR = Path("power_results")

files = glob.glob(str(OUTDIR / "results_chunk_*.csv"))

results_df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True,
)

clean_df = results_df.dropna(
    subset=[
        "seq_value",
        "seq_reject",
        "seq_stopping_time",
        "std_value",
        "std_reject",
        "std_stopping_time",
    ]
).copy()

summary_df = (
    clean_df
    .groupby(["theta", "N", "alpha", "beta"], as_index=False)
    .agg(
        seq_power=("seq_reject", "mean"),
        seq_stopping_time=("seq_stopping_time", "mean"),
        std_power=("std_reject", "mean"),
        std_stopping_time=("std_stopping_time", "mean"),

        seq_value_early_stop=(
            "seq_value",
            lambda x: x[
                clean_df.loc[x.index, "seq_stopping_time"]
                < clean_df.loc[x.index, "N"]
            ].mean()
        ),

        seq_early_stop_prob=(
            "seq_stopping_time",
            lambda x: (
                x < clean_df.loc[x.index, "N"]
            ).mean()
        ),

        seq_stop_at_N_and_reject=(
            "seq_stopping_time",
            lambda x: (
                (x == clean_df.loc[x.index, "N"])
                & (clean_df.loc[x.index, "seq_reject"] == 1)
            ).mean()
        ),
    )
)

summary_df["power_diff"] = summary_df["std_power"] - summary_df["seq_power"]
summary_df["power_diff_up"] = 1 - summary_df["beta"]

summary_df = summary_df[
    [
        "theta",
        "N",
        "alpha",
        "beta",
        "seq_power",
        "seq_stopping_time",
        "std_power",
        "std_stopping_time",
        "seq_value_early_stop",
        "seq_early_stop_prob",
        "seq_stop_at_N_and_reject",
        "power_diff",
        "power_diff_up",
    ]
]

summary_df.to_csv(OUTDIR / "power_summary.csv", index=False)