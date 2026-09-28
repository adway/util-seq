"""Run anytime-valid stopping-time, power, and utility simulations."""

from pathlib import Path
import argparse
import csv
from collections import defaultdict
import numpy as np
from statistics import NormalDist

from .martingale import anytime_stopping_time, standard_power, wald_upper_bound

_NORMAL = NormalDist()


DEFAULT_THETAS = (0.0, 0.01, 0.05, 0.10, 0.50, 1.0)
DEFAULT_NS = (1_000, 5_000, 10_000, 50_000)
DEFAULT_ALPHAS = (0.01, 0.05)
DEFAULT_RHOS = (1.0,)


def run_simulations(output, reps=500, thetas=DEFAULT_THETAS,
                    horizons=DEFAULT_NS, alphas=DEFAULT_ALPHAS,
                    rhos=DEFAULT_RHOS, seed=12345):
    """Run all scenarios and write replicate-level and summary CSV files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for theta in thetas:
        for N in horizons:
            for alpha in alphas:
                for rho in rhos:
                    ub = wald_upper_bound(N, theta, alpha, rho)
                    for rep in range(reps):
                        sample = rng.normal(theta, 1.0, N)
                        value, stop, reject = anytime_stopping_time(sample, alpha, rho)
                        std_reject = sample.sum() / np.sqrt(N) > _NORMAL.inv_cdf(1.0 - alpha)
                        rows.append({
                            "rep": rep, "theta": theta, "N": N,
                            "alpha": alpha, "rho": rho,
                            "martingale_at_stop": value,
                            "stopping_time": stop,
                            "stopping_fraction": stop / N,
                            "rejected": int(reject),
                            # Per the requested convention, power counts only
                            # crossings strictly before the administrative horizon.
                            "power": int(stop < N),
                            "stopped_before_N": int(stop < N),
                            "std_rejected": int(std_reject),
                            "wald_upper_bound_time": ub,
                            "wald_upper_bound_fraction": ub / N,
                        })
    for row in rows:
        row["utility"] = 0.5 * (row["stopping_fraction"] - row["power"])
        row["std_utility"] = 0.5 * (1.0 - row["std_rejected"])
        row["utility_difference"] = row["utility"] - row["std_utility"]
    _write_csv(output / "replicates.csv", rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["theta"], row["N"], row["alpha"], row["rho"])].append(row)
    summary = []
    for (theta, N, alpha, rho), group in groups.items():
        times = [r["stopping_time"] for r in group]
        mean_time = sum(times) / len(times)
        sd_time = _sample_sd(times)
        ub = wald_upper_bound(N, theta, alpha, rho)
        summary.append({
            "theta": theta, "N": N, "alpha": alpha, "rho": rho,
            "mean_stopping_time": mean_time, "sd_stopping_time": sd_time,
            "mean_stopping_fraction": _mean(group, "stopping_fraction"),
            "anytime_power": _mean(group, "power"),
            "standard_power_empirical": _mean(group, "std_rejected"),
            "standard_power_theoretical": standard_power(theta, N, alpha),
            "utility": _mean(group, "utility"),
            "standard_utility": _mean(group, "std_utility"),
            "utility_difference": _mean(group, "utility_difference"),
            "early_stop_probability": _mean(group, "stopped_before_N"),
            "wald_upper_bound_time": ub,
            "wald_upper_bound_fraction": ub / N,
            "stopping_time_minus_wald_bound": mean_time - ub,
        })
    _write_csv(output / "summary.csv", summary)
    return rows, summary


def _mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def _sample_sd(values):
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def _write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--rho", type=float, default=1.0)
    args = parser.parse_args()
    run_simulations(args.output, reps=args.reps, rhos=(args.rho,), seed=args.seed)
