import argparse
import os
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats


ALPHAS = [0.05]
LAMBDAS = [0.25, 0.50, 0.75]
PSIS = [0.10, 0.25, 0.50, 0.75, 0.90]
THETAS = [0.1, 0.5, 1.0]
N_GRID = [1000, 2000, 5000, 10000, 20000, 50000]
PLOTDIR = Path("plots")

RED = "\033[31m"
RESET = "\033[0m"


def g_z(z, alpha, lam, psi, theta, N):
    z_alpha = scipy.stats.norm.ppf(1 - alpha)
    seq_term = -(1 - psi) + (
        np.sqrt(1 - psi) * (z_alpha + z) / (theta * np.sqrt(N))
    )
    stop_term = scipy.stats.norm.sf(z)

    return lam * seq_term + (1 - lam) * stop_term


def g_b(b, alpha, lam, psi, theta, N):
    z_b = scipy.stats.norm.ppf(b)
    return g_z(z_b, alpha, lam, psi, theta, N)


def b_from_z(z):
    return np.clip(
        scipy.stats.norm.cdf(z),
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )


def critical_points(lam, psi, theta, N):
    """
    Critical points of g(z).

    g'(z) = lam * sqrt(1 - psi) / (theta sqrt(N)) - (1 - lam) phi(z).

    So critical points solve

        phi(z) = lam * sqrt(1 - psi) / ((1 - lam) theta sqrt(N)).
    """
    if not (0 < lam < 1):
        return []

    critical_density = (
        lam * np.sqrt(1 - psi) / ((1 - lam) * theta * np.sqrt(N))
    )
    max_density = scipy.stats.norm.pdf(0)

    if not (0 < critical_density <= max_density):
        return []

    radius = np.sqrt(
        -2 * np.log(critical_density * np.sqrt(2 * np.pi))
    )

    return [-radius, radius]


def minimum_allowed_b(alpha):
    return scipy.stats.norm.cdf(alpha)


def solve_minimum(alpha, lam, psi, theta, N):
    min_allowed_z = scipy.stats.norm.ppf(minimum_allowed_b(alpha))
    candidates = [min_allowed_z]
    candidates.extend(
        z for z in critical_points(lam, psi, theta, N) if z >= min_allowed_z
    )

    objective_values = [g_z(z, alpha, lam, psi, theta, N) for z in candidates]
    min_idx = int(np.argmin(objective_values))
    min_z = candidates[min_idx]
    min_b = b_from_z(min_z)

    return {
        "theta": theta,
        "psi": psi,
        "N": N,
        "alpha": alpha,
        "lambda": lam,
        "min_z": min_z,
        "min_b": min_b,
        "g_at_min": objective_values[min_idx],
    }


def solve_roots(alpha, lam, psi, theta, N):
    candidates = [-np.inf, *critical_points(lam, psi, theta, N), np.inf]
    roots_z = []

    for left, right in zip(candidates[:-1], candidates[1:]):
        if np.isneginf(left):
            left = -max(40.0, 2 * abs(right) if np.isfinite(right) else 40.0)
            while g_z(left, alpha, lam, psi, theta, N) > 0:
                left *= 2

        if np.isposinf(right):
            right = max(40.0, 2 * abs(left) if np.isfinite(left) else 40.0)
            while g_z(right, alpha, lam, psi, theta, N) < 0:
                right *= 2

        f_left = g_z(left, alpha, lam, psi, theta, N)
        f_right = g_z(right, alpha, lam, psi, theta, N)

        if np.isclose(f_left, 0.0, atol=1e-14):
            roots_z.append(left)
        elif np.isclose(f_right, 0.0, atol=1e-14):
            roots_z.append(right)
        elif f_left * f_right < 0:
            sol = scipy.optimize.root_scalar(
                g_z,
                args=(alpha, lam, psi, theta, N),
                bracket=[left, right],
                method="toms748",
                xtol=1e-12,
                rtol=1e-12,
                maxiter=200,
            )

            if sol.converged:
                roots_z.append(sol.root)

    roots_z = sorted(set(np.round(roots_z, 12)))
    roots_b = [b_from_z(z) for z in roots_z]

    min_allowed_b = minimum_allowed_b(alpha)
    filtered_pairs = [
        (z, b) for z, b in zip(roots_z, roots_b) if b >= min_allowed_b
    ]

    if not filtered_pairs:
        return [], []

    filtered_roots_z, filtered_roots_b = zip(*filtered_pairs)
    return list(filtered_roots_z), list(filtered_roots_b)


def solve_one(alpha, lam, psi, theta, N):
    roots_z, roots_b = solve_roots(alpha, lam, psi, theta, N)
    has_root = bool(roots_z)

    return {
        "theta": theta,
        "psi": psi,
        "N": N,
        "alpha": alpha,
        "lambda": lam,
        "n_roots": len(roots_z),
        "status": "OK" if has_root else f"{RED}NO ROOT{RESET}",
        "roots_z": roots_z,
        "roots_b": roots_b,
        "max_root_z": roots_z[-1] if has_root else np.nan,
        "max_root_b": roots_b[-1] if has_root else np.nan,
        "g_at_max_root": (
            g_z(roots_z[-1], alpha, lam, psi, theta, N)
            if has_root
            else np.nan
        ),
    }


def build_dataframe(mode):
    solver = solve_one if mode == "roots" else solve_minimum
    rows = [
        solver(alpha=alpha, lam=lam, psi=psi, theta=theta, N=N)
        for alpha, theta, lam, psi, N in product(
            ALPHAS, THETAS, LAMBDAS, PSIS, N_GRID
        )
    ]

    df = pd.DataFrame(rows)
    df = (
        df.sort_values(
            by=["theta", "lambda", "psi", "N"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )

    if mode == "roots":
        df = df[
            [
                "theta",
                "lambda",
                "psi",
                "N",
                "alpha",
                "status",
                "n_roots",
                "max_root_z",
                "max_root_b",
                "g_at_max_root",
                "roots_z",
                "roots_b",
            ]
        ]
    else:
        df = df[
            [
                "theta",
                "lambda",
                "psi",
                "N",
                "alpha",
                "min_z",
                "min_b",
                "g_at_min",
            ]
        ]

    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].round(3)

    if mode == "roots":
        df["roots_z"] = df["roots_z"].apply(
            lambda xs: [round(float(x), 3) for x in xs]
        )
        df["roots_b"] = df["roots_b"].apply(
            lambda xs: [round(float(x), 3) for x in xs]
        )

    return df


def plot_minimum_results(df):
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    for theta in sorted(df["theta"].unique()):
        theta_df = df[df["theta"] == theta]
        lambda_values = sorted(theta_df["lambda"].unique())
        fig, axes = plt.subplots(
            nrows=len(lambda_values),
            ncols=2,
            figsize=(12, 4 * len(lambda_values)),
            sharex=True,
            constrained_layout=True,
        )

        axes = np.atleast_2d(axes)

        for row_idx, lam in enumerate(lambda_values):
            lam_df = theta_df[theta_df["lambda"] == lam]
            for psi in sorted(lam_df["psi"].unique()):
                series = lam_df[lam_df["psi"] == psi].sort_values("N")
                label = f"psi={psi:.2f}"
                axes[row_idx, 0].plot(
                    series["N"], series["min_z"], marker="o", label=label
                )
                axes[row_idx, 1].plot(
                    series["N"], series["g_at_min"], marker="o", label=label
                )

            axes[row_idx, 0].set_title(f"lambda={lam:.2f}: min_z")
            axes[row_idx, 1].set_title(f"lambda={lam:.2f}: g_at_min")

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xscale("log")
                ax.set_xlabel("N")
                ax.grid(True, alpha=0.3)

        for row_idx in range(len(lambda_values)):
            axes[row_idx, 0].set_ylabel("Minimum z")
            axes[row_idx, 1].set_ylabel("Objective at minimum")
            axes[row_idx, 1].axhline(
                0.0, color="black", linewidth=1, linestyle="--", alpha=0.6
            )
            axes[row_idx, 1].legend(loc="best", fontsize=8)

        fig.suptitle(f"Split clipped objective minima, theta={theta}")
        outfile = PLOTDIR / f"split_clipped_minimum_theta_{theta}.png"
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_root_results(df):
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    for theta in sorted(df["theta"].unique()):
        theta_df = df[df["theta"] == theta]
        lambda_values = sorted(theta_df["lambda"].unique())
        fig, axes = plt.subplots(
            nrows=len(lambda_values),
            ncols=2,
            figsize=(12, 4 * len(lambda_values)),
            sharex=True,
            constrained_layout=True,
        )

        axes = np.atleast_2d(axes)

        for row_idx, lam in enumerate(lambda_values):
            lam_df = theta_df[theta_df["lambda"] == lam]
            for psi in sorted(lam_df["psi"].unique()):
                series = lam_df[lam_df["psi"] == psi].sort_values("N")
                label = f"psi={psi:.2f}"
                axes[row_idx, 0].plot(
                    series["N"], series["max_root_z"], marker="o", label=label
                )
                axes[row_idx, 1].plot(
                    series["N"], series["g_at_max_root"], marker="o", label=label
                )

            axes[row_idx, 0].set_title(f"lambda={lam:.2f}: max_root_z")
            axes[row_idx, 1].set_title(f"lambda={lam:.2f}: g_at_max_root")

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xscale("log")
                ax.set_xlabel("N")
                ax.grid(True, alpha=0.3)

        for row_idx in range(len(lambda_values)):
            axes[row_idx, 0].set_ylabel("Largest root z")
            axes[row_idx, 1].set_ylabel("Objective at largest root")
            axes[row_idx, 1].axhline(
                0.0, color="black", linewidth=1, linestyle="--", alpha=0.6
            )
            axes[row_idx, 1].legend(loc="best", fontsize=8)

        fig.suptitle(f"Split clipped root summary, theta={theta}")
        outfile = PLOTDIR / f"split_clipped_roots_theta_{theta}.png"
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["roots", "minimum"],
        default="roots",
        help="Whether to find roots or the minimum of the objective.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save plots for minimum-mode results to the plots directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = build_dataframe(args.mode)

    if args.plot:
        if args.mode == "minimum":
            plot_minimum_results(df)
        else:
            plot_root_results(df)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", None)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
