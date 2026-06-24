from itertools import product

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize


ALPHAS = [0.05]
LAMBDAS = [0.25, 0.50, 0.75]
THETAS = [0.1, 0.5, 1.0]
N_GRID = [1000, 2000, 5000, 10000, 20000, 50000]


def g_z(z, alpha, lam, theta, N):
    z_alpha = scipy.stats.norm.ppf(1 - alpha)

    seq_term = (z_alpha + z) / (theta * np.sqrt(N)) - 1
    stop_term = scipy.stats.norm.sf(z)  # 1 - Phi(z)

    return lam * seq_term + (1 - lam) * stop_term


def g_b(b, alpha, lam, theta, N):
    z_b = scipy.stats.norm.ppf(b)
    return g_z(z_b, alpha, lam, theta, N)


def b_from_z(z):
    return np.clip(
        scipy.stats.norm.cdf(z),
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )


def critical_points(lam, theta, N):
    """
    Critical points of g(z).

    g'(z) = lam/(theta sqrt(N)) - (1-lam) phi(z).

    So critical points solve

        phi(z) = lam / ((1-lam) theta sqrt(N)).
    """
    if not (0 < lam < 1):
        return []

    critical_density = lam / ((1 - lam) * theta * np.sqrt(N))
    max_density = scipy.stats.norm.pdf(0)

    if not (0 < critical_density <= max_density):
        return []

    radius = np.sqrt(
        -2 * np.log(critical_density * np.sqrt(2 * np.pi))
    )

    return [-radius, radius]


def solve_roots(alpha, lam, theta, N):
    candidates = [-np.inf, *critical_points(lam, theta, N), np.inf]
    roots_z = []

    for left, right in zip(candidates[:-1], candidates[1:]):

        if np.isneginf(left):
            left = -max(40.0, 2 * abs(right) if np.isfinite(right) else 40.0)
            while g_z(left, alpha, lam, theta, N) > 0:
                left *= 2

        if np.isposinf(right):
            right = max(40.0, 2 * abs(left) if np.isfinite(left) else 40.0)
            while g_z(right, alpha, lam, theta, N) < 0:
                right *= 2

        f_left = g_z(left, alpha, lam, theta, N)
        f_right = g_z(right, alpha, lam, theta, N)

        if np.isclose(f_left, 0.0, atol=1e-14):
            roots_z.append(left)

        elif np.isclose(f_right, 0.0, atol=1e-14):
            roots_z.append(right)

        elif f_left * f_right < 0:
            sol = scipy.optimize.root_scalar(
                g_z,
                args=(alpha, lam, theta, N),
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

    return roots_z, roots_b


def solve_one(alpha, lam, theta, N):
    roots_z, roots_b = solve_roots(alpha, lam, theta, N)

    return {
        "theta": theta,
        "N": N,
        "alpha": alpha,
        "lambda": lam,
        "n_roots": len(roots_z),
        "roots_z": roots_z,
        "roots_b": roots_b,
        "max_root_z": roots_z[-1] if roots_z else np.nan,
        "max_root_b": roots_b[-1] if roots_b else np.nan,
        "g_at_max_root": (
            g_z(roots_z[-1], alpha, lam, theta, N)
            if roots_z
            else np.nan
        ),
    }


def main():
    rows = [
        solve_one(alpha=alpha, lam=lam, theta=theta, N=N)
        for alpha, theta, lam, N in product(ALPHAS, THETAS, LAMBDAS, N_GRID)
    ]

    df = pd.DataFrame(rows)

    # theta first, then fixed lambda, then N increasing
    df = (
        df.sort_values(
            by=["theta", "lambda", "N"],
            ascending=[True, True, True],
        )
        .reset_index(drop=True)
    )

    df = df[
        [
            "theta",
            "lambda",
            "N",
            "alpha",
            "n_roots",
            "max_root_z",
            "max_root_b",
            "g_at_max_root",
            "roots_z",
            "roots_b",
        ]
    ]

    # Round scalar float columns to 3 decimals
    float_cols = df.select_dtypes(include=["float"]).columns
    df[float_cols] = df[float_cols].round(3)

    # Round list-valued root columns to 3 decimals
    df["roots_z"] = df["roots_z"].apply(lambda xs: [round(float(x), 3) for x in xs])
    df["roots_b"] = df["roots_b"].apply(lambda xs: [round(float(x), 3) for x in xs])

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", None)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()