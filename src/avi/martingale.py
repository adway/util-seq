"""The anytime-valid normal-mixture test and its simulation quantities."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm


def martingale_values(sample, rho=1.0):
    """Return M_1^+, ..., M_N^+ for observations with unit variance.

    The formula is evaluated on the log scale for numerical stability.  At
    t=0, M_0^+=1.  The input is allowed to be any one-dimensional array.
    """
    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 1:
        raise ValueError("sample must be one-dimensional")
    if rho <= 0:
        raise ValueError("rho must be positive")

    t = np.arange(1, sample.size + 1, dtype=float)
    s = np.cumsum(sample)
    scale = np.sqrt(rho + t)
    # log(2 Phi(x)) is stable enough for the simulation ranges here.
    log_m = (
        np.log(2.0)
        + 0.5 * np.log(rho / (rho + t))
        + s * s / (2.0 * (rho + t))
        + np.log(ndtr(s / scale))
    )
    # Once the threshold is crossed only the comparison matters.  Capping the
    # exponent avoids an avoidable overflow for very large positive signals.
    return np.exp(np.minimum(log_m, np.log(np.finfo(float).max)))


def anytime_stopping_time(sample, alpha, rho=1.0):
    """Return ``(M_at_stop, T, rejected)`` for a horizon-limited test.

    The process is stopped at the first t <= N for which M_t^+ >= 1/alpha.
    If there is no crossing, T=N and rejected is False.  Rejection is
    deterministic; there is deliberately no post-stopping coin flip.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    values = martingale_values(sample, rho=rho)
    crossing = np.flatnonzero(values >= 1.0 / alpha)
    if crossing.size:
        index = int(crossing[0])
        return float(values[index]), index + 1, True
    return float(values[-1]), values.size, False


def standard_power(theta, N, alpha):
    """Exact one-sided fixed-N Gaussian-test power."""
    return float(norm.cdf(theta * np.sqrt(N) - norm.ppf(1.0 - alpha)))


def terminal_power_lower_bound(N, theta, alpha, rho=1.0):
    """Exact oracle probability of the terminal event M_N^+ > 1/alpha."""
    cutoff = inverted_martingale_boundary(N, alpha, rho)
    return float(norm.sf((cutoff - N * theta) / np.sqrt(N)))


def inverted_martingale_boundary(t, alpha, rho=1.0):
    """Return b(t), defined by M_t^+(b(t)) = 1/alpha."""
    if t <= 0 or rho <= 0 or not 0 < alpha < 1:
        raise ValueError("require t > 0, rho > 0, and alpha in (0, 1)")
    scale = np.sqrt(rho + t)
    log_cutoff = np.log(1.0 / alpha)

    def equation(s):
        log_m = (np.log(2.0) + 0.5 * np.log(rho / (rho + t))
                 + s * s / (2.0 * (rho + t))
                 + norm.logcdf(s / scale))
        return log_m - log_cutoff

    lo, hi = -scale, scale
    while equation(lo) > 0:
        lo *= 2.0
    while equation(hi) < 0:
        hi *= 2.0
    return float(brentq(equation, lo, hi))


def wald_upper_bound(N, theta, alpha, rho=1.0):
    """Return the Wald boundary-crossing comparison for E[min(T, N)].

    Let b(t) be the Gaussian-sum boundary obtained by inverting
    M_t^+(s)=1/alpha. The Wald calculation uses theta E[T] = b(E[T]);
    this function solves theta*t=b(t) and truncates the result at N.
    """
    if N < 1 or rho <= 0 or not 0 < alpha < 1:
        raise ValueError("require N >= 1, rho > 0, and alpha in (0, 1)")
    if theta <= 0:
        return float(N)
    def equation(t):
        return theta * t - inverted_martingale_boundary(t, alpha, rho)

    if equation(N) <= 0:
        return float(N)
    return float(brentq(equation, 1e-10, float(N)))
