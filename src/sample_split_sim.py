import sys
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import submitit

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from helpers.sample import stopping_time


M = 500
INNER_M = 50

thetas = [0.05, 0.10, 0.5, 1]
N_grid = [1_000, 2_000, 5_000, 10_000]
alphas = [0.05]
lambdas = [0.50]
psis = [0.10, 0.25, 0.50, 0.75, 0.90]

N_JOBS = 500

OUTDIR = ROOT / "sample_split_results"
LOGDIR = ROOT / "run_logs"


def split_into_chunks(tasks, n_chunks):
    chunks = [[] for _ in range(n_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_chunks].append(task)
    return chunks


def root_objective(b, lam, alpha, psi, theta_hat, N):
    z_alpha = scipy.stats.norm.ppf(1 - alpha)
    z_b = scipy.stats.norm.ppf(b)
    seq_term = - (1 - psi) + (
        np.sqrt(1 - psi) * (z_alpha + z_b)
        / (theta_hat * np.sqrt(N))
    )
    return lam * seq_term + (1 - lam) * (1 - b)


def bounded_objective_b(b, lam, alpha, psi, theta_hat, N):
    seq_term = root_objective(b, 1, alpha, psi, theta_hat, N)
    return lam * np.clip(seq_term, 0, 1) + (1 - lam) * (1 - b)


def minimize_bounded_objective_b(lam, alpha, psi, theta_hat, N):
    if not np.isfinite(theta_hat) or theta_hat <= 0:
        return np.nan

    def z_to_open_probability(z):
        b = scipy.stats.norm.cdf(z)
        return np.clip(b, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    def seq_term_z(z):
        z_alpha = scipy.stats.norm.ppf(1 - alpha)
        return - (1 - psi) + (
            np.sqrt(1 - psi) * (z_alpha + z)
            / (theta_hat * np.sqrt(N))
        )

    def objective_z(z):
        b = scipy.stats.norm.cdf(z)
        return lam * np.clip(seq_term_z(z), 0, 1) + (1 - lam) * (1 - b)

    z_alpha = scipy.stats.norm.ppf(1 - alpha)
    sqrt_q = np.sqrt(1 - psi)
    theta_sqrt_n = theta_hat * np.sqrt(N)
    z_seq_zero = theta_sqrt_n * sqrt_q - z_alpha
    z_seq_one = theta_sqrt_n * (2 - psi) / sqrt_q - z_alpha

    candidates = [z_seq_zero, z_seq_one]

    if 0 < lam < 1:
        critical_density = lam * sqrt_q / ((1 - lam) * theta_sqrt_n)
        max_density = scipy.stats.norm.pdf(0)
        if 0 < critical_density <= max_density:
            radius = np.sqrt(-2 * np.log(critical_density * np.sqrt(2 * np.pi)))
            candidates.extend([-radius, radius])

    finite_candidates = [
        z for z in candidates
        if np.isfinite(z) and z_seq_zero <= z <= z_seq_one
    ]

    if not finite_candidates:
        return np.nan

    best_z = min(finite_candidates, key=objective_z)
    return z_to_open_probability(best_z)


def seq_power_rng(sample, alpha, beta, rng):
    val, st = stopping_time(sample, alpha, beta)
    if st < len(sample):
        return val, int(rng.uniform(0, 1) < val), st

    return val, int(val > scipy.stats.norm.ppf(1 - alpha)), st


def run_one_task(task):
    theta = task["theta"]
    N = task["N"]
    alpha = task["alpha"]
    lam = task["lambda"]
    psi = task["psi"]
    seed = task["seed"]

    rng = np.random.default_rng(seed)
    split_n = int(np.floor(psi * N))
    remaining_n = N - split_n

    if split_n < 1 or remaining_n < 1:
        raise ValueError(f"Invalid split sizes: split_n={split_n}, remaining_n={remaining_n}")

    split_sample = rng.normal(theta, 1, split_n)
    theta_hat = split_sample.mean()
    b_hat = minimize_bounded_objective_b(lam, alpha, psi, theta_hat, N)

    if not np.isfinite(b_hat) or b_hat <= 0 or b_hat >= 1:
        return {
            "theta_hat": theta_hat,
            "b_hat": b_hat,
            "split_n": split_n,
            "remaining_n": remaining_n,
            "inner_power": np.nan,
            "inner_seq_stopping_time": np.nan,
            "inner_total_stopping_time": np.nan,
            "inner_total_stopping_fraction": np.nan,
            "inner_early_stop_prob": np.nan,
            "inner_n_success": 0,
        }

    rejects = []
    seq_stopping_times = []

    for _ in range(INNER_M):
        sample = rng.normal(theta, 1, remaining_n)
        _, reject, seq_stopping_time = seq_power_rng(sample, alpha, b_hat, rng)
        rejects.append(reject)
        seq_stopping_times.append(seq_stopping_time)

    rejects = np.asarray(rejects)
    seq_stopping_times = np.asarray(seq_stopping_times)
    total_stopping_times = split_n + seq_stopping_times

    return {
        "theta_hat": theta_hat,
        "b_hat": b_hat,
        "split_n": split_n,
        "remaining_n": remaining_n,
        "inner_power": rejects.mean(),
        "inner_seq_stopping_time": seq_stopping_times.mean(),
        "inner_total_stopping_time": total_stopping_times.mean(),
        "inner_total_stopping_fraction": (total_stopping_times / N).mean(),
        "inner_early_stop_prob": (seq_stopping_times < remaining_n).mean(),
        "inner_n_success": INNER_M,
    }


def run_task_chunk(chunk_id, tasks):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for task in tasks:
        print(task, flush=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_one_task(task)

            rows.append({
                "chunk_id": chunk_id,
                "rep": task["rep"],
                "seed": task["seed"],
                "theta": task["theta"],
                "N": task["N"],
                "alpha": task["alpha"],
                "lambda": task["lambda"],
                "psi": task["psi"],
                "inner_m": INNER_M,
                **result,
                "error": None,
            })

        except Exception as e:
            rows.append({
                "chunk_id": chunk_id,
                "rep": task["rep"],
                "seed": task["seed"],
                "theta": task["theta"],
                "N": task["N"],
                "alpha": task["alpha"],
                "lambda": task["lambda"],
                "psi": task["psi"],
                "inner_m": INNER_M,
                "theta_hat": None,
                "b_hat": None,
                "split_n": None,
                "remaining_n": None,
                "inner_power": None,
                "inner_seq_stopping_time": None,
                "inner_total_stopping_time": None,
                "inner_total_stopping_fraction": None,
                "inner_early_stop_prob": None,
                "inner_n_success": 0,
                "error": repr(e),
            })

    path = OUTDIR / f"results_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return str(path)


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    tasks = []

    base_seed = 67890
    task_id = 0

    for theta in thetas:
        for N in N_grid:
            for alpha in alphas:
                for lam in lambdas:
                    for psi in psis:
                        for rep in range(M):
                            tasks.append({
                                "theta": theta,
                                "N": N,
                                "alpha": alpha,
                                "lambda": lam,
                                "psi": psi,
                                "rep": rep,
                                "seed": base_seed + task_id,
                            })
                            task_id += 1

    random.seed(123)
    random.shuffle(tasks)

    chunks = split_into_chunks(tasks, N_JOBS)

    print(f"Total outer simulations: {len(tasks)}")
    print(f"Inner simulations per outer simulation: {INNER_M}")
    print(f"Total inner simulations: {len(tasks) * INNER_M}")
    print(f"Total jobs: {len(chunks)}")
    print(f"Tasks per job: about {len(tasks) / len(chunks):.1f}")

    executor = submitit.AutoExecutor(folder=str(LOGDIR))
    executor.update_parameters(
        slurm_job_name="split",
        slurm_partition="standard",
        slurm_account="stats_dept1",
        slurm_time=360,
        slurm_mem="16G",
        cpus_per_task=1,
        tasks_per_node=1,
        slurm_setup=[
            f"export PYTHONPATH={SRC}:$PYTHONPATH",
            f"cd {ROOT}",
        ],
    )

    jobs = []
    with executor.batch():
        for chunk_id, chunk in enumerate(chunks):
            jobs.append(
                executor.submit(run_task_chunk, chunk_id, chunk)
            )

    print(f"Submitted {len(jobs)} jobs.")
    print([job.job_id for job in jobs[:10]])
