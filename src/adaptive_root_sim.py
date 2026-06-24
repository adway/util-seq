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


M = 500

thetas = [0.10, 0.5, 1, 2]
N_grid = [1_000, 2_000, 5_000, 10_000]
alphas = [0.05]
lambdas = [0.50]

N_JOBS = 500

OUTDIR = ROOT / "adaptive_root_results"
LOGDIR = ROOT / "run_logs"
B_LOWER = 0.00001
B_UPPER = 0.99999


def split_into_chunks(tasks, n_chunks):
    chunks = [[] for _ in range(n_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_chunks].append(task)
    return chunks


def root_objective(b, lam, alpha, theta_hat, N):
    z_alpha = scipy.stats.norm.ppf(1 - alpha)
    z_b = scipy.stats.norm.ppf(b)
    seq_term = (z_alpha + z_b) / (theta_hat * np.sqrt(N)) - 1
    return lam * seq_term + (1 - lam) * (1 - b)


def largest_root(lam, alpha, theta_hat, N):
    if (
        not np.isfinite(theta_hat)
        or theta_hat <= 0
        or not np.isfinite(N)
        or N <= 0
    ):
        return np.nan, np.nan

    def z_to_open_probability(z):
        b = scipy.stats.norm.cdf(z)
        return np.clip(b, B_LOWER, B_UPPER)

    def root_objective_z(z):
        z_alpha = scipy.stats.norm.ppf(1 - alpha)
        seq_term = (z_alpha + z) / (theta_hat * np.sqrt(N)) - 1
        return lam * seq_term + (1 - lam) * scipy.stats.norm.sf(z)

    if lam == 0:
        return np.nan, np.nan

    if lam == 1:
        z_alpha = scipy.stats.norm.ppf(1 - alpha)
        root_z = theta_hat * np.sqrt(N) - z_alpha
        return root_z, z_to_open_probability(root_z)

    # f'(z) = slope - (1 - lam) * phi(z), so possible extrema occur where
    # the normal density equals slope / (1 - lam). Splitting at those points
    # lets the right-to-left brentq scan find the largest root when multiple
    # roots are possible.
    slope = lam / (theta_hat * np.sqrt(N))
    density_threshold = slope / (1 - lam)
    critical_points = []
    max_density = scipy.stats.norm.pdf(0)
    if 0 < density_threshold < max_density:
        radius = np.sqrt(-2 * np.log(density_threshold * np.sqrt(2 * np.pi)))
        critical_points = [-radius, radius]

    left = -8.0
    while root_objective_z(left) > 0 and left > -1_000:
        left *= 2

    right = 8.0
    while root_objective_z(right) < 0 and right < 1_000:
        right *= 2

    points = [left, *critical_points, right]
    points = [point for point in points if left <= point <= right]
    points = sorted(set(points))

    root_z = np.nan
    for interval_left, interval_right in reversed(list(zip(points[:-1], points[1:]))):
        left_value = root_objective_z(interval_left)
        right_value = root_objective_z(interval_right)

        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if right_value == 0:
            root_z = interval_right
            break
        if left_value == 0:
            root_z = interval_left
            break
        if left_value * right_value < 0:
            root_z = scipy.optimize.brentq(
                root_objective_z,
                interval_left,
                interval_right,
            )
            break

    if not np.isfinite(root_z):
        return np.nan, np.nan

    return root_z, z_to_open_probability(root_z)


def adaptive_root_test(sample, alpha, lam, rng):
    N = sample.size
    z_alpha = scipy.stats.norm.ppf(1 - alpha)
    cumulative = np.cumsum(sample)

    valid_root_looks = 0
    invalid_root_looks = 0
    last_theta_hat = np.nan
    last_z_hat = np.nan
    last_b_hat = np.nan

    for i, cumulative_i in enumerate(cumulative[:-1]):
        observed_n = i + 1
        remaining_n = N - i
        theta_hat = cumulative_i / observed_n
        z_hat, b_hat = largest_root(lam, alpha, theta_hat, N)

        if not np.isfinite(b_hat) or b_hat <= 0 or b_hat >= 1:
            invalid_root_looks += 1
            continue

        valid_root_looks += 1
        last_theta_hat = theta_hat
        last_z_hat = z_hat
        last_b_hat = b_hat

        seq_value = scipy.stats.norm.cdf(
            (cumulative_i - np.sqrt(N) * z_alpha) / np.sqrt(remaining_n)
        )

        if seq_value > b_hat:
            return {
                "adaptive_value": seq_value,
                "adaptive_reject": int(rng.uniform(0, 1) < seq_value),
                "adaptive_stopping_time": observed_n,
                "adaptive_stopping_fraction": observed_n / N,
                "adaptive_stopped_early": True,
                "theta_hat_at_stop": theta_hat,
                "z_hat_at_stop": z_hat,
                "b_hat_at_stop": b_hat,
                "last_theta_hat": last_theta_hat,
                "last_z_hat": last_z_hat,
                "last_b_hat": last_b_hat,
                "valid_root_looks": valid_root_looks,
                "invalid_root_looks": invalid_root_looks,
            }

    final_value = cumulative[-1] / np.sqrt(N)
    return {
        "adaptive_value": final_value,
        "adaptive_reject": int(final_value > z_alpha),
        "adaptive_stopping_time": N,
        "adaptive_stopping_fraction": 1.0,
        "adaptive_stopped_early": False,
        "theta_hat_at_stop": cumulative[-1] / N,
        "z_hat_at_stop": np.nan,
        "b_hat_at_stop": np.nan,
        "last_theta_hat": last_theta_hat,
        "last_z_hat": last_z_hat,
        "last_b_hat": last_b_hat,
        "valid_root_looks": valid_root_looks,
        "invalid_root_looks": invalid_root_looks,
    }


def run_one_task(task):
    theta = task["theta"]
    N = task["N"]
    alpha = task["alpha"]
    lam = task["lambda"]
    seed = task["seed"]

    rng = np.random.default_rng(seed)
    sample = rng.normal(theta, 1, N)
    return adaptive_root_test(sample, alpha, lam, rng)


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
                "adaptive_value": None,
                "adaptive_reject": None,
                "adaptive_stopping_time": None,
                "adaptive_stopping_fraction": None,
                "adaptive_stopped_early": None,
                "theta_hat_at_stop": None,
                "z_hat_at_stop": None,
                "b_hat_at_stop": None,
                "last_theta_hat": None,
                "last_z_hat": None,
                "last_b_hat": None,
                "valid_root_looks": None,
                "invalid_root_looks": None,
                "error": repr(e),
            })

    path = OUTDIR / f"results_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return str(path)


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    tasks = []

    base_seed = 78901
    task_id = 0

    for theta in thetas:
        for N in N_grid:
            for alpha in alphas:
                for lam in lambdas:
                    for rep in range(M):
                        tasks.append({
                            "theta": theta,
                            "N": N,
                            "alpha": alpha,
                            "lambda": lam,
                            "rep": rep,
                            "seed": base_seed + task_id,
                        })
                        task_id += 1

    random.seed(123)
    random.shuffle(tasks)

    chunks = split_into_chunks(tasks, N_JOBS)

    print(f"Total simulations: {len(tasks)}")
    print(f"Total jobs: {len(chunks)}")
    print(f"Tasks per job: about {len(tasks) / len(chunks):.1f}")

    executor = submitit.AutoExecutor(folder=str(LOGDIR))
    executor.update_parameters(
        slurm_job_name="adapt-root",
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
