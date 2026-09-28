"""SLURM/submitit simulation of anytime-valid power and utility."""

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import submitit
from scipy.stats import norm

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from avi._config import (ALPHAS, AVI, LOGDIR, M, N_GRID, N_JOBS, RHOS,
                         THETAS, split_into_chunks)
from avi.martingale import anytime_stopping_time, standard_power, wald_upper_bound

OUTDIR = AVI / "power_results"


def run_task_chunk(chunk_id, tasks):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        try:
            rng = np.random.default_rng(task["seed"])
            sample = rng.normal(task["theta"], 1.0, task["N"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value, st, rejected = anytime_stopping_time(
                    sample, task["alpha"], task["rho"]
                )
            N = task["N"]
            power = int(rejected)
            std_value = sample.sum() / np.sqrt(N)
            std_reject = int(std_value > norm.ppf(1 - task["alpha"]))
            utility = 0.5 * (st / N - power)
            std_utility = 0.5 * (1 - std_reject)
            rows.append({**task, "seq_value": value,
                         "seq_reject": int(rejected), "seq_power": power,
                         "seq_stopping_time": st,
                         "seq_stopping_fraction": st / N,
                         "std_value": std_value, "std_reject": std_reject,
                         "std_power_theoretical": standard_power(
                             task["theta"], N, task["alpha"]),
                         "std_stopping_time": N, "utility": utility,
                         "std_utility": std_utility,
                         "utility_difference": utility - std_utility,
                         "wald_upper_bound_time": wald_upper_bound(
                             N, task["theta"], task["alpha"], task["rho"]),
                         "error": None})
        except Exception as exc:
            rows.append({**task, "seq_value": None, "seq_reject": None,
                         "seq_power": None,
                         "seq_stopping_time": None,
                         "seq_stopping_fraction": None, "std_value": None,
                         "std_reject": None, "std_power_theoretical": None,
                         "std_stopping_time": None, "utility": None,
                         "std_utility": None,
                         "utility_difference": None,
                         "wald_upper_bound_time": None, "error": repr(exc)})
    path = OUTDIR / f"results_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    task_id = 0
    for theta in THETAS:
        for N in N_GRID:
            for alpha in ALPHAS:
                for rho in RHOS:
                    for rep in range(M):
                        tasks.append({"theta": theta, "N": N, "alpha": alpha,
                                      "rho": rho, "rep": rep,
                                      "seed": 12345 + task_id})
                        task_id += 1
    random.seed(123)
    random.shuffle(tasks)
    chunks = split_into_chunks(tasks, N_JOBS)
    executor = submitit.AutoExecutor(folder=str(LOGDIR))
    executor.update_parameters(
        slurm_job_name="avi-power", slurm_partition="standard",
        slurm_account="stats_dept1", slurm_time=360, slurm_mem="16G",
        cpus_per_task=1, tasks_per_node=1,
        slurm_setup=[f"export PYTHONPATH={SRC}:$PYTHONPATH", f"cd {ROOT}"],
    )
    with executor.batch():
        jobs = [executor.submit(run_task_chunk, i, chunk)
                for i, chunk in enumerate(chunks)]
    print(f"Submitted {len(jobs)} anytime-valid power jobs.")
