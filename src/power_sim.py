import os
import sys
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import submitit

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from helpers.sample import make_sample, stopping_time
from helpers.power import seq_power, std_power

M = 500

thetas = [0.0, 0.01, 0.05, 0.5, 1]
N_grid = [1_000, 5_000, 10_000, 50_000]
alphas = [0.01, 0.05]
betas = [0.80, 0.90, 0.95, 0.99]

N_JOBS = 500

OUTDIR = ROOT / "power_results"
LOGDIR = ROOT / "run_logs"

def split_into_chunks(tasks, n_chunks):
    chunks = [[] for _ in range(n_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_chunks].append(task)
    return chunks

def run_task_chunk(chunk_id, tasks):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for task in tasks:
        theta = task["theta"]
        N = task["N"]
        alpha = task["alpha"]
        beta = task["beta"]
        rep = task["rep"]
        seed = task["seed"]

        print(task, flush=True)

        try:
            np.random.seed(seed)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                sample = make_sample(N, theta)
                seq_val, seq_reject, seq_st = seq_power(sample, alpha, beta)
                std_val, std_reject, std_st = std_power(sample, alpha)

            rows.append({
                "chunk_id": chunk_id,
                "rep": rep,
                "seed": seed,
                "theta": theta,
                "N": N,
                "alpha": alpha,
                "beta": beta,
                "seq_value": seq_val,
                "seq_reject": seq_reject,
                "seq_stopping_time": seq_st,
                "std_value": std_val,
                "std_reject": std_reject,
                "std_stopping_time": std_st,
                "error": None,
            })

        except Exception as e:
            rows.append({
                "chunk_id": chunk_id,
                "rep": rep,
                "seed": seed,
                "theta": theta,
                "N": N,
                "alpha": alpha,
                "beta": beta,
                "seq_value": None,
                "seq_reject": None,
                "seq_stopping_time": None,
                "std_value": None,
                "std_reject": None,
                "std_stopping_time": None,
                "error": repr(e),
            })

    path = OUTDIR / f"results_chunk_{chunk_id:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return str(path)

if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    tasks = []

    base_seed = 12345
    task_id = 0

    for theta in thetas:
        for N in N_grid:
            for alpha in alphas:
                for beta in betas:
                    for rep in range(M):
                        tasks.append({
                            "theta": theta,
                            "N": N,
                            "alpha": alpha,
                            "beta": beta,
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
        slurm_job_name="stop",
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