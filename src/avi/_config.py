"""Shared simulation configuration for the anytime-valid experiments."""

from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
AVI = SRC / "avi"
LOGDIR = ROOT / "run_logs" / "avi"

M = 500
THETAS = (0.0, 0.01, 0.05, 0.10, 0.50, 1.0)
N_GRID = (1_000, 5_000, 10_000, 50_000)
ALPHAS = (0.01, 0.05)
RHOS = (1.0,)
N_JOBS = 500


def split_into_chunks(tasks, n_chunks):
    n_chunks = min(n_chunks, max(1, len(tasks)))
    chunks = [[] for _ in range(n_chunks)]
    for idx, task in enumerate(tasks):
        chunks[idx % n_chunks].append(task)
    return chunks
