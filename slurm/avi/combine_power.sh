#!/bin/bash
#SBATCH --job-name=avi_combine_power
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00

module load gcc/11.2.0
cd "$HOME/util-seq"
mkdir -p logs
export PYTHONPATH="$HOME/util-seq/src:$PYTHONPATH"
uv sync
PYTHONPATH="$HOME/util-seq/src:$PYTHONPATH" uv run python -u src/avi/combine_power.py
