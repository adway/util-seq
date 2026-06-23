#!/bin/bash
#SBATCH --job-name=combine_power
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00

module load gcc/11.2.0

cd "$HOME/util-seq"

export PYTHONPATH="$HOME/util-seq/src:$PYTHONPATH"

echo "Project: $(pwd)"
echo "uv: $(which uv)"
echo "uv version:"
uv --version

echo "Syncing environment from pyproject.toml..."
uv sync

echo "Python version:"
uv run python --version

echo "Installed packages:"
uv run python -m pip list | head

echo "Python executable:"
uv run python -c "import sys; print(sys.executable)"

echo "Submitting combine job..."

PYTHONPATH="$HOME/util-seq/src:$PYTHONPATH" \
uv run python -u src/combine_power.py

echo "Submission complete."
