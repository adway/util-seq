# Anytime-valid simulation workflow

The test uses the exact cutoff

\[
M_t^+ \geq \frac{1}{\alpha}.
\]

The first crossing is deterministic; there is no coin flip. The replicate
variable `seq_reject` records a crossing at or before \(N\), and `seq_power`
equals that rejection indicator, including a crossing at \(N\).

The oracle comparison is calculated in `notebooks/avi.ipynb` from the
terminal-event power lower bound and the inverted-martingale Wald boundary.

Submit simulations from the repository root with the scripts in
`slurm/avi/`. Combine chunk files with the corresponding combine script. The
Wald comparison is the truncated first-order mixture bound implemented in
`martingale.py`.
