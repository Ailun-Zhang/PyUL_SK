#!/bin/bash
# Slurm template for one PyUL_SK Stone Skipping run.
# Submit from the folder where the outputs should go:
#     mkdir -p logs   # Slurm does not create the --output folder
#     sbatch --export=ALL,PYUL_REPO=$HOME/PyUL_SK,RUN_ARGS="--resol 192 --duration 3000" examples/slurm/stone_skipping.sbatch
# Adjust partition/account and the environment activation for your site.
# No node is pinned: add '--nodelist=<node>' only if your site requires it.
#SBATCH --job-name=pyul_sk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_%x_%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- software environment (edit) -------------------------------------------
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate pyul-sk
PYTHON="${PYTHON:-python3}"
PYUL_REPO="${PYUL_REPO:-$HOME/PyUL_SK}"

# --- threads: one explicit value for FFTW, numexpr and scipy.fft -------------
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYUL_NUM_THREADS="$NTHREADS"
export OMP_NUM_THREADS="$NTHREADS"
export NUMEXPR_MAX_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- run-time behaviour ------------------------------------------------------
export PYUL_PROGRESS=lines          # one log line per snapshot instead of '\r' bars
export PYUL_INTERACTIVE=0           # never prompt; existing outputs raise FileExistsError
export PYUL_CACHE_DIR="${PYUL_CACHE_DIR:-$PWD/pyul_cache}"   # IsoP Green functions
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

mkdir -p logs
LOG="logs/pyul_sk_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
echo "PyUL_SK job ${SLURM_JOB_ID:-manual} on $(hostname) with ${NTHREADS} threads; log: ${LOG}"

# The driver's exit status is the job's exit status (no '| tee').
srun --cpus-per-task="$NTHREADS" "$PYTHON" -B "$PYUL_REPO/examples/run_stone_skipping_local.py" \
    --out "$PWD/runs" ${RUN_ARGS:-} > "$LOG" 2>&1
echo "finished with status 0"
