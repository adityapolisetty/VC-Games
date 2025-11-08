#!/bin/bash
#PBS -l select=1:ncpus=12:mem=5gb
#PBS -l walltime=08:00:00
#PBS -J 0-39
#PBS -N il_frontier_array
#PBS -o ../logs/il_frontier.out
#PBS -e ../logs/il_frontier.err
#PBS -V

set -euo pipefail

# Conda environment (match vc_sweep_dyn.sh)
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate base

cd "$PBS_O_WORKDIR"
mkdir -p ../logs

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ID="${PBS_ARRAY_INDEX:-0}"

# Parameterization (keep a constant base seed across array tasks like dynamic)
SEED=${SEED:-12345}
ROUNDS=${ROUNDS:-100000}
MAX_SIGNALS=${MAX_SIGNALS:-9}

STRIDE=${STRIDE:-40}

python3 -u frontier.py \
  --seed "$SEED" \
  --rounds "$ROUNDS" \
  --max_signals "$MAX_SIGNALS" \
  --procs 12 \
  --sweep \
  --sweep_out ../frontier_output \
  --sweep_index "$ID" \
  --sweep_stride "$STRIDE"
  --debug_excel
