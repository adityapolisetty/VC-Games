#!/bin/bash
#PBS -l select=1:ncpus=128:mem=920gb:cpu_type=rome
#PBS -l walltime=02:00:00
#PBS -J 0-49
#PBS -N il_frontier_v5
#PBS -o ../logs/il_frontier_v5.out
#PBS -e ../logs/il_frontier_v5.err
#PBS -V

set -euo pipefail

# Conda environment
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

# Parameterization
SEED=${SEED:-12345}
ROUNDS=${ROUNDS:-3000}

python3 -u frontier_v5.py \
  --seed "$SEED" \
  --rounds "$ROUNDS" \
  --procs 128 \
  --sweep_out ../frontier_output_v5 \
  --array_index "$ID"
