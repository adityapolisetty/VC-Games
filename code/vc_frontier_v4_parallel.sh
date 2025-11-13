#!/bin/bash
#PBS -l select=1:ncpus=128:mem=920gb
#PBS -l walltime=02:00:00
#PBS -J 0-79
#PBS -N il_frontier_v4_parallel
#PBS -o ../logs/il_frontier_v4_parallel.out
#PBS -e ../logs/il_frontier_v4_parallel.err
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
ROUNDS=${ROUNDS:-1000}

echo "[v4p] ========================================"
echo "[v4p] Array job $ID of 80 (8 files × 10 n_sig)"
echo "[v4p] Configuration: ROUNDS=$ROUNDS, PROCS=128"
echo "[v4p] ========================================"
echo ""

python3 -u frontier_v4_parallel.py \
  --seed "$SEED" \
  --rounds "$ROUNDS" \
  --procs 128 \
  --sweep_out ../frontier_output_v4_parallel \
  --array_index "$ID"

echo ""
echo "[v4p] Job $ID completed at $(date)"
