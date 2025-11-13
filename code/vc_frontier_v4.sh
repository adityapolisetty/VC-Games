#!/bin/bash
#PBS -l select=1:ncpus=64:mem=20gb
#PBS -l walltime=02:00:00
## PBS -J 0-1
#PBS -N il_frontier_v4_test
#PBS -o ../logs/il_frontier_v4.out
#PBS -e ../logs/il_frontier_v4.err
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

ID=0

# Parameterization (keep a constant base seed across array tasks like dynamic)
SEED=${SEED:-12345}
ROUNDS=${ROUNDS:-100}
MAX_SIGNALS=${MAX_SIGNALS:-9}

STRIDE=${STRIDE:-16}  # Total number of files (4 alphas × 2 scale_pays × 2 signal_types)

echo "[v4] Starting frontier_v4.py with shared weight matrices"
echo "[v4] Configuration: ROUNDS=$ROUNDS, PROCS=64, MAX_SIGNALS=$MAX_SIGNALS"
echo "[v4] Memory optimization: Build Wm2_all once, share via fork"
echo ""

python3 -u frontier_v4.py \
  --seed "$SEED" \
  --rounds "$ROUNDS" \
  --max_signals "$MAX_SIGNALS" \
  --procs 64 \
  --sweep \
  --sweep_out ../frontier_output_v4_test \
  --sweep_index "$ID" \
  --sweep_stride "$STRIDE"

echo ""
echo "[v4] Completed at $(date)"
