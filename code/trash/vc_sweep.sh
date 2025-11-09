#!/bin/bash
#PBS -l select=1:ncpus=8:mem=5gb
#PBS -l walltime=02:00:00
#PBS -J 0-61
#PBS -N v7_sweep_array
#PBS -o ../logs/output.out
#PBS -e ../logs/output.err
#PBS -V

set -euo pipefail

# env
eval "$(~/miniforge3/bin/conda shell.bash hook)"    
conda activate base

cd "$PBS_O_WORKDIR"
mkdir -p logs output

# avoid thread oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ID="${PBS_ARRAY_INDEX}"
STRIDE=62

python3 -u card_game.py --seed 12345 --rounds 100000 --max_signals 9 \
 --procs 8 --sweep --sweep_out ../output --sweep_index "$ID" --sweep_stride "$STRIDE"