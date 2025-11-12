#!/bin/bash
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -l walltime=01:00:00
#PBS -J 0-41
#PBS -N expand_frontier
#PBS -o ../logs/expand.out
#PBS -e ../logs/expand.err
#PBS -V

set -euo pipefail

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate base

cd "$PBS_O_WORKDIR"
mkdir -p ../logs ../frontier_expanded

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ID="${PBS_ARRAY_INDEX:-0}"

FRONTIER_DIR="../frontier_output_v2"
OUTPUT_DIR="../frontier_expanded"
UNITS=${UNITS:-3}

mapfile -t NPZ_FILES < <(ls -1 "$FRONTIER_DIR"/sc3*sp1*.npz 2>/dev/null | sort)
N_FILES=${#NPZ_FILES[@]}

if [ "$ID" -ge "$N_FILES" ]; then
    exit 1
fi

NPZ_FILE="${NPZ_FILES[$ID]}"

python3 -u analyze_linear_combinations.py \
    --npz "$NPZ_FILE" \
    --units "$UNITS" \
    --out_dir "$OUTPUT_DIR" \
    --suffix "_expanded"
