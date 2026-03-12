#!/bin/bash
#SBATCH --array=1-10
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=02:59:00
#SBATCH --partition=unkillable
#SBATCH --job-name=cem
#SBATCH -o output-%A_%a.log
#SBATCH -e error-%A_%a.log

set -euo pipefail
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module purge
module load python
module load cuda/12.0
export JAX_PLATFORM_NAME=cpu
export _TYPER_STANDARD_TRACEBACK=1
uv run cem optimize geometry multi_task --jobs 8 --trials 1000 --failures 10 --seed $SLURM_ARRAY_TASK_ID
