#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=40
#SBATCH --mem=191000M
#SBATCH --time=00-23:59
#SBATCH --account=def-mbowling
#SBATCH --array=4-8
#SBATCH --output=alpha-zero-rnd-sweep-%j.out

module load StdEnv/2020
module load cuda/11.0
module load cudnn/8.0.3

nvidia-smi

# Fetch command on line $SLURM_ARRAY_TASK_ID from test file
EXE=`cat job-params-alpha-zero-rnd-parameter-sweep.txt | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`
# EXE=`cat job-params-alpha-zero-rnd-parameter-sweep.txt | head -n $SLURM_ARRAY_TASK_ID | tail -n 1`
`$EXE`
