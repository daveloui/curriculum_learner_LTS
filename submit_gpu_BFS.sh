#!/bin/bash
#SBATCH --nodes=1  # allocate 1 node for each job
#SBATCH --ntasks-per-node=1  # because each job has a single task
#SBATCH --gres=gpu:1   # Number of GPUs (per node)
#SBATCH --cpus-per-task=16  # maximum CPU cores per GPU request
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-23:59
#SBATCH --account=def-mbowling


module load python/3.6
module load StdEnv/2020
module load cuda/11.0
module load cudnn/8.0.3
source ~/tensorGPU/bin/activate

nvidia-smi

python src/main_no_debug_data.py -a ${algorithm} -l ${loss} -m ${model} -p ${problem} -d Witness -b 2000 -g 10 -scheduler online -use_epsilon ${use_epsilon} --learn