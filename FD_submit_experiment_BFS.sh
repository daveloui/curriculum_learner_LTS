#!/bin/bash
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham, 10 Beluga, 8 on Niagara.
#SBATCH --mem-per-cpu=4000M       # memory per node
#SBATCH --time=3-23:59      # time (DD-HH:MM) # maximum allocation time: 7 days on Beluga, 24 hrs on Niagara.
#SBATCH --account=def-mbowling

module load python/3.6
module load cuda cudnn
source ~/tensorflow/bin/activate
python src/main.py -a ${algorithm} -m ${model} -p ${problem} -l ${loss} -d Witness -b 1 -g 10 -parall 1 -scheduler online --learn
