#!/bin/bash


algorithm="bootstrap_dfs_lvn_learning_planner"
domain_name="Witness"
problems_dir="problems/witness/puzzles_4x4"
size_puzzle="4x4"
output="puzzles_4x4_output/"
dropout_rate="1.0"
batch_size="1024"

output_exp="${output}${algorithm}"
sbatch --output="${output_exp}" --export=algorithm=${algorithm},problem=${problems_dir},model=${output},dropout_rate=${dropout_rate},batch_size=${batch_size} FD_submit_experiment_DepthFS.sh
echo "finished calling sbatch"
