#!/bin/bash

loss="CrossEntropyLoss"
algorithm="Levin"
domain_name="Witness"
problems_dir="problems/witness/puzzles_4x4/"
size_puzzle="4x4"
use_epsilons=("1")  # "0" "1"
output="output_test_witness_4x4/"
mkdir -p ${output}
echo "output folder = ${output}"


for use_eps in "${use_epsilons[@]}"; do
  output_exp="${output}${algorithm}-use_epsilon-${use_eps}-%N-%j.out"
  echo "output file = ${output_exp}"
  model=${size_puzzle}-${domain_name}-${loss}
  sbatch --output="${output_exp}" --export=algorithm=${algorithm},model=${model},problem=${problems_dir},loss=${loss},use_epsilon=${use_eps} submit_gpu_BFS.sh
  echo "finished calling sbatch"
done

