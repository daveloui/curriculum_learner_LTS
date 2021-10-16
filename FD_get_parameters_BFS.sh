#!/bin/bash

loss="CrossEntropyLoss"
algorithm="Levin"
domain_name="Witness"
problems_dir="problems/witness/puzzles_4x4/"
size_puzzle="4x4"
output="output_test_witness_4x4/"
mkdir -p ${output}
echo "output folder = ${output}"


output_exp="${output}${algorithm}-%N-%j.out"
echo "output file = ${output_exp}"
model=${size_puzzle}-${domain_name}-${loss}
sbatch --output="${output_exp}" --export=algorithm=${algorithm},model=${model},problem=${problems_dir},loss=${loss} FD_submit_experiment_BFS.sh
echo "finished calling sbatch"
