#!/bin/bash

declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss" "RegLevinLoss")
#declare -a losses=("CrossEntropyLoss")
output="output_test_sokoban_large/"
domain_name="10x10-sokoban-"
problems_dir="problems/sokoban/test/000.txt"

heuristic_scheme=("--learned-heuristic" "")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic" "")
algorithm="Levin"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}"
			model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}

			#echo ${output_exp}
			#echo ${model}

			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir} run_bootstrap_test_sokoban.sh
		done
	done
done