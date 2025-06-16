#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --partition=edu-medium

#SBATCH --job-name=sort-lib
#SBATCH --output=slurm-%j.out.txt
#SBATCH --error=slurm-%j.err.txt


script/bitonic_sort_lib_measure.sh 11 23 5 bin/thrust_sort > csv/thrust_sort.csv
script/bitonic_sort_lib_measure.sh 11 30 5 bin/cub_merge_sort > csv/cub_merge_sort.csv
script/bitonic_sort_lib_measure.sh 11 30 5 bin/cub_radix_sort > csv/cub_radix_sort.csv

