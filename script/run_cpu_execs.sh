#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --partition=edu-medium

#SBATCH --job-name=sort-cpu
#SBATCH --output=slurm-%j.out.txt
#SBATCH --error=slurm-%j.err.txt


bin/bitonic_sort_cpu_measure0 11 23 10 csv/bitonic_sort_cpu_measure0.csv
bin/bitonic_sort_cpu_measure1 11 23 10 csv/bitonic_sort_cpu_measure1.csv
bin/bitonic_sort_cpu_measure2 11 23 10 csv/bitonic_sort_cpu_measure2.csv
bin/bitonic_sort_cpu_measure3 11 23 10 csv/bitonic_sort_cpu_measure3.csv

