#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --partition=edu-medium

#SBATCH --job-name=sort-gpu
#SBATCH --output=slurm-%j.out.txt
#SBATCH --error=slurm-%j.err.txt


bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 A csv/global_memory_step_by_step.csv
bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 B csv/global_memory_grouped_steps.csv
bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 C csv/shared_memory_grouped_steps.csv
bin/bitonic_sort_gpu_measure 11 26 5 10 0 4 10 D csv/shared_memory_grouped_steps_warp.csv

