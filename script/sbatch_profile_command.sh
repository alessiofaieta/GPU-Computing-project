#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --partition=edu-short

#SBATCH --job-name=profile
#SBATCH --output=slurm-%j.out.txt
#SBATCH --error=slurm-%j.err.txt

sudo $CUDA_BIN/ncu -o profile $@

