#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --partition=edu-short

#SBATCH --job-name=sort
#SBATCH --output=slurm-%j.out.txt
#SBATCH --error=slurm-%j.err.txt

# $@    # This does not work if there are redirects in the command passed as input, `eval` is needed in that case
eval "$@"

