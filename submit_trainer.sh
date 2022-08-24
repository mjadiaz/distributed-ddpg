#!/bin/bash
#SBATCH --job-name=trainer
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=30

source /home/mjad1g20/.bashrc
source activate rrlib

python test.py


