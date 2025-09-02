#!/bin/bash
#SBATCH --job-name="Train"
#SBATCH --account=ehrtas
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --partition=booster
#SBATCH --mail-type=NONE
#SBATCH --output=logs/Train-%j-%t.out

srun -N2 mamba run -n OL python train.py $1
