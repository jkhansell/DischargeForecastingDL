#!/bin/bash
#SBATCH --job-name="Train"
#SBATCH --account=ehrtas
#SBATCH --time=00:25:00
#SBATCH --nodes=2
#SBATCH --partition=booster
#SBATCH --mail-type=NONE
#SBATCH --output=logs/Train-%j-%t.out


srun -N2 mamba run -n OL python train.py $1

sleep 5

CUDA_VISIBLE_DEVICES=0 srun -N1 -n1 mamba run -n OL python test.py $1
