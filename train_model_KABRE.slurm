#!/bin/bash
#SBATCH --job-name="Train LSTM"
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca,
#SBATCH --partition=nukwa-wide
#SBATCH --mail-type=NONE
#SBATCH --output=logs/TrainDL-%j-%t.out

CUDA_VISIBLE_DEVICES=0 srun -N2 mamba run -n OL python train.py $1
CUDA_VISIBLE_DEVICES=0 srun -N1 mamba run -n OL python test.py $1
