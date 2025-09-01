#!/bin/bash
#SBATCH --job-name="Train LSTM"
#SBATCH --time=01:00:00
#SBATCH --nodes=3
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca,
#SBATCH --partition=nukwa-wide
#SBATCH --mail-type=NONE
#SBATCH --output=serghei-%j-%t.out

srun --exclusive mamba run -n OL python experiments.py lstm & 
srun --exclusive mamba run -n OL python experiments.py ltcn & 
srun --exclusive mamba run -n OL python experiments.py lstm & 

wait
