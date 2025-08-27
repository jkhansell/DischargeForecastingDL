#!/bin/bash
#SBATCH --job-name="Train LSTM"
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --partition=nukwa-wide
#SBATCH --mail-type=NONE
#SBATCH --output=serghei-%j-%t.out

srun --exclusive mamba run -n OL python experiments.py lstm & 
srun --exclusive mamba run -n OL python experiments.py ltcn & 

wait
