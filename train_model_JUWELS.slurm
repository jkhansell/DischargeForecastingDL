#!/bin/bash
#SBATCH --job-name="Train LSTM"
#SBATCH --account=ehrtas
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --partition=booster
#SBATCH --mail-type=NONE
#SBATCH --output=logs/LSTMTrain-%j-%t.out

srun --exclusive mamba run -N2 OL python experiments.py lstm
