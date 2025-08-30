#!/bin/bash
#SBATCH --job-name="Train LSTM"
#SBATCH --account=ehrtas
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --partition=booster
#SBATCH --mail-type=NONE
#SBATCH --output=logs/LSTMTrain-%j-%t.out

srun -N2 mamba run -n OL python experiments.py $1
