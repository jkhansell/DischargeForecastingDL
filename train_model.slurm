#!/bin/bash
#SBATCH --job-name="Train LSTM"
#SBATCH --account=ehrtas
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --partition=develbooster
#SBATCH --mail-type=NONE
#SBATCH --output=serghei-%j-%t.out

mamba run -n OL python experiments.py