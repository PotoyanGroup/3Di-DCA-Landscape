#!/bin/bash
#SBATCH --time=10:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=nova    # gpu node(s)

module load cuda
python PATH_TO_TRANSLATE.PY -i INPUT_FASTA_PATH -o OUTPUT_FASTA_PATH