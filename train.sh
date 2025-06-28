#!/bin/bash
#SBATCH --time=20:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 36 processor core(s) per node
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --partition=nova    # gpu node(s)

module load cuda

cd ../git-repos/3Di_Landscape
python run_vae.py INPUT_FASTA PATH_TO_SAVE_MODEL/saved_model.keras  
